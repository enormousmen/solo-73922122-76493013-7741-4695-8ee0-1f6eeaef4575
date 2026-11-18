"""Single-process micro-batch inference engine.

Concurrency model
-----------------
* Everything except the blocking model forward pass runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch is ever actually
  inferring at any moment.
* Capacity accounting: an order occupies a slot while it is queued AND while
  its batch is computing (even if it was cancelled or expired mid-flight).
  The slot is released only when the batch compute actually finishes.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held)
processing -> expired          (deadline passed mid-flight; result discarded)
processing -> failed           (model raised; defensive only)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from . import calibration as calib

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"


class CalibrationError(Exception):
    """Base class for calibration request failures."""


class CalibrationValidationError(CalibrationError):
    """Request content is invalid (bad count, duplicates, labels ...)."""


class CalibrationOrderError(CalibrationError):
    """A referenced order is known but not usable for calibration."""


class CalibrationUnknownOrder(CalibrationError):
    """A referenced order_id does not exist (maps to 404)."""


class CalibrationInProgress(CalibrationError):
    """Another calibration request is currently being fitted."""


class CalibrationFitError(CalibrationError):
    """The numerical fit failed; the current temperature is retained."""


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    tensor: torch.Tensor
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    # Temperature captured at acceptance. Queued/processing orders keep using
    # it even if a calibration changes the global temperature mid-flight.
    temperature: float = 1.0
    # Filled on completion: the original model output and its softmax(T=1).
    # Calibration always reads these *uncalibrated* scores, even when the same
    # order is submitted as a calibration sample more than once.
    logits: Optional[list[float]] = None
    raw_scores: Optional[list[float]] = None


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
        initial_temperature: float = 1.0,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0

        # Global confidence-calibration temperature. Only updated on the event
        # loop thread by `calibrate`; read when an order is accepted.
        t0 = float(initial_temperature)
        if not (calib.TEMP_MIN <= t0 <= calib.TEMP_MAX):
            raise ValueError(
                f"initial_temperature must be within [{calib.TEMP_MIN}, {calib.TEMP_MAX}]"
            )
        self.temperature = t0
        self._calibration_in_progress = False

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        self._orders: dict[str, Order] = {}
        self._inflight = 0  # orders inside the batch currently being computed
        self._not_empty = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        # Separate single-thread executor: temperature fitting must never block
        # (or queue behind) model inference, and vice versa.
        self._calib_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="calibrate"
        )
        self._worker: Optional[asyncio.Task] = None
        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)
        self._calib_executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Mark queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        if o.state in (QUEUED, PROCESSING) and o.deadline_at <= time.monotonic():
            if o.state == QUEUED:
                try:
                    self._pending.remove(o)
                except ValueError:
                    pass
            o.state = EXPIRED

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return len(self._pending) + self._inflight

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        for o in self._orders.values():
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "capacity_used": len(self._pending) + self._inflight,
            "max_capacity": self.max_capacity,
            "queued": len(self._pending),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "temperature": self.temperature,
            "calibration_in_progress": self._calibration_in_progress,
            "orders": counts,
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(self, tensor: torch.Tensor, deadline_ms: int) -> Optional[Order]:
        """Accept an image, or return None when at capacity (16 by default)."""
        if self.capacity_used() >= self.max_capacity:
            return None
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
            # Snapshot the current temperature: later recalibration must not
            # change the score this order will be scored with.
            temperature=self.temperature,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self._not_empty.set()
        return order

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)
        return o

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            return None
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the batch
            # compute actually ends; no new batch is started because of it.
            o.state = CANCELLED
        return o

    # ------------------------------------------------------------------ #
    # confidence calibration (all checks run on the event-loop thread)
    # ------------------------------------------------------------------ #
    def _build_calibration_data(
        self, samples: Sequence[tuple[str, int]]
    ) -> tuple[list[list[float]], list[int]]:
        if not (calib.MIN_SAMPLES <= len(samples) <= calib.MAX_SAMPLES):
            raise CalibrationValidationError(
                f"expected between {calib.MIN_SAMPLES} and {calib.MAX_SAMPLES} "
                f"samples, got {len(samples)}"
            )
        ids = [s[0] for s in samples]
        if len(set(ids)) != len(ids):
            raise CalibrationValidationError("order_id values must be distinct")
        raw_scores: list[list[float]] = []
        labels: list[int] = []
        for order_id, label in samples:
            if not isinstance(label, int) or not (0 <= label <= 9):
                raise CalibrationValidationError(
                    f"label for {order_id} must be an integer in [0, 9]"
                )
            o = self._orders.get(order_id)
            if o is None:
                raise CalibrationUnknownOrder(f"unknown order_id: {order_id}")
            if o.state != COMPLETED or o.raw_scores is None:
                raise CalibrationOrderError(
                    f"order {order_id} is not completed (state={o.state}); "
                    "only completed orders can calibrate"
                )
            # Always the uncalibrated original-model scores, even if this
            # order already served as a calibration sample before.
            raw_scores.append(o.raw_scores)
            labels.append(label)
        return raw_scores, labels

    async def calibrate(
        self, samples: Sequence[tuple[str, int]]
    ) -> dict:
        """Fit a candidate temperature and keep it only when NLL improves.

        Overlapping calls fail with ``CalibrationInProgress``; fit failures
        leave the current temperature untouched.
        """
        if self._calibration_in_progress:
            raise CalibrationInProgress("a calibration request is already running")
        self._calibration_in_progress = True
        try:
            # Content validation while the overlap flag is held, so a second
            # request racing an invalid one still observes 409.
            raw_scores, labels = self._build_calibration_data(samples)
            current_temp = self.temperature
            loop = asyncio.get_running_loop()

            def _fit() -> tuple[float, float, float, float]:
                # Runs on the dedicated calibration thread. Both losses use
                # the same uncalibrated scores and the same sample set.
                cur_loss = calib.mean_nll(current_temp, raw_scores, labels)
                cand_temp, cand_loss = calib.fit_temperature(raw_scores, labels)
                return cur_loss, cand_temp, cand_loss

            try:
                current_loss, candidate_temp, candidate_loss = await loop.run_in_executor(
                    self._calib_executor, _fit
                )
            except Exception as exc:  # noqa: BLE001 - fit failure keeps old T
                raise CalibrationFitError(f"temperature fitting failed: {exc}") from exc

            if not (
                math.isfinite(candidate_temp)
                and math.isfinite(candidate_loss)
                and calib.TEMP_MIN <= candidate_temp <= calib.TEMP_MAX
            ):
                raise CalibrationFitError("fitted temperature/loss is invalid")

            applied = current_loss - candidate_loss > calib.IMPROVEMENT_EPS
            if applied:
                # Only affects orders accepted from this point on; existing
                # orders carry their own captured temperature.
                self.temperature = candidate_temp
            effective_temp = candidate_temp if applied else current_temp
            return {
                "current_temperature": current_temp,
                "current_loss": current_loss,
                "candidate_temperature": candidate_temp,
                "candidate_loss": candidate_loss,
                "applied": applied,
                "temperature": effective_temp,
                "sample_count": len(samples),
            }
        finally:
            self._calibration_in_progress = False

    # ------------------------------------------------------------------ #
    # batching worker
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._not_empty.wait()

            # Batch-formation window: the idle model waits at most
            # `batch_wait_s` for more work, up to `batch_size` images.
            batch: list[Order] = []
            window_end = loop.time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                self._purge_expired_pending()
                while self._pending and len(batch) < self.batch_size:
                    batch.append(self._pending.popleft())
                if len(batch) >= self.batch_size:
                    break
                remaining = window_end - loop.time()
                if remaining <= 0:
                    break
                self._not_empty.clear()
                try:
                    await asyncio.wait_for(self._not_empty.wait(), remaining)
                except asyncio.TimeoutError:
                    break

            # Final gate: cancelled/expired orders never enter the model.
            now = time.monotonic()
            batch = [o for o in batch if o.state == QUEUED and o.deadline_at > now]
            if not batch:
                continue

            batch_id = uuid.uuid4().hex[:8]
            for o in batch:
                o.state = PROCESSING
                o.batch_id = batch_id
            self._inflight = len(batch)
            self.batches_started += 1
            self.last_batch_size = len(batch)
            tensors = torch.stack([o.tensor for o in batch])

            try:
                logits_rows = await loop.run_in_executor(
                    self._executor, self._infer, tensors
                )
            except Exception:  # noqa: BLE001 - defensive: model failure must not kill the worker
                log.exception("inference failed for batch %s", batch_id)
                for o in batch:
                    if o.state == PROCESSING:
                        o.state = FAILED
                self._inflight = 0
                continue

            now = time.monotonic()
            for o, row in zip(batch, logits_rows):
                if o.state != PROCESSING:
                    # Cancelled or expired mid-flight: result is discarded and
                    # the late result must NOT revive the order.
                    continue
                if o.deadline_at <= now:
                    o.state = EXPIRED
                    continue
                o.state = COMPLETED
                o.logits = list(row)
                # Uncalibrated original-model scores (T = 1). Kept verbatim so
                # later calibration always fits against the raw model output.
                o.raw_scores = calib.softmax(row, 1.0)
                # The receipt is scaled with the temperature captured at
                # acceptance; later recalibrations never rewrite it.
                scores = calib.softmax(row, o.temperature)
                o.result = {
                    "predicted_class": max(range(len(row)), key=lambda i: row[i]),
                    "scores": scores,
                }
            self._inflight = 0

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — never touches shared state
    # ------------------------------------------------------------------ #
    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        with torch.inference_mode():
            # Return raw logits: the event-loop thread derives both the
            # uncalibrated (T=1) and temperature-scaled probabilities.
            logits = self.model(batch_tensor)
        return [[float(v) for v in row] for row in logits]
