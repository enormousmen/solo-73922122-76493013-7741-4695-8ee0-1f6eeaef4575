"""Tests for the temperature-calibration feature.

Covers
------
* unit-level: NLL math, bounds, grid optimizer vs brute-force scan
* API-level: happy path, no-improvement retention, raw-score reuse,
  validation/status mapping, overlapping 409, fit failure keeps T=1,
  acceptance-time temperature snapshot for queued/in-flight/completed orders,
  calibration not blocking inference/query.
"""
from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import httpx
import pytest
import torch

from app import calibration as calib
from app.engine import BatchEngine
from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MODEL = Path(__file__).resolve().parent.parent / "app" / "model.pt"
SAMPLE_NAMES = ["sample_gradient.png", "sample_circles.png", "sample_stripes.png", "sample_noise.png"]


def make_settings(**kw) -> Settings:
    kw.setdefault("model_path", str(MODEL))
    return Settings(**kw)


class Client:
    def __init__(self, http: httpx.AsyncClient, app):
        self.http = http
        self.app = app

    async def submit(self, name: str, deadline_ms: int = 60_000):
        data = (SAMPLES / name).read_bytes()
        return await self.http.post(
            "/predict",
            files={"file": (name, data, "image/png")},
            data={"deadline_ms": str(deadline_ms)},
        )

    async def get(self, order_id: str):
        return await self.http.get(f"/orders/{order_id}")

    async def get_json(self, order_id: str):
        return (await self.get(order_id)).json()

    async def wait_state(self, order_id: str, states, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = await self.get_json(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")

    async def calibrate(self, payload):
        return await self.http.post("/calibration", json=payload)

    async def complete_n(self, n: int):
        """Submit n images and wait for completion; return list of order views."""
        responses = await asyncio.gather(
            *[self.submit(SAMPLE_NAMES[i % 4]) for i in range(n)]
        )
        ids = [r.json()["order_id"] for r in responses]
        bodies = await asyncio.gather(
            *[self.wait_state(i, {"completed"}) for i in ids]
        )
        return list(bodies)


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        app = create_app(make_settings(**settings_kw))
        from asgi_lifespan import LifespanManager

        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        client = Client(http, app)
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


def _calib_payload(bodies, labels):
    return {
        "samples": [
            {"order_id": b["order_id"], "label": y}
            for b, y in zip(bodies, labels)
        ]
    }


# --------------------------------------------------------------------- #
# calibration.py unit tests (no network / model)
# --------------------------------------------------------------------- #
def test_softmax_normalizes_and_scaling_keeps_argmax():
    logits = [1.0, 2.0, 3.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for t in (0.25, 1.0, 4.0):
        p = calib.softmax(logits, t)
        assert len(p) == 10
        assert abs(sum(p) - 1.0) < 1e-12
        assert all(x > 0 for x in p)
        assert max(range(10), key=lambda i: p[i]) == 2  # argmax invariant
    sharp = calib.softmax(logits, 0.25)
    flat = calib.softmax(logits, 4.0)
    assert sharp[2] > flat[2]  # lower temperature => sharper peak


def test_mean_nll_matches_manual_softmax_formula():
    torch.manual_seed(7)
    raw = torch.softmax(torch.randn(12, 10), dim=1).tolist()
    labels = torch.randint(0, 10, (12,)).tolist()
    t = 1.7
    expected = -sum(
        math.log(calib.softmax([math.log(p) for p in row], t)[y])
        for row, y in zip(raw, labels)
    ) / len(raw)
    assert calib.mean_nll(t, raw, labels) == pytest.approx(expected, rel=1e-10)


def test_fit_temperature_bounds_and_bruteforce_agreement():
    torch.manual_seed(23)
    # Over-confident logits with noisy labels benefit from T > 1.
    logits = torch.randn(20, 10) * 5.0
    raw = torch.softmax(logits, dim=1).tolist()
    labels = (torch.argmax(logits, dim=1) + torch.randint(0, 3, (20,))).remainder(10).tolist()

    t_star, nll_star = calib.fit_temperature(raw, labels)
    assert calib.TEMP_MIN <= t_star <= calib.TEMP_MAX

    scan = torch.linspace(calib.TEMP_MIN, calib.TEMP_MAX, 4001, dtype=torch.float64)
    log_scores = torch.log(torch.tensor(raw, dtype=torch.float64))
    labels_t = torch.tensor(labels)
    nlls = [
        float(
            -torch.log_softmax(log_scores / float(t), dim=1)
            .gather(1, labels_t.unsqueeze(1))
            .mean()
        )
        for t in scan
    ]
    brute_t = float(scan[int(torch.tensor(nlls).argmin())])
    brute_nll = min(nlls)
    assert t_star == pytest.approx(brute_t, abs=1e-2)
    assert nll_star <= brute_nll + 1e-8


def test_fit_clamps_to_bounds_when_optimum_outside():
    # Fully separable labels => NLL monotonically decreasing toward T_min.
    raw = []
    labels = []
    for i in range(10):
        row = [1e-6] * 10
        row[i] = 1.0 - 9 * 1e-6
        raw.append(row)
        labels.append(i)
    t_star, nll_star = calib.fit_temperature(raw, labels)
    assert t_star == pytest.approx(calib.TEMP_MIN, abs=1e-4)
    assert nll_star == pytest.approx(calib.mean_nll(calib.TEMP_MIN, raw, labels), abs=1e-10)


def test_fit_rejects_bad_inputs():
    good = torch.softmax(torch.randn(8, 10), dim=1).tolist()
    bad = [[1.0] + [0.0] * 9 for _ in range(8)]  # zero probability -> invalid
    with pytest.raises(ValueError):
        calib.fit_temperature(bad, list(range(8)))
    with pytest.raises(ValueError):
        calib.mean_nll(-1.0, good, list(range(8)))
    with pytest.raises(ValueError):
        calib.fit_temperature(good, list(range(7)) + [10])  # label out of range
    with pytest.raises(ValueError):
        calib.softmax([1.0, 2.0], float("nan"))


# --------------------------------------------------------------------- #
# API: happy path and no-improvement retention
# --------------------------------------------------------------------- #
async def test_calibration_applies_improving_temperature(make_client):
    c = await make_client()
    bodies = await c.complete_n(10)
    engine: BatchEngine = c.app.state.engine
    assert engine.temperature == 1.0

    # Misaligned labels make a higher temperature strictly better.
    labels = [(b["result"]["predicted_class"] + 3) % 10 for b in bodies]
    r = await c.calibrate(_calib_payload(bodies, labels))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 10
    assert body["candidate_loss"] <= body["current_loss"] + 1e-9
    if body["applied"]:
        assert body["candidate_loss"] < body["current_loss"]
        assert body["temperature"] == body["candidate_temperature"]
        assert engine.temperature == body["candidate_temperature"]
    else:
        # Candidate cannot improve -> current value retained.
        assert body["temperature"] == engine.temperature == 1.0
        assert body["candidate_loss"] >= body["current_loss"] - 1e-9
    stats = (await c.http.get("/stats")).json()
    assert stats["temperature"] == engine.temperature
    assert stats["calibration_in_progress"] is False


async def test_no_improvement_keeps_current_temperature(make_client):
    c = await make_client()
    bodies = await c.complete_n(8)
    # Labels matching argmax on easy samples: T=1 (or T_min) is already best;
    # first calibrate at T_min so the current value equals the optimum -> no move.
    labels = [b["result"]["predicted_class"] for b in bodies]
    r1 = await c.calibrate(_calib_payload(bodies, labels))
    assert r1.status_code == 200
    first = r1.json()
    t_min = calib.TEMP_MIN
    if first["applied"]:
        assert first["temperature"] == pytest.approx(t_min, abs=1e-3)
    # Re-fit on the exact same data: candidate cannot improve on the optimum.
    r2 = await c.calibrate(_calib_payload(bodies, labels))
    second = r2.json()
    assert r2.status_code == 200
    assert second["applied"] is False
    assert second["temperature"] == first["temperature"]
    assert second["candidate_loss"] >= second["current_loss"] - 1e-9
    assert c.app.state.engine.temperature == first["temperature"]


# --------------------------------------------------------------------- #
# uncalibrated scores are always used, even after prior calibration
# --------------------------------------------------------------------- #
async def test_calibration_always_uses_uncalibrated_scores(make_client):
    c = await make_client()
    bodies = await c.complete_n(12)
    engine: BatchEngine = c.app.state.engine
    labels = [(b["result"]["predicted_class"] + 1) % 10 for b in bodies]
    raw_snapshot = [engine.get(b["order_id"]).raw_scores for b in bodies]

    r1 = await c.calibrate(_calib_payload(bodies, labels))
    body1 = r1.json()
    t_after_first = engine.temperature

    # Same orders reused as samples: raw scores unchanged and the second fit
    # on the same raw data returns the identical candidate.
    raw_snapshot_2 = [engine.get(b["order_id"]).raw_scores for b in bodies]
    assert raw_snapshot_2 == raw_snapshot
    r2 = await c.calibrate(_calib_payload(bodies, labels))
    body2 = r2.json()
    assert body2["candidate_temperature"] == pytest.approx(
        body1["candidate_temperature"], abs=1e-9
    )
    assert body2["candidate_loss"] == pytest.approx(
        body1["candidate_loss"], abs=1e-9
    )
    # Whatever the first decision, the second fit cannot move the temperature.
    assert body2["applied"] is False
    assert engine.temperature == t_after_first == body2["temperature"]

    # Existing receipts were never rewritten.
    for b in bodies:
        again = await c.get_json(b["order_id"])
        assert again["result"]["scores"] == b["result"]["scores"]
        assert again["temperature"] == 1.0  # accepted before any calibration


async def test_new_orders_use_new_temperature_and_scores_match_formula(make_client):
    c = await make_client()
    bodies = await c.complete_n(8)
    labels = [(b["result"]["predicted_class"] + 4) % 10 for b in bodies]
    r = await c.calibrate(_calib_payload(bodies, labels))
    t_effective = r.json()["temperature"]
    assert t_effective == c.app.state.engine.temperature
    # Shifted labels on a confident model should sharpen/soften away from T=1.
    assert t_effective != 1.0

    new_bodies = await c.complete_n(4)
    for b in new_bodies:
        assert b["temperature"] == pytest.approx(t_effective)
        order = c.app.state.engine.get(b["order_id"])
        expected = calib.softmax(order.logits, t_effective)
        assert b["result"]["scores"] == pytest.approx(expected, abs=1e-12)
        # raw_scores stay the T=1 model output
        assert order.raw_scores == pytest.approx(
            calib.softmax(order.logits, 1.0), abs=1e-12
        )


# --------------------------------------------------------------------- #
# validation / status codes
# --------------------------------------------------------------------- #
async def test_calibration_validation_errors(make_client):
    c = await make_client(inference_delay_ms=1000)
    bodies = await c.complete_n(10)
    labels = [0] * 10
    good = _calib_payload(bodies, labels)

    # too few / too many
    few = {"samples": good["samples"][:7]}
    assert (await c.calibrate(few)).status_code == 400
    many = {"samples": good["samples"] * 3 + good["samples"][:3]}  # 33 items
    assert len(many["samples"]) == 33
    assert (await c.calibrate(many)).status_code == 400

    # duplicate order ids
    dup = {"samples": good["samples"][:8] + [good["samples"][0]]}
    r = await c.calibrate(dup)
    assert r.status_code == 400 and "distinct" in r.text

    # label out of range (schema level)
    bad_label = {
        "samples": [
            {**s, "label": 10 if i == 0 else s["label"]}
            for i, s in enumerate(good["samples"][:8])
        ]
    }
    assert (await c.calibrate(bad_label)).status_code == 400

    # unknown order id -> 404
    unknown = {
        "samples": [
            {"order_id": "deadbeef00", "label": 1}
        ] + good["samples"][:7]
    }
    assert (await c.calibrate(unknown)).status_code == 404

    # known but not completed -> 409
    pending = await c.submit("sample_gradient.png", deadline_ms=60_000)
    pid = pending.json()["order_id"]
    not_done = {
        "samples": [{"order_id": pid, "label": 0}] + good["samples"][:7]
    }
    assert (await c.calibrate(not_done)).status_code == 409


# --------------------------------------------------------------------- #
# overlapping calibrations -> 409
# --------------------------------------------------------------------- #
async def test_overlapping_calibration_returns_409(make_client, monkeypatch):
    c = await make_client()
    bodies = await c.complete_n(8)
    payload = _calib_payload(bodies, [1] * 8)

    entered = asyncio.Event()
    release = asyncio.Event()
    real_fit = calib.fit_temperature
    loop = asyncio.get_running_loop()

    def patched_fit(raw, labels):
        loop.call_soon_threadsafe(entered.set)
        # Block the calibration thread until the test releases it.
        for _ in range(200):
            if release.is_set():
                break
            time.sleep(0.01)
        return real_fit(raw, labels)

    monkeypatch.setattr(calib, "fit_temperature", patched_fit)
    first = asyncio.create_task(c.calibrate(payload))
    await entered.wait()
    second = await c.calibrate(payload)
    assert second.status_code == 409
    assert "already running" in second.text
    release.set()
    first_body = (await first).json()
    assert "candidate_temperature" in first_body
    monkeypatch.undo()


# --------------------------------------------------------------------- #
# fit failure keeps the current temperature; service stays responsive
# --------------------------------------------------------------------- #
async def test_fit_failure_keeps_temperature_and_service_alive(make_client, monkeypatch):
    c = await make_client()
    bodies = await c.complete_n(8)
    payload = _calib_payload(bodies, [2] * 8)

    def boom(raw, labels):
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(calib, "fit_temperature", boom)
    r = await c.calibrate(payload)
    assert r.status_code == 500
    assert c.app.state.engine.temperature == 1.0
    stats = (await c.http.get("/stats")).json()
    assert stats["calibration_in_progress"] is False

    # inference and queries still respond
    more = await c.complete_n(2)
    assert more[0]["status"] == "completed"
    assert (await c.get(bodies[0]["order_id"])).status_code == 200


# --------------------------------------------------------------------- #
# key interleaving: calibration lands while orders are queued / computing
# --------------------------------------------------------------------- #
async def test_acceptance_temperature_snapshot_during_calibration(make_client):
    c = await make_client(inference_delay_ms=1200, batch_wait_ms=50)
    # 8 completed calibration samples first.
    cal_bodies = await c.complete_n(8)
    labels = [(b["result"]["predicted_class"] + 2) % 10 for b in cal_bodies]
    payload = _calib_payload(cal_bodies, labels)

    # Occupy the model with one batch; two extra orders queue behind it.
    inflight = await asyncio.gather(*[c.submit(SAMPLE_NAMES[i]) for i in range(4)])
    await asyncio.sleep(0.2)  # first batch computing
    queued = await asyncio.gather(*[c.submit(SAMPLE_NAMES[i]) for i in range(2)])
    inflight_ids = [r.json()["order_id"] for r in inflight]
    queued_ids = [r.json()["order_id"] for r in queued]
    for oid in inflight_ids + queued_ids:
        assert (await c.get_json(oid))["temperature"] == 1.0

    # Calibrate while the old orders are still queued/computing.
    r = await c.calibrate(payload)
    assert r.status_code == 200, r.text
    t_effective = r.json()["temperature"]
    assert t_effective == c.app.state.engine.temperature

    # Old orders keep the acceptance-time temperature (1.0) in their receipts.
    old_bodies = await asyncio.gather(
        *[c.wait_state(i, {"completed", "expired", "cancelled"}) for i in inflight_ids + queued_ids]
    )
    for b in old_bodies:
        assert b["status"] == "completed"
        assert b["temperature"] == 1.0
        order = c.app.state.engine.get(b["order_id"])
        assert b["result"]["scores"] == pytest.approx(
            calib.softmax(order.logits, 1.0), abs=1e-12
        )

    # Orders accepted after calibration use the (possibly new) temperature.
    fresh = await c.complete_n(3)
    for b in fresh:
        assert b["temperature"] == pytest.approx(t_effective)
        order = c.app.state.engine.get(b["order_id"])
        assert b["result"]["scores"] == pytest.approx(
            calib.softmax(order.logits, t_effective), abs=1e-12
        )


async def test_calibration_does_not_block_inference_api(make_client, monkeypatch):
    c = await make_client(inference_delay_ms=500)
    bodies = await c.complete_n(8)
    payload = _calib_payload(bodies, [0] * 8)

    entered = asyncio.Event()
    release = asyncio.Event()
    real_fit = calib.fit_temperature
    loop = asyncio.get_running_loop()

    def slow_fit(raw, labels):
        loop.call_soon_threadsafe(entered.set)
        for _ in range(200):
            if release.is_set():
                break
            time.sleep(0.01)
        return real_fit(raw, labels)

    monkeypatch.setattr(calib, "fit_temperature", slow_fit)
    task = asyncio.create_task(c.calibrate(payload))
    await entered.wait()

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    # While fitting runs, submit/query/stats stay responsive.
    r1, dt1 = await timed(c.submit("sample_gradient.png"))
    r2, dt2 = await timed(c.http.get("/stats"))
    r3, dt3 = await timed(c.get(bodies[0]["order_id"]))
    for dt in (dt1, dt2, dt3):
        assert dt < 0.5, f"API blocked for {dt:.3f}s during calibration"
    assert r1.status_code == 201 and r2.status_code == 200 and r3.status_code == 200
    release.set()
    assert (await task).status_code == 200
    monkeypatch.undo()
