"""FastAPI HTTP layer for the micro-batch image classifier.

Endpoints
---------
POST /predict                 submit an image + deadline_ms (from acceptance)
GET  /orders/{order_id}       query status / result by order id
POST /orders/{order_id}/cancel  cancel a queued or in-flight order
POST /calibration             fit a confidence-calibration temperature
GET  /stats                   capacity and state counters (demo/observability)
GET  /health                  liveness
Interactive docs are served at /docs (Swagger UI).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import calibration as calib
from .config import Settings
from .engine import (
    COMPLETED,
    BatchEngine,
    CalibrationFitError,
    CalibrationInProgress,
    CalibrationOrderError,
    CalibrationUnknownOrder,
    CalibrationValidationError,
    Order,
)
from .preprocessing import preprocess


class CalibrationSample(BaseModel):
    order_id: str = Field(..., min_length=1)
    label: int = Field(..., ge=0, le=9, description="ground-truth class 0..9")


class CalibrationRequest(BaseModel):
    # 8..32 mutually distinct completed orders, each with a 0..9 true label.
    samples: list[CalibrationSample] = Field(
        ...,
        min_length=calib.MIN_SAMPLES,
        max_length=calib.MAX_SAMPLES,
    )


def order_view(o: Order) -> dict:
    remaining_ms = max(0, round((o.deadline_at - time.monotonic()) * 1000))
    return {
        "order_id": o.id,
        "status": o.state,
        "deadline_ms": o.deadline_ms,
        "remaining_ms": remaining_ms,
        "batch_id": o.batch_id,
        # Temperature applied to this order's receipt: captured at acceptance,
        # so later recalibrations never change an existing order.
        "temperature": o.temperature,
        "result": o.result if o.state == COMPLETED else None,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = BatchEngine(
        model_path=settings.model_path,
        max_capacity=settings.max_capacity,
        batch_size=settings.batch_size,
        batch_wait_ms=settings.batch_wait_ms,
        inference_delay_ms=settings.inference_delay_ms,
        initial_temperature=settings.initial_temperature,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(
        title="Micro-batch Image Classifier",
        version="1.1.0",
        description=(
            "Time-limited image classification for multiple cameras. "
            "Single-process micro-batching: at most 16 queued/in-flight images, "
            "batches of up to 4, idle model waits up to 50 ms to fill a batch, "
            "one batch infers at a time. Cancelled/expired orders never yield results. "
            "POST /calibration fits a temperature in [0.25, 4] on 8..32 verified "
            "completed orders; only newly accepted orders use the new temperature."
        ),
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.settings = settings

    # Request-body problems are bad requests (consistent with the 400 used
    # elsewhere in this API rather than FastAPI's default 422).
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.post("/predict", status_code=201)
    async def predict(file: UploadFile = File(...), deadline_ms: int = Form(...)):
        """Submit an image with a deadline in milliseconds counted from acceptance."""
        if deadline_ms <= 0:
            raise HTTPException(400, "deadline_ms must be a positive integer")
        data = await file.read()
        try:
            tensor = preprocess(data)
        except ValueError as exc:
            raise HTTPException(400, f"invalid image: {exc}") from exc
        order = engine.submit(tensor, deadline_ms)
        if order is None:
            raise HTTPException(
                429,
                f"capacity full: at most {engine.max_capacity} images queued or in-flight",
            )
        return order_view(order)

    @app.get("/orders/{order_id}")
    async def get_order(order_id: str):
        order = engine.get(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        return order_view(order)

    @app.post("/orders/{order_id}/cancel")
    async def cancel_order(order_id: str):
        order = engine.cancel(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        if order.state == COMPLETED:
            raise HTTPException(409, "order already completed; cannot cancel")
        return order_view(order)

    @app.post("/calibration")
    async def calibrate(payload: CalibrationRequest):
        """Fit a temperature on verified completed orders.

        The candidate temperature (clipped to [0.25, 4]) replaces the current
        one only when its mean NLL is strictly lower. Fitting always uses the
        orders' uncalibrated original-model scores.
        """
        samples = [(s.order_id, s.label) for s in payload.samples]
        try:
            return await engine.calibrate(samples)
        except CalibrationInProgress as exc:
            raise HTTPException(409, str(exc)) from exc
        except CalibrationUnknownOrder as exc:
            raise HTTPException(404, str(exc)) from exc
        except CalibrationOrderError as exc:
            raise HTTPException(409, str(exc)) from exc
        except CalibrationValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        except CalibrationFitError as exc:
            # Temperature untouched; inference and queries keep working.
            raise HTTPException(500, str(exc)) from exc

    @app.get("/stats")
    async def stats():
        return engine.stats()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
