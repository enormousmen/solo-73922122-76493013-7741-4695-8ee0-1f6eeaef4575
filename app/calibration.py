"""Post-hoc confidence calibration via scalar temperature scaling.

Input
-----
The *uncalibrated* model scores ``p_i`` (the 10-class softmax probabilities
emitted by the original model, i.e. temperature = 1) of human-verified orders,
together with their ground-truth labels ``y_i``.

Model
-----
A single scalar temperature ``T`` scales the scores before softmax::

    q_i(T) = softmax( log(p_i) / T )

Using ``log(p_i)`` instead of the network's raw logits is exact: the logits
and ``log(p)`` differ only by the log-partition constant
``log sum exp(logits)``, which softmax cancels. The mean negative
log-likelihood is therefore::

    NLL(T) = -mean_i( log q_i(T)[y_i] )

``NLL(T)`` is convex in ``T`` (it is a log-sum-exp of affine functions of
``1/T``), so a log-space grid plus golden-section refinement finds the global
minimum on the closed interval [TEMP_MIN, TEMP_MAX].
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

# Spec-mandated bounds and sample-size limits.
TEMP_MIN = 0.25
TEMP_MAX = 4.0
MIN_SAMPLES = 8
MAX_SAMPLES = 32

# A candidate replaces the current temperature only when the measured NLL
# improvement exceeds this tolerance (guards against float64 noise).
IMPROVEMENT_EPS = 1e-12

_GRID_POINTS = 101          # log-spaced grid over [TEMP_MIN, TEMP_MAX]
_GOLDEN_TOL = 1e-5          # final bracket width for the scalar temperature
_EPS = 1e-12                # floor before taking log of a probability


def softmax(logits: Sequence[float], temperature: float) -> list[float]:
    """Numerically stable softmax of ``logits / temperature`` (pure Python).

    Kept off the Torch executor so the event-loop thread can build per-order
    receipts without crossing thread boundaries.
    """
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("temperature must be a finite positive number")
    scaled = [float(x) / t for x in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _prepare(
    raw_scores: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate inputs and return (log uncalibrated probs, labels) in float64."""
    n = len(raw_scores)
    if n == 0 or n != len(labels):
        raise ValueError("raw_scores and labels must be non-empty and of equal length")
    probs = torch.tensor(list(raw_scores), dtype=torch.float64)
    if probs.ndim != 2:
        raise ValueError("raw_scores must be a 2-D matrix of probability vectors")
    if torch.any(~torch.isfinite(probs)) or torch.any(probs <= 0.0):
        raise ValueError("raw_scores must contain finite positive probabilities")
    labels_t = torch.tensor(list(labels), dtype=torch.long)
    if labels_t.ndim != 1 or labels_t.shape[0] != probs.shape[0]:
        raise ValueError("labels must be one integer per score vector")
    num_classes = probs.shape[1]
    if torch.any(labels_t < 0) or torch.any(labels_t >= num_classes):
        raise ValueError(f"labels must be integers in [0, {num_classes - 1}]")
    return torch.log(torch.clamp(probs, min=_EPS)), labels_t


def mean_nll(
    temperature: float,
    raw_scores: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> float:
    """Mean negative log-likelihood under a given temperature."""
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("temperature must be a finite positive number")
    log_scores, labels_t = _prepare(raw_scores, labels)
    log_probs = torch.log_softmax(log_scores / t, dim=1)
    picked = log_probs.gather(1, labels_t.unsqueeze(1)).squeeze(1)
    return float(-picked.mean())


def _golden_section(f, a: float, b: float, tol: float) -> float:
    """Minimum of a unimodal function on [a, b] (golden-section search)."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - ratio * (b - a)
        d = a + ratio * (b - a)
    return (a + b) / 2.0


def fit_temperature(
    raw_scores: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[float, float]:
    """Fit the temperature in [TEMP_MIN, TEMP_MAX] minimizing mean NLL.

    Returns ``(temperature, nll)``. Raises ``ValueError`` on malformed input.
    """
    log_scores, labels_t = _prepare(raw_scores, labels)
    n, num_classes = log_scores.shape

    # Coarse global pass: vectorized log-spaced grid (covers both endpoints).
    grid = torch.exp(
        torch.linspace(
            math.log(TEMP_MIN), math.log(TEMP_MAX), _GRID_POINTS, dtype=torch.float64
        )
    )
    # (G, N, C): scaled log-scores for every candidate temperature at once.
    scaled = log_scores.unsqueeze(0) / grid.view(-1, 1, 1)
    log_probs = torch.log_softmax(scaled, dim=2)
    nlls = -log_probs.gather(
        2, labels_t.view(1, -1, 1).expand(_GRID_POINTS, n, 1)
    ).squeeze(2).mean(dim=1)
    best_idx = int(torch.argmin(nlls).item())
    grid_best_t = float(grid[best_idx].item())
    grid_best_loss = float(nlls[best_idx].item())

    # Refine inside the grid bracket containing the best point.
    if best_idx == 0:
        lo, hi = TEMP_MIN, float(grid[1].item())
    elif best_idx == _GRID_POINTS - 1:
        lo, hi = float(grid[-2].item()), TEMP_MAX
    else:
        lo, hi = float(grid[best_idx - 1].item()), float(grid[best_idx + 1].item())

    def scalar_loss(t: float) -> float:
        lp = torch.log_softmax(log_scores / t, dim=1)
        return float(-lp.gather(1, labels_t.unsqueeze(1)).mean())

    refined_t = _golden_section(scalar_loss, lo, hi, _GOLDEN_TOL)
    refined_t = min(TEMP_MAX, max(TEMP_MIN, refined_t))
    refined_loss = scalar_loss(refined_t)

    # Never do worse than the grid (or an endpoint); stay inside the bounds.
    candidates = [
        (grid_best_t, grid_best_loss),
        (refined_t, refined_loss),
        (TEMP_MIN, scalar_loss(TEMP_MIN)),
        (TEMP_MAX, scalar_loss(TEMP_MAX)),
    ]
    best_t, best_loss = min(candidates, key=lambda pair: pair[1])
    best_t = min(TEMP_MAX, max(TEMP_MIN, float(best_t)))
    return float(best_t), float(best_loss)
