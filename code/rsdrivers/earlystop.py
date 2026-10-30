"""Early attribution with a finite-sample guarantee (learn-then-test), after Ringel et al. (ICML 2024).

A model gives class probabilities after each year t = 0..T-1 of accumulated evidence. Rule(lambda): stop at the first
t with max probability >= lambda, at the last year otherwise. The loss of an event is 1 when the early decision is
wrong while the full-length decision is right (an upper bound of the per-event accuracy gap), so the risk R(lambda) is
the probability that stopping early costs a correct answer. Learn-then-test with fixed-sequence testing over a grid
from lambda = 1 (never stop early, R = 0) downwards, with exact binomial p-values, gives lambda_hat such that
P(R(lambda_hat) <= alpha) >= 1 - delta (Angelopoulos et al., "Learn then test", 2021/2025).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def stop_times(probs: np.ndarray, lam: float) -> np.ndarray:
    """probs [n, T, C] -> index of the stopping year per event."""
    conf = probs.max(-1)
    hit = conf >= lam
    hit[:, -1] = True
    return hit.argmax(1)


def losses(probs, y, lam):
    t = stop_times(probs, lam)
    pred_early = probs[np.arange(len(y)), t].argmax(-1)
    pred_full = probs[:, -1].argmax(-1)
    return ((pred_early != y) & (pred_full == y)).astype(int), t, pred_early


def calibrate(probs, y, alpha=0.05, delta=0.1, grid=None):
    """Smallest lambda (earliest stopping) certified by fixed-sequence testing; 1.0+ means 'always wait'."""
    grid = np.sort(np.unique(np.r_[np.linspace(0.25, 1.0, 76), 1.0 + 1e-9]))[::-1] if grid is None else np.sort(grid)[::-1]
    n = len(y)
    chosen = 1.0 + 1e-9
    for lam in grid:
        k = losses(probs, y, lam)[0].sum()
        p = binom.cdf(k, n, alpha)  # H0: R(lam) > alpha ; small p -> reject -> lam is safe
        if p <= delta:
            chosen = lam
        else:
            break
    return float(chosen)


def evaluate(probs, y, lam, years=None):
    l, t, pred = losses(probs, y, lam)
    full = probs[:, -1].argmax(-1)
    T = probs.shape[1]
    return {"lambda": lam, "gap_loss": float(l.mean()), "acc_early": float((pred == y).mean()),
            "acc_full": float((full == y).mean()), "mean_stop": float(t.mean() if years is None else np.asarray(years)[t].mean()),
            "stop_share": [float((t == i).mean()) for i in range(T)]}
