"""v2 / v2.1 additions to the learn-then-test early stop (rsdrivers/earlystop.py is frozen with v1 and not changed):
a year-decreasing threshold lambda_k = lambda + beta * (T - 1 - k) / (T - 1) (design notes) and temperature scaling
per step k fitted on the holdout by negative log-likelihood (design notes)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

T_GRID = np.round(np.arange(0.25, 5.0001, 0.05), 2)


def thresholds(lam: float, beta: float, T: int = 5) -> np.ndarray:
    return lam + beta * (T - 1 - np.arange(T)) / (T - 1)


def stop_times(probs: np.ndarray, lam: float, beta: float = 0.0) -> np.ndarray:
    hit = probs.max(-1) >= thresholds(lam, beta, probs.shape[1])[None, :]
    hit[:, -1] = True
    return hit.argmax(1)


def losses(probs, y, lam, beta=0.0):
    t = stop_times(probs, lam, beta)
    pred_early = probs[np.arange(len(y)), t].argmax(-1)
    pred_full = probs[:, -1].argmax(-1)
    return ((pred_early != y) & (pred_full == y)).astype(int), t, pred_early


def calibrate(probs, y, alpha=0.05, delta=0.10, beta=0.0):
    """Fixed-sequence learn-then-test over lambda = 1.00, 0.99, ..., 0.25 (design notes); 1+ = always wait."""
    chosen = 1.0 + 1e-9
    for lam in np.round(np.arange(1.0, 0.2499, -0.01), 2):
        k = losses(probs, y, lam, beta)[0].sum()
        if binom.cdf(k, len(y), alpha) <= delta:
            chosen = float(lam)
        else:
            break
    return chosen


def evaluate(probs, y, lam, beta=0.0):
    l, t, pred = losses(probs, y, lam, beta)
    T = probs.shape[1]
    return {"lambda": lam, "beta": beta, "gap_loss": float(l.mean()), "acc_early": float((pred == y).mean()),
            "acc_full": float((probs[:, -1].argmax(-1) == y).mean()), "mean_stop": float(t.mean()),
            "stop_share": [float((t == i).mean()) for i in range(T)]}, t, pred


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One temperature per step k (probs [n, T, C]) minimising holdout NLL on the grid 0.25..5."""
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    out = []
    for k in range(probs.shape[1]):
        best = None
        for T in T_GRID:
            z = logp[:, k] / T
            z = z - z.max(1, keepdims=True)
            nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
            if best is None or nll < best[0]:
                best = (nll, T)
        out.append(best[1])
    return np.array(out)


def apply_temperature(probs: np.ndarray, temps: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1.0)) / temps[None, :, None]
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
