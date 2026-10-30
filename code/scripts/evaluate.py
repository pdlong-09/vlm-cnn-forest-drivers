"""Early attribution evaluation (design notes sections 4-6).

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/evaluate.py M1 M3a ... [--alpha 0.05] [--final]

Per model, final (--final, once per frozen model): lambda from learn-then-test on val_cal, scored on test. Development
(default): val_cal is not read; lambda is cross-fitted between two halves of val_sel. Reported:
gap loss, accuracy at stop, mean stop year, share stopping at each k, macro-F1 per k. With --final and two or more
models, the paired bootstrap of the mean-stop difference between M3c and the best of M1/M2 is reported (P-A2).
Probabilities are read from runs/rs_forestnet/probs/<model>*.npz (files are merged by event id).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsdrivers.earlystop import calibrate, evaluate, stop_times  # noqa: E402
from rsdrivers.io import RUNS, events  # noqa: E402


def load(model):
    files = sorted((RUNS / "probs").glob(f"{model}.npz")) + sorted((RUNS / "probs").glob(f"{model}_*.npz"))
    if not files:
        raise SystemExit(f"no probabilities for {model}")
    data = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for i, e in enumerate(z["eid"]):
            data[int(e)] = (z["probs"][i], z["part"][i], int(z["y"][i]))
    eids = np.array(sorted(data))
    return eids, np.stack([data[e][0] for e in eids]), np.array([data[e][1] for e in eids]), np.array([data[e][2] for e in eids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--final", action="store_true", help="score the test part (once per frozen model)")
    ap.add_argument("--since", type=int, help="sensitivity: only target events with loss year >= this (design notes sec. 2)")
    a = ap.parse_args()
    target = "test" if a.final else "val_sel"
    res, stops = {}, {}
    year_of = dict(zip(events().eid, events().year))
    for m in a.models:
        eids, P, part, y = load(m)
        tgt = part == target
        if a.since:
            tgt &= np.array([year_of[int(e)] >= a.since for e in eids])
        if a.final:
            cal = part == "val_cal"
            if not cal.any() or not tgt.any():
                print(f"{m}: missing val_cal or test; skipped")
                continue
            lam = calibrate(P[cal], y[cal], a.alpha, a.delta)
            e = evaluate(P[tgt], y[tgt], lam)
        else:
            # development: val_cal is never read before the freeze; lambda is cross-fitted on two halves of val_sel
            idx = np.flatnonzero(tgt)
            half = np.random.default_rng(7).permutation(len(idx)) % 2
            lam_ev, t_dev = [], np.zeros(len(idx), dtype=int)
            for h in (0, 1):
                c, v = idx[half == h], idx[half != h]
                lam_h = calibrate(P[c], y[c], a.alpha, a.delta)
                lam_ev.append(lam_h)
                t_dev[half != h] = stop_times(P[v], lam_h)
            pred = P[idx][np.arange(len(idx)), t_dev].argmax(-1)
            full = P[idx][:, -1].argmax(-1)
            e = {"lambda": lam_ev, "gap_loss": float(((pred != y[idx]) & (full == y[idx])).mean()),
                 "acc_early": float((pred == y[idx]).mean()), "acc_full": float((full == y[idx]).mean()),
                 "mean_stop": float(t_dev.mean()), "stop_share": [float((t_dev == i).mean()) for i in range(P.shape[1])]}
            lam = None
        e["macro_f1_by_k"] = [round(f1_score(y[tgt], P[tgt][:, k].argmax(1), average="macro"), 3) for k in range(P.shape[1])]
        t = stop_times(P[tgt], lam) if a.final else t_dev
        e["macro_f1_at_stop"] = round(f1_score(y[tgt], P[tgt][np.arange(tgt.sum()), t].argmax(1), average="macro"), 3)
        e["n"] = int(tgt.sum())
        res[m] = e
        stops[m] = (eids[tgt], t)
        print(m, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}))
    if a.final and "M3c" in stops:
        base = [m for m in ("M1", "M2") if m in stops]
        if base:
            best = min(base, key=lambda m: stops[m][1].mean())
            e3, t3 = stops["M3c"]
            eb, tb = stops[best]
            assert (e3 == eb).all()
            rng = np.random.default_rng(20261021)
            d = t3 - tb
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"P-A2: mean stop M3c - {best} = {d.mean():.3f} (95% CI {lo:.3f} .. {hi:.3f})")
    out = RUNS / f"eval_{target}_alpha{a.alpha}{f'_since{a.since}' if a.since else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
