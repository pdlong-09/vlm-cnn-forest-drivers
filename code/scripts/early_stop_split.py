"""Driver attribution: learn-then-test early attribution with local calibration, evaluated on spatially independent test cells.

    .venv/Scripts/python paper/rs_forestnet/scripts/early_stop_split.py

For each target set (Indonesia outside ForestNet: clusters = 1-degree cells; E2: clusters = countries) and each of
1000 repetitions, the clusters are shuffled and assigned to a calibration pool until it holds half of the cells; the
other clusters form the test set. n_cal cells are drawn from the pool, the threshold is calibrated by learn-then-test
(rsdrivers/earlystop_v2.calibrate: exact binomial p-values, fixed-sequence testing over lambda = 1.00..0.25, alpha 0.05,
delta 0.10), and the rule is evaluated on the test clusters only. Reported per model and n_cal: share of repetitions
without a certified threshold, share of repetitions whose test gap exceeds alpha, mean and 95th percentile of the test
gap over repetitions, the mean one-sided 95% Clopper-Pearson upper bound of the test risk (valid only for independent cells), share of cells attributed before k = 4 and at k = 0, mean stopping
year, and the cell and error counts. Binary task (conversion to agriculture). Writes runs/rs_forestnet_robust/
early_stop_split.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers import earlystop_v2 as es  # noqa: E402

ALPHA, DELTA, REPS, SEED = 0.05, 0.10, 1000, 20261026
R = ROOT / "runs"


def to_bin(P):
    a = P[..., 0] + P[..., 1]
    return np.stack([1 - a, a], -1)


def load(path, eids):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    return np.stack([probs[pos[int(e)]] for e in eids])


def cp_upper(k, n, conf=0.95):
    return 1.0 if k == n else float(beta_dist.ppf(conf, k + 1, n - k))


def target_sets():
    out = {}
    ev = pd.read_csv(ROOT / "data/interim/rs_forestnet_sims/events.csv")
    ids = 100000 + ev.sid
    P = {"CNN": load(R / "rs_forestnet_v3/probs/M2.npz", ids), "VLM": load(R / "rs_forestnet_v3/probs/M3c.npz", ids)}
    cl = (np.floor(ev.Latitude).astype(int).astype(str) + "_" + np.floor(ev.Longitude).astype(int).astype(str)).to_numpy()
    out["Indonesia"] = (ev.label_agri.to_numpy(), P, cl, "1-degree cells")
    ev = pd.read_csv(ROOT / "data/interim/rs_forestnet_sims2/events.csv")
    ids = 200000 + ev.sid
    P = {"CNN": load(R / "rs_forestnet_v4/probs/M2_e2.npz", ids), "VLM": load(R / "rs_forestnet_v4/probs/M3c.npz", ids)}
    out["E2"] = (ev.label_agri.to_numpy(), P, ev.country.to_numpy(), "countries")
    for name in list(out):
        y, P, cl, unit = out[name]
        P["CNN + VLM"] = (P["CNN"] + P["VLM"]) / 2
        out[name] = (y, {m: to_bin(p) for m, p in P.items()}, cl, unit)
    return out


def run_set(y, P, clusters, n_cals, rng):
    uc = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uc}
    res = {m: {n: [] for n in n_cals} for m in P}
    for _ in range(REPS):
        order = rng.permutation(uc)
        pool, size = [], 0
        for c in order:
            if size >= len(y) / 2:
                break
            pool.append(c)
            size += len(members[c])
        cal_pool = np.concatenate([members[c] for c in pool])
        test = np.setdiff1d(np.arange(len(y)), cal_pool)
        for n in n_cals:
            if n > len(cal_pool):
                continue
            c = rng.choice(cal_pool, n, replace=False)
            for m, p in P.items():
                lam = es.calibrate(p[c], y[c], ALPHA, DELTA, 0.0)
                l, t, _ = es.losses(p[test], y[test], lam, 0.0)
                res[m][n].append((lam <= 1.0, int(l.sum()), len(test), float((t < p.shape[1] - 1).mean()),
                                  float((t == 0).mean()), float(t.mean())))
    out = {}
    for m in P:
        out[m] = {}
        for n, rows in res[m].items():
            if not rows:
                continue
            a = np.array(rows, dtype=float)
            gap = a[:, 1] / a[:, 2]
            out[m][str(n)] = {"repetitions": len(rows), "share_no_certified_threshold": round(1 - a[:, 0].mean(), 3),
                              "share_test_gap_above_alpha": round(float((gap > ALPHA).mean()), 3),
                              "mean_test_gap": round(float(gap.mean()), 4),
                              "p95_test_gap": round(float(np.percentile(gap, 95)), 4),
                              "mean_cp95_upper_test_risk": round(float(np.mean([cp_upper(int(k), int(t)) for k, t in a[:, 1:3]])), 4),
                              "mean_test_cells": round(float(a[:, 2].mean()), 1), "mean_test_errors": round(float(a[:, 1].mean()), 1),
                              "share_attributed_before_k4": round(float(a[:, 3].mean()), 3),
                              "share_attributed_at_k0": round(float(a[:, 4].mean()), 3), "mean_stop_year": round(float(a[:, 5].mean()), 3)}
    return out


def main():
    rng = np.random.default_rng(SEED)
    out = {"design": {"alpha": ALPHA, "delta": DELTA, "repetitions": REPS, "seed": SEED,
                      "split": "clusters shuffled; calibration pool = first clusters reaching half of the cells; test = the rest"}}
    for name, (y, P, cl, unit) in target_sets().items():
        n_cals = [50, 100, 200] if name == "Indonesia" else [50, 100, 200, 400]
        out[name] = {"cluster_unit": unit, "n_cells": int(len(y)), "n_clusters": int(len(np.unique(cl))), "models": run_set(y, P, cl, n_cals, rng)}
        print(name, json.dumps(out[name]["models"], indent=None)[:2500], flush=True)
    path = R / "rs_forestnet_robust/early_stop_split.json"
    path.write_text(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
