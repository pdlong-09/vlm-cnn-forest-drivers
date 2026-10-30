"""design notes-4 scoring on E2 (tropical Sims plots outside Indonesia; binary agricultural conversion), run once.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/evaluate_v4.py

P(agriculture) = P(Plantation) + P(Smallholder agriculture), decision at 0.5, balanced accuracy (BA) at k = 4, paired
bootstraps (2,000, seed 20261021). P5-3: binary early stop for M5e (alpha 0.05, delta 0.10, beta 0) calibrated on a
random 40% of each region group (seed 20261028), gap pooled over the other 60%; plus lambda from ForestNet val_cal.
Writes runs/rs_forestnet_v4/eval_v4.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers import earlystop_v2 as es  # noqa: E402
from rsdrivers.features import LABELS4  # noqa: E402
from rsdrivers.io import INTERIM, events  # noqa: E402

E2 = ROOT / "data/interim/rs_forestnet_sims2"
P4 = ROOT / "runs/rs_forestnet_v4/probs"
OFF, B = 200000, 2000


def to_bin(P):
    a = P[..., 0] + P[..., 1]
    return np.stack([1 - a, a], -1)


def load(name, eids):
    z = np.load(P4 / f"{name}.npz", allow_pickle=True)
    probs = z["probs"]  # read once (each NpzFile access decompresses the array again)
    pos = {int(e): i for i, e in enumerate(z["eid"])}
    return np.stack([probs[pos[e]] for e in eids])


def boot(fn, n, seed=20261021):
    rng = np.random.default_rng(seed)
    return [round(float(x), 4) for x in np.percentile([fn(rng.integers(0, n, n)) for _ in range(B)], [2.5, 97.5])]


def main():
    se = pd.read_csv(E2 / "events.csv")
    ids = (OFF + se.sid).to_numpy()
    y = se.label_agri.to_numpy()
    P = {m: load(m, ids) for m in ("M5e", "M5t", "M5tr", "M3c")}
    P["M2"] = load("M2_e2", ids)
    P["M4"] = (P["M2"] + P["M3c"]) / 2
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    tr = fn[fn.part == "train"]
    loc = KNeighborsClassifier(1, metric="haversine").fit(np.radians(tr[["latitude", "longitude"]].to_numpy()),
                                                          tr.merged_label.map(LABELS4.index).to_numpy())
    pred = {m: (to_bin(p)[:, 4, 1] >= 0.5).astype(int) for m, p in P.items()}
    pred["location 1-NN"] = np.isin(loc.predict(np.radians(se[["Latitude", "Longitude"]].to_numpy())), [0, 1]).astype(int)
    ba = lambda m, i: balanced_accuracy_score(y[i], pred[m][i])  # noqa: E731
    allidx = np.arange(len(y))
    res = {"n": int(len(y)), "n_agri": int(y.sum()), "by_group_n": se.group.value_counts().to_dict(), "by_model": {},
           "criteria": {}, "secondary": {}}
    for m in pred:
        res["by_model"][m] = {"BA_k4": round(ba(m, allidx), 4), "macro_f1_k4": round(f1_score(y, pred[m], average="macro"), 4),
                              **({"BA_by_k": [round(balanced_accuracy_score(y, to_bin(P[m])[:, k, 1] >= 0.5), 4) for k in range(5)]}
                                 if m in P else {}),
                              "BA_by_group": {g: round(ba(m, np.flatnonzero((se.group == g).to_numpy())), 4)
                                              for g in sorted(se.group.unique())}}
    diff = lambda a_, b_: (lambda i: ba(a_, i) - ba(b_, i))  # noqa: E731
    for name, (a_, b_), crit in (("P5-1 M5e - M2", ("M5e", "M2"), True), ("P5-2 M5e - M4", ("M5e", "M4"), True),
                                 ("M5t - M2", ("M5t", "M2"), False), ("M5tr - M2", ("M5tr", "M2"), False),
                                 ("M5t - M4", ("M5t", "M4"), False), ("M5tr - M4", ("M5tr", "M4"), False),
                                 ("M5e - M5t", ("M5e", "M5t"), False), ("M5tr - M5t", ("M5tr", "M5t"), False),
                                 ("M3c - M2", ("M3c", "M2"), False), ("M4 - M2", ("M4", "M2"), False)):
        ci = boot(diff(a_, b_), len(y))
        rec = {"diff": round(diff(a_, b_)(allidx), 4), "ci": ci}
        if crit:
            rec["met"] = ci[0] > 0
        (res["criteria"] if crit else res["secondary"])[name] = rec
    # P5-3: binary early stop for M5e, calibrated within each region group
    m5e = to_bin(P["M5e"])
    rng = np.random.default_rng(20261028)
    losses, per = [], {}
    for g in sorted(se.group.unique()):
        idx = np.flatnonzero((se.group == g).to_numpy())
        idx = idx[rng.permutation(len(idx))]
        c, t = idx[: round(0.4 * len(idx))], idx[round(0.4 * len(idx)):]
        lam = es.calibrate(m5e[c], y[c], 0.05, 0.10, 0.0)
        l, st, pr = es.losses(m5e[t], y[t], lam, 0.0)
        losses.append(l)
        per[g] = {"lambda": lam, "gap": round(float(l.mean()), 4), "mean_stop": round(float(st.mean()), 3),
                  "acc_at_stop": round(float((pr == y[t]).mean()), 4), "n_cal": int(len(c)), "n_eval": int(len(t))}
    pooled = float(np.concatenate(losses).mean())
    zc = np.load(P4 / "M5e.npz", allow_pickle=True)
    cal_ids = [e for e in fn.eid[fn.part == "val_cal"] if e in set(zc["eid"].tolist())]
    yfn = fn.set_index("eid").merged_label.map(lambda l: int(LABELS4.index(l) in (0, 1)))
    lam_fn = es.calibrate(to_bin(load("M5e", cal_ids)), yfn.loc[cal_ids].to_numpy(), 0.05, 0.10, 0.0)
    ea, _, _ = es.evaluate(m5e, y, lam_fn, 0.0)
    res["criteria"]["P5-3 M5e early stop, local calibration per region"] = {
        "pooled_gap_on_60pct": round(pooled, 4), "met": pooled <= 0.05, "per_group": per,
        "lambda_from_ForestNet_val_cal": {"lambda": lam_fn, "gap_on_E2": round(ea["gap_loss"], 4), "mean_stop": ea["mean_stop"]}}
    for sub, mask in {"validation file only": (se.src == "validation").to_numpy(),
                      "High confidence only": (se.Confidence_primary == "High").to_numpy(),
                      ">=100 km from ForestNet train": (se.dist_fn_train_km >= 100).to_numpy()}.items():
        idx = np.flatnonzero(mask)
        res["secondary"][sub] = {"n": int(len(idx)), **{m: round(ba(m, idx), 4) for m in pred}}
    path = ROOT / "runs/rs_forestnet_v4/eval_v4.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("n", "n_agri", "by_group_n", "criteria")}, indent=1, ensure_ascii=False))
    for m, v in res["by_model"].items():
        print(f"{m:14s} BA {v['BA_k4']:.3f} | by group {v['BA_by_group']}")
    print("secondary:", json.dumps(res["secondary"], ensure_ascii=False))
    print("saved", path)


if __name__ == "__main__":
    main()
