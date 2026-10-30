"""design notes-5 scoring on the external Sims set E (binary: agricultural conversion or not), run once.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/evaluate_v3.py

P(agriculture) = P(Plantation) + P(Smallholder agriculture); decision at 0.5. Balanced accuracy at k = 4 with paired
bootstraps (2,000, seed 20261021). P4-2: binary early stop for M4 (confidence max(p, 1 - p), alpha 0.05, delta 0.10,
beta 0): (a) lambda from ForestNet val_cal, (b) lambda from a random 40% of E (seed 20261026), gap on the other 60%.
Writes runs/rs_forestnet_v3/eval_v3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import BallTree, KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers import earlystop_v2 as es  # noqa: E402
from rsdrivers.features import LABELS4  # noqa: E402
from rsdrivers.io import INTERIM, RUNS, events  # noqa: E402

SIMS = ROOT / "data/interim/rs_forestnet_sims"
RUNS3 = ROOT / "runs/rs_forestnet_v3"
OFFSET = 100000
B = 2000


def to_bin(P):
    """[..., 4] class probabilities -> [..., 2] (non-agricultural, agricultural)."""
    a = P[..., 0] + P[..., 1]
    return np.stack([1 - a, a], -1)


def boot(fn, n, seed=20261021):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]


def load(path, eids):
    z = np.load(path, allow_pickle=True)
    pos = {int(e): i for i, e in enumerate(z["eid"])}
    return np.stack([z["probs"][pos[e]] for e in eids])


def main():
    se = pd.read_csv(SIMS / "events.csv")
    e_ids = (OFFSET + se.sid).to_numpy()
    y = se.label_agri.to_numpy()
    P = {"M2": load(RUNS3 / "probs/M2.npz", e_ids), "M3c": load(RUNS3 / "probs/M3c.npz", e_ids),
         "M3v": load(RUNS3 / "probs/M3v.npz", e_ids)}
    P["M4"] = (P["M2"] + P["M3c"]) / 2
    P["M4v"] = (P["M2"] + P["M3v"]) / 2
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    tr = fn[fn.part == "train"]
    loc4 = KNeighborsClassifier(1, metric="haversine").fit(np.radians(tr[["latitude", "longitude"]].to_numpy()),
                                                           tr.merged_label.map(LABELS4.index).to_numpy())
    loc = np.isin(loc4.predict(np.radians(se[["Latitude", "Longitude"]].to_numpy())), [0, 1]).astype(int)
    pred = {m: (to_bin(p)[:, 4, 1] >= 0.5).astype(int) for m, p in P.items()} | {"location 1-NN": loc}
    ba = lambda m, i: balanced_accuracy_score(y[i], pred[m][i])  # noqa: E731
    allidx = np.arange(len(y))
    res = {"n": int(len(y)), "n_agri": int(y.sum()), "by_model": {}, "criteria": {}, "subsets": {}}
    for m in pred:
        res["by_model"][m] = {"balanced_acc_k4": round(ba(m, allidx), 4),
                              "macro_f1_k4": round(f1_score(y, pred[m], average="macro"), 4)}
        if m in P:
            res["by_model"][m]["balanced_acc_by_k"] = [round(balanced_accuracy_score(y, (to_bin(P[m])[:, k, 1] >= 0.5)), 4)
                                                      for k in range(5)]
    d = lambda a_, b_: (lambda i: ba(a_, i) - ba(b_, i))  # noqa: E731
    for name, (a_, b_) in {"P4-1 M4 - M2": ("M4", "M2"), "P4-3 M3v - M3c": ("M3v", "M3c"),
                           "M4v - M2": ("M4v", "M2"), "M4v - M4": ("M4v", "M4")}.items():
        ci = boot(d(a_, b_), len(y))
        res["criteria" if name.startswith("P4") else "subsets"][name] = {"diff": round(d(a_, b_)(allidx), 4), "ci": ci,
                                                                         **({"met": ci[0] > 0} if name.startswith("P4") else {})}
    # P4-2: binary early stop for M4
    yfn = fn.set_index("eid").merged_label.map(lambda l: int(LABELS4.index(l) in (0, 1)))
    m2fn = np.load(RUNS3 / "probs/M2.npz", allow_pickle=True)
    cal_ids = fn.eid[fn.part == "val_cal"].to_numpy()
    zc = [np.load(f, allow_pickle=True) for f in sorted((RUNS / "probs").glob("M3c_*.npz"))]
    m3c_all = {int(e): p for z in zc for e, p in zip(z["eid"], z["probs"])}
    pos2 = {int(e): i for i, e in enumerate(m2fn["eid"])}
    m4_cal = to_bin(np.stack([(m2fn["probs"][pos2[e]] + m3c_all[e]) / 2 for e in cal_ids]))
    y_cal = yfn.loc[cal_ids].to_numpy()
    m4_e = to_bin(P["M4"])
    out = {}
    lam_a = es.calibrate(m4_cal, y_cal, 0.05, 0.10, 0.0)
    ea, _, _ = es.evaluate(m4_e, y, lam_a, 0.0)
    out["a_lambda_from_ForestNet_val_cal"] = {"lambda": lam_a, "gap_on_E": round(ea["gap_loss"], 4), "mean_stop": ea["mean_stop"]}
    rng = np.random.default_rng(20261026)
    perm = rng.permutation(len(y))
    c_idx, t_idx = perm[: round(0.4 * len(y))], perm[round(0.4 * len(y)):]
    lam_b = es.calibrate(m4_e[c_idx], y[c_idx], 0.05, 0.10, 0.0)
    eb, _, _ = es.evaluate(m4_e[t_idx], y[t_idx], lam_b, 0.0)
    out["b_lambda_from_40pct_of_E"] = {"lambda": lam_b, "gap_on_60pct": round(eb["gap_loss"], 4), "mean_stop": eb["mean_stop"],
                                       "acc_at_stop": round(eb["acc_early"], 4), "n_cal": int(len(c_idx)), "n_eval": int(len(t_idx))}
    out["met"] = eb["gap_loss"] <= 0.05
    res["criteria"]["P4-2 M4 binary early stop"] = out
    # secondary subsets
    dist = se.dist_fn_train_km.to_numpy()
    for sub, mask in {">=10 km from ForestNet train": dist >= 10, "validation file only": (se.src == "validation").to_numpy(),
                      "High confidence only": (se.Confidence_primary == "High").to_numpy()}.items():
        idx = np.flatnonzero(mask)
        res["subsets"][sub] = {"n": int(len(idx)), **{m: round(ba(m, idx), 4) for m in pred}}
    res["subsets"]["4-class prediction share by Sims class (M4, k4)"] = (
        pd.crosstab(se.Driver_primary, pd.Series(P["M4"][:, 4].argmax(1)).map(dict(enumerate(LABELS4))), normalize="index")
        .round(3).to_dict(orient="index"))
    # M2 retrained vs the v1 M2 on the ForestNet test (agreement check, design notes)
    v1 = np.load(RUNS / "probs/M2.npz", allow_pickle=True)
    pv1 = {int(e): p for e, p in zip(v1["eid"], v1["probs"])}
    t_ids = fn.eid[fn.part == "test"].to_numpy()
    a1 = np.array([pv1[e][4].argmax() for e in t_ids])
    a3 = np.array([m2fn["probs"][pos2[e]][4].argmax() for e in t_ids])
    yt = fn.set_index("eid").merged_label.map(LABELS4.index).loc[t_ids].to_numpy()
    res["M2_retrained_check"] = {"argmax_agreement_test_k4": round(float((a1 == a3).mean()), 4),
                                 "macro_f1_test_k4_v1": round(f1_score(yt, a1, average="macro"), 4),
                                 "macro_f1_test_k4_retrained": round(f1_score(yt, a3, average="macro"), 4)}
    path = RUNS3 / "eval_v3.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("n", "n_agri", "criteria", "by_model")}, indent=1, ensure_ascii=False))
    print("subsets:", json.dumps({k: v for k, v in res["subsets"].items() if "prediction share" not in k}, ensure_ascii=False))
    print("M2 check:", res["M2_retrained_check"])
    print("saved", path)


if __name__ == "__main__":
    main()
