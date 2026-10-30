"""design notes scoring: binary agricultural conversion at k = 4, balanced accuracy (BA), paired bootstraps (2,000, seed
20261021). P(agriculture) = P(Plantation) + P(Smallholder agriculture), decision at 0.5.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/evaluate_v5.py dev     # E2 (development; exploratory)
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/evaluate_v5.py final   # E3, run once (§4-5)

Models: M2 (CNN, 3 seeds), M3c (v1 VLM), M4 = (M2 + M3c) / 2, A-probe, M4A = (M3c + A-probe) / 2, M6 (chosen; on E2 all four
candidates), and (M2 + M3c + A-probe) / 3. Writes runs/rs_forestnet_v5/eval_v5_<dev|final>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[3]
P4, P5 = ROOT / "runs/rs_forestnet_v4/probs", ROOT / "runs/rs_forestnet_v5/probs"
SETS = {"dev": (ROOT / "data/interim/rs_forestnet_sims2", 200000), "final": (ROOT / "data/interim/rs_forestnet_sims3", 300000)}
B = 2000


def agri(path, ids, k4=False):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    if k4:
        probs = probs[:, 4]
    pos = {int(e): i for i, e in enumerate(eid)}
    return np.array([probs[pos[e], 0] + probs[pos[e], 1] for e in ids])


def boot(fn, n, seed=20261021):
    rng = np.random.default_rng(seed)
    return [round(float(x), 4) for x in np.percentile([fn(rng.integers(0, n, n)) for _ in range(B)], [2.5, 97.5])]


def main():
    mode = sys.argv[1]
    base, off = SETS[mode]
    se = pd.read_csv(base / "events.csv")
    ids = (off + se.sid).to_numpy()
    y = se.label_agri.to_numpy()
    P = {}
    if mode == "dev":
        P["M2"], P["M3c"] = agri(P4 / "M2_e2.npz", ids, True), agri(P4 / "M3c.npz", ids, True)
        for t in ("p0", "p05"):
            for e in (1, 2):
                f = P5 / f"M6_{t}_ep{e}_E2.npz"
                if f.exists():
                    P[f"M6_{t}_ep{e}"] = agri(f, ids)
    else:
        P["M2"], P["M3c"], P["M6"] = (agri(P5 / f"{m}_E3.npz", ids) for m in ("M2", "M3c", "M6"))
    P["A-probe"] = agri(P5 / "A_probe.npz", ids)
    P["M4"] = (P["M2"] + P["M3c"]) / 2
    P["M4A"] = (P["M3c"] + P["A-probe"]) / 2
    P["M2+M3c+A"] = (P["M2"] + P["M3c"] + P["A-probe"]) / 3
    pred = {m: (p >= 0.5).astype(int) for m, p in P.items()}
    ba = lambda m, i: balanced_accuracy_score(y[i], pred[m][i])  # noqa: E731
    allidx = np.arange(len(y))
    res = {"set": mode, "n": int(len(y)), "n_agri": int(y.sum()), "by_model": {}, "criteria": {}, "secondary": {}}
    for m in pred:
        res["by_model"][m] = {"BA_k4": round(ba(m, allidx), 4),
                              "BA_by_group": {g: round(ba(m, np.flatnonzero((se.group == g).to_numpy())), 4)
                                              for g in sorted(se.group.unique())}}
    diff = lambda a_, b_: (lambda i: ba(a_, i) - ba(b_, i))  # noqa: E731
    m6s = [m for m in P if m.startswith("M6")]
    pairs = [(m, b_) for m in m6s for b_ in ("M4", "M4A", "M3c", "M2")] + [("A-probe", "M2"), ("M4A", "M4"), ("M2+M3c+A", "M4")]
    crit = {("M6", "M4"): "P6-1 M6 - M4", ("M6", "M4A"): "P6-2 M6 - M4A", ("M6", "M3c"): "P6-3 M6 - M3c"} if mode == "final" else {}
    for a_, b_ in pairs:
        ci = boot(diff(a_, b_), len(y))
        rec = {"diff": round(diff(a_, b_)(allidx), 4), "ci": ci}
        if (a_, b_) in crit:
            rec["met"] = ci[0] > 0
            res["criteria"][crit[(a_, b_)]] = rec
        else:
            res["secondary"][f"{a_} - {b_}"] = rec
    for sub, mask in {"validation file only": (se.src == "validation").to_numpy(),
                      "High confidence only": (se.Confidence_primary == "High").to_numpy(),
                      ">=100 km from ForestNet train": (se.dist_fn_train_km >= 100).to_numpy()}.items():
        idx = np.flatnonzero(mask)
        res["secondary"][sub] = {"n": int(len(idx)), **{m: round(ba(m, idx), 4) for m in pred}}
    path = ROOT / f"runs/rs_forestnet_v5/eval_v5_{mode}.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for m, v in res["by_model"].items():
        print(f"{m:12s} BA {v['BA_k4']:.3f} | by group {v['BA_by_group']}")
    print(json.dumps({k: res[k] for k in ("criteria", "secondary")}, indent=1, ensure_ascii=False))
    print("saved", path)


if __name__ == "__main__":
    main()
