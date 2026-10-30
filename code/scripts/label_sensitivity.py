"""Driver attribution: sensitivity of the VLM - CNN difference to the mapping of the external labels (26/10/2026).

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/label_sensitivity.py

ForestNet counts plantation (including timber plantations) and smallholder agriculture as conversion to agriculture;
the external sets count permanent agriculture and shifting cultivation as agriculture and every other driver of Sims et
al. as other. The classes that do not map one to one are logging (timber plantations fall under ForestNet's plantation
class) and shifting cultivation (part of it regrows to forest, which ForestNet calls other). The same predictions at
k = 4 are scored under alternative mappings and subsets; intervals from 2000 paired bootstrap resamples of cells and of
clusters (countries for E2 and E3, 1-degree cells for Indonesia). Writes runs/rs_forestnet_robust/label_sensitivity.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SETS = {"Indonesia": ("rs_forestnet_sims", 100000, "rs_forestnet_v3/probs/M2.npz", "rs_forestnet_v3/probs/M3c.npz"),
        "E2": ("rs_forestnet_sims2", 200000, "rs_forestnet_v4/probs/M2_e2.npz", "rs_forestnet_v4/probs/M3c.npz"),
        "E3": ("rs_forestnet_sims3", 300000, "rs_forestnet_v5/probs/M2_E3.npz", "rs_forestnet_v5/probs/M3c_E3.npz")}
AGRI = {"Permanent agriculture", "Shifting cultivation"}
VARIANTS = {
    "main mapping": (lambda d: np.ones(len(d), bool), lambda d: d.Driver_primary.isin(AGRI)),
    "logging cells excluded": (lambda d: d.Driver_primary != "Logging", lambda d: d.Driver_primary.isin(AGRI)),
    "logging counted as agriculture": (lambda d: np.ones(len(d), bool), lambda d: d.Driver_primary.isin(AGRI | {"Logging"})),
    "shifting cultivation excluded": (lambda d: d.Driver_primary != "Shifting cultivation", lambda d: d.Driver_primary.isin(AGRI)),
    "logging and shifting cultivation excluded": (lambda d: ~d.Driver_primary.isin(["Logging", "Shifting cultivation"]),
                                                  lambda d: d.Driver_primary.isin(AGRI)),
    "high-confidence labels only": (lambda d: d.Confidence_primary == "High", lambda d: d.Driver_primary.isin(AGRI)),
}


def ba(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def load(path, ids):
    z = np.load(ROOT / "runs" / path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    P = np.stack([probs[pos[int(e)]] for e in ids])
    P = P[:, 4] if P.ndim == 3 else P
    return ((P[:, 0] + P[:, 1]) >= 0.5).astype(int)


def main():
    rng = np.random.default_rng(20261021)
    out = {}
    for name, (d, off, pc, pv) in SETS.items():
        ev = pd.read_csv(ROOT / "data/interim" / d / "events.csv")
        ids = off + ev.sid
        cnn, vlm = load(pc, ids), load(pv, ids)
        cl = ev.country.to_numpy() if "country" in ev else (
            np.floor(ev.Latitude).astype(int).astype(str) + "_" + np.floor(ev.Longitude).astype(int).astype(str)).to_numpy()
        out[name] = {}
        for v, (keep, pos) in VARIANTS.items():
            m = np.asarray(keep(ev), bool)
            y = np.asarray(pos(ev), int)[m]
            a2, a3, c = cnn[m], vlm[m], cl[m]
            n = len(y)
            cell = [(lambda i: ba(y[i], a3[i]) - ba(y[i], a2[i]))(rng.integers(0, n, n)) for _ in range(2000)]
            uc = np.unique(c)
            mem = {u: np.flatnonzero(c == u) for u in uc}
            clus = []
            for _ in range(2000):
                i = np.concatenate([mem[u] for u in rng.choice(uc, len(uc), replace=True)])
                clus.append(ba(y[i], a3[i]) - ba(y[i], a2[i]))
            out[name][v] = {"n": int(n), "n_agri": int(y.sum()), "CNN": round(ba(y, a2), 4), "VLM": round(ba(y, a3), 4),
                            "diff": round(ba(y, a3) - ba(y, a2), 4),
                            "ci_cell": [round(float(x), 4) for x in np.percentile(cell, [2.5, 97.5])],
                            "ci_cluster": [round(float(x), 4) for x in np.percentile(clus, [2.5, 97.5])]}
            r = out[name][v]
            print(f"{name:9s} {v:42s} n={n:4d} CNN {r['CNN']:.3f} VLM {r['VLM']:.3f} diff {r['diff']:+.3f} "
                  f"cell {r['ci_cell']} cluster {r['ci_cluster']}", flush=True)
    (ROOT / "runs/rs_forestnet_robust/label_sensitivity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
