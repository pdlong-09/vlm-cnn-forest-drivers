"""Driver attribution: the image-only VLM over three seeds against the VLM with the full card over three seeds and against the CNNs.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fair_seeds_eval.py

Image-only VLM: seed 0 from raw/fair_vlmimg, seeds 1 and 2 from raw/fair_vlmimgseeds (same recipe, epoch chosen on the
selection half). Full-card VLM: seed 0 as in the article, seeds 1 and 2 from rs_forestnet_v5/raw/seeds (E2 and E3 only).
Three-seed models average the class probabilities. BA of conversion to agriculture at k = 4; differences with 2000 paired
bootstrap resamples of cells and of countries. Writes runs/rs_forestnet_robust/fair_vlmimg_seeds.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
sys.path.insert(0, str(ROOT / "paper/rs_forestnet/scripts"))
from fn_v5 import k4_probs  # noqa: E402
from rsdrivers.features import LABELS4  # noqa: E402
from rsdrivers.io import INTERIM, events  # noqa: E402

R = ROOT / "runs"
RAW = R / "rs_forestnet_robust/raw"
SETS = {"fn": None, "sidn": ("rs_forestnet_sims", 100000), "e2": ("rs_forestnet_sims2", 200000), "e3": ("rs_forestnet_sims3", 300000)}
JSONL = {"fn": "fntest", "sidn": "sidn", "e2": "e2", "e3": "e3"}


def jl(path, ids):
    eids, P = k4_probs(path)
    pos = {int(e): i for i, e in enumerate(eids)}
    P = np.asarray(P)
    return np.stack([P[pos[int(e)]] for e in ids])


def npz(path, ids):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    P = np.stack([probs[pos[int(e)]] for e in ids])
    return P[:, 4] if P.ndim == 3 else P


def agri(P):
    return ((P[:, 0] + P[:, 1]) >= 0.5).astype(int)


def ba(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def macro_f1(y, p):
    f = []
    for c in range(4):
        tp, fp, fn_ = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn_))
    return float(np.mean(f))


def boot(y, a1, a0, groups, rng, B=2000):
    n = len(y)
    cell = [(lambda i: ba(y[i], a1[i]) - ba(y[i], a0[i]))(rng.integers(0, n, n)) for _ in range(B)]
    out = {"diff": round(ba(y, a1) - ba(y, a0), 4), "ci_cell": [round(float(v), 4) for v in np.percentile(cell, [2.5, 97.5])]}
    if groups is not None:
        uc = np.unique(groups)
        mem = {c: np.flatnonzero(groups == c) for c in uc}
        g = [(lambda i: ba(y[i], a1[i]) - ba(y[i], a0[i]))(np.concatenate([mem[c] for c in rng.choice(uc, len(uc))]))
             for _ in range(B)]
        out["ci_country"] = [round(float(v), 4) for v in np.percentile(g, [2.5, 97.5])]
    return out


def main():
    rng = np.random.default_rng(20261021)
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    te = fn[fn.part == "test"]
    data = {"fn": (te.eid.to_numpy(), te.merged_label.isin(LABELS4[:2]).astype(int).to_numpy(), None)}
    for t, v in SETS.items():
        if v:
            ev = pd.read_csv(ROOT / "data/interim" / v[0] / "events.csv")
            data[t] = ((v[1] + ev.sid).to_numpy(), ev.label_agri.to_numpy(), ev.country.to_numpy() if "country" in ev else None)
    yfn4 = te.merged_label.map(LABELS4.index).to_numpy()
    img, full, res = {}, {}, {"image_only": {}, "full_card": {}, "differences": {}}
    for t, (ids, y, groups) in data.items():
        img[t] = [jl(RAW / f"fair_vlmimg/out/img_{JSONL[t]}.jsonl", ids)] + \
                 [jl(RAW / f"fair_vlmimgseeds/out/img_s{s}_{JSONL[t]}.jsonl", ids) for s in (1, 2)]
        per = [round(ba(y, agri(P)), 4) for P in img[t]]
        m3 = np.mean(img[t], 0)
        res["image_only"][t] = {"BA_seeds": per, "BA_mean_of_seeds": round(float(np.mean(per)), 4),
                                "BA_sd_of_seeds": round(float(np.std(per, ddof=1)), 4), "BA_three_seed_model": round(ba(y, agri(m3)), 4)}
        if t == "fn":
            res["image_only"][t]["macro_f1_seeds"] = [round(macro_f1(yfn4, P.argmax(1)), 4) for P in img[t]]
            res["image_only"][t]["macro_f1_three_seed_model"] = round(macro_f1(yfn4, m3.argmax(1)), 4)
        if t in ("e2", "e3"):
            seed0 = {"e2": R / "rs_forestnet_v4/probs/M3c.npz", "e3": R / "rs_forestnet_v5/probs/M3c_E3.npz"}[t]
            full[t] = [npz(seed0, ids)] + [jl(R / f"rs_forestnet_v5/raw/seeds/out/m3c_s{s}_{t}.jsonl", ids) for s in (1, 2)]
            perf = [round(ba(y, agri(P)), 4) for P in full[t]]
            f3 = np.mean(full[t], 0)
            res["full_card"][t] = {"BA_seeds": perf, "BA_mean_of_seeds": round(float(np.mean(perf)), 4),
                                   "BA_sd_of_seeds": round(float(np.std(perf, ddof=1)), 4), "BA_three_seed_model": round(ba(y, agri(f3)), 4)}
            res["differences"][f"{t}: image-only (3 seeds) - full card (3 seeds)"] = boot(y, agri(m3), agri(f3), groups, rng)
        cnn = {"fn": R / "rs_forestnet_v3/probs/M2.npz", "sidn": R / "rs_forestnet_v3/probs/M2.npz",
               "e2": R / "rs_forestnet_v4/probs/M2_e2.npz", "e3": R / "rs_forestnet_v5/probs/M2_E3.npz"}[t]
        res["differences"][f"{t}: image-only (3 seeds) - CNN 2.25 km"] = boot(y, agri(m3), agri(npz(cnn, ids)), groups, rng)
        c3 = np.mean([npz(RAW / f"fair_cnn/out/img3km_s{s}_{'fn' if t == 'fn' else t}.npz", ids) for s in (0, 1, 2)], 0)
        res["differences"][f"{t}: image-only (3 seeds) - CNN 3 km"] = boot(y, agri(m3), agri(c3), groups, rng)
    (R / "rs_forestnet_robust/fair_vlmimg_seeds.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
