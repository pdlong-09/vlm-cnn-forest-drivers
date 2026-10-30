"""Driver attribution: early attribution with the threshold calibrated on the ForestNet calibration half and applied unchanged to
every target set, with one CNN throughout (the ResNet-18 run whose weights were kept and applied to the external sets).

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/early_stop_source.py

ForestNet test: four classes, lambda calibrated on the four-class probabilities of the calibration half. Sims sets:
binary conversion to agriculture, lambda calibrated on the binary probabilities of the calibration half (plantation and
smallholder agriculture counted as agriculture). Learn-then-test as in rsdrivers/earlystop_v2.py (alpha 0.05,
delta 0.10). Writes runs/rs_forestnet_robust/early_stop_source.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers import earlystop_v2 as es  # noqa: E402
from rsdrivers.features import LABELS4  # noqa: E402
from rsdrivers.io import INTERIM, events  # noqa: E402

R = ROOT / "runs"
CNN_FN = R / "rs_forestnet_v3/probs/M2.npz"
VLM_FN = R / "rs_forestnet/probs/M3c_val_cal+test.npz"
SETS = {"Indonesia outside ForestNet": ("rs_forestnet_sims", 100000, R / "rs_forestnet_v3/probs/M2.npz", R / "rs_forestnet_v3/probs/M3c.npz"),
        "E2": ("rs_forestnet_sims2", 200000, R / "rs_forestnet_v4/probs/M2_e2.npz", R / "rs_forestnet_v4/probs/M3c.npz")}


def load(path, ids):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    return np.stack([probs[pos[int(e)]] for e in ids])


def to_bin(P):
    a = P[..., 0] + P[..., 1]
    return np.stack([1 - a, a], -1)


def row(lam, P, y):
    r = es.evaluate(P, y, lam, 0.0)[0]
    return {"lambda": round(float(lam), 2), "gap": round(r["gap_loss"], 4), "mean_stop": round(r["mean_stop"], 3),
            "share_k0": round(r["stop_share"][0], 3), "n": int(len(y))}


def main():
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    cal, te = fn[fn.part == "val_cal"], fn[fn.part == "test"]
    yc, yt = cal.merged_label.map(LABELS4.index).to_numpy(), te.merged_label.map(LABELS4.index).to_numpy()
    Pc = {"CNN": load(CNN_FN, cal.eid), "VLM": load(VLM_FN, cal.eid)}
    Pt = {"CNN": load(CNN_FN, te.eid), "VLM": load(VLM_FN, te.eid)}
    for P in (Pc, Pt):
        P["CNN + VLM"] = (P["CNN"] + P["VLM"]) / 2
    out = {"ForestNet test (4 classes)": {}}
    lam4 = {m: es.calibrate(Pc[m], yc, 0.05, 0.10, 0.0) for m in Pc}
    lamb = {m: es.calibrate(to_bin(Pc[m]), (yc <= 1).astype(int), 0.05, 0.10, 0.0) for m in Pc}
    for m in Pc:
        out["ForestNet test (4 classes)"][m] = row(lam4[m], Pt[m], yt)
    for name, (d, off, pc, pv) in SETS.items():
        ev = pd.read_csv(ROOT / "data/interim" / d / "events.csv")
        ids = off + ev.sid
        P = {"CNN": load(pc, ids), "VLM": load(pv, ids)}
        P["CNN + VLM"] = (P["CNN"] + P["VLM"]) / 2
        y = ev.label_agri.to_numpy()
        groups = [(name, np.ones(len(ev), bool))]
        if "group" in ev:
            groups += [(f"E2 {g}", (ev.group == g).to_numpy()) for g in ("Africa", "Asia-Pacific", "Latin America")]
        for gname, m_ in groups:
            out[gname] = {m: row(lamb[m], to_bin(P[m])[m_], y[m_]) for m in P}
    (R / "rs_forestnet_robust/early_stop_source.json").write_text(json.dumps(out, indent=1))
    for s, v in out.items():
        print(f"{s:28s}", " | ".join(f"{m}: lam {r['lambda']:.2f} gap {r['gap']:.3f} stop {r['mean_stop']:.2f}" for m, r in v.items()))


if __name__ == "__main__":
    main()
