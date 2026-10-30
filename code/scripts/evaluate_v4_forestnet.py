"""design notes, exploratory: M5e, M5t, M5tr on the ForestNet test split (earlier development stages also scored this split; nothing reported in the article was selected on it).

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/evaluate_v4_forestnet.py

4-class macro-F1 and accuracy at k = 4, and binary BA (agriculture = plantation + smallholder), against the v3 CNN M2
(3-seed mean), the v1 VLM M3c and their average M4, on the same test events. Writes runs/rs_forestnet_v4/eval_v4_forestnet.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
K = 4


def load(path):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read once
    return {int(e): probs[i] for i, e in enumerate(eid)}


def main():
    ref = np.load(ROOT / "runs/rs_forestnet/probs/M3c_val_cal+test.npz", allow_pickle=True)
    part, y, eid = ref["part"], ref["y"], ref["eid"]
    test = [int(e) for e, p in zip(eid, part) if p == "test"]
    ytest = np.array([int(v) for v, p in zip(y, part) if p == "test"])
    P = {"M2": load(ROOT / "runs/rs_forestnet_v3/probs/M2.npz"), "M3c": load(ROOT / "runs/rs_forestnet/probs/M3c_val_cal+test.npz")}
    for m in ("M5e", "M5t", "M5tr"):
        P[m] = load(ROOT / f"runs/rs_forestnet_v4/probs/{m}.npz")
    X = {m: np.stack([p[e][K] for e in test]) for m, p in P.items()}
    X["M4"] = (X["M2"] + X["M3c"]) / 2
    ybin = np.isin(ytest, [0, 1]).astype(int)  # LABELS4 order: plantation, smallholder agriculture, grassland/shrub, other
    out = {"n_test": len(test)}
    for m, x in X.items():
        pred = x.argmax(1)
        out[m] = {"macro_f1_4class": round(f1_score(ytest, pred, average="macro"), 4),
                  "accuracy_4class": round(accuracy_score(ytest, pred), 4),
                  "BA_binary": round(balanced_accuracy_score(ybin, (x[:, 0] + x[:, 1] >= 0.5).astype(int)), 4)}
        print(m, out[m])
    (ROOT / "runs/rs_forestnet_v4/eval_v4_forestnet.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
