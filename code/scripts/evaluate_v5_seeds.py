"""design notes (post hoc): seed robustness of M3c on E2 and E3, binary agricultural conversion at k = 4, BA.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/evaluate_v5_seeds.py

Seed 0 is the v1 adapter (E2: runs/rs_forestnet_v4/probs/M3c.npz, E3: runs/rs_forestnet_v5/probs/M3c_E3.npz); seeds 1, 2
come from the seeds batch (runs/rs_forestnet_v5/raw/seeds). Reports BA per seed, BA of the mean of the 3 seeds'
probabilities, and M3c - M2 with paired bootstraps (2,000, seed 20261021). Writes runs/rs_forestnet_v5/eval_v5_seeds.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet/scripts"))
import evaluate_v5 as E  # noqa: E402
from fn_v5 import k4_probs  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

RAW = ROOT / "runs/rs_forestnet_v5/raw/seeds/out"


def seed_agri(path, ids):
    eids, P = k4_probs(path)
    pos = {int(e): i for i, e in enumerate(eids)}
    return np.array([P[pos[e], 0] + P[pos[e], 1] for e in ids])


def main():
    res = {}
    for mode, tag in (("dev", "e2"), ("final", "e3")):
        base, off = E.SETS[mode]
        se = pd.read_csv(base / "events.csv")
        ids, y = (off + se.sid).to_numpy(), se.label_agri.to_numpy()
        if mode == "dev":
            m3 = {0: E.agri(E.P4 / "M3c.npz", ids, True)}
            m2 = E.agri(E.P4 / "M2_e2.npz", ids, True)
        else:
            m3 = {0: E.agri(E.P5 / "M3c_E3.npz", ids)}
            m2 = E.agri(E.P5 / "M2_E3.npz", ids)
        for s in (1, 2):
            m3[s] = seed_agri(RAW / f"m3c_s{s}_{tag}.jsonl", ids)
        m3["mean of 3"] = np.mean([m3[0], m3[1], m3[2]], 0)
        ba = lambda p, i: balanced_accuracy_score(y[i], p[i] >= 0.5)  # noqa: E731
        allidx = np.arange(len(y))
        r = {"n": int(len(y)), "M2 BA": round(ba(m2, allidx), 4)}
        for s, p in m3.items():
            d = lambda i, p=p: ba(p, i) - ba(m2, i)  # noqa: E731
            r[f"M3c seed {s}"] = {"BA": round(ba(p, allidx), 4), "minus M2": round(d(allidx), 4), "ci": E.boot(d, len(y))}
        res["E2" if mode == "dev" else "E3"] = r
    (ROOT / "runs/rs_forestnet_v5/eval_v5_seeds.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
