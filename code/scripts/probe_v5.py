"""design notes, A-probe: multinomial logistic regression on the 128-d AlphaEarth features of the ForestNet training
events; C in {0.01, 0.1, 1, 10} chosen by 4-class macro-F1 on val_sel (tie: smaller C).

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/probe_v5.py [--e3]

Writes runs/rs_forestnet_v5/probs/A_probe.npz (eid, probs [n, 4] for ForestNet val/test and E2; E3 rows only with --e3,
which runs after the E2 choice is hashed) and A_probe_choice.json. Reads no E2 or E3 label.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet/scripts"))
from fn_v4 import fn_events  # noqa: E402
from rsdrivers.features import LABELS4  # noqa: E402

V5 = ROOT / "data/interim/rs_forestnet_v5"
OUT = ROOT / "runs/rs_forestnet_v5/probs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", action="store_true")
    a = ap.parse_args()
    z = np.load(V5 / "aef.npz")
    eid, F = z["eid"], z["feat"]
    pos = {int(e): i for i, e in enumerate(eid)}
    fn = fn_events()
    part = {p: g for p, g in fn.groupby("part")}

    def xy(p):
        g = part[p]
        g = g[g.eid.isin(pos)]
        return F[[pos[e] for e in g.eid]], g.merged_label.map(LABELS4.index).to_numpy(), g.eid.to_numpy()
    Xtr, ytr, _ = xy("train")
    Xvs, yvs, _ = xy("val_sel")
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda X: (X - mean) / std  # noqa: E731
    scores, models = {}, {}
    for C in (0.01, 0.1, 1, 10):
        m = LogisticRegression(C=C, max_iter=5000).fit(norm(Xtr), ytr)
        scores[C] = f1_score(yvs, m.predict(norm(Xvs)), average="macro")
        models[C] = m
    best = max(scores, key=lambda c: (round(scores[c], 10), -c))
    m = models[best]
    keep = (eid < 300000) if not a.e3 else np.ones(len(eid), bool)
    train_ids = set(part["train"].eid)
    keep &= ~np.isin(eid, list(train_ids))  # no in-sample probabilities
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "A_probe.npz", eid=eid[keep], probs=m.predict_proba(norm(F[keep])))
    (OUT / "A_probe_choice.json").write_text(json.dumps({"C": best, "val_sel_macro_f1": scores, "n_train": int(len(ytr)),
                                                         "rows": int(keep.sum()), "with_e3": a.e3}, indent=1))
    print("C", best, {c: round(s, 4) for c, s in scores.items()}, "rows", int(keep.sum()))


if __name__ == "__main__":
    main()
