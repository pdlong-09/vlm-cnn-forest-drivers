"""Recompute the main results of the article from the files of this release alone (data/ and predictions/).

    python code/scripts/reproduce_main_results.py

Needs numpy and pandas only: no imagery, no GPU and no Earth Engine access. For the CNN (ResNet-18, mean of three seeds)
and the fine-tuned VLM (seed 0) at k = 4 it prints the four-class macro-F1 on the ForestNet test set, the balanced
accuracy (BA) of conversion to agriculture on the four evaluation sets, and the VLM - CNN difference with 95% paired
bootstrap intervals (2000 resamples of cells and, for E2 and E3, of countries; same resampling and seed as
code/server/robust_a.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
B, SEED = 2000, 20261021
AGRI = ("Plantation", "Smallholder agriculture")
PROBS = ["p_plantation", "p_smallholder", "p_grassland", "p_other"]
CLASSES = [*AGRI, "Grassland shrubland", "Other"]


def probs(name, eids):
    """Class probabilities at k = 4, in the order of eids."""
    d = pd.read_csv(ROOT / "predictions" / f"{name}.csv")
    return d[d.k == 4].set_index("eid").loc[eids, PROBS].to_numpy()


def ba(y, pred):
    return 0.5 * (pred[y == 1].mean() + 1 - pred[y == 0].mean())


def macro_f1(y, pred):
    f1 = []
    for c in range(4):
        tp, fp, fn = ((pred == c) & (y == c)).sum(), ((pred == c) & (y != c)).sum(), ((pred != c) & (y == c)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def cell_ci(y, a, b, rng):
    n = len(y)
    diffs = [ba(y[i], a[i]) - ba(y[i], b[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(diffs, [2.5, 97.5])


def country_ci(y, a, b, country, rng):
    names = np.unique(country)
    members = {c: np.flatnonzero(country == c) for c in names}
    diffs = []
    for _ in range(B):
        i = np.concatenate([members[c] for c in rng.choice(names, len(names))])
        if y[i].min() != y[i].max():
            diffs.append(ba(y[i], a[i]) - ba(y[i], b[i]))
    return np.percentile(diffs, [2.5, 97.5])


def agri(p):
    return (p[:, 0] + p[:, 1] >= 0.5).astype(int)


def main():
    fn = pd.read_csv(ROOT / "data/forestnet_events.csv")
    fn = fn[fn.part == "test"]
    cnn, vlm = probs("forestnet_cnn_resnet18", fn.eid), probs("forestnet_vlm", fn.eid)
    y4 = fn.merged_label.map(CLASSES.index).to_numpy()
    print(f"ForestNet test, four classes (n = {len(fn)}): macro-F1 CNN {macro_f1(y4, cnn.argmax(1)):.3f}, "
          f"VLM {macro_f1(y4, vlm.argmax(1)):.3f}, CNN + VLM {macro_f1(y4, ((cnn + vlm) / 2).argmax(1)):.3f}\n")
    sets = [("ForestNet test", "forestnet", fn.assign(label_agri=fn.merged_label.isin(AGRI).astype(int))),
            ("Indonesia outside ForestNet", "indonesia", pd.read_csv(ROOT / "data/indonesia_cells.csv")),
            ("E2, regional validation", "e2", pd.read_csv(ROOT / "data/e2_cells.csv")),
            ("E3, held-out test", "e3", pd.read_csv(ROOT / "data/e3_cells.csv"))]
    print("Balanced accuracy of conversion to agriculture at k = 4")
    for title, name, d in sets:
        y = d.label_agri.to_numpy()
        a, b = agri(probs(f"{name}_vlm", d.eid)), agri(probs(f"{name}_cnn_resnet18", d.eid))
        rng = np.random.default_rng(SEED)
        lo, hi = cell_ci(y, a, b, rng)
        line = (f"{title} (n = {len(d)}): CNN {ba(y, b):.3f}, VLM {ba(y, a):.3f}, difference {ba(y, a) - ba(y, b):+.3f} "
                f"(95% interval, cells {lo:.4f} to {hi:.4f}")
        if "country" in d:
            lo, hi = country_ci(y, a, b, d.country.to_numpy(), rng)
            line += f"; countries {lo:.4f} to {hi:.4f}"
        print(line + ")")


if __name__ == "__main__":
    main()
