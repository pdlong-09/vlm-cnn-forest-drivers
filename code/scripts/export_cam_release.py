"""Export the Cam-ForestNet in-domain check (CNN and VLM, spatial split) to the release, in the release format."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers import cam  # noqa: E402

RUNS = ROOT / "runs/rs_forestnet_cam/probs"
OUT = ROOT / "release/rs_forestnet_release"
NAMES = ["Plantation", "Smallholder agriculture", "Grassland shrubland", "Other"]
COLS = ["p_plantation", "p_smallholder", "p_grassland", "p_other"]


def main():
    ev = cam.events()
    parts = pd.read_csv(ROOT / "data/interim/rs_forestnet_cam/parts.csv")
    ev = ev.merge(parts, on="eid", how="inner")
    ev = ev.rename(columns={"split": "split_official", "part": "part_spatial"})
    ev[["eid", "split_official", "part_spatial", "label", "merged_label", "latitude", "longitude", "year"]].to_csv(
        OUT / "data/cameroon_events.csv", index=False)
    for m, name in (("M2", "cnn_resnet18"), ("M3c", "vlm")):
        z = np.load(RUNS / f"{m}.npz", allow_pickle=True)
        eid, y, P = z["eid"].astype(int), z["y"].astype(int), z["probs"]
        lab = ev.set_index("eid").loc[eid, "merged_label"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()
        assert (lab == y).all(), f"class order differs for {m}"
        rows = [{"eid": e, "k": k, **dict(zip(COLS, np.round(P[i, k], 5)))} for i, e in enumerate(eid) for k in range(5)]
        pd.DataFrame(rows).to_csv(OUT / f"predictions/cameroon_{name}.csv", index=False)
        t = z["part"] == "test"
        print(name, "test events", int(t.sum()), "macro-F1 k=4",
              round(f1_score(y[t], P[t, 4].argmax(1), average="macro", labels=[0, 1, 2, 3]), 3),
              "grassland share of predictions", round((P[t, 4].argmax(1) == 2).mean(), 3))


if __name__ == "__main__":
    main()
