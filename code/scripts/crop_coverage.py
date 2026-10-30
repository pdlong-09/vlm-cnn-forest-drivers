"""Share of the loss polygon inside the central 2.25 km crop that the CNN reads (3 km chips, central 96 of 128 px).

    .venv/Scripts/python paper/rs_forestnet/scripts/crop_coverage.py

ForestNet polygons are in the 15 m pixel grid of the 332 px images centred on the event; the crop is the central
2250 m, i.e. 150 px. External polygons (Sims cells) are in lon/lat around the cell centre. Prints the area-weighted and
the median per-event share of the loss area inside the crop.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box, shape
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers.io import INTERIM, PIX_M, SIZE, events, load_region  # noqa: E402

HALF_M = 2250 / 2


def report(name, shares, areas):
    shares, areas = np.array(shares), np.array(areas)
    print(f"{name}: n={len(shares)}, area-weighted {np.sum(shares * areas) / np.sum(areas):.3f}, "
          f"median {np.median(shares):.3f}, share of events fully inside {(shares > 0.999).mean():.3f}")


def main():
    half = HALF_M / PIX_M
    crop = box(SIZE / 2 - half, SIZE / 2 - half, SIZE / 2 + half, SIZE / 2 + half)
    sh, ar = [], []
    for r in events().itertuples():
        g = make_valid(load_region(r.example_path))
        if g.area > 0:
            sh.append(g.intersection(crop).area / g.area)
            ar.append(g.area)
    report("ForestNet", sh, ar)
    for name, d in (("Indonesia", "rs_forestnet_sims"), ("E2", "rs_forestnet_sims2"), ("E3", "rs_forestnet_sims3")):
        gj = json.loads((INTERIM.parent / d / "polygons.geojson").read_text(encoding="utf-8"))
        ev = pd.read_csv(INTERIM.parent / d / "events.csv").set_index("sid")
        sh, ar = [], []
        for f in gj["features"]:
            sid = f["properties"].get("sid")
            if sid not in ev.index:
                continue
            r = ev.loc[sid]
            dlat = HALF_M / 110574.0
            dlon = HALF_M / (111320.0 * math.cos(math.radians(r.Latitude)))
            c = box(r.Longitude - dlon, r.Latitude - dlat, r.Longitude + dlon, r.Latitude + dlat)
            g = make_valid(shape(f["geometry"]))
            if g.area > 0:
                sh.append(g.intersection(c).area / g.area)
                ar.append(g.area)
        report(name, sh, ar)


if __name__ == "__main__":
    main()
