"""Driver attribution v3, on server1: pack the chips for the v1 CNN (server/cnn_train.py, unchanged) exactly like scripts/m2_cnn.py.

    python fn_pack_v3.py meta_cnn.csv chips chips.npz

meta_cnn.csv: eid, part, y, year, chipdir (relative to chips/; ForestNet "<eid>", Sims "sims/<sid>"). Years -1..4 of
each event, central 96 px of the 128 px chip, missing chips as zeros. Sims rows have part "ext" and y = -1.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CROP = 96

rows = list(csv.DictReader(open(sys.argv[1])))
chips = Path(sys.argv[2])
X = np.zeros((len(rows), 6, CROP, CROP, 3), dtype=np.uint8)
o = (128 - CROP) // 2
missing = 0
for i, r in enumerate(rows):
    for j, t in enumerate(range(-1, 5)):
        p = chips / r["chipdir"] / f"k{t}_{int(r['year']) + t}.png"
        if p.exists():
            X[i, j] = np.asarray(Image.open(p).convert("RGB"))[o:o + CROP, o:o + CROP]
        else:
            missing += 1
np.savez_compressed(sys.argv[3], X=X, y=np.array([int(r["y"]) for r in rows]), part=np.array([r["part"] for r in rows]),
                    eid=np.array([int(r["eid"]) for r in rows]))
print(f"packed {len(rows)} events, {missing} missing chips")
