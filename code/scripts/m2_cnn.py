"""M2: pack the yearly chips, train server/cnn_train.py on server1 GPU 1, fetch probabilities, clean the server.

    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/m2_cnn.py [--epochs 40] [--seeds 0,1,2]

Trains one model per seed (default 0,1,2, design notes) and writes the mean probabilities to runs/rs_forestnet/probs/M2.npz. Missing chips (no data for a year) are zeros, like black chip pixels.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))

from rsdrivers.features import LABELS4  # noqa: E402
from rsdrivers.io import INTERIM, RUNS, events  # noqa: E402
from rsdrivers.lab import Lab, wait_gpu  # noqa: E402

REMOTE = "/home/user/llm4deforest/paperA"
CROP = 96  # central 2.25 km of the 3 km chip


def pack(path: Path):
    ev = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    X = np.zeros((len(ev), 6, CROP, CROP, 3), dtype=np.uint8)
    o = (128 - CROP) // 2
    missing = 0
    for i, r in enumerate(ev.itertuples()):
        for j, t in enumerate(range(-1, 5)):
            p = INTERIM / "chips" / str(r.eid) / f"k{t}_{int(r.year) + t}.png"
            if p.exists():
                X[i, j] = np.asarray(Image.open(p).convert("RGB"))[o:o + CROP, o:o + CROP]
            else:
                missing += 1
    y = ev.merged_label.map({l: i for i, l in enumerate(LABELS4)}).to_numpy()
    np.savez_compressed(path, X=X, y=y, part=ev.part.to_numpy(), eid=ev.eid.to_numpy())
    print(f"packed {len(ev)} events, {missing} missing chips, {path.stat().st_size / 1e6:.0f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    local = INTERIM / "chips_packed.npz"
    if not local.exists():
        pack(local)
    run = f"M2_{time.strftime('%Y%m%d%H%M%S')}"
    lab = Lab()
    try:
        lab.ssh(f"mkdir -p {REMOTE}/data/{run}")
        lab.put_chunked(local, f"{REMOTE}/data/{run}/chips.npz")
        lab.put(ROOT / "paper/rs_forestnet/server/cnn_train.py", f"{REMOTE}/cnn_train.py")
        gpu = wait_gpu(lab)
        print("using GPU", gpu, flush=True)
        seeds = [int(s) for s in a.seeds.split(",")]
        (RUNS / "raw").mkdir(parents=True, exist_ok=True)
        for s in seeds:
            cmd = (f"cd {REMOTE}/data/{run} && source ~/llm4deforest/env.sh >/dev/null 2>&1; deactivate 2>/dev/null; "
                   f"export CUDA_VISIBLE_DEVICES={gpu}; {REMOTE}/venv/bin/python {REMOTE}/cnn_train.py --data chips.npz "
                   f"--out probs_s{s}.npz --epochs {a.epochs} --seed {s} > log_s{s}.txt 2>&1; tail -2 log_s{s}.txt")
            print(f"seed {s}:", lab.ssh(cmd, timeout=6 * 3600), flush=True)
            lab.get(f"{REMOTE}/data/{run}/probs_s{s}.npz", RUNS / "raw" / f"{run}_probs_s{s}.npz")
            lab.get(f"{REMOTE}/data/{run}/log_s{s}.txt", RUNS / "raw" / f"{run}_log_s{s}.txt")
    finally:
        print("server data left:", lab.retry(lambda: lab.ssh(f"rm -rf {REMOTE}/data/{run}; ls -A {REMOTE}/data | wc -l"),
                                                "clean server").strip())
        lab.close()
    zs = [np.load(RUNS / "raw" / f"{run}_probs_s{s}.npz", allow_pickle=True) for s in seeds]
    (RUNS / "probs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / "probs" / "M2.npz", probs=np.mean([z["probs"] for z in zs], axis=0), eid=zs[0]["eid"],
                        part=zs[0]["part"], y=zs[0]["y"])
    print("seeds:", seeds, "val_sel best per seed:", [float(z["best_val_sel"]) for z in zs])
    print("saved", RUNS / "probs" / "M2.npz (mean of seeds)")


if __name__ == "__main__":
    main()
