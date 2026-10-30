"""Check that cards rendered on the lab server (render.py, venv_render) equal local cards byte for byte.

    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/verify_render.py [--n 60]

Local reference: rsdrivers.card.grid_image saved as JPEG q92 (the renderer used for every run so far). Writes
runs/rs_forestnet/verify_render.json; server-side rendering may only be used if all cards are identical.
"""
import argparse
import io
import json
import random
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers.card import grid_image  # noqa: E402
from rsdrivers.io import INTERIM, RUNS, events, load_region  # noqa: E402
from rsdrivers.lab import Lab  # noqa: E402
from rsdrivers.render import job_for  # noqa: E402

REMOTE = "/home/user/llm4deforest/paperA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    ev = events()
    random.seed(20261022)
    picks = [(int(e), random.randint(0, 4)) for e in random.sample(range(len(ev)), a.n)]
    run = f"verify_{time.strftime('%Y%m%d%H%M%S')}"
    local, jobs = {}, []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "run"
        (root / "chips").mkdir(parents=True)
        for eid, k in picks:
            r = ev.iloc[eid]
            buf = io.BytesIO()
            grid_image(r, k).save(buf, "JPEG", quality=92)
            local[f"{eid}_k{k}.jpg"] = buf.getvalue()
            j = job_for(r, k, load_region(r.example_path), INTERIM / "chips")
            jobs.append(j)
            for c in j["chips"]:
                if c:
                    dst = root / "chips" / c
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes((INTERIM / "chips" / c).read_bytes())
        (root / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
        for f in ("server/render_cards.py", "rsdrivers/render.py"):
            (root / Path(f).name).write_bytes((ROOT / "paper/rs_forestnet" / f).read_bytes())
        tar = Path(td) / f"{run}.tar"
        with tarfile.open(tar, "w") as t:
            t.add(root, arcname="run")
        lab = Lab()
        d = f"{REMOTE}/data/{run}"
        try:
            lab.ssh(f"mkdir -p {d}")
            lab.put_chunked(tar, f"{d}/p.tar")
            print(lab.ssh(f"cd {d} && tar xf p.tar && rm p.tar && {REMOTE}/venv_render/bin/python run/render_cards.py "
                          f"--jobs run/jobs.json --chips run/chips --out out && tar cf out.tar out"), flush=True)
            lab.get(f"{d}/out.tar", Path(td) / "out.tar")
        finally:
            print("server data left:", lab.retry(lambda: lab.ssh(f"rm -rf {d}; ls -A {REMOTE}/data | wc -l"), "clean").strip())
            lab.close()
        with tarfile.open(Path(td) / "out.tar") as t:
            t.extractall(td, filter="data")
        same_bytes = same_px = 0
        for name, ref in local.items():
            got = (Path(td) / "out" / name).read_bytes()
            same_bytes += got == ref
            same_px += np.array_equal(np.asarray(Image.open(io.BytesIO(got))), np.asarray(Image.open(io.BytesIO(ref))))
    res = {"n": len(local), "identical_bytes": same_bytes, "identical_pixels": same_px, "all_identical": same_bytes == len(local)}
    print(json.dumps(res))
    (RUNS / "verify_render.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
