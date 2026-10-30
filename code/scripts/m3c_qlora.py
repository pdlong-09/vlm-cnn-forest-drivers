"""M3c training: build train cards (every event x step k x 4 option orders), train server/qlora_train.py on server1
GPU 1, fetch the per-epoch adapters to runs/rs_forestnet/adapters/<run>/, clean the server.

    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/m3c_qlora.py [--epochs 2] [--limit N]

Selection afterwards (design notes): score each epoch on val_sel with
    m3_vlm.py --name M3c_ep<N> --parts val_sel --lora runs/rs_forestnet/adapters/<run>/epoch<N>
and keep the epoch with the higher mean macro-F1 over k.
"""
import argparse
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers.card import fact_sheet, grid_image, question  # noqa: E402
from rsdrivers.features import LABELS4, series_wide, usable  # noqa: E402
from rsdrivers.io import INTERIM, RUNS, events, load_region  # noqa: E402
from rsdrivers.render import job_for  # noqa: E402
from rsdrivers.lab import Lab, wait_gpu  # noqa: E402

REMOTE = "/home/user/llm4deforest/paperA"
MODEL = "/home/user/llm4deforest/models/Qwen3.5-9B"
ORDERS = [LABELS4[i:] + LABELS4[:i] for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, help="smoke test on a few training events")
    ap.add_argument("--server-render", action="store_true", help="send raw chips; render identical cards on the server")
    a = ap.parse_args()
    ev = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    tr = ev[ev.part == "train"]
    if a.limit:
        tr = tr.sample(a.limit, random_state=0)
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    run = f"M3c_{time.strftime('%Y%m%d%H%M%S')}"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "run"
        (root / "images").mkdir(parents=True)
        rows, jobs = [], []
        for r in tr.itertuples():
            for k in range(5):
                name = f"{r.eid}_k{k}.jpg"
                if a.server_render:  # identical cards rendered on the server (scripts/verify_render.py)
                    j = job_for(r, k, load_region(r.example_path), INTERIM / "chips")
                    jobs.append(j)
                    for c in j["chips"]:
                        if c and not (root / "chips" / c).exists():
                            (root / "chips" / c).parent.mkdir(parents=True, exist_ok=True)
                            (root / "chips" / c).write_bytes((INTERIM / "chips" / c).read_bytes())
                else:
                    grid_image(r, k).save(root / "images" / name, quality=92)
                facts = fact_sheet(r, wide.loc[r.eid].to_dict(), static.loc[r.eid], k)
                texts, answers = [], []
                for order in ORDERS:
                    q, mapping = question(order)
                    texts.append(facts + "\n\n" + q)
                    answers.append(next(L for L, lab in mapping.items() if lab == r.merged_label))
                rows.append({"eid": int(r.eid), "k": k, "image": name, "texts": texts, "answers": answers})
        (root / "train.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
        if a.server_render:
            (root / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            for f in ("server/render_cards.py", "rsdrivers/render.py"):
                (root / Path(f).name).write_bytes((ROOT / "paper/rs_forestnet" / f).read_bytes())
        tar = Path(td) / f"{run}.tar"
        with tarfile.open(tar, "w") as t:
            t.add(root, arcname="run")
        print(f"{len(tr)} events, {len(rows)} cards, payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
        lab = Lab()
        local = RUNS / "adapters" / run
        try:
            d = f"{REMOTE}/data/{run}"
            lab.ssh(f"mkdir -p {d}")
            lab.put_chunked(tar, f"{d}/payload.tar")
            lab.put(ROOT / "paper/rs_forestnet/server/qlora_train.py", f"{REMOTE}/qlora_train.py")
            gpu = wait_gpu(lab)
            print("using GPU", gpu, flush=True)
            render = (f"{REMOTE}/venv_render/bin/python run/render_cards.py --jobs run/jobs.json --chips run/chips "
                      f"--out run/images > render_log.txt 2>&1 && " if a.server_render else "")
            cmd = (f"cd {d} && tar xf payload.tar && rm payload.tar && {render}source ~/llm4deforest/env.sh >/dev/null 2>&1; "
                   f"deactivate 2>/dev/null; export CUDA_VISIBLE_DEVICES={gpu}; {REMOTE}/venv/bin/python "
                   f"{REMOTE}/qlora_train.py --model {MODEL} --train run/train.jsonl --images run/images --out adapters "
                   f"--epochs {a.epochs} > log.txt 2>&1; tail -4 log.txt; ls adapters")
            print(lab.ssh(cmd, timeout=24 * 3600), flush=True)
            local.mkdir(parents=True, exist_ok=True)
            lab.ssh(f"cd {d} && tar cf adapters.tar adapters log.txt")
            lab.get(f"{d}/adapters.tar", local / "adapters.tar", timeout=3600)
        finally:
            print("server data left:", lab.retry(lambda: lab.ssh(f"rm -rf {REMOTE}/data/{run}; ls -A {REMOTE}/data | wc -l"),
                                                "clean server").strip())
            lab.close()
    with tarfile.open(local / "adapters.tar") as t:
        t.extractall(local, filter="data")
    (local / "adapters.tar").unlink()
    print("adapters in", local, sorted(p.name for p in (local / "adapters").iterdir()))


if __name__ == "__main__":
    main()
