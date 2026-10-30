"""M3a/M3b (and M3c with --lora): VLM probabilities on evidence cards, run on server1 GPU 1, one batch at a time.

    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/m3_vlm.py --name M3a --parts val_sel [--shots 0]
    ... --name M3b --parts val_sel --shots 8

Per run: build cards (image grid + fact sheet) for every event x step k x 4 cyclic option orders -> upload a tar with the
JSONL to ~/llm4deforest/paperA/data/<run> -> run server/vlm_infer.py with the project venv on GPU 1 -> download the
output -> delete the server folder and check it is gone (lab-server batch rule). Probabilities are the softmax over
the option letters, averaged over the four orders, and saved to runs/rs_forestnet/probs/<name>_<parts>.npz.
"""
import argparse
import io
import json
import math
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))

from rsdrivers.card import OPTIONS, fact_sheet, grid_image, question  # noqa: E402
from rsdrivers.features import LABELS4, series_wide, usable  # noqa: E402
from rsdrivers.io import INTERIM, RUNS, events, load_region  # noqa: E402
from rsdrivers.render import job_for  # noqa: E402
from rsdrivers.lab import Lab, wait_gpu  # noqa: E402

REMOTE = "/home/user/llm4deforest/paperA"
MODEL = "/home/user/llm4deforest/models/Qwen3.5-9B"
ORDERS = [LABELS4[i:] + LABELS4[:i] for i in range(4)]  # 4 cyclic orders: every class once in every position
STEPS = range(0, 5)


def build(ev, wide, static, steps, shots_ev, tmp: Path, server_render: bool = False):
    """server_render=False (default, used by every run up to the frozen test run): cards rendered here as JPEG.
    server_render=True: only the raw chips and the render jobs travel; the server renders the identical cards with
    rsdrivers/render.py under Pillow 13.0.0 (verified byte for byte: runs/rs_forestnet/verify_render.json)."""
    imgs = tmp / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    items, jobs = [], {}

    def card(r, k):
        name = f"{r.eid}_k{k}.jpg"
        if server_render:
            if name not in jobs:
                j = job_for(r, k, load_region(r.example_path), INTERIM / "chips")
                jobs[name] = j
                for c in j["chips"]:
                    if c and not (tmp / "chips" / c).exists():
                        (tmp / "chips" / c).parent.mkdir(parents=True, exist_ok=True)
                        (tmp / "chips" / c).write_bytes((INTERIM / "chips" / c).read_bytes())
        elif not (imgs / name).exists():
            grid_image(r, k).save(imgs / name, quality=92)
        w = wide.loc[r.eid].to_dict() if r.eid in wide.index else {}
        return name, fact_sheet(r, w, static.loc[r.eid], k)

    for r in ev.itertuples():
        for k in steps:
            name, facts = card(r, k)
            for o, order in enumerate(ORDERS):
                q, mapping = question(order)
                shots = []
                for s in shots_ev.itertuples():
                    sname, sfacts = card(s, k)
                    sq, smap = question(order)
                    ans = next(L for L, lab in smap.items() if lab == s.merged_label)
                    shots.append({"text": sfacts + "\n\n" + sq, "image": sname, "answer": ans})
                items.append({"id": f"{r.eid}|{k}|{o}", "text": facts + "\n\n" + q, "image": name, "shots": shots})
    (tmp / "batch.jsonl").write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    if server_render:
        (tmp / "jobs.json").write_text(json.dumps(list(jobs.values())), encoding="utf-8")
        for f in ("server/render_cards.py", "rsdrivers/render.py"):
            (tmp / Path(f).name).write_bytes((ROOT / "paper/rs_forestnet" / f).read_bytes())
    tar = tmp.parent / (tmp.name + ".tar")
    with tarfile.open(tar, "w") as t:
        t.add(tmp, arcname="run")
    return tar, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", default="val_sel")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--lora", help="local adapter folder (uploaded with the batch, deleted afterwards)")
    ap.add_argument("--server-render", action="store_true", help="send raw chips; render identical cards on the server")
    a = ap.parse_args()
    ev = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    target = ev[ev.part.isin(a.parts.split(","))]
    if a.limit:
        target = target.sample(a.limit, random_state=0)
    shots_ev = ev.iloc[0:0]
    if a.shots:
        tr = ev[ev.part == "train"]
        shots_ev = pd.concat([g.sample(a.shots // 4, random_state=1) for _, g in tr.groupby("merged_label")])
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    run = f"{a.name}_{a.parts.replace(',', '+')}_{time.strftime('%Y%m%d%H%M%S')}"
    with tempfile.TemporaryDirectory() as td:
        if a.lora:  # the adapter travels with the batch so nothing derived from the data stays on the server
            import shutil
            shutil.copytree(a.lora, Path(td) / run / "adapter")
        tar, n = build(target, wide, static, STEPS, shots_ev, Path(td) / run, server_render=a.server_render)
        print(f"{n} prompts, payload {tar.stat().st_size / 1e6:.1f} MB", flush=True)
        lab = Lab()
        try:
            lab.ssh(f"mkdir -p {REMOTE}/data")
            lab.put_chunked(tar, f"{REMOTE}/data/{run}.tar")
            lab.put(ROOT / "paper/rs_forestnet/server/vlm_infer.py", f"{REMOTE}/vlm_infer.py")
            lora = "--lora run/adapter" if a.lora else ""
            # few-shot prompts carry 9 images: fewer parallel sequences and more headroom (OOM on 22/10 otherwise)
            if a.shots:
                lora += " --max-num-seqs 1 --headroom-gib 1.0"
            render = (f" && {REMOTE}/venv_render/bin/python run/render_cards.py --jobs run/jobs.json --chips run/chips "
                      f"--out run/images > run/render_log.txt 2>&1" if a.server_render else "")
            # 1) unpack (and render) once, detached; 2) inference, detached, retried on start-up OOM without re-upload
            code = lab.run_detached(f"{REMOTE}/data", f"tar xf {run}.tar && rm {run}.tar{render}",
                                    f"{REMOTE}/data/{run}.prep", poll_s=20)
            if code != 0:
                raise RuntimeError(f"unpack/render failed on the server (exit {code})")
            for attempt in range(3):
                gpu = wait_gpu(lab)  # chosen right before the run: other users come and go
                print("using GPU", gpu, flush=True)
                cmd = (f"source ~/llm4deforest/env.sh >/dev/null 2>&1; deactivate 2>/dev/null; export CUDA_VISIBLE_DEVICES={gpu}; "
                       f"{REMOTE}/venv/bin/python {REMOTE}/vlm_infer.py --model {MODEL} --input run/batch.jsonl "
                       f"--images run/images --output run/out.jsonl {lora} > run/log.txt 2>&1")
                marker = f"{REMOTE}/data/{run}.exit{attempt}"
                # detached: a long ssh session over bore can drop (it killed the first M3b test run on 22/10/2026)
                code = lab.run_detached(f"{REMOTE}/data", cmd, marker)
                tail = lab.retry(lambda: lab.ssh(f"tail -c 2000 {REMOTE}/data/run/log.txt"), "log")
                print("server exit code", code, "|", tail.strip()[-300:], flush=True)
                if code == 0:
                    break
                if "out of memory" not in tail.lower():
                    raise RuntimeError(f"inference failed on the server (exit {code})")
                print("start-up OOM (GPU taken by another job); waiting and retrying", flush=True)
            else:
                raise RuntimeError("inference failed 3 times with OOM")
            out_local = RUNS / "raw" / f"{run}.jsonl"
            out_local.parent.mkdir(parents=True, exist_ok=True)
            lab.get(f"{REMOTE}/data/run/out.jsonl", out_local)
        finally:
            try:  # keep the server log locally, also when the run failed
                lab.get(f"{REMOTE}/data/run/log.txt", RUNS / "raw" / f"{run}_log.txt")
            except RuntimeError:
                pass
            left = lab.retry(lambda: lab.ssh(f"rm -rf {REMOTE}/data/run {REMOTE}/data/{run}.tar {REMOTE}/data/{run}.exit* "
                                             f"{REMOTE}/data/{run}.prep*; ls -A {REMOTE}/data | wc -l"),
                             "clean server")
            print("server data left:", left.strip(), flush=True)
            lab.close()
    rows = [json.loads(l) for l in open(out_local, encoding="utf-8")]
    ids = target.eid.to_numpy()
    pos = {e: i for i, e in enumerate(ids)}
    acc = np.zeros((len(ids), len(STEPS), 4))
    cnt = np.zeros((len(ids), len(STEPS)))
    for r in rows:
        eid, k, o = map(int, r["id"].split("|"))
        _, mapping = question(ORDERS[o])
        lp = np.array([r["logprobs"][L] for L in OPTIONS])
        p = np.exp(lp - np.max(lp)) if np.isfinite(lp).any() else np.full(4, 0.25)
        p = p / p.sum()
        for L, pr in zip(OPTIONS, p):
            acc[pos[eid], k, LABELS4.index(mapping[L])] += pr
        cnt[pos[eid], k] += 1
    probs = acc / np.maximum(cnt[..., None], 1)
    y = target.merged_label.map({l: i for i, l in enumerate(LABELS4)}).to_numpy()
    out = RUNS / "probs" / f"{a.name}_{a.parts.replace(',', '+')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, probs=probs, eid=ids, part=target.part.to_numpy(), y=y)
    from sklearn.metrics import f1_score
    if "test" not in a.parts:
        for k in STEPS:
            print(f"k={k}: macro-F1 {f1_score(y, probs[:, k].argmax(1), average='macro'):.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
