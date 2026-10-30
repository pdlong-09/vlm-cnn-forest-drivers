"""Shared helpers of the GPU pipelines: a pool of GPUs that waits until a GPU has room, a stage runner with out-of-memory
retries, model paths, the four cyclic option orders and the holdout macro-F1 of letter probabilities."""
import argparse
import csv
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # run/code
RUN = HERE.parent
BASE = Path("/home/user/llm4deforest/paperA")
MODEL = "/home/user/llm4deforest/models/Qwen3.5-9B"
PY, PY_RENDER = BASE / "venv/bin/python", BASE / "venv_render/bin/python"
LABELS4 = ["Plantation", "Smallholder agriculture", "Grassland shrubland", "Other"]
ORDERS = [LABELS4[i:] + LABELS4[:i] for i in range(4)]
ENV = "source ~/llm4deforest/env.sh >/dev/null 2>&1; deactivate 2>/dev/null; "


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(i): int(m) for i, m in (l.split(",") for l in out.strip().splitlines())}


class Pool:
    """GPUs in preference order; one stage per GPU; a GPU used by others (>= 2000 MiB) is skipped."""

    def __init__(self, gpus):
        self.gpus, self.held, self.lock = gpus, set(), threading.Lock()

    def acquire(self, stage, hours=24):
        t0, said = time.time(), 0
        while True:
            with self.lock:
                used = gpu_used()
                for g in self.gpus:
                    if g not in self.held and used.get(g, 10 ** 9) < 2000:
                        self.held.add(g)
                        return g
            if time.time() - t0 > hours * 3600:
                raise SystemExit(f"[{stage}] no free GPU for {hours} h")
            if time.time() - said > 1800:
                print(f"[{stage}] waiting for a free GPU (in use: {used}, held here: {sorted(self.held)})", flush=True)
                said = time.time()
            time.sleep(120)

    def release(self, g):
        with self.lock:
            self.held.discard(g)


POOL = None


def run(stage, cmd, gpu, log):
    """gpu=True: take one from the pool for this stage; gpu=None: CPU only."""
    for attempt in range(3):
        g = POOL.acquire(stage) if gpu else None
        try:
            print(f"[{stage}] attempt {attempt + 1} on GPU {g}: {cmd[:150]}", flush=True)
            code = subprocess.run(["bash", "-c", f"{ENV}export CUDA_VISIBLE_DEVICES={g if g is not None else ''}; "
                                                 f"cd {RUN} && {cmd} > {log} 2>&1"]).returncode
        finally:
            if g is not None:
                POOL.release(g)
        if code == 0:
            return
        text = Path(log).read_text(errors="replace").lower()
        tail = text[-4000:]
        print(f"[{stage}] exit {code}: {tail[-600:]}", flush=True)
        # memory taken by another user between the check and the start: OOM, or vLLM finding no room for the KV cache
        # (23/10/2026: "Available KV cache memory: -0.13 GiB" -> "engine core initialization failed")
        if not any(s in text for s in ("out of memory", "available kv cache memory: -", "no available memory for the cache")):
            raise SystemExit(f"stage {stage} failed")
        time.sleep(120)
    raise SystemExit(f"stage {stage} failed 3 times with OOM")


def holdout_f1(path, y_of):
    """Mean over k of the holdout macro-F1 of letter probabilities averaged over the 4 option orders."""
    acc = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e, k, o = map(int, r["id"].split("|"))
        lp = [r["logprobs"][L] for L in "ABCD"]
        m = max(lp)
        p = [math.exp(v - m) if math.isfinite(v) else 0.0 for v in lp] if math.isfinite(m) else [0.25] * 4
        s = sum(p)
        v = acc.setdefault((e, k), [0.0] * 4)
        for i, L in enumerate("ABCD"):
            v[LABELS4.index(ORDERS[o][i])] += p[i] / s
    f1s = []
    for k in range(5):
        ys, ps = [], []
        for (e, kk), v in acc.items():
            if kk == k:
                ys.append(y_of[e]); ps.append(max(range(4), key=lambda i: v[i]))
        f = []
        for c in range(4):
            tp = sum(1 for a, b in zip(ys, ps) if a == c and b == c)
            fp = sum(1 for a, b in zip(ys, ps) if a != c and b == c)
            fn = sum(1 for a, b in zip(ys, ps) if a == c and b != c)
            f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        f1s.append(sum(f) / 4)
    return sum(f1s) / 5, f1s
