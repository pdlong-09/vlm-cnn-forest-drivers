"""Driver attribution, fact-sheet country ablation on server1: the v1 M3c adapter on E3 cards whose first line names the cell's
country instead of Indonesia (k = 4, 4 option orders). Resumable stages, GPU pool and OOM retries from gpu_pipeline.py.

    python pipeline_country.py --gpus 1

render -> m3c_e3_country (vlm_infer.py with adapters/m3c). No label is on the server.
"""
import argparse
import sys
import time
from pathlib import Path

import gpu_pipeline as cp

RUN = Path(__file__).resolve().parent.parent
PY = "/home/user/llm4deforest/paperA/venv/bin/python"
PYR = "/home/user/llm4deforest/paperA/venv_render/bin/python"
MODEL = "/home/user/llm4deforest/models/Qwen3.5-9B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1")
    a = ap.parse_args()
    cp.POOL = cp.Pool([int(g) for g in a.gpus.split(",")])
    state, out, logs = RUN / "state", RUN / "out", RUN / "logs"
    for d in (state, out, logs):
        d.mkdir(exist_ok=True)

    def stage(name, cmd, gpu):
        if (state / name).exists():
            print(f"[{name}] already done", flush=True)
            return
        t0 = time.time()
        cp.run(name, cmd, gpu, logs / f"{name}.txt")
        (state / name).write_text(f"{time.time() - t0:.0f} s\n")
        print(f"[{name}] done in {time.time() - t0:.0f} s", flush=True)

    stage("render", f"{PYR} code/render_cards.py --jobs jobs.json --chips chips --out images", None)
    stage("m3c_e3_country", f"{PY} code/vlm_infer.py --model {MODEL} --images images --input m3c_e3_country.jsonl "
                            f"--lora adapters/m3c --output out/m3c_e3_country.jsonl", True)
    print("pipeline finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
