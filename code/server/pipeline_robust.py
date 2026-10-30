"""Driver attribution robustness batches (25/10/2026), resumable stage by stage.

    python pipeline_robust.py --phase ablate|backbones|cpu [--gpus 1]

ablate (server1): render -> the v1 M3c adapter on E2 and E3 at k = 4 with three inputs each: the full card (control),
         image only (every number on the fact sheet replaced by n/a) and text only (a black image grid without the outline).
backbones (server2): the M2 recipe with ResNet-50 and ConvNeXt-Tiny, seeds 0-2, on ForestNet, E2 and E3
         (cnn_backbones.py; ImageNet weights from torch_home/, no download on the server).
cpu (server2): spatial block bootstrap, repeated calibration splits and the tabular baseline (robust_a.py).
GPU pool, waits and OOM retries come from gpu_pipeline.py.
"""
import argparse
import sys
import time
from pathlib import Path

import gpu_pipeline as cp

RUN = Path(__file__).resolve().parent.parent
PY = {"server1": "/home/user/llm4deforest/paperA/venv/bin/python",
      "server1_render": "/home/user/llm4deforest/paperA/venv_render/bin/python",
      "server2": "/home/user/llm4deforest_v4/venv/bin/python"}
MODEL = "/home/user/llm4deforest/models/Qwen3.5-9B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ablate", "backbones", "cpu"], required=True)
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--parts", default="1,3,2")
    a = ap.parse_args()
    cp.POOL = cp.Pool([int(g) for g in a.gpus.split(",")])
    state, out, logs = RUN / "state", RUN / "out", RUN / "logs"
    for d in (state, out, logs):
        d.mkdir(exist_ok=True)

    def stage(name, cmd, gpu=True):
        if (state / name).exists():
            print(f"[{name}] already done", flush=True)
            return
        t0 = time.time()
        cp.run(name, cmd, gpu, logs / f"{name}.txt")
        (state / name).write_text(f"{time.time() - t0:.0f} s\n")
        print(f"[{name}] done in {time.time() - t0:.0f} s", flush=True)

    if a.phase == "ablate":
        stage("render", f"{PY['server1_render']} code/render_cards.py --jobs jobs.json --chips chips --out images", None)
        for t in ("e3", "e2"):
            for v in ("full", "img", "txt"):
                stage(f"m3c_{t}_{v}", f"{PY['server1']} code/vlm_infer.py --model {MODEL} --images images "
                                      f"--input m3c_{t}_{v}.jsonl --lora adapters/m3c --output out/m3c_{t}_{v}.jsonl")
    elif a.phase == "backbones":
        for arch in ("resnet50", "convnext_tiny"):
            for s in (0, 1, 2):
                stage(f"{arch}_s{s}", f"TORCH_HOME=torch_home {PY['server2']} code/cnn_backbones.py --arch {arch} --seed {s} "
                                      f"--data chips_fn.npz --ext e2=chips_e2.npz e3=chips_e3.npz sidn=chips_sidn.npz --outdir out")
    else:
        stage("robust_cpu", f"{PY['server2']} code/robust_a.py --bundle bundle --out out/robust_a.json --parts {a.parts}", None)
    print("pipeline finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
