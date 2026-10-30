"""Driver attribution v3 GPU pipeline on server1 (design notes), batches in /data/tmp (NVMe), resumable stage by stage.

    python pipeline_v3.py --phase train|ext --gpus 0,1 [--smoke]

phase train (ForestNet only, while the Sims chips are still being fetched):
  render, pack (CPU) -> chain 1: M2 (server/cnn_v3.py = the v1 CNN loop, seeds 0, 1, 2, weights kept)
                     -> chain 2: m3v_train (fn_qlora_v3.py), m3v_select (val_sel, epoch 1 vs 2 by mean 4-class macro-F1
                        over k, tie epoch 1), m3v_valcaltest (ForestNet val_cal + test; exploratory, and the M4v calibration)
phase ext (the external set E, models from phase train and the frozen v1 M3c):
  render, pack -> chain 1: M2_ext (cnn_v3.py predict, 3 seeds), m3c_ext (v1 server/vlm_infer.py) -> chain 2: m3v_ext
GPU pool, waits and OOM retries come from gpu_pipeline.py.
"""
import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import gpu_pipeline as cp

RUN = Path(__file__).resolve().parent.parent
MODEL = cp.MODEL
PY, PY_RENDER = cp.PY, cp.PY_RENDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "ext"], required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cp.POOL = cp.Pool([int(g) for g in a.gpus.split(",")])
    state, out, logs = RUN / "state", RUN / "out", RUN / "logs"
    for d in (state, out, logs, RUN / "adapters", RUN / "weights"):
        d.mkdir(exist_ok=True)
    epochs = 2 if a.smoke else 40
    seeds = [0] if a.smoke else [0, 1, 2]
    infer1 = f"{PY} code/vlm_infer.py --model {MODEL} --images images"
    infer3 = f"{PY} code/vlm_infer_v3.py --model {MODEL} --images images --tower-lora"

    def stage(name, fn):
        if (state / name).exists():
            print(f"[{name}] already done", flush=True)
            return
        t0 = time.time()
        fn()
        (state / name).write_text(f"{time.time() - t0:.0f} s\n")
        print(f"[{name}] done in {time.time() - t0:.0f} s", flush=True)

    def ep():
        return json.loads((out / "m3v_choice.json").read_text())["epoch"]

    stage("render", lambda: cp.run("render", f"{PY_RENDER} code/render_cards.py --jobs jobs.json --chips chips --out images",
                                   None, logs / "render.txt"))
    stage("pack", lambda: cp.run("pack", f"{PY} code/fn_pack_v3.py meta_cnn.csv chips chips.npz", None, logs / "pack.txt"))

    if a.phase == "train":
        y_valsel = {int(r["eid"]): int(r["y"]) for r in csv.DictReader(open(RUN / "meta_valsel.csv"))}

        def m2():
            for s in seeds:
                cp.run("M2", f"{PY} code/cnn_v3.py train --data chips.npz --out out/M2_s{s}.npz --weights weights/M2_s{s}.pt "
                             f"--epochs {epochs} --seed {s}", True, logs / f"M2_s{s}.txt")

        def select():
            scores = {}
            for e in (1, 2):
                cp.run("m3v_select", f"{infer3} --input valsel.jsonl --lora adapters/m3v/epoch{e} "
                                     f"--output out/m3v_valsel_ep{e}.jsonl", True, logs / f"m3v_valsel_ep{e}.txt")
                scores[e] = cp.holdout_f1(out / f"m3v_valsel_ep{e}.jsonl", y_valsel)
            e = 1 if scores[1][0] >= scores[2][0] else 2
            (out / "m3v_choice.json").write_text(json.dumps({"epoch": e, "val_sel_mean_f1": {k: s[0] for k, s in scores.items()},
                                                             "val_sel_f1_by_k": {k: s[1] for k, s in scores.items()}}))
            print(f"[m3v_select] epoch {e} chosen: {scores}", flush=True)

        chains = {"cnn": [("M2", m2)],
                  "m3v": [("m3v_train", lambda: cp.run("m3v_train", f"{PY} code/fn_qlora_v3.py --model {MODEL} --train "
                                                                    f"train.jsonl --images images --out adapters/m3v --epochs 2",
                                                       True, logs / "m3v_train.txt")),
                          ("m3v_select", select),
                          # vLLM marks tower LoRA experimental: check it against transformers + PEFT (informational)
                          ("m3v_check", lambda: cp.run("m3v_check", f"head -n 16 valsel.jsonl > check16.jsonl && "
                                                                    f"{PY} code/vlm_infer.py --model {MODEL} --images images "
                                                                    f"--input check16.jsonl --lora adapters/m3v/epoch{ep()} "
                                                                    f"--output out/m3v_check_notower.jsonl && "
                                                                    f"{PY} code/check_tower_lora.py --model {MODEL} "
                                                                    f"--adapter adapters/m3v/epoch{ep()} --input valsel.jsonl "
                                                                    f"--images images --vllm out/m3v_valsel_ep{ep()}.jsonl "
                                                                    f"--vllm-notower out/m3v_check_notower.jsonl "
                                                                    f"--out out/m3v_check.json || echo CHECK_FAILED",
                                                       True, logs / "m3v_check.txt")),
                          ("m3v_valcaltest", lambda: cp.run("m3v_valcaltest", f"{infer3} --input valcaltest.jsonl --lora "
                                                                              f"adapters/m3v/epoch{ep()} --output "
                                                                              f"out/m3v_valcaltest.jsonl", True,
                                                            logs / "m3v_valcaltest.txt"))]}
    else:
        def m2_ext():
            for s in seeds:
                cp.run("M2_ext", f"{PY} code/cnn_v3.py predict --data chips.npz --weights weights/M2_s{s}.pt "
                                 f"--out out/M2ext_s{s}.npz", True, logs / f"M2ext_s{s}.txt")
        chains = {"cnn+m3c": [("M2_ext", m2_ext),
                              ("m3c_ext", lambda: cp.run("m3c_ext", f"{infer1} --input ext.jsonl --lora adapters/m3c "
                                                                    f"--output out/m3c_ext.jsonl", True, logs / "m3c_ext.txt"))],
                  "m3v": [("m3v_ext", lambda: cp.run("m3v_ext", f"{infer3} --input ext.jsonl --lora adapters/m3v/epoch{ep()} "
                                                                f"--output out/m3v_ext.jsonl", True, logs / "m3v_ext.txt"))]}
    failed = {}

    def go(name, steps):
        try:
            for n, fn in steps:
                stage(n, fn)
        except BaseException as e:
            failed[name] = repr(e)
            print(f"chain {name} stopped: {e!r}", flush=True)
    threads = [threading.Thread(target=go, args=(n, s)) for n, s in chains.items()]
    for t in threads:
        t.start()
        time.sleep(60)
    for t in threads:
        t.join()
    print("pipeline finished" + (f" with failures: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
