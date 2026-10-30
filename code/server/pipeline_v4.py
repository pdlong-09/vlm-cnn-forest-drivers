"""design notes GPU pipeline, resumable stage by stage, on two servers (split as in design notes).

    python pipeline_v4.py --role server2|server1 --phase cnn|cnn_e2|train|ext [--gpus 0] [--smoke]

CNN phases (either server, whichever is ready; design notes):
  cnn:    cnn_folds (5 out-of-fold CNNs, fn_v4_cnn.py) -> cnn_full (seed-0 v3 CNN, ForestNet features) -> merge
  cnn_e2: seed-0 features on E2; M2 seeds 0-2 predictions on E2
VLM phases (server1, RTX 4090):
  train: render -> chain m5e: m5e_train (fn_m5e.py) -> m5e_select -> m5e_valcal
                 | chain text: for m in (m5t, m5tr): train (v1 qlora_train.py) -> select (v1 vlm_infer.py) -> valcaltest
  ext:   render -> chain m5e: m5e_ext -> m5e_test (exploratory) | chain text: m3c_ext, m5t_ext, m5tr_ext
Selection: val_sel, epoch 1 vs 2 by mean 4-class macro-F1 over k, tie epoch 1. GPU pool, waits and OOM retries come
from gpu_pipeline.py.
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
ROLE = {"server1": {"py": "/home/user/llm4deforest/paperA/venv/bin/python",
                   "pyr": "/home/user/llm4deforest/paperA/venv_render/bin/python",
                   "model": "/home/user/llm4deforest/models/Qwen3.5-9B"},
        "server2": {"py": "/home/user/llm4deforest_v4/venv/bin/python", "pyr": None, "model": None}}
CNN_PHASES = ("cnn", "cnn_e2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=list(ROLE), required=True)
    ap.add_argument("--phase", choices=["cnn", "cnn_e2", "train", "ext"], required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cp.POOL = cp.Pool([int(g) for g in a.gpus.split(",")])
    P = ROLE[a.role]
    PY, PYR, MODEL = P["py"], P["pyr"], P["model"]
    state, out, logs = RUN / "state", RUN / "out", RUN / "logs"
    for d in (state, out, logs, RUN / "adapters", RUN / "cnnv4"):
        d.mkdir(exist_ok=True)

    def stage(name, fn):
        if (state / name).exists():
            print(f"[{name}] already done", flush=True)
            return
        t0 = time.time()
        fn()
        (state / name).write_text(f"{time.time() - t0:.0f} s\n")
        print(f"[{name}] done in {time.time() - t0:.0f} s", flush=True)

    def run(name, cmd, gpu=True):
        cp.run(name, cmd, gpu, logs / f"{name}.txt")

    def select(m, infer_cmd):
        y = {int(r["eid"]): int(r["y"]) for r in csv.DictReader(open(RUN / "meta_valsel.csv"))}
        scores = {}
        for e in (1, 2):
            run(f"{m}_select_ep{e}", infer_cmd(e, f"{m}_valsel.jsonl", f"out/{m}_valsel_ep{e}.jsonl"))
            scores[e] = cp.holdout_f1(out / f"{m}_valsel_ep{e}.jsonl", y)
        e = 1 if scores[1][0] >= scores[2][0] else 2
        (out / f"{m}_choice.json").write_text(json.dumps({"epoch": e, "val_sel_mean_f1": {k: s[0] for k, s in scores.items()},
                                                          "val_sel_f1_by_k": {k: s[1] for k, s in scores.items()}}))
        print(f"[{m}_select] epoch {e} chosen: {scores}", flush=True)

    def ep(m):
        return json.loads((out / f"{m}_choice.json").read_text())["epoch"]

    def chains_run(chains):
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
        return failed

    failed = {}
    if a.phase in CNN_PHASES:
        if a.phase == "cnn":
            stage("cnn_folds", lambda: run("cnn_folds", f"{PY} code/fn_v4_cnn.py folds --data chips_fn.npz --folds folds.csv "
                                                        f"--outdir cnnv4"))
            stage("cnn_full", lambda: run("cnn_full", f"{PY} code/fn_v4_cnn.py extract --data chips_fn.npz --weights "
                                                      f"weights/M2_s0.pt --out cnnv4/full_fn.npz"))
            stage("merge", lambda: run("merge", f"{PY} code/fn_v4_merge.py cnnv4/oof_feats.npz cnnv4/full_fn.npz feats_v4.npz",
                                       None))
        else:
            def cnn_e2():
                run("cnn_e2_feats", f"{PY} code/fn_v4_cnn.py extract --data chips_e2.npz --weights weights/M2_s0.pt "
                                    f"--out cnnv4/full_e2.npz")
                for s in (0, 1, 2):
                    run(f"M2_e2_s{s}", f"{PY} code/cnn_v3.py predict --data chips_e2.npz --weights weights/M2_s{s}.pt "
                                       f"--out out/M2e2_s{s}.npz")
            stage("cnn_e2", cnn_e2)
    else:
        m5e = lambda e, inp, o: (f"{PY} code/fn_m5e.py infer --model {MODEL} --ckpt adapters/m5e/epoch{e} --input {inp} "  # noqa: E731
                                 f"--images images --feats feats_v4.npz --output {o} --batch 8")
        vl = lambda lora: (lambda e, inp, o: f"{PY} code/vlm_infer.py --model {MODEL} --images images --input {inp} "  # noqa: E731
                                             f"--lora {lora.format(e=e)} --output {o}")
        stage("render", lambda: run("render", f"{PYR} code/render_cards.py --jobs jobs.json --chips chips --out images", None))
        if a.phase == "train":
            text = []
            for m in ("m5t", "m5tr"):
                text += [(f"{m}_train", lambda m=m: run(f"{m}_train", f"{PY} code/qlora_train.py --model {MODEL} --train "
                                                                      f"{m}_train.jsonl --images images --out adapters/{m} --epochs 2")),
                         (f"{m}_select", lambda m=m: select(m, vl(f"adapters/{m}/epoch{{e}}"))),
                         (f"{m}_valcaltest", lambda m=m: run(f"{m}_valcaltest", vl(f"adapters/{m}/epoch{{e}}")(
                             ep(m), f"{m}_valcaltest.jsonl", f"out/{m}_valcaltest.jsonl")))]
            failed = chains_run({
                "m5e": [("m5e_train", lambda: run("m5e_train", f"{PY} code/fn_m5e.py train --model {MODEL} --train m5e_train.jsonl "
                                                               f"--images images --feats feats_v4.npz --out adapters/m5e")),
                        ("m5e_select", lambda: select("m5e", m5e)),
                        ("m5e_valcal", lambda: run("m5e_valcal", m5e(ep("m5e"), "m5e_valcal.jsonl", "out/m5e_valcal.jsonl")))],
                "text": text})
        else:
            failed = chains_run({
                "m5e": [("m5e_ext", lambda: run("m5e_ext", m5e(ep("m5e"), "m5e_ext.jsonl", "out/m5e_ext.jsonl"))),
                        ("m5e_test", lambda: run("m5e_test", m5e(ep("m5e"), "m5e_test.jsonl", "out/m5e_test.jsonl")))],
                "text": [("m3c_ext", lambda: run("m3c_ext", vl("adapters/m3c")(0, "m3c_ext.jsonl", "out/m3c_ext.jsonl")))] +
                        [(f"{m}_ext", lambda m=m: run(f"{m}_ext", vl(f"adapters/{m}/epoch{{e}}")(ep(m), f"{m}_ext.jsonl",
                                                                                                 f"out/{m}_ext.jsonl")))
                         for m in ("m5t", "m5tr")]})
    print("pipeline finished" + (f" with failures: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
