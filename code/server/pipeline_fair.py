"""Driver attribution fair-comparison batches (26/10/2026), resumable stage by stage; GPU pool and OOM retries from gpu_pipeline.py.

    python pipeline_fair.py --phase cnn --gpus 0          (server2: cnn_fair.py img3km and facts, seeds 0-2)
    python pipeline_fair.py --phase vlmimg --gpus 1,0     (server1: M3c recipe on image-only cards)

vlmimg: render -> qlora_train.py (seed 0, 2 epochs) -> val_sel at both epochs -> epoch with the best mean macro-F1 over
k (as M3c) -> ForestNet test, Indonesia, E2 and E3 at k = 4. Only the val_sel labels (meta_valsel.csv) are on the server.
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
LAB = {"py": "/home/user/llm4deforest/paperA/venv/bin/python", "pyr": "/home/user/llm4deforest/paperA/venv_render/bin/python",
       "model": "/home/user/llm4deforest/models/Qwen3.5-9B"}
HB = {"py": "/home/user/llm4deforest_v4/venv/bin/python"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cnn", "vlmimg", "vlmimgseeds"], required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    cp.POOL = cp.Pool([int(g) for g in a.gpus.split(",")])
    state, out, logs = RUN / "state", RUN / "out", RUN / "logs"
    for d in (state, out, logs, RUN / "adapters"):
        d.mkdir(exist_ok=True)

    def stage(name, cmd=None, gpu=True, fn=None):
        if (state / name).exists():
            print(f"[{name}] already done", flush=True)
            return
        t0 = time.time()
        fn() if fn else cp.run(name, cmd, gpu, logs / f"{name}.txt")
        (state / name).write_text(f"{time.time() - t0:.0f} s\n")
        print(f"[{name}] done in {time.time() - t0:.0f} s", flush=True)

    if a.phase == "cnn":
        ext = "sidn=chips_sidn.npz:tab_sidn.npz e2=chips_e2.npz:tab_e2.npz e3=chips_e3.npz:tab_e3.npz"
        for variant in ("img3km", "facts"):
            for s in (0, 1, 2):
                stage(f"{variant}_s{s}", f"TORCH_HOME=torch_home {HB['py']} code/cnn_fair.py --variant {variant} --seed {s} "
                                         f"--data chips_fn.npz --tab tab_fn.npz --ext {ext} --outdir out")
    elif a.phase == "vlmimgseeds":  # the vlmimg recipe with other seeds, one chain per seed (one GPU each when free)
        PY, MODEL = LAB["py"], LAB["model"]
        stage("render", f"{LAB['pyr']} code/render_cards.py --jobs jobs.json --chips chips --out images", None)
        y = {int(r["eid"]): int(r["y"]) for r in csv.DictReader(open(RUN / "meta_valsel.csv"))}
        failed = {}

        def vl(lora, inp, o):
            return f"{PY} code/vlm_infer.py --model {MODEL} --images images --input {inp} --lora {lora} --output {o}"

        def chain(sd):
            try:
                stage(f"s{sd}_train", f"{PY} code/qlora_train.py --model {MODEL} --train img_train.jsonl --images images "
                                      f"--out adapters/img_s{sd} --epochs 2 --seed {sd}")
                for e in (1, 2):
                    stage(f"s{sd}_valsel_ep{e}", vl(f"adapters/img_s{sd}/epoch{e}", "img_valsel.jsonl",
                                                    f"out/img_s{sd}_valsel_ep{e}.jsonl"))

                def select():
                    sc = {e: cp.holdout_f1(out / f"img_s{sd}_valsel_ep{e}.jsonl", y) for e in (1, 2)}
                    e = 1 if sc[1][0] >= sc[2][0] else 2
                    (out / f"img_s{sd}_choice.json").write_text(json.dumps({"seed": sd, "epoch": e,
                                                                            "val_sel_mean_f1": {k: v[0] for k, v in sc.items()}}))
                    print(f"[s{sd}_select] epoch {e}: {sc}", flush=True)
                stage(f"s{sd}_select", fn=select)
                ep = json.loads((out / f"img_s{sd}_choice.json").read_text())["epoch"]
                for t in ("fntest", "sidn", "e2", "e3"):
                    stage(f"s{sd}_{t}", vl(f"adapters/img_s{sd}/epoch{ep}", f"img_{t}.jsonl", f"out/img_s{sd}_{t}.jsonl"))
            except BaseException as ex:
                failed[sd] = repr(ex)
        threads = [threading.Thread(target=chain, args=(int(sd),)) for sd in a.seeds.split(",")]
        for t in threads:
            t.start()
            time.sleep(90)
        for t in threads:
            t.join()
        if failed:
            print(f"pipeline finished with failures: {failed}", flush=True)
            return 1
    else:
        PY, MODEL = LAB["py"], LAB["model"]

        def vl(lora, inp, o):
            return f"{PY} code/vlm_infer.py --model {MODEL} --images images --input {inp} --lora {lora} --output {o}"
        stage("render", f"{LAB['pyr']} code/render_cards.py --jobs jobs.json --chips chips --out images", None)
        stage("train", f"{PY} code/qlora_train.py --model {MODEL} --train img_train.jsonl --images images "
                       f"--out adapters/img --epochs 2 --seed 0")
        for e in (1, 2):
            stage(f"valsel_ep{e}", vl(f"adapters/img/epoch{e}", "img_valsel.jsonl", f"out/img_valsel_ep{e}.jsonl"))
        y = {int(r["eid"]): int(r["y"]) for r in csv.DictReader(open(RUN / "meta_valsel.csv"))}

        def select():
            sc = {e: cp.holdout_f1(out / f"img_valsel_ep{e}.jsonl", y) for e in (1, 2)}
            e = 1 if sc[1][0] >= sc[2][0] else 2
            (out / "img_choice.json").write_text(json.dumps({"epoch": e, "val_sel_mean_f1": {k: v[0] for k, v in sc.items()}}))
            print(f"[select] epoch {e}: {sc}", flush=True)
        stage("select", fn=select)
        ep = json.loads((out / "img_choice.json").read_text())["epoch"]
        failed = {}

        def go(sets):
            for t in sets:
                try:
                    stage(f"img_{t}", vl(f"adapters/img/epoch{ep}", f"img_{t}.jsonl", f"out/img_{t}.jsonl"))
                except BaseException as ex:
                    failed[t] = repr(ex)
        threads = [threading.Thread(target=go, args=(s,)) for s in (["fntest", "sidn"], ["e2", "e3"])]
        for t in threads:
            t.start()
            time.sleep(60)
        for t in threads:
            t.join()
        if failed:
            print(f"pipeline finished with failures: {failed}", flush=True)
            return 1
    print("pipeline finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
