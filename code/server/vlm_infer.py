"""Driver attribution, runs on server1 (GPU 1, ~/llm4deforest/paperA/venv): VLM letter probabilities for evidence cards.

    python vlm_infer.py --model <dir> --input batch.jsonl --images <dir> --output out.jsonl [--lora <dir>]

Each input line: {"id": str, "text": str, "image": "<file in --images>", "shots": [ {"text", "image", "answer"} ... ]}.
The model answers with one token; the log-probabilities of the option letters A-L in the top-20 are returned
(missing letters get -inf). Temperature 0, thinking off.
"""
import argparse
import base64
import json
import math
import time
from pathlib import Path

LLM_KW = dict(max_num_seqs=8, max_num_batched_tokens=8192, mm_processor_kwargs={"max_pixels": 1024 * 1024},
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]}, seed=42)
LETTERS = list("ABCDEFGHIJKL")  # 4 coarse options (A-D) or 12 fine options (A-L, P-A4)


def uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def user(text, image, images_dir):
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri(images_dir / image)}},
                                        {"type": "text", "text": text}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--headroom-gib", type=float, default=0.6, help="memory left free for the desktop and the vision encoder")
    ap.add_argument("--gpu-mem", type=float, help="fraction of GPU memory for vLLM; default: what is free now minus 0.6 GB, at most 0.92")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    items = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    n_img = 1 + max((len(it.get("shots", [])) for it in items), default=0)
    images_dir = Path(a.images)
    system = {"role": "system", "content": "You are a remote sensing analyst who attributes tropical forest loss "
                                           "to its direct driver from satellite evidence."}
    convs = []
    for it in items:
        msgs = [system]
        for s in it.get("shots", []):
            msgs += [user(s["text"], s["image"], images_dir), {"role": "assistant", "content": s["answer"]}]
        msgs.append(user(it["text"], it["image"], images_dir))
        convs.append(msgs)
    kw = dict(LLM_KW, limit_mm_per_prompt={"image": n_img, "video": 0}, max_num_seqs=a.max_num_seqs)
    if a.lora:
        kw.update(enable_lora=True, max_lora_rank=64)
    if a.gpu_mem is None:  # GPU 0 also drives the desktop: size vLLM to what is actually free
        import torch
        free, total = torch.cuda.mem_get_info()
        a.gpu_mem = round(min(0.92, (free - a.headroom_gib * 2 ** 30) / total), 3)
        print(f"free {free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GiB -> gpu_memory_utilization {a.gpu_mem}", flush=True)
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=42)
    lreq = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lreq = LoRARequest("paperA", 1, a.lora)
    t1 = time.time()
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False, lora_request=lreq)
    t2 = time.time()
    with open(a.output, "w", encoding="utf-8") as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            lp = {L: -math.inf for L in LETTERS}
            for cand in (c.logprobs[0] if c.logprobs else {}).values():
                tok = (cand.decoded_token or "").strip()
                if tok in lp and cand.logprob > lp[tok]:
                    lp[tok] = cand.logprob
            f.write(json.dumps({"id": it["id"], "text_out": c.text, "logprobs": lp}) + "\n")
    print(json.dumps({"items": len(items), "load_s": round(t1 - t0, 1), "run_s": round(t2 - t1, 1)}))


if __name__ == "__main__":
    main()
