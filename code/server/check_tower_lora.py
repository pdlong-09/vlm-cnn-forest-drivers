"""Driver attribution v3: does vLLM really apply the vision-tower part of the M3v adapter? (vLLM marks tower LoRA experimental.)

    python check_tower_lora.py --model <dir> --adapter <dir> --input valsel.jsonl --images images --vllm out.jsonl --out check.json

For the first --n prompts: letter log-probabilities from transformers + PEFT (bf16 base, as vLLM serves it) with the
adapter (a) as is and (b) with the vision-tower LoRA scaled to zero, compared with the vLLM output for the same ids.
vLLM should agree with (a) and differ from (b).
"""
import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

LETTERS = list("ABCD")
SYSTEM = ("You are a remote sensing analyst who attributes tropical forest loss to its direct driver from satellite "
          "evidence.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--vllm-notower", help="vLLM output for the same prompts without --tower-lora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    items = [json.loads(l) for l in open(a.input, encoding="utf-8")][: a.n]
    v = {r["id"]: r for r in (json.loads(l) for l in open(a.vllm, encoding="utf-8"))}
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=1024 * 1024)
    base = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    letter_ids = {L: proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS}

    def letters():
        out = []
        for it in items:
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["text"]}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            enc = proc(text=[text], images=[Image.open(Path(a.images) / it["image"]).convert("RGB")], return_tensors="pt")
            with torch.no_grad():
                lp = torch.log_softmax(model(**{k: t.to(0) for k, t in enc.items()}).logits[0, -1].float(), -1)
            out.append({L: float(lp[i]) for L, i in letter_ids.items()})
        return out

    with_tower = letters()
    saved = {}
    for name, mod in model.named_modules():  # switch off the LoRA of the vision tower and merger only
        if "visual" in name and hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            saved[name] = dict(mod.scaling)
            mod.scaling = {k: 0.0 for k in mod.scaling}
    no_tower = letters()
    for name, mod in model.named_modules():
        if name in saved:
            mod.scaling = saved[name]

    def cmp(ref):
        diffs, agree = [], 0
        for it, r in zip(items, ref):
            vl = v[it["id"]]["logprobs"]
            d = [abs(vl[L] - r[L]) for L in LETTERS if math.isfinite(vl[L])]
            diffs.append(max(d) if d else float("nan"))
            agree += max(LETTERS, key=lambda L: vl[L]) == max(LETTERS, key=lambda L: r[L])
        return {"mean_max_abs_logprob_diff": sum(diffs) / len(diffs), "argmax_agreement": agree / len(items)}
    tower_effect = sum(max(abs(a_[L] - b_[L]) for L in LETTERS) for a_, b_ in zip(with_tower, no_tower)) / len(items)
    res = {"n": len(items), "vllm_vs_hf_with_tower": cmp(with_tower), "vllm_vs_hf_without_tower": cmp(no_tower),
           "hf_tower_effect_mean_max_abs_diff": tower_effect, "lora_modules_switched_off": len(saved)}
    if a.vllm_notower:  # direct test: vLLM with vs without the tower flag, against the HF tower effect
        import numpy as np
        vn = {r["id"]: r for r in (json.loads(l) for l in open(a.vllm_notower, encoding="utf-8"))}
        dv, dh = [], []
        for it, wt, nt in zip(items, with_tower, no_tower):
            for L in LETTERS:
                x, y = v[it["id"]]["logprobs"][L], vn[it["id"]]["logprobs"][L]
                if math.isfinite(x) and math.isfinite(y):
                    dv.append(x - y)
                    dh.append(wt[L] - nt[L])
        dv, dh = np.array(dv), np.array(dh)
        res["vllm_tower_effect_mean_abs"] = float(np.abs(dv).mean())
        res["corr_vllm_vs_hf_tower_effect"] = float(np.corrcoef(dv, dh)[0, 1]) if dv.std() > 0 and dh.std() > 0 else None
        res["verdict"] = ("vLLM applies the tower LoRA" if res["vllm_tower_effect_mean_abs"] > 0.02
                          and (res["corr_vllm_vs_hf_tower_effect"] or 0) > 0.5 else "NOT confirmed")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
