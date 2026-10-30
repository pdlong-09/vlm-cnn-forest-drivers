"""Driver attribution v3, M3v on server1: the v1 M3c recipe (copied from server/qlora_train.py, v1 file unchanged) with the two
design notes changes: LoRA also on the vision tower blocks and the merger, and a class-balanced loss (weight of the
answer's class = 1 / its frequency among training events, normalised to mean 1 over events; needs "label" per row).

    python fn_qlora_v3.py --model <dir> --train train.jsonl --images <dir> --out <adapter dir> [--epochs 2]

The rest of the v1 description holds:

train.jsonl, one line per event and step: {"eid", "k", "image", "texts": [4 prompts, one per cyclic option order],
"answers": [4 letters]}. Each epoch uses, per event, one random step k and one random option order. Loss on the
answer token only. 4-bit NF4 base, LoRA r=16, alpha=32, dropout 0.05 on the language model's attention and MLP
projections, lr 1e-4, effective batch 8. An adapter is saved after every epoch (<out>/epoch<N>).
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

SYSTEM = ("You are a remote sensing analyst who attributes tropical forest loss to its direct driver from satellite "
          "evidence.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    proc = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, quantization_config=bnb, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=(r".*(language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                                      r"|visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
                                      r"|visual\.merger\.(linear_fc1|linear_fc2))"))
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["eid"]].append(r)
    from collections import Counter
    freq = Counter(rs[0]["label"] for rs in by_event.values())
    raw_w = {c: 1.0 / n for c, n in freq.items()}
    norm = sum(raw_w[rs[0]["label"]] for rs in by_event.values()) / len(by_event)
    class_w = {c: w / norm for c, w in raw_w.items()}  # mean 1 over training events
    print("class weights:", {c: round(w, 3) for c, w in sorted(class_w.items())}, flush=True)
    images = Path(a.images)

    def encode(r, o):
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["texts"][o]}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        full = prompt + r["answers"][o] + proc.tokenizer.eos_token
        img = Image.open(images / r["image"]).convert("RGB")
        enc = proc(text=[full], images=[img], return_tensors="pt")
        n_prompt = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100
        enc["labels"] = labels
        return {k: v.to(model.device) for k, v in enc.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps_per_epoch = len(by_event) // a.accum
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(total, 1)))
    model.train()
    t0, step = time.time(), 0
    for ep in range(1, a.epochs + 1):
        eids = list(by_event)
        random.shuffle(eids)
        running = 0.0
        for i, e in enumerate(eids, 1):
            r = random.choice(by_event[e])
            o = random.randrange(len(r["texts"]))
            loss = model(**encode(r, o)).loss * class_w[r["label"]] / a.accum
            loss.backward()
            running += loss.item()
            if i % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"epoch {ep} step {step}/{total} loss {running / 20:.4f} {time.time() - t0:.0f} s", flush=True)
                    running = 0.0
        out = Path(a.out) / f"epoch{ep}"
        model.save_pretrained(out)
        print(f"saved {out} after {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
