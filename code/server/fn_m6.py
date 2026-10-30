"""design notes, M6: the M5e recipe (fn_m5e.py) with AlphaEarth Foundations embeddings instead of the CNN features, and
modality dropout. The 128-d embedding of the event (polygon and 150 m ring means, one snapshot) goes through a trained
MLP (128 -> 1024 -> 4 x hidden, GELU) whose output replaces the embeddings of four <|fim_pad|> placeholder tokens.
With probability p per training example the placeholder sentence is removed from the prompt (no features that time);
at inference the features are always present. LoRA on the language model only, 4-bit NF4 base for training and inference.

    python fn_m6.py train --model <dir> --train train.jsonl --images images --feats aef.npz --out m6_p05 --p 0.5 [--epochs 2]
    python fn_m6.py infer --model <dir> --ckpt m6_p05/epoch2 --input items.jsonl --images images --feats aef.npz --output out.jsonl

aef.npz: eid, feat [n, 128], standardised with the mean and std over the training events (saved with the checkpoint).
Train rows carry eid, k, texts (with the placeholder sentence `ph`), answers; item ids are "eid|k|o".
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fn_m5e import PH, N_TOK, LETTERS, Injector, load_base, messages  # noqa: E402  (same base, hook and prompts)


class Projector(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, N_TOK * hidden))
        nn.init.normal_(self.net[2].weight, std=0.02)
        nn.init.zeros_(self.net[2].bias)
        self.hidden = hidden

    def forward(self, f):
        return self.net(f).view(-1, N_TOK, self.hidden)


def feats_table(path):
    z = np.load(path, allow_pickle=True)
    F = z["feat"].astype(np.float32)
    return {int(e): F[i] for i, e in enumerate(z["eid"])}, F.shape[1]


def train(a):
    from peft import LoraConfig, get_peft_model
    random.seed(0); torch.manual_seed(0)
    proc, model, prep = load_base(a.model)
    model = prep(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                             target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    hidden = model.get_input_embeddings().weight.shape[1]
    raw, dim = feats_table(a.feats)
    proj = Projector(dim, hidden).to(model.device)
    inj = Injector(proc.tokenizer.convert_tokens_to_ids(PH))
    model.get_input_embeddings().register_forward_hook(inj)
    rows = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["eid"]].append(r)
    train_f = np.stack([raw[e] for e in by_event])  # standardise over the training events
    mean, std = train_f.mean(0), train_f.std(0) + 1e-6
    params = [p for p in model.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = (len(by_event) // a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(total, 1)))
    model.train()
    t0, step, dropped = time.time(), 0, 0
    for ep in range(1, a.epochs + 1):
        eids = list(by_event)
        random.shuffle(eids)
        running = 0.0
        for i, e in enumerate(eids, 1):
            r = random.choice(by_event[e])
            o = random.randrange(4)
            text = r["texts"][o]
            if random.random() < a.p:  # modality dropout: the placeholder sentence and its tokens leave the prompt
                text = text.replace(r["ph"] + "\n\n", "")
                assert PH not in text
                inj.current, dropped = None, dropped + 1
            else:
                f = torch.from_numpy((raw[e] - mean) / std).to(model.device, torch.float32)
                inj.current = proj(f[None])[0]
            prompt = proc.apply_chat_template(messages(text), add_generation_prompt=True, tokenize=False, enable_thinking=False)
            img = Image.open(Path(a.images) / r["image"]).convert("RGB")
            enc = proc(text=[prompt + r["answers"][o] + proc.tokenizer.eos_token], images=[img], return_tensors="pt")
            n_prompt = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
            labels = enc["input_ids"].clone()
            labels[:, :n_prompt] = -100
            loss = model(**{k: v.to(model.device) for k, v in enc.items()}, labels=labels.to(model.device)).loss / a.accum
            loss.backward()
            running += loss.item()
            if i % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"epoch {ep} step {step}/{total} loss {running / 20:.4f} dropped {dropped} {time.time() - t0:.0f} s",
                          flush=True)
                    running = 0.0
        out = Path(a.out) / f"epoch{ep}"
        model.save_pretrained(out)
        torch.save({"proj": proj.state_dict(), "mean": mean, "std": std, "hidden": hidden, "dim": dim, "p": a.p},
                   out / "projector.pt")
        print(f"saved {out} after {time.time() - t0:.0f} s", flush=True)


def infer(a):
    from peft import PeftModel
    proc, base, _ = load_base(a.model)
    model = PeftModel.from_pretrained(base, a.ckpt).eval()
    ck = torch.load(Path(a.ckpt) / "projector.pt", map_location="cuda", weights_only=False)
    proj = Projector(ck["dim"], ck["hidden"]).to(model.device)
    proj.load_state_dict(ck["proj"])
    proj.eval()
    inj = Injector(proc.tokenizer.convert_tokens_to_ids(PH))
    model.get_input_embeddings().register_forward_hook(inj)
    raw, _ = feats_table(a.feats)
    letter_ids = {L: proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS}
    proc.tokenizer.padding_side = "left"  # batched prompts: the last position is the answer slot of every row
    items = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    t0, n = time.time(), 0
    with open(a.output, "w", encoding="utf-8") as fo:
        for s in range(0, len(items), a.batch):
            chunk = items[s:s + a.batch]
            prompts = [proc.apply_chat_template(messages(it["text"]), add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False) for it in chunk]
            imgs = [Image.open(Path(a.images) / it["image"]).convert("RGB") for it in chunk]
            enc = proc(text=prompts, images=imgs, return_tensors="pt", padding=True)
            f = np.stack([(raw[int(it["id"].split("|")[0])] - ck["mean"]) / ck["std"] for it in chunk])
            with torch.no_grad():
                inj.current = proj(torch.from_numpy(f).to(model.device, torch.float32))
                lp = torch.log_softmax(model(**{kk: v.to(model.device) for kk, v in enc.items()}).logits[:, -1].float(), -1)
            for it, row in zip(chunk, lp):
                fo.write(json.dumps({"id": it["id"], "logprobs": {L: float(row[i]) for L, i in letter_ids.items()}}) + "\n")
            n += len(chunk)
            if (s // a.batch) % 50 == 0:
                fo.flush()
                print(f"{n}/{len(items)} prompts, {time.time() - t0:.0f} s", flush=True)
    print(f"done {n} prompts in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--train")
    ap.add_argument("--out")
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    (train if a.mode == "train" else infer)(a)


if __name__ == "__main__":
    main()
