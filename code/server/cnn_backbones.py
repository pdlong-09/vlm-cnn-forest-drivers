"""Driver attribution, stronger image baselines (robustness, 25/10/2026): the M2 recipe (server/cnn_v3.py) with larger ImageNet
backbones. Same input (years -1..4 stacked, 18 channels, later years zeroed), loss, optimiser, schedule, augmentation and
val_sel selection; only the network changes.

    python cnn_backbones.py --arch resnet50|convnext_tiny --seed 0 --data chips.npz --ext e2=chips_e2.npz e3=chips_e3.npz --outdir out

Writes <outdir>/<arch>_s<seed>_{fn,e2,e3}.npz with probabilities for every event and step k = 0..4.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_train import macro_f1, to_input  # noqa: E402


def adapt_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """First convolution with in_ch inputs, ImageNet RGB filters repeated and rescaled as in cnn_train.model."""
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    new.weight.data = conv.weight.data.repeat(1, in_ch // 3, 1, 1) / (in_ch // 3)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    return new


def build(arch: str, in_ch: int = 18, n_cls: int = 4) -> nn.Module:
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.conv1 = adapt_conv(m.conv1, in_ch)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
    elif arch == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.features[0][0] = adapt_conv(m.features[0][0], in_ch)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_cls)
    else:
        raise SystemExit(f"unknown arch {arch}")
    return m


def predict_all(net, X, dev="cuda"):
    net.eval()
    out = np.zeros((len(X), 5, 4))
    with torch.no_grad():
        for s in range(0, len(X), 128):
            xb = X[s:s + 128].to(dev)
            for k in range(5):
                out[s:s + len(xb), k] = torch.softmax(net(to_input(xb, torch.full((len(xb),), k, device=dev))), 1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ext", nargs="*", default=[])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    z = np.load(a.data, allow_pickle=True)
    X, y, part, eid = torch.from_numpy(z["X"]), z["y"].astype(int), z["part"], z["eid"]  # read each key once
    dev = "cuda"
    tr, sel = np.flatnonzero(part == "train"), np.flatnonzero(part == "val_sel")
    net = build(a.arch).to(dev)
    w = torch.tensor(len(tr) / (4 * np.bincount(y[tr], minlength=4)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    yt = torch.from_numpy(y)
    best, state, t0 = -1, None, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            xb = X[b].to(dev)
            if np.random.rand() < 0.5:
                xb = xb.flip(3)
            if np.random.rand() < 0.5:
                xb = xb.flip(2)
            xb = torch.rot90(xb, int(np.random.randint(4)), dims=(2, 3))
            k = torch.randint(0, 5, (len(b),), device=dev)
            loss = crit(net(to_input(xb, k)), yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        p = predict_all(net, X[sel])
        f1 = np.mean([macro_f1(y[sel], p[:, k].argmax(1)) for k in range(5)])
        if f1 > best:
            best, state = f1, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep}: loss {loss.item():.3f} val_sel mean macro-F1 {f1:.3f} (best {best:.3f}) {time.time() - t0:.0f} s",
              flush=True)
    net.load_state_dict(state)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arch}_s{a.seed}"
    np.savez_compressed(out / f"{tag}_fn.npz", probs=predict_all(net, X), eid=eid, part=part, y=y, best_val_sel=best)
    for item in a.ext:
        name, path = item.split("=", 1)
        ze = np.load(path, allow_pickle=True)
        np.savez_compressed(out / f"{tag}_{name}.npz", probs=predict_all(net, torch.from_numpy(ze["X"])), eid=ze["eid"])
    print(f"done {tag}: best val_sel {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
