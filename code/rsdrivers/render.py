"""Evidence-card image grid as a pure Pillow function, so the same file renders cards locally and on the lab server.

Mirrors rsdrivers.card.grid_image exactly (checked pixel by pixel and byte by byte in scripts/verify_render.py).
No imports from the project: the server runs it with only Pillow (same version as locally, 13.0.0).

job = {"name": "<eid>_k<k>.jpg", "chips": ["<file or null>", ...] (years -1..k), "labels": ["year -1", ...],
       "polys": [[[x, y], ...], ...] (polygon exteriors already in scaled chip pixels)}
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

CHIP_PX = 128
SCALE = 2
QUALITY = 92


def render_grid(job: dict, chips_dir: Path) -> Image.Image:
    tiles = []
    for chip, label in zip(job["chips"], job["labels"]):
        p = Path(chips_dir) / chip if chip else None
        im = Image.open(p).convert("RGB") if p is not None and p.exists() else Image.new("RGB", (CHIP_PX, CHIP_PX), (0, 0, 0))
        im = im.resize((CHIP_PX * SCALE, CHIP_PX * SCALE), Image.NEAREST)
        d = ImageDraw.Draw(im)
        for poly in job["polys"]:
            pts = [tuple(v) for v in poly]
            d.line(pts + [pts[0]], fill=(255, 230, 0), width=2)
        d.rectangle([0, 0, 52, 16], fill=(0, 0, 0))
        d.text((3, 2), label, fill=(255, 255, 255))
        tiles.append(im)
    w = tiles[0].width
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (cols * w, rows * w), (255, 255, 255))
    for i, im in enumerate(tiles):
        out.paste(im, ((i % cols) * w, (i // cols) * w))
    return out


def job_for(row, k: int, region, interim_chips: Path | None = None) -> dict:
    """Build the job of one event and step locally (needs the ForestNet polygon in image pixels)."""
    from .card import _to_chip  # local only: the server receives finished jobs
    polys = list(region.geoms) if hasattr(region, "geoms") else [region]
    chips, labels = [], []
    for t in range(-1, k + 1):
        y = int(row.year) + t
        rel = f"{row.eid}/k{t}_{y}.png"
        chips.append(rel if interim_chips is None or (Path(interim_chips) / rel).exists() else None)
        labels.append(f"year {t:+d}")
    pts = [[[v * SCALE for v in _to_chip(x, yy)] for x, yy in p.exterior.coords] for p in polys]
    return {"name": f"{row.eid}_k{k}.jpg", "chips": chips, "labels": labels, "polys": pts}
