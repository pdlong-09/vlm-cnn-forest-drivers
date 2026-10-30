"""Evidence card for the VLM at step k: a fact sheet (no label hints) and an image grid of the yearly chips -1..k.

The loss polygon is drawn as a yellow outline on every chip. Chips cover 3 km x 3 km at 128 px around the event
centre; ForestNet pixel coordinates (15 m, 332 px, same centre) are mapped onto the chip.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .io import INTERIM, PIX_M, SIZE, load_region

CHIP_PX, CHIP_M = 128, 3000.0
OPTIONS = {"A": "Plantation", "B": "Smallholder agriculture", "C": "Grassland shrubland", "D": "Other"}
# wording condensed from the class definitions of Austin et al. 2019 (ERL 14:024007, table 1), the labelling scheme
# behind ForestNet
OPTION_TEXT = {
    "Plantation": "large-scale plantation: a large network of plantation blocks connected by a well-defined road grid "
                  "(oil palm, timber or other large estates), even where trees are not yet planted",
    "Smallholder agriculture": "smallholder agriculture: a matrix of small, clearly defined fields (crops, bare ground or "
                               "fallow) or small plots of planted tree rows (mixed or oil palm smallholdings)",
    "Grassland shrubland": "grassland or shrubland: large homogeneous areas with few or sparse shrubs or trees that "
                           "persist, without clearly defined field boundaries",
    "Other": "other: mining (pits, terraces, ponds), fish ponds, logging roads inside forest, rapid regrowth to closed "
             "forest (secondary forest), urban expansion or natural disturbances",
}


def _to_chip(x, y):
    s = PIX_M / (CHIP_M / CHIP_PX)
    return (x - SIZE / 2) * s + CHIP_PX / 2, (y - SIZE / 2) * s + CHIP_PX / 2


def grid_image(row, k: int, scale: int = 2) -> Image.Image | None:
    g = load_region(row.example_path)
    polys = list(g.geoms) if hasattr(g, "geoms") else [g]
    tiles = []
    for t in range(-1, k + 1):
        y = int(row.year) + t
        p = INTERIM / "chips" / str(row.eid) / f"k{t}_{y}.png"
        im = Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", (CHIP_PX, CHIP_PX), (0, 0, 0))
        im = im.resize((CHIP_PX * scale, CHIP_PX * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        for poly in polys:
            pts = [tuple(v * scale for v in _to_chip(x, yy)) for x, yy in poly.exterior.coords]
            d.line(pts + [pts[0]], fill=(255, 230, 0), width=2)
        d.rectangle([0, 0, 52, 16], fill=(0, 0, 0))
        d.text((3, 2), f"year {t:+d}", fill=(255, 255, 255))  # relative years only (design notes)
        tiles.append(im)
    w = tiles[0].width
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (cols * w, rows * w), (255, 255, 255))
    for i, im in enumerate(tiles):
        out.paste(im, ((i % cols) * w, (i // cols) * w))
    return out


def _fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def fact_sheet(row, wide_row, static_row, k: int) -> str:
    # no calendar years: in ForestNet the grassland, logging-road and secondary-forest events all date from 2012 on,
    # so the loss year itself would leak the class (design notes)
    lines = [f"Forest loss event in Indonesia (Hansen GFC). Year 0 is the year of the tree cover loss. "
             f"Evidence available up to year {k:+d} after the loss.",
             f"Loss area: {_fmt(static_row['area_ha'], 1)} ha in {int(static_row['n_parts'])} patch(es). "
             f"Tree cover in 2000 inside the area: {_fmt(static_row['tc2000_in'], 0)}%. "
             f"Peat soil: {'yes' if static_row['peat'] else 'no'}. Mean elevation: {_fmt(static_row['srtm0_mean'], 0)} m.",
             # no OSM distances and no Hansen gain: both carry information from after the loss (design notes)
             "Yearly Landsat median composites (inside the loss area | 150 m ring around it):",
             "year | NDVI in | NDVI ring | NBR in | NBR ring | NDMI in | clear obs in"]
    for t in range(-2, k + 1):
        g = lambda c: wide_row.get(f"{c}_y{t}", np.nan)  # noqa: E731
        lines.append(f"{t:+d} | {_fmt(g('ndvi_in'))} | {_fmt(g('ndvi_ring'))} | {_fmt(g('nbr_in'))} | "
                     f"{_fmt(g('nbr_ring'))} | {_fmt(g('ndmi_in'))} | {_fmt(g('nobs_in'), 0)}")
    lines.append("The image grid shows false-colour (SWIR1, NIR, red) yearly composites of 3 km x 3 km around the event, "
                 "with the loss area outlined in yellow; black means no cloud-free data that year.")
    return "\n".join(lines)


def question(order: list[str]) -> tuple[str, dict]:
    letters = list(OPTIONS)
    mapping = {letters[i]: lab for i, lab in enumerate(order)}
    opts = "\n".join(f"{L}) {OPTION_TEXT[lab]}" for L, lab in mapping.items())
    q = ("What is the direct driver of this forest loss? Choose one option.\n" + opts +
         "\nAnswer with the letter only.")
    return q, mapping
