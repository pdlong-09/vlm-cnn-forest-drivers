"""Driver attribution, on server1 with ~/llm4deforest/paperA/venv_render (Pillow 13.0.0 only): render evidence-card grids.

    python render_cards.py --jobs jobs.json --chips <dir> --out <dir>

render.py (a copy of rsdrivers/render.py) must sit next to this file. Writes JPEG quality 92 files named by job.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import QUALITY, render_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--chips", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(a.jobs).read_text(encoding="utf-8"))
    for j in jobs:
        render_grid(j, Path(a.chips)).save(out / j["name"], quality=QUALITY)
    print(f"rendered {len(jobs)} cards")


if __name__ == "__main__":
    main()
