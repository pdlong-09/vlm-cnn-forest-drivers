"""ForestNet I/O: paths, the event table, and a restricted loader for the loss-region pickles."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import shapely.geometry.multipolygon as _mp
import shapely.geometry.polygon as _p

ROOT = Path(__file__).resolve().parents[3]
FN = ROOT / "data/raw/forestnet_A/deep/downloads/ForestNetDataset"
INTERIM = ROOT / "data/interim/rs_forestnet"
RUNS = ROOT / "runs/rs_forestnet"
PIX_M = 15.0  # ForestNet images: 332 x 332 pan-sharpened Landsat 8 pixels of 15 m, centred on latitude/longitude
SIZE = 332

_ALLOWED = {("shapely.geometry.polygon", "Polygon"): _p.Polygon,
            ("shapely.geometry.multipolygon", "MultiPolygon"): _mp.MultiPolygon}


class _SafeUnpickler(pickle.Unpickler):
    """The downloaded pickles hold shapely geometries only (checked for all 2,757 files); anything else is refused."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def load_region(example_path: str):
    """Loss polygon of one event, in image pixel coordinates (x = column, y = row)."""
    import warnings
    with open(FN / example_path / "forest_loss_region.pkl", "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # shapely<2 pickles: compatibility warning only
        return _SafeUnpickler(f).load()


def events() -> pd.DataFrame:
    """All 2,757 events with the official split; `eid` is a stable id (row order of train, val, test)."""
    d = pd.concat([pd.read_csv(FN / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
    d.insert(0, "eid", range(len(d)))
    return d


def region_lonlat(row):
    """Loss polygon in lon/lat: the image is north-up, centred on (latitude, longitude), 15 m pixels.

    Checked 21/10/2026 against Hansen GFC treecover2000 (auxiliary/gfc.npy channel 1) for 6 random events:
    correlation 0.88-0.98 at zero offset, best offset at most 2 px (one 30 m Hansen cell)."""
    import math
    from shapely.affinity import affine_transform
    dlat = PIX_M / 110574.0
    dlon = PIX_M / (111320.0 * math.cos(math.radians(row.latitude)))
    half = SIZE / 2
    # lon = lon0 + (x - half) * dlon ; lat = lat0 - (y - half) * dlat
    return affine_transform(load_region(row.example_path),
                            [dlon, 0, 0, -dlat, row.longitude - half * dlon, row.latitude + half * dlat])
