"""Yearly Landsat statistics for ForestNet events computed on Google Earth Engine (server-side reductions only).

Reading Landsat COGs from Planetary Computer was too slow from Vietnam (about 1 MB/s, 73 s per event-year after GDAL
tuning, hundreds of GB for the whole set), so the reductions run where the data are and only numbers come back.

Composite: all Landsat 5/7/8/9 Collection 2 Tier 1 Level-2 scenes of a calendar year, QA_PIXEL clear or water and no
cloud, shadow, dilated cloud or cirrus, scaled to surface reflectance, per-pixel median. Statistics per event and year:
median NDVI, NBR and NDMI and the number of clear observations, inside the loss polygon and in a 150 m ring.
"""
from __future__ import annotations

import os
from pathlib import Path

import ee

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
_L57 = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
_L89 = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]


def init():
    project = os.environ.get("GEE_PROJECT")
    if not project:  # the project id lives in the repository .env, which is not committed
        env = Path(__file__).resolve().parents[3] / ".env"
        for line in env.read_text(encoding="utf-8").splitlines() if env.exists() else []:
            if line.startswith("GEE_PROJECT="):
                project = line.split("=", 1)[1].strip().strip('"')
    ee.Initialize(project=project)
    return ee


def _prep(img, src):
    qa = img.select("QA_PIXEL")
    bad = (qa.bitwiseAnd(1 << 1).neq(0)  # dilated cloud
           .Or(qa.bitwiseAnd(1 << 2).neq(0))  # cirrus
           .Or(qa.bitwiseAnd(1 << 3).neq(0))  # cloud
           .Or(qa.bitwiseAnd(1 << 4).neq(0))  # cloud shadow
           .Or(qa.bitwiseAnd(1 << 0).neq(0)))  # fill
    ok = qa.bitwiseAnd(1 << 6).neq(0).Or(qa.bitwiseAnd(1 << 7).neq(0)).And(bad.Not())
    sr = img.select(src, BANDS).multiply(0.0000275).add(-0.2)
    # residual bright cloud and haze that QA misses (seen on Landsat 5 chips, 21/10/2026): blue reflectance > 0.15
    ok = ok.And(sr.select("blue").lt(0.15)).And(sr.reduce(ee.Reducer.min()).gt(-0.05))
    return sr.updateMask(ok).copyProperties(img, ["system:time_start"])


SENSORS = {"LT05": _L57, "LE07": _L57, "LC08": _L89, "LC09": _L89}
ALL = ("LT05", "LE07", "LC08", "LC09")  # v1 (ForestNet)
L8 = ("LC08",)  # v2.1: one sensor, so composites do not reveal the calendar year


def landsat(year: int, sensors=ALL, first_month: int = 1, region=None) -> ee.ImageCollection:
    """region (24/10/2026, speed only): scenes are filtered to it first; scenes outside it never contribute to pixels
    inside it, so a composite read inside the region is unchanged (E2 chips re-made this way: byte-identical PNGs)."""
    d0, d1 = f"{year}-{first_month:02d}-01", f"{year + 1}-01-01"
    c = ee.ImageCollection([])
    for s in sensors:
        col = ee.ImageCollection(f"LANDSAT/{s}/C02/T1_L2").filterDate(d0, d1)
        if region is not None:
            col = col.filterBounds(region)
        c = c.merge(col.filter(ee.Filter.lt("CLOUD_COVER", 70)).map(lambda i, src=SENSORS[s]: _prep(i, src)))
    return c


def year_stack(year: int, **kw) -> ee.Image:
    c = landsat(year, **kw)
    med = c.median()
    idx = ee.Image.cat(med.normalizedDifference(["nir", "red"]).rename("ndvi"),
                       med.normalizedDifference(["nir", "swir2"]).rename("nbr"),
                       med.normalizedDifference(["nir", "swir1"]).rename("ndmi"))
    nobs = c.select("nir").count().rename("nobs")
    return idx.addBands(nobs)


def reduce_year(year: int, features: list[dict], **kw) -> list[dict]:
    """features: [{'id': str, 'geom': GeoJSON}] -> per feature the median of each index and of nobs (30 m)."""
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geom"]), {"fid": f["id"]}) for f in features])
    red = ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True)
    out = year_stack(year, **kw).reduceRegions(collection=fc, reducer=red, scale=30, tileScale=4)
    return [f["properties"] for f in out.getInfo()["features"]]


def _box(lon: float, lat: float, half_m: float):
    import math
    dlat = half_m / 110574.0
    dlon = half_m / (111320.0 * math.cos(math.radians(lat)))
    return ee.Geometry.Rectangle([lon - dlon, lat - dlat, lon + dlon, lat + dlat])


def chip_url(year: int, lon: float, lat: float, half_m: float = 1500, px: int = 128, kind: str = "false", bounded=False,
             **kw) -> str:
    """PNG thumbnail of the yearly median composite around an event (false colour SWIR1/NIR/red, or true colour).
    bounded (design notes E3, speed only): the scenes are filtered to the chip first (see landsat)."""
    region = _box(lon, lat, half_m)
    med = landsat(year, region=region if bounded else None, **kw).median()
    if kind == "false":
        vis = med.visualize(bands=["swir1", "nir", "red"], min=[0.0, 0.0, 0.0], max=[0.35, 0.5, 0.2])
    else:
        vis = med.visualize(bands=["red", "green", "blue"], min=0.0, max=0.18, gamma=1.3)
    return vis.getThumbURL({"region": region, "dimensions": px, "format": "png"})


def s2a(d0: str, d1: str, region=None) -> ee.ImageCollection:
    """v2.1: Sentinel-2A only (one platform, as Landsat 8 only), L1C harmonised TOA, cloud probability < 40.
    region (22/10/2026, speed only): both collections are filtered to the chip before the join; scenes outside the
    chip never contributed to it, so the pixels are unchanged (checked against chips made without it)."""
    a = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(d0, d1)
         .filter(ee.Filter.eq("SPACECRAFT_NAME", "Sentinel-2A")).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    p = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterDate(d0, d1)
    if region is not None:
        a, p = a.filterBounds(region), p.filterBounds(region)
    j = ee.Join.saveFirst("cp").apply(a, p, ee.Filter.equals(leftField="system:index", rightField="system:index"))
    return ee.ImageCollection(j).map(
        lambda i: i.select(["B4", "B8", "B11"], ["red", "nir", "swir1"]).divide(10000)
        .updateMask(ee.Image(i.get("cp")).select("probability").lt(40)))


def s2_chip_url(d0: str, d1: str, lon: float, lat: float, half_m: float = 1500, px: int = 300) -> str:
    """False colour SWIR1/NIR/red of the Sentinel-2A median in [d0, d1), same stretch as the Landsat chips."""
    box = _box(lon, lat, half_m)
    vis = s2a(d0, d1, box).median().visualize(bands=["swir1", "nir", "red"], min=[0.0, 0.0, 0.0], max=[0.35, 0.5, 0.2])
    return vis.getThumbURL({"region": box, "dimensions": px, "format": "png"})


def static_stats(features: list[dict]) -> list[dict]:
    """v2.1 static layers per polygon: SRTM elevation/slope/aspect (mean, std), Hansen treecover2000 (mean) and the
    most frequent Hansen loss year in 2015-2020 (cross-check of the authors' year only, never a feature)."""
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geom"]), {"fid": f["id"]}) for f in features])
    dem = ee.Image("USGS/SRTMGL1_003")
    terrain = ee.Terrain.products(dem).select(["elevation", "slope", "aspect"])
    gfc = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    ly = gfc.select("lossyear")
    ly = ly.updateMask(ly.gte(15).And(ly.lte(20))).rename("lossyear")
    a = terrain.reduceRegions(fc, ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), 30, tileScale=4)
    b = gfc.select("treecover2000").reduceRegions(a, ee.Reducer.mean().setOutputs(["tc2000_in"]), 30, tileScale=4)
    c = ly.reduceRegions(b, ee.Reducer.mode().setOutputs(["hansen_year"]), 30, tileScale=4)
    return [f["properties"] for f in c.getInfo()["features"]]
