"""
STAGE 1a — Build tile_meta.csv from the combined annotation export.

What this does: groups every annotation by tileId, computes each tile's
bounding box (+10% buffer), attaches its locked imagery date / composite
window, maps tileId -> tile name, drops anything outside the agreed 24-tile
set, and applies the fixed train/test split.

Run from gls_model/:
    !python data/build_tile_metadata.py --dry-run
    !python data/build_tile_metadata.py
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANNOTATIONS_EXPORT_PATH, TILE_META_PATH, TEST_NAMES
from data.tile_name_mapping import ID_TO_NAME


def build(export_path, output_path, force=False):
    if os.path.exists(output_path) and not force:
        print(f"[skip] {output_path} already exists. Use --force to rebuild.")
        return pd.read_csv(output_path)

    with open(export_path) as f:
        data = json.load(f)
    features = data["features"]
    print(f"Total annotations in export: {len(features)}")

    tiles = {}
    for feat in features:
        props = feat["properties"]
        tile_id = props["tileId"]
        coords = feat["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        if tile_id not in tiles:
            tiles[tile_id] = {
                "min_lon": min(lons), "max_lon": max(lons),
                "min_lat": min(lats), "max_lat": max(lats),
                "n_annotations": 0,
                "imageryDate": props.get("imageryDate"),
                "imageryDateFrom": props.get("imageryDateFrom"),
                "imageryDateTo": props.get("imageryDateTo"),
                "compositeMethod": props.get("compositeMethod"),
            }
        t = tiles[tile_id]
        t["min_lon"] = min(t["min_lon"], min(lons))
        t["max_lon"] = max(t["max_lon"], max(lons))
        t["min_lat"] = min(t["min_lat"], min(lats))
        t["max_lat"] = max(t["max_lat"], max(lats))
        t["n_annotations"] += 1

    rows = []
    for tile_id, t in tiles.items():
        name = ID_TO_NAME.get(tile_id)
        if name is None:
            continue  # not in the agreed 24-tile set

        w, h = t["max_lon"] - t["min_lon"], t["max_lat"] - t["min_lat"]
        buf_lon, buf_lat = w * 0.10, h * 0.10
        mode = "single" if t["imageryDate"] else "composite"

        rows.append({
            "tile_id": tile_id, "tile_name": name,
            "min_lon": t["min_lon"] - buf_lon, "max_lon": t["max_lon"] + buf_lon,
            "min_lat": t["min_lat"] - buf_lat, "max_lat": t["max_lat"] + buf_lat,
            "mode": mode,
            "date_single": t["imageryDate"],
            "date_from": t["imageryDateFrom"], "date_to": t["imageryDateTo"],
            "composite_method": t["compositeMethod"],
            "n_annotations": t["n_annotations"],
        })

    tile_meta = pd.DataFrame(rows)
    tile_meta["split"] = tile_meta["tile_name"].apply(lambda n: "test" if n in TEST_NAMES else "train")

    print(f"Tiles in agreed set: {len(tile_meta)} (expect 24)")
    print(tile_meta.groupby("split")["n_annotations"].sum())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tile_meta.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return tile_meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", default=ANNOTATIONS_EXPORT_PATH)
    p.add_argument("--output", default=TILE_META_PATH)
    p.add_argument("--force", action="store_true", help="Rebuild even if output already exists")
    p.add_argument("--dry-run", action="store_true", help="Validate paths only, do not process")
    args = p.parse_args()

    if args.dry_run:
        print(f"Would read:  {args.export} (exists: {os.path.exists(args.export)})")
        print(f"Would write: {args.output}")
    else:
        build(args.export, args.output, force=args.force)
