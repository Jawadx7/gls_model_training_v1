"""
STAGE 2 — Convert lat/lon polygon annotations into per-tile training labels:
  - YOLO-seg: normalized pixel-coordinate polygon lines (RGB-space, lightly simplified)
  - U-Net: rasterized per-pixel class masks (full precision, no simplification)

Skips any tile whose label + mask already exist — safe to rerun.

Run from gls_model/:
    !python data/convert_annotations.py --dry-run
    !python data/convert_annotations.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ANNOTATIONS_EXPORT_PATH, TILE_META_PATH, RASTERS_DIR,
                     LABELS_YOLO_DIR, MASKS_UNET_DIR, CLASS_TO_ID, IGNORE_ID)


def run(export_path, tile_meta_path, rasters_dir, labels_dir, masks_dir, force=False):
    with open(export_path) as f:
        features = json.load(f)["features"]
    by_tile = {}
    for feat in features:
        by_tile.setdefault(feat["properties"]["tileId"], []).append(feat)

    tile_meta = pd.read_csv(tile_meta_path)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    total = 0
    for _, row in tile_meta.iterrows():
        name, tid = row.tile_name, row.tile_id
        label_path = f"{labels_dir}/{name}.txt"
        mask_path = f"{masks_dir}/{name}.npy"
        if os.path.exists(label_path) and os.path.exists(mask_path) and not force:
            print(f"[skip] {name}: already converted")
            continue

        with rasterio.open(f"{rasters_dir}/{name}.tif") as src:
            transform, H, W = src.transform, src.height, src.width
        inv = ~transform

        yolo_lines, shapes_for_mask = [], []
        for feat in by_tile.get(tid, []):
            cls_id = CLASS_TO_ID[feat["properties"]["label"]]
            coords = feat["geometry"]["coordinates"][0]
            px_coords = [inv * (lon, lat) for lon, lat in coords]
            poly_px = Polygon(px_coords).simplify(1.5, preserve_topology=True)
            if poly_px.is_empty or not poly_px.exterior:
                continue

            norm = []
            for x, y in poly_px.exterior.coords:
                norm += [max(0, min(1, x / W)), max(0, min(1, y / H))]
            yolo_lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm))
            shapes_for_mask.append((poly_px, cls_id))

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        mask = np.full((H, W), IGNORE_ID, dtype=np.uint8)
        if shapes_for_mask:
            mask = rasterize(shapes_for_mask, out_shape=(H, W), fill=IGNORE_ID, dtype=np.uint8)
        np.save(mask_path, mask)

        total += len(yolo_lines)
        print(f"{name}: {len(yolo_lines)} polygons")

    print(f"\nTotal polygons written: {total} (expect ~1341)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", default=ANNOTATIONS_EXPORT_PATH)
    p.add_argument("--tile-meta", default=TILE_META_PATH)
    p.add_argument("--rasters-dir", default=RASTERS_DIR)
    p.add_argument("--labels-dir", default=LABELS_YOLO_DIR)
    p.add_argument("--masks-dir", default=MASKS_UNET_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print(f"Would read: {args.export}, {args.tile_meta}, rasters from {args.rasters_dir}")
        print(f"Would write to: {args.labels_dir}, {args.masks_dir}")
    else:
        run(args.export, args.tile_meta, args.rasters_dir, args.labels_dir, args.masks_dir, force=args.force)
