"""
STAGE 3 — Organize converted labels + rasters into the train/test folder
structure YOLOv8 and the U-Net training loop expect, and write data.yaml.

Skips any file that's already been copied — safe to rerun.

Run from gls_model/:
    !python data/build_dataset_folders.py --dry-run
    !python data/build_dataset_folders.py
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TILE_META_PATH, RASTERS_DIR, LABELS_YOLO_DIR, MASKS_UNET_DIR,
                     DATASET_DIR, CLASS_NAMES)


def run(tile_meta_path, rasters_dir, labels_dir, masks_dir, dataset_dir):
    tile_meta = pd.read_csv(tile_meta_path)

    for split in ["train", "test"]:
        for sub in ["yolo/images", "yolo/labels", "unet/images", "unet/masks"]:
            os.makedirs(f"{dataset_dir}/{sub}/{split}", exist_ok=True)

    for _, row in tile_meta.iterrows():
        name, split = row.tile_name, row.split

        yolo_img_path = f"{dataset_dir}/yolo/images/{split}/{name}.png"
        if not os.path.exists(yolo_img_path):
            with rasterio.open(f"{rasters_dir}/{name}.tif") as src:
                r, g, b = src.read(3), src.read(2), src.read(1)
            rgb = np.stack([r, g, b], axis=-1)
            rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
            Image.fromarray((rgb * 255).astype("uint8")).save(yolo_img_path)

        yolo_lbl_path = f"{dataset_dir}/yolo/labels/{split}/{name}.txt"
        if not os.path.exists(yolo_lbl_path):
            shutil.copy(f"{labels_dir}/{name}.txt", yolo_lbl_path)

        unet_img_path = f"{dataset_dir}/unet/images/{split}/{name}.tif"
        if not os.path.exists(unet_img_path):
            shutil.copy(f"{rasters_dir}/{name}.tif", unet_img_path)

        unet_mask_path = f"{dataset_dir}/unet/masks/{split}/{name}.npy"
        if not os.path.exists(unet_mask_path):
            shutil.copy(f"{masks_dir}/{name}.npy", unet_mask_path)

    yaml_content = f"path: {dataset_dir}/yolo\ntrain: images/train\nval: images/test\nnames:\n"
    for i, cname in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {cname}\n"
    with open(f"{dataset_dir}/yolo/data.yaml", "w") as f:
        f.write(yaml_content)

    for split in ["train", "test"]:
        n_img = len(os.listdir(f"{dataset_dir}/yolo/images/{split}"))
        n_lbl = len(os.listdir(f"{dataset_dir}/yolo/labels/{split}"))
        print(f"YOLO {split}: {n_img} images, {n_lbl} labels")
        n_tif = len(os.listdir(f"{dataset_dir}/unet/images/{split}"))
        n_mask = len(os.listdir(f"{dataset_dir}/unet/masks/{split}"))
        print(f"U-Net {split}: {n_tif} images, {n_mask} masks")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tile-meta", default=TILE_META_PATH)
    p.add_argument("--rasters-dir", default=RASTERS_DIR)
    p.add_argument("--labels-dir", default=LABELS_YOLO_DIR)
    p.add_argument("--masks-dir", default=MASKS_UNET_DIR)
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print(f"Would build dataset folders under: {args.dataset_dir}")
    else:
        run(args.tile_meta, args.rasters_dir, args.labels_dir, args.masks_dir, args.dataset_dir)
