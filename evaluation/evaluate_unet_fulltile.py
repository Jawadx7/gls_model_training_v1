"""
STAGE 6a — Evaluate U-Net on the 6 held-out test tiles using FULL-TILE inference
(each tile fed to the model whole, padded to a multiple of 32, cropped back).

This is the PRIMARY reported number for Ch4 — gives the model full spatial
context per tile, at the cost of a train/test input-size mismatch (model only
ever trained on 256x256 patches). See evaluate_unet_patchgrid.py for the
matching-input-size robustness cross-check.

Run from gls_model/:
    !python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best_v2.pt --dry-run
    !python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best_v2.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, jaccard_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TILE_META_PATH, DATASET_DIR, CHECKPOINTS_DIR, NUM_CLASSES, IGNORE_ID, CLASS_NAMES
from training.dataset import normalize_image


def predict_full_tile(name, model, device, dataset_dir):
    with rasterio.open(f"{dataset_dir}/unet/images/test/{name}.tif") as src:
        img = src.read().astype("float32")
    mask = np.load(f"{dataset_dir}/unet/masks/test/{name}.npy")
    img = normalize_image(img)

    C, H, W = img.shape
    pad_h, pad_w = (32 - H % 32) % 32, (32 - W % 32) % 32
    img_p = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    with torch.no_grad():
        x = torch.from_numpy(img_p).unsqueeze(0).to(device)
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy()
    return pred[:H, :W], mask


def run(checkpoint_path, tile_meta_path, dataset_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", in_channels=6, classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    tile_meta = pd.read_csv(tile_meta_path)
    test_names = tile_meta[tile_meta.split == "test"].tile_name.tolist()

    all_preds, all_true = [], []
    for name in test_names:
        pred, true = predict_full_tile(name, model, device, dataset_dir)
        valid = true != IGNORE_ID
        all_preds.append(pred[valid])
        all_true.append(true[valid])
        print(f"{name}: {valid.sum()} labeled pixels evaluated")

    y_pred, y_true = np.concatenate(all_preds), np.concatenate(all_true)
    print(f"\nTotal labeled pixels: {len(y_true)}")

    print("\n" + classification_report(y_true, y_pred, labels=range(NUM_CLASSES),
                                        target_names=CLASS_NAMES, zero_division=0))

    iou = jaccard_score(y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
    print("Per-class IoU:")
    for name, v in zip(CLASS_NAMES, iou):
        print(f"  {name}: {v:.4f}")

    print(f"\nOverall Accuracy: {(y_true == y_pred).mean():.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(CLASS_NAMES)
    print(cm)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Filename inside checkpoints/, e.g. unet_best_v2.pt")
    p.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR)
    p.add_argument("--tile-meta", default=TILE_META_PATH)
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    ckpt_path = f"{args.checkpoints_dir}/{args.checkpoint}"

    if args.dry_run:
        print(f"Would evaluate checkpoint: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")
    else:
        run(ckpt_path, args.tile_meta, args.dataset_dir)
