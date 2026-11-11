"""
STAGE 6b — Evaluate U-Net on the 6 held-out test tiles using PATCH-GRID
inference (each tile chopped into non-overlapping 256x256 patches, matching
the exact input size the model trained on).

Robustness cross-check alongside evaluate_unet_fulltile.py — report BOTH in
Ch4 rather than picking one. Expect Forest Land to drop slightly here
(patch-boundary seams on a spatially contiguous class); expect no dramatic
swing on rare classes if the model is genuinely stable.

Run from gls_model/:
    !python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best_v2.pt --dry-run
    !python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best_v2.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, jaccard_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TILE_META_PATH, DATASET_DIR, CHECKPOINTS_DIR, NUM_CLASSES, IGNORE_ID, CLASS_NAMES, PATCH_SIZE
from training.dataset import normalize_image

STRIDE = PATCH_SIZE  # non-overlapping


def predict_patchgrid(name, model, device, dataset_dir):
    with rasterio.open(f"{dataset_dir}/unet/images/test/{name}.tif") as src:
        img = src.read().astype("float32")
    mask = np.load(f"{dataset_dir}/unet/masks/test/{name}.npy")
    img = normalize_image(img)

    C, H, W = img.shape
    pad_h, pad_w = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE, (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
    padded_H, padded_W = H + pad_h, W + pad_w
    img_p = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    logits_sum = np.zeros((NUM_CLASSES, padded_H, padded_W), dtype=np.float32)
    pred_count = np.zeros((padded_H, padded_W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, padded_H, STRIDE):
            for x in range(0, padded_W, STRIDE):
                patch = img_p[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                if patch.shape[1] != PATCH_SIZE or patch.shape[2] != PATCH_SIZE:
                    continue
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(tensor).squeeze(0).cpu().numpy()
                logits_sum[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += logits
                pred_count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    logits_sum /= np.maximum(pred_count[None, :, :], 1)
    pred = np.argmax(logits_sum, axis=0)[:H, :W]
    return pred, mask


def run(checkpoint_path, tile_meta_path, dataset_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", in_channels=6, classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    tile_meta = pd.read_csv(tile_meta_path)
    test_names = sorted(tile_meta[tile_meta.split == "test"].tile_name.tolist())

    all_preds, all_true = [], []
    for name in test_names:
        pred, true = predict_patchgrid(name, model, device, dataset_dir)
        valid = true != IGNORE_ID
        all_preds.append(pred[valid])
        all_true.append(true[valid])
        print(f"{name}: {valid.sum():,} labeled pixels evaluated")

    y_pred, y_true = np.concatenate(all_preds), np.concatenate(all_true)
    print(f"\nTotal labeled pixels: {len(y_true):,}")

    print("\n" + classification_report(y_true, y_pred, labels=range(NUM_CLASSES),
                                        target_names=CLASS_NAMES, zero_division=0))

    iou = jaccard_score(y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
    print("Per-class IoU / F1:")
    for name, i, f in zip(CLASS_NAMES, iou, f1):
        print(f"  {name:<22}: IoU={i:.4f}  F1={f:.4f}")

    print(f"\nOverall Accuracy: {(y_true == y_pred).mean():.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")

    present = sorted(np.unique(y_true))
    print(f"Mean IoU (present test classes): {np.mean([iou[i] for i in present]):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(CLASS_NAMES)
    print(cm)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
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
