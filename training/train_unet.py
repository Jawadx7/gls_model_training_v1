"""
STAGE 5 — Train U-Net (6-band, from scratch, per-pixel class-weighted loss).

--weighting inverse : 1/count            (v1 — Agricultural Land's rarity let it
                                           hijack the loss and false-positive onto
                                           real Forest/Mining pixels)
--weighting sqrt     : 1/sqrt(count)      (v2 — reported result, compresses the
                                           extremes so no single rare class dominates)

Run from gls_model/:
    !python training/train_unet.py --dry-run
    !python training/train_unet.py --weighting sqrt --out-name unet_best_v2.pt
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TILE_META_PATH, CHECKPOINTS_DIR, NUM_CLASSES, IGNORE_ID, CLASS_NAMES
from training.dataset import TileDataset


def compute_weights(train_ds, scheme):
    pixel_counts = np.zeros(NUM_CLASSES)
    for _, mask in train_ds.samples:
        for c in range(NUM_CLASSES):
            pixel_counts[c] += (mask == c).sum()
    print("Pixel counts per class:", dict(zip(CLASS_NAMES, pixel_counts.astype(int))))

    if scheme == "inverse":
        weights = 1.0 / (pixel_counts + 1e-6)
    elif scheme == "sqrt":
        weights = 1.0 / np.sqrt(pixel_counts + 1e-6)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    weights[pixel_counts == 0] = 0
    if weights.sum() > 0:
        weights = weights / weights.sum() * NUM_CLASSES
    print(f"Class weights ({scheme}):", dict(zip(CLASS_NAMES, weights)))
    return torch.tensor(weights, dtype=torch.float32)


def run(args):
    tile_meta = pd.read_csv(args.tile_meta)
    train_names = tile_meta[tile_meta.split == "train"].tile_name.tolist()

    random.seed(0)
    val_names = random.sample(train_names, 3)
    fit_names = [n for n in train_names if n not in val_names]
    print(f"Fit: {len(fit_names)} tiles | Val: {val_names}")

    train_ds = TileDataset(fit_names, split="train", patches_per_tile=10, augment=True)
    val_ds = TileDataset(val_names, split="train", patches_per_tile=4, augment=False)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_t = compute_weights(train_ds, args.weighting).to(device)

    model = smp.Unet(encoder_name="resnet34", in_channels=6, classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_t, ignore_index=IGNORE_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    out_path = f"{args.checkpoints_dir}/{args.out_name}"

    best_val_loss, patience_counter = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), out_path)
            print(f"  saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f} | Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tile-meta", default=TILE_META_PATH)
    p.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR)
    p.add_argument("--out-name", default="unet_best_v2.pt")
    p.add_argument("--weighting", choices=["inverse", "sqrt"], default="sqrt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print(f"Would train U-Net, weighting={args.weighting}, save to "
              f"{args.checkpoints_dir}/{args.out_name} (tile_meta exists: {os.path.exists(args.tile_meta)})")
    else:
        run(args)
