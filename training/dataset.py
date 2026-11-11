"""
Shared TileDataset — imported by train_unet.py AND both evaluation scripts,
so split/augment logic is defined exactly once. This is the fix for the
earlier bug where "augment" was used to infer "train vs test" (they are NOT
the same thing — a validation tile is a held-out TRAIN tile, not a test tile).

split and augment are always passed explicitly and independently.
"""
import os
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from config import DATASET_DIR, PATCH_SIZE


def normalize_image(img):
    """Per-band 1st-99th percentile normalization. MUST match at train and eval time."""
    img = img.astype("float32", copy=True)
    for b in range(img.shape[0]):
        p1, p99 = np.percentile(img[b], [1, 99])
        img[b] = np.clip((img[b] - p1) / (p99 - p1 + 1e-6), 0, 1)
    return img


class TileDataset(Dataset):
    def __init__(self, tile_names, split, patches_per_tile=8, augment=True, dataset_dir=DATASET_DIR):
        """
        split:   "train" or "test" — which physical folder to read from.
        augment: whether to apply flips/rotations AND randomly sample patches
                 (True for fitting; False for validation/test, even though
                 validation tiles still live in the "train" split folder).
        """
        self.samples = []
        self.augment = augment

        for name in tile_names:
            with rasterio.open(f"{dataset_dir}/unet/images/{split}/{name}.tif") as src:
                img = src.read().astype("float32")
            mask = np.load(f"{dataset_dir}/unet/masks/{split}/{name}.npy")
            img = normalize_image(img)

            H, W = mask.shape
            n = patches_per_tile if augment else max(1, (H // PATCH_SIZE) * (W // PATCH_SIZE))

            for i in range(n):
                if H <= PATCH_SIZE or W <= PATCH_SIZE:
                    self.samples.append((img, mask))
                    continue

                if augment:
                    y = random.randint(0, H - PATCH_SIZE)
                    x = random.randint(0, W - PATCH_SIZE)
                else:
                    grid_w = W // PATCH_SIZE
                    row_i, col_i = i // grid_w, i % grid_w
                    y = min(row_i * PATCH_SIZE, H - PATCH_SIZE)
                    x = min(col_i * PATCH_SIZE, W - PATCH_SIZE)

                self.samples.append((img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE],
                                      mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        img, mask = img.copy(), mask.copy()

        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, :, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[::-1, :].copy()
            k = random.choice([0, 1, 2, 3])
            img, mask = np.rot90(img, k, axes=(1, 2)).copy(), np.rot90(mask, k).copy()

        return torch.from_numpy(img), torch.from_numpy(mask.astype("int64"))
