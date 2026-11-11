# GLS Model Pipeline — Notebook Reference

Copy `gls_model/` to Drive as a **sibling** of `GLS_Model_Training/` (both under
`.../GhanaLandScan/`). If your layout differs, only `config.py`'s `DATA_DIR`
needs to change.

---

## Notebook cells, in order

```notebook-python
# ── Setup: mount Drive, cd into code folder ──
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/04_Projects/GhanaLandScan/gls_model
!pip install ultralytics sentinelhub rasterio segmentation-models-pytorch scikit-learn -q
```

```notebook-python
# ── Set Sentinel Hub credentials (Colab Secrets — never hardcoded) ──
import os
from google.colab import userdata
os.environ["SH_CLIENT_ID"] = userdata.get("SH_CLIENT_ID")
os.environ["SH_CLIENT_SECRET"] = userdata.get("SH_CLIENT_SECRET")
```

```notebook-python
# ── Stage 1a: build tile_meta.csv from the annotation export ──
# Skips rebuild if tile_meta.csv already exists (add --force to redo)
!python data/build_tile_metadata.py --dry-run
!python data/build_tile_metadata.py
```

```notebook-python
# ── Stage 1b: pull Sentinel-2 rasters per tile ──
# Skips any tile whose .tif already exists — safe to rerun after a disconnect
!python data/pull_rasters.py --dry-run
!python data/pull_rasters.py
```

```notebook-python
# ── Stage 2: convert polygons to YOLO-seg labels + U-Net masks ──
# Skips any tile already converted (add --force to redo)
!python data/convert_annotations.py --dry-run
!python data/convert_annotations.py
```

```notebook-python
# ── Stage 3: build train/test folder structure + data.yaml ──
# Skips any file already copied
!python data/build_dataset_folders.py --dry-run
!python data/build_dataset_folders.py
```

```notebook-python
# ── Stage 4: train YOLOv8-seg ──
# v2 config (reported result): freeze=0, mosaic=0
!python training/train_yolo.py --dry-run
!python training/train_yolo.py --name yolo_ashanti_v2 --freeze 0 --mosaic 0
```

```notebook-python
# ── Stage 5: train U-Net ──
# --weighting sqrt = v2 (reported result). --weighting inverse reproduces the
# original run that let Agricultural Land hijack the loss.
!python training/train_unet.py --dry-run
!python training/train_unet.py --weighting sqrt --out-name unet_best_v2.pt
```

```notebook-python
# ── Stage 6a: evaluate — full-tile inference (primary Ch4 numbers) ──
!python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best_v2.pt --dry-run
!python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best_v2.pt
```

```notebook-python
# ── Stage 6b: evaluate — patch-grid inference (robustness cross-check) ──
!python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best_v2.pt --dry-run
!python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best_v2.pt
```

---

## Folder map — what lives where

```
gls_model/                          (code — upload to Drive, this is your repo)
├── config.py                       All paths + class list. Edit DATA_DIR here if layout changes.
├── data/
│   ├── tile_name_mapping.py        Static name -> tileId dict
│   ├── build_tile_metadata.py      Stage 1a
│   ├── pull_rasters.py             Stage 1b (needs SH_CLIENT_ID/SECRET env vars)
│   ├── convert_annotations.py      Stage 2
│   └── build_dataset_folders.py    Stage 3
├── training/
│   ├── dataset.py                  Shared TileDataset — do not duplicate this logic elsewhere
│   ├── train_yolo.py               Stage 4
│   └── train_unet.py               Stage 5
└── evaluation/
    ├── evaluate_unet_fulltile.py   Stage 6a
    └── evaluate_unet_patchgrid.py  Stage 6b

GLS_Model_Training/                 (data — generated files, gitignore-sized, stays in Drive only)
├── annotations_export.geojson      Stage 1a input (your raw combined export)
├── tile_meta.csv                   Stage 1a output
├── rasters/, rasters_rgb/          Stage 1b output
├── labels_yolo/, masks_unet/       Stage 2 output
├── dataset/                        Stage 3 output (yolo/ + unet/ train+test folders)
└── checkpoints/
    ├── unet_best.pt, unet_best_v2.pt
    └── yolo_runs/                  Ultralytics' own run logs/plots/weights
```
