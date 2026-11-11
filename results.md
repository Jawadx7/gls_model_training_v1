Results after running all cells one after the other.

```notebook-python
# ── Stage 1a: build tile_meta.csv from the annotation export ──
# Skips rebuild if tile_meta.csv already exists (add --force to redo)
!python data/build_tile_metadata.py --dry-run
!python data/build_tile_metadata.py

RESULTS:
Would read:  /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/annotations_export.geojson (exists: True)
Would write: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/tile_meta.csv
Total annotations in export: 1349
Tiles in agreed set: 24 (expect 24)
split
test      219
train    1122
Name: n_annotations, dtype: int64
Saved: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/tile_meta.csv
```

```notebook-python
# ── Stage 1b: pull Sentinel-2 rasters per tile ──
# Skips any tile whose .tif already exists — safe to rerun after a disconnect
!python data/pull_rasters.py --dry-run
!python data/pull_rasters.py

RESULTS:
Would read tile_meta: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/tile_meta.csv (exists: True)
Would write to: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/rasters, /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/rasters_rgb
Credentials set: True
[skip] AS-(Sanso): already exists
[skip] AS-(Manso-Adubia): already exists
[skip] AS-(Dokyiwa): already exists
[skip] AS-005: already exists
[skip] AS-006: already exists
[skip] AS-(Anyinam): already exists
[skip] AS-003: already exists
[skip] AS-(Adubia): already exists
[skip] AS-(Anwia-Nkwanta): already exists
[skip] AS-(Poano): already exists
[skip] AS-(Nyinahin): already exists
[skip] AS-(Jacobu): already exists
[skip] AS-(Esaase): already exists
[skip] AS-(Asankrangwa): already exists
[skip] AS-(kunsu): already exists
[skip] AS-001: already exists
[skip] AS-(Fomena): already exists
[skip] AS-(Bibiani): already exists
[skip] AS-007: already exists
[skip] AS-002: already exists
[skip] AS-(Samreboi): already exists
[skip] AS-(Tweapease): already exists
[skip] AS-(Nkroful): already exists
[skip] AS-(Fomena)-1: already exists
```

```notebook-python
# ── Stage 2: convert polygons to YOLO-seg labels + U-Net masks ──
# Skips any tile already converted (add --force to redo)
!python data/convert_annotations.py --dry-run
!python data/convert_annotations.py

RESULTS:
Would read: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/annotations_export.geojson, /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/tile_meta.csv, rasters from /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/rasters
Would write to: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/labels_yolo, /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/masks_unet
[skip] AS-(Sanso): already converted
[skip] AS-(Manso-Adubia): already converted
[skip] AS-(Dokyiwa): already converted
[skip] AS-005: already converted
[skip] AS-006: already converted
[skip] AS-(Anyinam): already converted
[skip] AS-003: already converted
[skip] AS-(Adubia): already converted
[skip] AS-(Anwia-Nkwanta): already converted
[skip] AS-(Poano): already converted
[skip] AS-(Nyinahin): already converted
[skip] AS-(Jacobu): already converted
[skip] AS-(Esaase): already converted
[skip] AS-(Asankrangwa): already converted
[skip] AS-(kunsu): already converted
[skip] AS-001: already converted
[skip] AS-(Fomena): already converted
[skip] AS-(Bibiani): already converted
[skip] AS-007: already converted
[skip] AS-002: already converted
[skip] AS-(Samreboi): already converted
[skip] AS-(Tweapease): already converted
[skip] AS-(Nkroful): already converted
[skip] AS-(Fomena)-1: already converted

Total polygons written: 0 (expect ~1341)
```

```notebook-python
# ── Stage 3: build train/test folder structure + data.yaml ──
# Skips any file already copied
!python data/build_dataset_folders.py --dry-run
!python data/build_dataset_folders.py

RESULTS:
Would build dataset folders under: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/dataset
YOLO train: 18 images, 18 labels
U-Net train: 18 images, 18 masks
YOLO test: 6 images, 6 labels
U-Net test: 6 images, 6 masks
```

```notebook-python
# ── Stage 4: train YOLOv8-seg ──
# v2 config (reported result): freeze=0, mosaic=0
!python training/train_yolo.py --dry-run
!python training/train_yolo.py --name yolo_ashanti_v2 --freeze 0 --mosaic 0

RESULTS:
Would train with data=/content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/dataset/yolo/data.yaml, freeze=0, mosaic=0, name=yolo_ashanti_v2 (data.yaml exists: True)
Ultralytics 8.4.120 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=True, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=False, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/dataset/yolo/data.yaml, degrees=0.0, deterministic=True, device=, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=150, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.1, mode=train, model=yolov8n-seg.pt, momentum=0.937, mosaic=0.0, multi_scale=0.0, name=yolo_ashanti_v2-3, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=20, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs/yolo_ashanti_v2-3, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=segment, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=6

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1   1005250  ultralytics.nn.modules.head.Segment          [6, 32, 64, 16, None, [64, 128, 256]]
YOLOv8n-seg summary: 152 layers, 3,264,786 parameters, 3,264,770 gradients, 11.5 GFLOPs

Transferred 381/417 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.6±0.2 ms, read: 85.2±186.4 MB/s, size: 1618.8 KB)
train: Scanning /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/dataset/yolo/labels/train... 18 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 18/18 1.4s/it 25.3s
train: New cache created: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/dataset/yolo/labels/train.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
WARNING ⚠️ val: Slow image access detected (ping: 1.2±0.6 ms, read: 1.7±0.8 MB/s, size: 1566.0 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/dataset/yolo/labels/test.cache... 6 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 6/6 198.2Kit/s 0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001, momentum=0.9) with parameter groups 66 weight(decay=0.0), 77 weight(decay=0.0005), 76 bias(decay=0.0)
Plotting labels to /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs/yolo_ashanti_v2-3/labels.jpg... 
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs/yolo_ashanti_v2-3
Starting training for 150 epochs...

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      1/150      2.07G      2.979      5.569      4.208      2.537          0        108        640: 100% ━━━━━━━━━━━━ 3/3 1.4s/it 4.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 3.6s/it 3.6s
                   all          6        219    0.00301      0.048    0.00309   0.000482   0.000501      0.008   3.33e-05   4.76e-06

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      2/150      2.49G      2.904      5.717        4.2      2.509          0        195        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          6        219    0.00466     0.0562    0.00368   0.000692    0.00112    0.00808   6.51e-05   1.11e-05

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      3/150      2.49G      2.903      5.546      4.183      2.523          0        106        640: 100% ━━━━━━━━━━━━ 3/3 5.9it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219    0.00466     0.0562    0.00368   0.000692    0.00112    0.00808   6.51e-05   1.11e-05

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      4/150      2.49G      3.018      5.931      4.181      2.614          0        127        640: 100% ━━━━━━━━━━━━ 3/3 2.8it/s 1.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219    0.00306      0.052    0.00325   0.000781    0.00102    0.00808   0.000207   2.22e-05

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      5/150      2.49G      2.874      5.838      4.157      2.595          0        107        640: 100% ━━━━━━━━━━━━ 3/3 3.1it/s 1.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          6        219    0.00306      0.052    0.00325   0.000781    0.00102    0.00808   0.000207   2.22e-05

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      6/150      2.82G      2.976      5.601       4.28      2.455          0        126        640: 100% ━━━━━━━━━━━━ 3/3 4.2it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219    0.00335     0.0481    0.00224   0.000519   0.000945    0.00808   0.000103   1.03e-05

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      7/150      3.21G       2.79      5.799      4.141      2.455          0        199        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219    0.00335     0.0481    0.00224   0.000519   0.000945    0.00808   0.000103   1.03e-05

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      8/150      3.21G      2.912      6.041      4.118      2.585          0        136        640: 100% ━━━━━━━━━━━━ 3/3 5.0it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219    0.00335     0.0481    0.00224   0.000519   0.000945    0.00808   0.000103   1.03e-05

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
      9/150      3.21G      2.884      5.678      4.269      2.429          0        126        640: 100% ━━━━━━━━━━━━ 3/3 5.8it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.5s
                   all          6        219    0.00346     0.0481    0.00227   0.000465   0.000508      0.008   2.61e-05   3.17e-06

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     10/150      3.21G      2.879      5.715      4.069      2.573          0        226        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219    0.00346     0.0481    0.00227   0.000465   0.000508      0.008   2.61e-05   3.17e-06

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     11/150      3.41G      2.964      5.464      4.173      2.495          0        107        640: 100% ━━━━━━━━━━━━ 3/3 4.4it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219    0.00843     0.0441    0.00192   0.000486    0.00553      0.028    0.00256   0.000327

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     12/150      3.42G      2.886      4.874      4.064      2.171          0        174        640: 100% ━━━━━━━━━━━━ 3/3 5.9it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219    0.00843     0.0441    0.00192   0.000486    0.00553      0.028    0.00256   0.000327

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     13/150      3.42G      2.799      4.807      3.979      2.182          0        150        640: 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.4it/s 0.7s
                   all          6        219     0.0118     0.0762    0.00938    0.00142    0.00886     0.0562    0.00697    0.00109

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     14/150      3.42G      2.745      4.907      3.808       2.16          0         49        640: 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          6        219     0.0118     0.0762    0.00938    0.00142    0.00886     0.0562    0.00697    0.00109

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     15/150      3.42G      2.886       4.99      3.801       2.21          0        133        640: 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.4it/s 0.7s
                   all          6        219     0.0118     0.0762    0.00938    0.00142    0.00886     0.0562    0.00697    0.00109

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     16/150      3.86G      2.683       4.41      3.653      2.169          0         39        640: 100% ━━━━━━━━━━━━ 3/3 5.3it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          6        219     0.0122     0.0805     0.0183    0.00264     0.0108     0.0724      0.011    0.00177

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     17/150      3.86G      2.658      4.223      3.642      2.118          0        116        640: 100% ━━━━━━━━━━━━ 3/3 5.7it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219     0.0122     0.0805     0.0183    0.00264     0.0108     0.0724      0.011    0.00177

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     18/150      4.32G      2.783      4.506      3.497       2.13          0        127        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219      0.012     0.0887    0.00798    0.00177    0.00893     0.0647     0.0059    0.00123

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     19/150      4.32G      2.607      4.147      3.461       2.05          0        114        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219      0.012     0.0887    0.00798    0.00177    0.00893     0.0647     0.0059    0.00123

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     20/150      4.32G      2.575       3.98      3.323      1.991          0        172        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219     0.0139      0.126      0.022    0.00486     0.0122      0.109     0.0211     0.0045

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     21/150      4.32G      2.607      3.932      3.142      2.027          0         49        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219     0.0139      0.126      0.022    0.00486     0.0122      0.109     0.0211     0.0045

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     22/150      4.32G      2.584      4.134      3.341      2.001          0        111        640: 100% ━━━━━━━━━━━━ 3/3 4.0it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          6        219     0.0139      0.126      0.022    0.00486     0.0122      0.109     0.0211     0.0045

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     23/150      4.32G      2.566      3.836      3.161      1.927          0        113        640: 100% ━━━━━━━━━━━━ 3/3 3.4it/s 0.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3it/s 0.8s
                   all          6        219     0.0148      0.146     0.0254    0.00894     0.0123      0.118     0.0218    0.00638

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     24/150      4.32G      2.608      3.885      3.169      1.951          0        125        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          6        219     0.0148      0.146     0.0254    0.00894     0.0123      0.118     0.0218    0.00638

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     25/150      4.32G      2.655      4.103      3.119      2.052          0         39        640: 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.5s
                   all          6        219      0.266     0.0896     0.0272    0.00823     0.0181       0.16     0.0296    0.00681

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     26/150      4.32G      2.581      3.983      2.968      1.987          0         62        640: 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219      0.266     0.0896     0.0272    0.00823     0.0181       0.16     0.0296    0.00681

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     27/150      4.32G       2.62      4.064       3.07      2.009          0        184        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219     0.0976     0.0448     0.0223    0.00524       0.08     0.0552     0.0201    0.00391

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     28/150      4.32G      2.722      3.992      2.955      2.021          0         91        640: 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219     0.0976     0.0448     0.0223    0.00524       0.08     0.0552     0.0201    0.00391

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     29/150      4.32G      2.556      4.027      2.996      2.001          0         97        640: 100% ━━━━━━━━━━━━ 3/3 5.0it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219     0.0976     0.0448     0.0223    0.00524       0.08     0.0552     0.0201    0.00391

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     30/150      4.32G      2.527      3.889      2.885      2.012          0        210        640: 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219     0.0501     0.0915     0.0286     0.0108     0.0452      0.083     0.0239    0.00651

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     31/150      4.32G      2.667      4.101      2.832      2.042          0         97        640: 100% ━━━━━━━━━━━━ 3/3 3.8it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          6        219     0.0501     0.0915     0.0286     0.0108     0.0452      0.083     0.0239    0.00651

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     32/150      4.32G      2.536      4.047      2.835      1.992          0        123        640: 100% ━━━━━━━━━━━━ 3/3 4.1it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          6        219     0.0715     0.0828     0.0364     0.0121     0.0575     0.0718     0.0237    0.00614

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     33/150      4.32G      2.481      3.899       2.72      1.925          0        195        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.5s
                   all          6        219     0.0715     0.0828     0.0364     0.0121     0.0575     0.0718     0.0237    0.00614

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     34/150      4.32G      2.494      3.585      2.617      1.906          0         56        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219     0.0936      0.103     0.0405     0.0133     0.0722      0.088     0.0286    0.00916

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     35/150      4.32G      2.379      3.617      2.616      1.865          0         50        640: 100% ━━━━━━━━━━━━ 3/3 5.5it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          6        219     0.0936      0.103     0.0405     0.0133     0.0722      0.088     0.0286    0.00916

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     36/150      4.32G      2.434      3.808      2.821      1.916          0        110        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219     0.0936      0.103     0.0405     0.0133     0.0722      0.088     0.0286    0.00916

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     37/150      4.32G      2.546      3.659      2.627      1.964          0         94        640: 100% ━━━━━━━━━━━━ 3/3 6.1it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219     0.0799      0.119      0.048     0.0158     0.0862      0.115     0.0417      0.014

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     38/150      4.32G      2.577       3.95      2.636      1.922          0        195        640: 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219     0.0799      0.119      0.048     0.0158     0.0862      0.115     0.0417      0.014

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     39/150      4.32G      2.587      4.017      2.594      1.977          0        135        640: 100% ━━━━━━━━━━━━ 3/3 3.8it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          6        219      0.103      0.104     0.0496     0.0193     0.0686      0.118     0.0378     0.0157

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     40/150      4.32G      2.495      3.688      2.533       1.86          0        151        640: 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0it/s 1.0s
                   all          6        219      0.103      0.104     0.0496     0.0193     0.0686      0.118     0.0378     0.0157

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     41/150      4.32G      2.509      4.094      2.561      1.951          0        148        640: 100% ━━━━━━━━━━━━ 3/3 5.0it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          6        219      0.097      0.111     0.0561     0.0215      0.073      0.109       0.05     0.0162

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     42/150      4.32G      2.302      3.483       2.51      1.855          0        168        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219      0.097      0.111     0.0561     0.0215      0.073      0.109       0.05     0.0162

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     43/150      4.32G      2.477      3.603      2.503      1.843          0        148        640: 100% ━━━━━━━━━━━━ 3/3 5.3it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.097      0.111     0.0561     0.0215      0.073      0.109       0.05     0.0162

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     44/150      4.32G      2.523      3.781      2.461      1.846          0        182        640: 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219       0.15      0.106     0.0708     0.0293     0.0788     0.0877     0.0648     0.0271

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     45/150      4.32G      2.385      3.708      2.629      1.853          0        113        640: 100% ━━━━━━━━━━━━ 3/3 5.7it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219       0.15      0.106     0.0708     0.0293     0.0788     0.0877     0.0648     0.0271

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     46/150      4.32G      2.481      3.662      2.425      1.817          0         89        640: 100% ━━━━━━━━━━━━ 3/3 5.8it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219       0.29     0.0816     0.0707     0.0268      0.303     0.0836     0.0703     0.0229

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     47/150      4.32G      2.718      3.787      2.609      1.904          0        124        640: 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0it/s 1.0s
                   all          6        219       0.29     0.0816     0.0707     0.0268      0.303     0.0836     0.0703     0.0229

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     48/150      4.32G      2.301      3.708       2.47      1.772          0         91        640: 100% ━━━━━━━━━━━━ 3/3 3.3it/s 0.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.4it/s 0.7s
                   all          6        219      0.148      0.104     0.0651     0.0201     0.0771      0.164     0.0575     0.0189

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     49/150      4.32G      2.355      3.728      2.404      1.847          0        127        640: 100% ━━━━━━━━━━━━ 3/3 3.7it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219      0.148      0.104     0.0651     0.0201     0.0771      0.164     0.0575     0.0189

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     50/150      4.32G      2.356      3.604      2.473      1.795          0        115        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219      0.148      0.104     0.0651     0.0201     0.0771      0.164     0.0575     0.0189

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     51/150      4.32G      2.446      3.532      2.325      1.809          0         91        640: 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.2it/s 0.5s
                   all          6        219      0.106      0.127     0.0779     0.0256      0.156      0.108      0.079     0.0242

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     52/150      4.32G      2.262      3.472      2.422      1.835          0         86        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219      0.106      0.127     0.0779     0.0256      0.156      0.108      0.079     0.0242

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     53/150      4.32G      2.287      3.776      2.357      1.814          0        127        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.118      0.136     0.0854      0.032      0.128      0.154     0.0962     0.0326

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     54/150      4.32G      2.516      3.946      2.432      1.929          0        208        640: 100% ━━━━━━━━━━━━ 3/3 5.1it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          6        219      0.118      0.136     0.0854      0.032      0.128      0.154     0.0962     0.0326

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     55/150      4.32G      2.353      3.792      2.418      1.904          0        131        640: 100% ━━━━━━━━━━━━ 3/3 3.6it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.5s
                   all          6        219      0.103      0.157     0.0858     0.0302     0.0992      0.151     0.0907     0.0296

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     56/150      4.32G      2.348      3.577      2.369      1.767          0        133        640: 100% ━━━━━━━━━━━━ 3/3 3.7it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0it/s 1.0s
                   all          6        219      0.103      0.157     0.0858     0.0302     0.0992      0.151     0.0907     0.0296

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     57/150      4.32G      2.237      3.622      2.387      1.806          0         55        640: 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.4it/s 0.7s
                   all          6        219      0.103      0.157     0.0858     0.0302     0.0992      0.151     0.0907     0.0296

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     58/150      4.32G      2.232      3.606      2.268       1.75          0         98        640: 100% ━━━━━━━━━━━━ 3/3 6.2it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219     0.0792      0.198     0.0828     0.0308      0.112      0.129     0.0831     0.0282

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     59/150      4.32G      2.326      3.627      2.349       1.81          0         56        640: 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0s/it 1.0s
                   all          6        219     0.0792      0.198     0.0828     0.0308      0.112      0.129     0.0831     0.0282

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     60/150      4.32G      2.253      3.526      2.253      1.792          0         99        640: 100% ━━━━━━━━━━━━ 3/3 5.3it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219     0.0863      0.195     0.0822     0.0324     0.0872        0.2     0.0889     0.0301

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     61/150      4.79G       2.32      3.609      2.276      1.827          0        111        640: 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.2it/s 0.4s
                   all          6        219     0.0863      0.195     0.0822     0.0324     0.0872        0.2     0.0889     0.0301

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     62/150      4.79G      2.291      3.478      2.206      1.723          0        143        640: 100% ━━━━━━━━━━━━ 3/3 5.5it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219     0.0834       0.19     0.0902     0.0354     0.0959      0.165     0.0903     0.0333

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     63/150      4.79G      2.199      3.281      2.095      1.757          0         56        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219     0.0834       0.19     0.0902     0.0354     0.0959      0.165     0.0903     0.0333

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     64/150      4.79G      2.307        3.8       2.31      1.788          0        166        640: 100% ━━━━━━━━━━━━ 3/3 3.5it/s 0.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3it/s 0.8s
                   all          6        219     0.0834       0.19     0.0902     0.0354     0.0959      0.165     0.0903     0.0333

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     65/150      4.79G      2.298      3.592      2.335      1.787          0        180        640: 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          6        219      0.116      0.159     0.0857     0.0326     0.0962      0.164     0.0793     0.0284

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     66/150      4.79G      2.206      3.487      2.166       1.67          0         96        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.116      0.159     0.0857     0.0326     0.0962      0.164     0.0793     0.0284

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     67/150      4.79G      2.154      3.401      2.167      1.687          0        108        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219     0.0985      0.172     0.0839     0.0252     0.0873      0.168     0.0725     0.0223

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     68/150      4.79G      2.348      3.519      2.461       1.87          0         58        640: 100% ━━━━━━━━━━━━ 3/3 5.5it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219     0.0985      0.172     0.0839     0.0252     0.0873      0.168     0.0725     0.0223

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     69/150      4.79G      2.262      3.533      2.201      1.725          0        194        640: 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219      0.112      0.163     0.0789     0.0227     0.0983      0.147     0.0656     0.0195

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     70/150      4.79G      2.142      3.461      2.226      1.662          0        166        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219      0.112      0.163     0.0789     0.0227     0.0983      0.147     0.0656     0.0195

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     71/150      4.79G      2.398      3.766      2.342      1.806          0        260        640: 100% ━━━━━━━━━━━━ 3/3 4.4it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0s/it 1.0s
                   all          6        219      0.112      0.163     0.0789     0.0227     0.0983      0.147     0.0656     0.0195

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     72/150      4.79G      2.248      3.438      2.164      1.752          0        158        640: 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0s/it 1.0s
                   all          6        219      0.136      0.183      0.104     0.0289      0.107      0.153     0.0857     0.0268

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     73/150      4.79G      2.268      3.516      2.201      1.649          0        154        640: 100% ━━━━━━━━━━━━ 3/3 3.8it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          6        219      0.136      0.183      0.104     0.0289      0.107      0.153     0.0857     0.0268

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     74/150      4.79G      2.274      3.626       2.19      1.739          0         56        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219       0.15      0.156      0.108     0.0368      0.145      0.149      0.107     0.0366

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     75/150      4.79G      2.141       3.44      2.157      1.666          0         97        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219       0.15      0.156      0.108     0.0368      0.145      0.149      0.107     0.0366

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     76/150      4.79G      2.222      3.496      2.096       1.74          0        135        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219      0.121       0.16      0.102     0.0348      0.112      0.156     0.0972      0.033

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     77/150      4.79G      2.191      3.385      2.109      1.738          0        157        640: 100% ━━━━━━━━━━━━ 3/3 6.0it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.6s
                   all          6        219      0.121       0.16      0.102     0.0348      0.112      0.156     0.0972      0.033

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     78/150      4.79G      2.142       3.53      2.106      1.669          0        106        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.121       0.16      0.102     0.0348      0.112      0.156     0.0972      0.033

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     79/150      4.79G      2.291      3.648      2.059      1.728          0         86        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.135      0.156     0.0927     0.0316      0.129      0.149        0.1      0.032

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     80/150      4.79G      2.076      3.301      2.083      1.626          0        117        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.135      0.156     0.0927     0.0316      0.129      0.149        0.1      0.032

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     81/150      4.79G      2.297      3.833      2.327      1.761          0        227        640: 100% ━━━━━━━━━━━━ 3/3 3.7it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          6        219      0.139      0.158     0.0946      0.035      0.137      0.149     0.0917     0.0323

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     82/150      4.79G      1.983       3.21      2.003       1.56          0        118        640: 100% ━━━━━━━━━━━━ 3/3 4.5it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7s/it 1.7s
                   all          6        219      0.139      0.158     0.0946      0.035      0.137      0.149     0.0917     0.0323

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     83/150      4.79G      2.222      3.735      2.207      1.745          0        238        640: 100% ━━━━━━━━━━━━ 3/3 4.1it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.146       0.15     0.0998     0.0379      0.144      0.146     0.0986     0.0354

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     84/150      4.79G      2.263      3.477       2.17      1.695          0         89        640: 100% ━━━━━━━━━━━━ 3/3 5.9it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.146       0.15     0.0998     0.0379      0.144      0.146     0.0986     0.0354

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     85/150      4.79G      2.072      3.299      2.041       1.62          0        101        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219      0.146       0.15     0.0998     0.0379      0.144      0.146     0.0986     0.0354

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     86/150      4.79G      1.985      3.298      1.942       1.63          0         81        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219      0.328      0.174     0.0962     0.0379      0.332      0.178     0.0991     0.0364

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     87/150      4.79G      2.262      3.415      2.023      1.651          0        131        640: 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219      0.328      0.174     0.0962     0.0379      0.332      0.178     0.0991     0.0364

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     88/150      4.79G      2.129      3.476       2.06      1.731          0         92        640: 100% ━━━━━━━━━━━━ 3/3 5.1it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.347       0.17      0.101     0.0398      0.343      0.164      0.101     0.0372

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     89/150      4.79G      1.922      3.307      2.055      1.674          0         49        640: 100% ━━━━━━━━━━━━ 3/3 3.9it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0it/s 1.0s
                   all          6        219      0.347       0.17      0.101     0.0398      0.343      0.164      0.101     0.0372

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     90/150      4.79G      2.051      3.314      2.027      1.654          0        159        640: 100% ━━━━━━━━━━━━ 3/3 3.6it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          6        219       0.37       0.19      0.109     0.0413      0.354      0.172      0.106      0.038

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     91/150      4.79G      2.036      3.229      1.811      1.545          0        101        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219       0.37       0.19      0.109     0.0413      0.354      0.172      0.106      0.038

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     92/150      4.79G      1.916      3.143      1.852      1.593          0         56        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219       0.37       0.19      0.109     0.0413      0.354      0.172      0.106      0.038

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     93/150      4.79G      2.053      3.352      2.057      1.707          0         81        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.346      0.178      0.105     0.0413      0.364      0.149      0.104      0.036

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     94/150      4.79G      2.063      3.425      1.908      1.636          0        254        640: 100% ━━━━━━━━━━━━ 3/3 4.9it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219      0.346      0.178      0.105     0.0413      0.364      0.149      0.104      0.036

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     95/150      4.79G      2.215      3.618       2.04       1.63          0        170        640: 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219      0.366      0.146      0.105     0.0399      0.375      0.152      0.106     0.0352

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     96/150      4.79G      2.039       3.33      2.083      1.655          0         43        640: 100% ━━━━━━━━━━━━ 3/3 6.0it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3it/s 0.8s
                   all          6        219      0.366      0.146      0.105     0.0399      0.375      0.152      0.106     0.0352

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     97/150      4.79G      2.022      3.345      1.857      1.598          0        155        640: 100% ━━━━━━━━━━━━ 3/3 5.7it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.5it/s 0.7s
                   all          6        219      0.352      0.146      0.102     0.0397      0.363      0.153      0.106     0.0357

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     98/150      4.79G      2.168       3.34      2.131      1.527          0        130        640: 100% ━━━━━━━━━━━━ 3/3 4.1it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          6        219      0.352      0.146      0.102     0.0397      0.363      0.153      0.106     0.0357

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     99/150      5.27G      2.076       3.48      2.073      1.717          0         89        640: 100% ━━━━━━━━━━━━ 3/3 3.4it/s 0.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.4s/it 1.4s
                   all          6        219      0.352      0.146      0.102     0.0397      0.363      0.153      0.106     0.0357

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    100/150      5.27G      2.142      3.483      1.968      1.658          0        112        640: 100% ━━━━━━━━━━━━ 3/3 4.3it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.355      0.149      0.109     0.0373      0.358      0.145      0.102     0.0345

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    101/150      5.27G       2.12       3.56      1.933      1.636          0        124        640: 100% ━━━━━━━━━━━━ 3/3 5.5it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219      0.355      0.149      0.109     0.0373      0.358      0.145      0.102     0.0345

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    102/150      5.27G      2.289      3.574      2.209      1.627          0        120        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.322      0.161     0.0976     0.0387      0.374      0.131      0.112     0.0358

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    103/150      5.27G      2.035      3.322      1.956      1.642          0         88        640: 100% ━━━━━━━━━━━━ 3/3 5.6it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219      0.322      0.161     0.0976     0.0387      0.374      0.131      0.112     0.0358

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    104/150      5.27G      1.914       3.18      1.786      1.484          0        106        640: 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.1it/s 0.5s
                   all          6        219      0.339      0.138     0.0991     0.0399      0.355      0.157      0.116     0.0387

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    105/150      5.27G      2.227      3.648      2.065      1.769          0        224        640: 100% ━━━━━━━━━━━━ 3/3 4.2it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.339      0.138     0.0991     0.0399      0.355      0.157      0.116     0.0387

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    106/150      5.27G      2.017      3.293      1.994      1.616          0         38        640: 100% ━━━━━━━━━━━━ 3/3 4.4it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3it/s 0.7s
                   all          6        219      0.339      0.138     0.0991     0.0399      0.355      0.157      0.116     0.0387

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    107/150      5.27G      1.997      3.436      2.031       1.63          0        155        640: 100% ━━━━━━━━━━━━ 3/3 3.8it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.1it/s 0.9s
                   all          6        219       0.37      0.123      0.103     0.0406      0.378       0.13      0.117     0.0392

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    108/150      5.27G      2.011      3.332      2.017      1.554          0        117        640: 100% ━━━━━━━━━━━━ 3/3 3.5it/s 0.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219       0.37      0.123      0.103     0.0406      0.378       0.13      0.117     0.0392

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    109/150      5.27G      2.017      3.187      1.882      1.502          0        158        640: 100% ━━━━━━━━━━━━ 3/3 5.8it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.4it/s 0.7s
                   all          6        219      0.321      0.158      0.105     0.0426      0.326      0.167      0.121      0.043

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    110/150      5.27G      1.959      3.261      1.905       1.59          0        112        640: 100% ━━━━━━━━━━━━ 3/3 6.1it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.321      0.158      0.105     0.0426      0.326      0.167      0.121      0.043

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    111/150      5.27G      1.893      3.081      1.783      1.521          0         94        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219      0.334       0.14      0.102     0.0413      0.331      0.153      0.118     0.0385

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    112/150      5.27G      1.778      3.107      1.699      1.468          0        135        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219      0.334       0.14      0.102     0.0413      0.331      0.153      0.118     0.0385

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    113/150      5.27G      1.811      3.072      1.767      1.521          0        113        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.334       0.14      0.102     0.0413      0.331      0.153      0.118     0.0385

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    114/150      5.27G      1.923      3.168      1.767      1.475          0        103        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.359      0.151      0.106     0.0414      0.354      0.156      0.117     0.0374

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    115/150      5.27G      2.039      3.365      1.876      1.429          0        145        640: 100% ━━━━━━━━━━━━ 3/3 3.6it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0it/s 1.0s
                   all          6        219      0.359      0.151      0.106     0.0414      0.354      0.156      0.117     0.0374

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    116/150      5.27G      2.137      3.546      1.993      1.609          0        207        640: 100% ━━━━━━━━━━━━ 3/3 3.2it/s 0.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.8s
                   all          6        219      0.375      0.147      0.107     0.0428      0.357      0.167      0.115     0.0372

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    117/150      5.27G      1.921      3.263      1.792      1.537          0        162        640: 100% ━━━━━━━━━━━━ 3/3 4.7it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.375      0.147      0.107     0.0428      0.357      0.167      0.115     0.0372

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    118/150      5.27G      1.927      3.289      1.823      1.519          0        115        640: 100% ━━━━━━━━━━━━ 3/3 5.5it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.374      0.139       0.11     0.0446      0.371      0.157      0.117     0.0376

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    119/150      5.27G      1.937      3.106      1.777      1.502          0        156        640: 100% ━━━━━━━━━━━━ 3/3 5.9it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219      0.374      0.139       0.11     0.0446      0.371      0.157      0.117     0.0376

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    120/150      5.27G      1.882      3.199      1.826      1.529          0         92        640: 100% ━━━━━━━━━━━━ 3/3 5.0it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219      0.374      0.139       0.11     0.0446      0.371      0.157      0.117     0.0376

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    121/150      5.27G      1.932      3.152      1.729      1.537          0        126        640: 100% ━━━━━━━━━━━━ 3/3 5.9it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.6it/s 0.6s
                   all          6        219      0.381      0.142      0.107     0.0417       0.38      0.142      0.113     0.0367

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    122/150      5.27G      1.942      3.133      1.875      1.538          0        131        640: 100% ━━━━━━━━━━━━ 3/3 5.4it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.381      0.142      0.107     0.0417       0.38      0.142      0.113     0.0367

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    123/150      5.27G      1.818      3.124      1.752      1.474          0         89        640: 100% ━━━━━━━━━━━━ 3/3 3.8it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3it/s 0.8s
                   all          6        219      0.422      0.138      0.109     0.0426      0.397      0.146      0.113     0.0376

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    124/150      5.27G       2.12      3.533      2.017      1.629          0        202        640: 100% ━━━━━━━━━━━━ 3/3 3.7it/s 0.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.2it/s 0.9s
                   all          6        219      0.422      0.138      0.109     0.0426      0.397      0.146      0.113     0.0376

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    125/150      5.27G      1.922      3.219      1.885      1.487          0         59        640: 100% ━━━━━━━━━━━━ 3/3 4.6it/s 0.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219        0.4      0.141      0.111     0.0418      0.381      0.144      0.105     0.0362

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    126/150      5.27G      1.759      3.033      1.675      1.436          0        169        640: 100% ━━━━━━━━━━━━ 3/3 5.2it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.0it/s 0.5s
                   all          6        219        0.4      0.141      0.111     0.0418      0.381      0.144      0.105     0.0362

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    127/150      5.27G      1.821      3.028      1.642      1.482          0         73        640: 100% ━━━━━━━━━━━━ 3/3 5.0it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.8it/s 0.6s
                   all          6        219        0.4      0.141      0.111     0.0418      0.381      0.144      0.105     0.0362

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    128/150      5.27G       1.96      3.092      1.735       1.48          0        114        640: 100% ━━━━━━━━━━━━ 3/3 5.7it/s 0.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.7it/s 0.6s
                   all          6        219      0.402      0.144      0.109     0.0389      0.383      0.137      0.102     0.0337

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    129/150      5.27G      1.742      3.243      1.767      1.477          0         69        640: 100% ━━━━━━━━━━━━ 3/3 4.8it/s 0.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.9it/s 0.5s
                   all          6        219      0.402      0.144      0.109     0.0389      0.383      0.137      0.102     0.0337
EarlyStopping: Training stopped early as no improvement observed in last 20 epochs. Best results observed at epoch 109, best model saved as best.pt.
To update EarlyStopping(patience=20) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

129 epochs completed in 0.063 hours.
Optimizer stripped from /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs/yolo_ashanti_v2-3/weights/last.pt, 6.8MB
Optimizer stripped from /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs/yolo_ashanti_v2-3/weights/best.pt, 6.8MB

Validating /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs/yolo_ashanti_v2-3/weights/best.pt...
Ultralytics 8.4.120 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-seg summary (fused): 86 layers, 3,259,234 parameters, 0 gradients, 11.3 GFLOPs
WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.3it/s 0.4s
                   all          6        219      0.321      0.157      0.104     0.0426      0.327      0.167      0.119     0.0418
   Illegal Mining Site          6         66      0.152     0.0606     0.0223    0.00507       0.12     0.0455      0.018    0.00548
           Forest Land          6         50      0.229       0.26      0.206     0.0919       0.26       0.28      0.238     0.0818
            Water Body          4         37          0          0          0          0          0          0          0          0
            Urban Land          6         49      0.226      0.465      0.291      0.116      0.257       0.51       0.34      0.122
      Suspected Mining          2         17          1          0          0          0          1          0          0          0
Speed: 0.2ms preprocess, 2.6ms inference, 0.0ms loss, 4.0ms postprocess per image
Results saved to /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/yolo_runs/yolo_ashanti_v2-3
{'metrics/precision(B)': 0.32149185603254316, 'metrics/recall(B)': 0.15712109976395694, 'metrics/mAP50(B)': 0.10383255037574086, 'metrics/mAP50-95(B)': 0.04261860823153106, 'metrics/precision(M)': 0.3274314382695533, 'metrics/recall(M)': 0.16713172541743973, 'metrics/mAP50(M)': 0.11909502769860769, 'metrics/mAP50-95(M)': 0.041825256757820596, 'fitness': 0.08444386498935166}
```

```notebook-python
# ── Stage 5: train U-Net ──
# --weighting sqrt = v2 (reported result). --weighting inverse reproduces the
# original run that let Agricultural Land hijack the loss.
!python training/train_unet.py --dry-run
!python training/train_unet.py --weighting sqrt --out-name unet_best_v2.pt

RESULTS:
Would train U-Net, weighting=sqrt, save to /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/unet_best_v2.pt (tile_meta exists: True)
Fit: 15 tiles | Val: ['AS-(Fomena)', 'AS-(Bibiani)', 'AS-005']
Pixel counts per class: {'Illegal Mining Site': np.int64(561877), 'Forest Land': np.int64(4731807), 'Agricultural Land': np.int64(4450), 'Water Body': np.int64(13724), 'Urban Land': np.int64(1153026), 'Suspected Mining': np.int64(29108)}
Class weights (sqrt): {'Illegal Mining Site': np.float64(0.24925758727042546), 'Forest Land': np.float64(0.08589255134060515), 'Agricultural Land': np.float64(2.800844570341002), 'Water Body': np.float64(1.5948823785006208), 'Urban Land': np.float64(0.17400008647631363), 'Suspected Mining': np.float64(1.0951228260710335)}
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100% 156/156 [00:00<00:00, 600kB/s]

model.safetensors: downloading bytes:  92% 80.6M/87.3M [00:02<00:00, 59.2MB/s, 6.99MB/s  ]
model.safetensors: downloading bytes: 100% 80.6M/80.6M [00:02<00:00, 36.5MB/s, 7.54MB/s  ]
model.safetensors: reconstructing file: 100% 87.3M/87.3M [00:02<00:00, 39.5MB/s, 8.32MB/s  ]
Epoch 001/150: train_loss=1.6097 val_loss=1.5701
  saved best (val_loss=1.5701)
Epoch 002/150: train_loss=1.1677 val_loss=1.1737
  saved best (val_loss=1.1737)
Epoch 003/150: train_loss=0.9295 val_loss=0.8655
  saved best (val_loss=0.8655)
Epoch 004/150: train_loss=0.7787 val_loss=0.8602
  saved best (val_loss=0.8602)
Epoch 005/150: train_loss=0.6937 val_loss=0.5298
  saved best (val_loss=0.5298)
Epoch 006/150: train_loss=0.6376 val_loss=0.5194
  saved best (val_loss=0.5194)
Epoch 007/150: train_loss=0.5551 val_loss=0.4964
  saved best (val_loss=0.4964)
Epoch 008/150: train_loss=0.5949 val_loss=0.3835
  saved best (val_loss=0.3835)
Epoch 009/150: train_loss=0.5483 val_loss=0.5240
Epoch 010/150: train_loss=0.5805 val_loss=0.6690
Epoch 011/150: train_loss=0.4955 val_loss=0.5042
Epoch 012/150: train_loss=0.4898 val_loss=0.4024
Epoch 013/150: train_loss=0.4337 val_loss=0.4218
Epoch 014/150: train_loss=0.4207 val_loss=0.3816
  saved best (val_loss=0.3816)
Epoch 015/150: train_loss=0.4112 val_loss=0.4192
Epoch 016/150: train_loss=0.3725 val_loss=0.4289
Epoch 017/150: train_loss=0.3992 val_loss=0.5110
Epoch 018/150: train_loss=0.3920 val_loss=0.4426
Epoch 019/150: train_loss=0.3520 val_loss=0.4217
Epoch 020/150: train_loss=0.3606 val_loss=0.3958
Epoch 021/150: train_loss=0.3827 val_loss=0.3982
Epoch 022/150: train_loss=0.3043 val_loss=0.4270
Epoch 023/150: train_loss=0.3259 val_loss=0.4961
Epoch 024/150: train_loss=0.2990 val_loss=0.3823
Epoch 025/150: train_loss=0.3300 val_loss=0.6059
Epoch 026/150: train_loss=0.2957 val_loss=0.4837
Epoch 027/150: train_loss=0.2712 val_loss=0.4765
Epoch 028/150: train_loss=0.2975 val_loss=0.5773
Epoch 029/150: train_loss=0.2746 val_loss=0.5106
Early stopping at epoch 29

Best val loss: 0.3816 | Saved: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/unet_best_v2.pt
```

```notebook-python
# ── Stage 6a: evaluate — full-tile inference (primary Ch4 numbers) ──
!python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best.pt --dry-run
!python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best.pt

RESULTS:
Would evaluate checkpoint: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/unet_best.pt (exists: True)
AS-(Sanso): 317257 labeled pixels evaluated
AS-(Dokyiwa): 382335 labeled pixels evaluated
AS-(Anwia-Nkwanta): 408654 labeled pixels evaluated
AS-(Jacobu): 56442 labeled pixels evaluated
AS-007: 319370 labeled pixels evaluated
AS-(Fomena)-1: 647642 labeled pixels evaluated

Total labeled pixels: 2131700

                     precision    recall  f1-score   support

Illegal Mining Site       0.52      0.73      0.61     66535
        Forest Land       0.98      0.81      0.89   1528371
  Agricultural Land       0.00      0.00      0.00         0
         Water Body       0.03      0.02      0.03     10185
         Urban Land       0.84      0.89      0.86    521658
   Suspected Mining       0.00      0.00      0.00      4951

           accuracy                           0.82   2131700
          macro avg       0.40      0.41      0.40   2131700
       weighted avg       0.93      0.82      0.87   2131700

Per-class IoU:
  Illegal Mining Site: 0.4370
  Forest Land: 0.8005
  Agricultural Land: 0.0000
  Water Body: 0.0129
  Urban Land: 0.7577
  Suspected Mining: 0.0000

Overall Accuracy: 0.8220
Cohen's Kappa: 0.6518

Confusion matrix (rows=true, cols=predicted):
['Illegal Mining Site', 'Forest Land', 'Agricultural Land', 'Water Body', 'Urban Land', 'Suspected Mining']
[[  48421    4941    9195     323    2804     851]
 [  32403 1239443  154042    5614   86271   10598]
 [      0       0       0       0       0       0]
 [   1031    1821    5245     224    1598     266]
 [   6819   12487   34778    1080  464103    2391]
 [   4006     664       0     126     155       0]]

---

# ── Stage 6a: evaluate — full-tile inference (primary Ch4 numbers) ──
!python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best_v2.pt --dry-run
!python evaluation/evaluate_unet_fulltile.py --checkpoint unet_best_v2.pt

RESULTS:
Would evaluate checkpoint: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/unet_best_v2.pt (exists: True)
AS-(Sanso): 317257 labeled pixels evaluated
AS-(Dokyiwa): 382335 labeled pixels evaluated
AS-(Anwia-Nkwanta): 408654 labeled pixels evaluated
AS-(Jacobu): 56442 labeled pixels evaluated
AS-007: 319370 labeled pixels evaluated
AS-(Fomena)-1: 647642 labeled pixels evaluated

Total labeled pixels: 2131700

                     precision    recall  f1-score   support

Illegal Mining Site       0.55      0.66      0.60     66535
        Forest Land       0.99      0.91      0.95   1528371
  Agricultural Land       0.00      0.00      0.00         0
         Water Body       0.33      0.37      0.35     10185
         Urban Land       0.84      0.97      0.90    521658
   Suspected Mining       0.03      0.20      0.05      4951

           accuracy                           0.91   2131700
          macro avg       0.46      0.52      0.47   2131700
       weighted avg       0.93      0.91      0.92   2131700

Per-class IoU:
  Illegal Mining Site: 0.4262
  Forest Land: 0.8978
  Agricultural Land: 0.0000
  Water Body: 0.2121
  Urban Land: 0.8205
  Suspected Mining: 0.0260

Overall Accuracy: 0.9111
Cohen's Kappa: 0.8055

Confusion matrix (rows=true, cols=predicted):
['Illegal Mining Site', 'Forest Land', 'Agricultural Land', 'Water Body', 'Urban Land', 'Suspected Mining']
[[  44013    5368       0    1052    4785   11317]
 [  28077 1388311       0    6555   83976   21452]
 [      0       0       0       0       0       0]
 [   1078     294       0    3790    4687     336]
 [   4857   11572       0      49  505138      42]
 [   2717     672       0      29     544     989]]
```

```notebook-python
# ── Stage 6b: evaluate — patch-grid inference (robustness cross-check) ──
!python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best.pt --dry-run
!python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best.pt

RESULTS:
Would evaluate checkpoint: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/unet_best.pt (exists: True)
AS-(Anwia-Nkwanta): 408,654 labeled pixels evaluated
AS-(Dokyiwa): 382,335 labeled pixels evaluated
AS-(Fomena)-1: 647,642 labeled pixels evaluated
AS-(Jacobu): 56,442 labeled pixels evaluated
AS-(Sanso): 317,257 labeled pixels evaluated
AS-007: 319,370 labeled pixels evaluated

Total labeled pixels: 2,131,700

                     precision    recall  f1-score   support

Illegal Mining Site       0.51      0.74      0.61     66535
        Forest Land       0.98      0.76      0.86   1528371
  Agricultural Land       0.00      0.00      0.00         0
         Water Body       0.19      0.29      0.23     10185
         Urban Land       0.83      0.89      0.86    521658
   Suspected Mining       0.03      0.11      0.05      4951

           accuracy                           0.79   2131700
          macro avg       0.43      0.46      0.43   2131700
       weighted avg       0.93      0.79      0.84   2131700

Per-class IoU / F1:
  Illegal Mining Site   : IoU=0.4344  F1=0.6057
  Forest Land           : IoU=0.7480  F1=0.8558
  Agricultural Land     : IoU=0.0000  F1=0.0000
  Water Body            : IoU=0.1282  F1=0.2273
  Urban Land            : IoU=0.7557  F1=0.8609
  Suspected Mining      : IoU=0.0267  F1=0.0520

Overall Accuracy: 0.7856
Cohen's Kappa: 0.6018
Mean IoU (present test classes): 0.4186

Confusion matrix (rows=true, cols=predicted):
['Illegal Mining Site', 'Forest Land', 'Agricultural Land', 'Water Body', 'Urban Land', 'Suspected Mining']
[[  48929    4747    6669     952    3987    1251]
 [  33785 1158058  227400   10529   87054   11545]
 [      0       0       0       0       0       0]
 [   1451    3375    1555    2914     890       0]
 [   7753   11267   34564    1050  464260    2764]
 [   3119     524       0       6     754     548]]
---

# ── Stage 6b: evaluate — patch-grid inference (robustness cross-check) ──
!python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best_v2.pt --dry-run
!python evaluation/evaluate_unet_patchgrid.py --checkpoint unet_best_v2.pt

RESULTS:
Would evaluate checkpoint: /content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training/checkpoints/unet_best_v2.pt (exists: True)
AS-(Anwia-Nkwanta): 408,654 labeled pixels evaluated
AS-(Dokyiwa): 382,335 labeled pixels evaluated
AS-(Fomena)-1: 647,642 labeled pixels evaluated
AS-(Jacobu): 56,442 labeled pixels evaluated
AS-(Sanso): 317,257 labeled pixels evaluated
AS-007: 319,370 labeled pixels evaluated

Total labeled pixels: 2,131,700

                     precision    recall  f1-score   support

Illegal Mining Site       0.51      0.66      0.57     66535
        Forest Land       0.98      0.91      0.95   1528371
  Agricultural Land       0.00      0.00      0.00         0
         Water Body       0.36      0.55      0.44     10185
         Urban Land       0.85      0.96      0.90    521658
   Suspected Mining       0.00      0.02      0.01      4951

           accuracy                           0.91   2131700
          macro avg       0.45      0.52      0.48   2131700
       weighted avg       0.93      0.91      0.92   2131700

Per-class IoU / F1:
  Illegal Mining Site   : IoU=0.4033  F1=0.5748
  Forest Land           : IoU=0.8977  F1=0.9461
  Agricultural Land     : IoU=0.0000  F1=0.0000
  Water Body            : IoU=0.2796  F1=0.4370
  Urban Land            : IoU=0.8249  F1=0.9040
  Suspected Mining      : IoU=0.0038  F1=0.0076

Overall Accuracy: 0.9121
Cohen's Kappa: 0.8068
Mean IoU (present test classes): 0.4818

Confusion matrix (rows=true, cols=predicted):
['Illegal Mining Site', 'Forest Land', 'Agricultural Land', 'Water Body', 'Urban Land', 'Suspected Mining']
[[  43999    7985       0    1529    3234    9788]
 [  31835 1392851       0    8106   80366   15213]
 [      0       0       0       0       0       0]
 [   1052    1080       0    5594    2442      17]
 [   5947   13876       0      21  501792      22]
 [   3738     296       0     168     634     115]]
```
---
