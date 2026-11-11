"""
STAGE 4 — Train YOLOv8-seg (RGB, transfer learning from COCO).

v1 used freeze=10, mosaic=0.5 (weak result — domain gap too large for frozen layers).
v2 used freeze=0, mosaic=0 (reported result — full fine-tune, no mosaic on 18 tiles).

Run from gls_model/:
    !python training/train_yolo.py --dry-run
    !python training/train_yolo.py --name yolo_ashanti_v2 --freeze 0 --mosaic 0
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_DATA_YAML, YOLO_RUNS_DIR


def run(args):
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs, patience=args.patience,
        imgsz=640, batch=8,
        augment=True, mosaic=args.mosaic, mixup=0.1,
        freeze=args.freeze,
        project=args.runs_dir, name=args.name,
    )
    print(results.results_dict)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-yaml", default=YOLO_DATA_YAML)
    p.add_argument("--runs-dir", default=YOLO_RUNS_DIR)
    p.add_argument("--name", default="yolo_ashanti_v2")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--freeze", type=int, default=0)
    p.add_argument("--mosaic", type=float, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print(f"Would train with data={args.data_yaml}, freeze={args.freeze}, mosaic={args.mosaic}, "
              f"name={args.name} (data.yaml exists: {os.path.exists(args.data_yaml)})")
    else:
        run(args)
