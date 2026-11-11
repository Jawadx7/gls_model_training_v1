"""
Shared constants for the GhanaLandScan model training pipeline.
Every other script imports from here — never redefine these values locally.
"""

# ── Data location (all generated files, NOT the code) ──
DATA_DIR = "/content/drive/MyDrive/04_Projects/GhanaLandScan/GLS_Model_Training"

RASTERS_DIR = f"{DATA_DIR}/rasters"
RASTERS_RGB_DIR = f"{DATA_DIR}/rasters_rgb"
LABELS_YOLO_DIR = f"{DATA_DIR}/labels_yolo"
MASKS_UNET_DIR = f"{DATA_DIR}/masks_unet"
DATASET_DIR = f"{DATA_DIR}/dataset"
CHECKPOINTS_DIR = f"{DATA_DIR}/checkpoints"
YOLO_RUNS_DIR = f"{CHECKPOINTS_DIR}/yolo_runs"
TILE_META_PATH = f"{DATA_DIR}/tile_meta.csv"
ANNOTATIONS_EXPORT_PATH = f"{DATA_DIR}/annotations_export.geojson"

YOLO_DATA_YAML = f"{DATASET_DIR}/yolo/data.yaml"

# ── Class schema ──
CLASS_TO_ID = {
    "Illegal Mining Site": 0,
    "Forest Land": 1,
    "Agricultural Land": 2,
    "Water Body": 3,
    "Urban Land": 4,
    "Suspected Mining": 5,
}
CLASS_NAMES = list(CLASS_TO_ID.keys())
NUM_CLASSES = len(CLASS_TO_ID)

# ── U-Net specifics ──
IGNORE_ID = 255          # sparse-annotation background, excluded from loss/eval
PATCH_SIZE = 256

# ── Fixed train/test split (agreed for class coverage + composite/region reasons — do not re-split) ──
TEST_NAMES = {"AS-(Jacobu)", "AS-007", "AS-(Sanso)", "AS-(Fomena)-1",
              "AS-(Dokyiwa)", "AS-(Anwia-Nkwanta)"}
