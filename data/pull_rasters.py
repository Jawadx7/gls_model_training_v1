"""
STAGE 1b — Pull real Sentinel-2 imagery per tile from CDSE via Sentinel Hub.

Requires SH_CLIENT_ID / SH_CLIENT_SECRET set as environment variables
BEFORE running this (set them in the notebook from Colab Secrets — never
hardcode or save them to a file).

Skips any tile whose raster already exists on disk — safe to rerun after a
disconnect without re-pulling everything.

Run from gls_model/:
    !python data/pull_rasters.py --dry-run
    !python data/pull_rasters.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from sentinelhub import SHConfig, SentinelHubRequest, BBox, CRS, bbox_to_dimensions, MimeType, DataCollection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TILE_META_PATH, RASTERS_DIR, RASTERS_RGB_DIR

SINGLE_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: ["B02","B03","B04","B08","B11","B12"], output: { bands: 6, sampleType: "FLOAT32" } };
}
function evaluatePixel(s) { return [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12]; }
"""

COMPOSITE_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B08","B11","B12","SCL","dataMask"] }],
    output: { bands: 6, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  let sum = [0,0,0,0,0,0], count = 0;
  for (let i = 0; i < samples.length; i++) {
    let s = samples[i];
    let cloudy = (s.SCL == 3 || s.SCL == 8 || s.SCL == 9 || s.SCL == 10);
    if (s.dataMask == 1 && !cloudy) {
      sum[0]+=s.B02; sum[1]+=s.B03; sum[2]+=s.B04; sum[3]+=s.B08; sum[4]+=s.B11; sum[5]+=s.B12;
      count += 1;
    }
  }
  if (count == 0) return [0,0,0,0,0,0];
  return sum.map(v => v / count);
}
"""


def get_config():
    config = SHConfig()
    config.sh_client_id = os.environ["SH_CLIENT_ID"]
    config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    return config


def pull_tile(row, config, cdse_s2l2a):
    bbox = BBox(bbox=[row.min_lon, row.min_lat, row.max_lon, row.max_lat], crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=10)

    if row["mode"] == "single":
        date = str(row.date_single)[:10]
        time_interval, evalscript = (date, date), SINGLE_EVALSCRIPT
    else:
        time_interval = (str(row.date_from)[:10], str(row.date_to)[:10])
        evalscript = COMPOSITE_EVALSCRIPT

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(data_collection=cdse_s2l2a, time_interval=time_interval)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox, size=size, config=config,
    )
    arr = request.get_data()[0]
    return arr, bbox, size


def run(tile_meta_path, rasters_dir, rasters_rgb_dir):
    tile_meta = pd.read_csv(tile_meta_path)
    os.makedirs(rasters_dir, exist_ok=True)
    os.makedirs(rasters_rgb_dir, exist_ok=True)

    config = get_config()
    cdse_s2l2a = DataCollection.SENTINEL2_L2A.define_from("cdse_s2l2a", service_url=config.sh_base_url)

    for _, row in tile_meta.iterrows():
        out_path = f"{rasters_dir}/{row.tile_name}.tif"
        if os.path.exists(out_path):
            print(f"[skip] {row.tile_name}: already exists")
            continue

        print(f"Pulling {row.tile_name} ({row['mode']})...", end=" ")
        try:
            arr, bbox, size = pull_tile(row, config, cdse_s2l2a)
            transform = from_bounds(row.min_lon, row.min_lat, row.max_lon, row.max_lat, size[0], size[1])

            with rasterio.open(out_path, "w", driver="GTiff", height=size[1], width=size[0],
                                count=6, dtype="float32", crs="EPSG:4326", transform=transform) as dst:
                for b in range(6):
                    dst.write(arr[:, :, b], b + 1)

            rgb = arr[:, :, [2, 1, 0]]
            rgb_norm = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
            Image.fromarray((rgb_norm * 255).astype("uint8")).save(f"{rasters_rgb_dir}/{row.tile_name}.png")
            print(f"OK — {size[0]}x{size[1]}px")
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tile-meta", default=TILE_META_PATH)
    p.add_argument("--rasters-dir", default=RASTERS_DIR)
    p.add_argument("--rasters-rgb-dir", default=RASTERS_RGB_DIR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        creds_ok = "SH_CLIENT_ID" in os.environ and "SH_CLIENT_SECRET" in os.environ
        print(f"Would read tile_meta: {args.tile_meta} (exists: {os.path.exists(args.tile_meta)})")
        print(f"Would write to: {args.rasters_dir}, {args.rasters_rgb_dir}")
        print(f"Credentials set: {creds_ok}")
    else:
        run(args.tile_meta, args.rasters_dir, args.rasters_rgb_dir)
