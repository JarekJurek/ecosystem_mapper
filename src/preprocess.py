import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from tqdm import tqdm


def tif_to_rgb_image(path: Path) -> Image.Image:
    """Load a (200, 200, 3) TIFF and convert to uint8 RGB PIL.Image."""
    arr = tiff.imread(str(path))  # shape (H, W, 3), float or int

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected TIFF shape {arr.shape} in {path}")

    # Normalize to [0, 255] uint8 (same logic as before, but simpler)
    arr = arr.astype(np.float32)
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)  # RGB


def preprocess_tiffs(csv_path: Path, tif_dir: Path, out_dir: Path, resize_to: int = 256):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    ids = df["id"].astype(str).unique()

    print(f"Found {len(ids)} unique IDs. Converting to PNG in {out_dir} ...")

    skipped = 0
    for sample_id in tqdm(ids, desc="Preprocessing TIFFs"):
        tif_path = tif_dir / f"{sample_id}.tif"
        png_path = out_dir / f"{sample_id}.png"

        if png_path.exists():
            continue

        if not tif_path.exists():
            print(f"[WARN] TIFF not found: {tif_path}")
            skipped += 1
            continue

        try:
            img = tif_to_rgb_image(tif_path)
        except Exception as e:
            print(f"[ERROR] Failed to read/convert {tif_path}: {e}")
            skipped += 1
            continue

        if resize_to is not None and resize_to > 0:
            img = img.resize((resize_to, resize_to), Image.BILINEAR)

        try:
            img.save(png_path)
        except Exception as e:
            print(f"[ERROR] Failed to save {png_path}: {e}")
            skipped += 1
            continue

    print(f"Done. Skipped {skipped} files (missing/corrupt).")


def main():
    parser = argparse.ArgumentParser(description="Preprocess 200x200x3 TIFFs to RGB PNGs.")
    parser.add_argument(
        "--csv-path", type=str, default="data/dataset_split.csv",
        help="Path to dataset_split.csv",
    )
    parser.add_argument(
        "--tif-dir", type=str, default="data/data",
        help="Directory containing original .tif files",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/preprocessed_png_256",
        help="Directory to save preprocessed PNGs",
    )
    parser.add_argument(
        "--resize", type=int, default=256,
        help="Resize images to (resize x resize). Set <=0 to keep original 200x200.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    tif_dir = Path(args.tif_dir)
    out_dir = Path(args.out_dir)
    resize_to = args.resize if args.resize and args.resize > 0 else None

    preprocess_tiffs(csv_path, tif_dir, out_dir, resize_to)


if __name__ == "__main__":
    main()
