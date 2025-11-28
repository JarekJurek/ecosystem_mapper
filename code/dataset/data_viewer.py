#!/usr/bin/env python3
"""Simple viewer for .tif tiles with their EUNIS labels."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


MAIN_DIR = Path(__file__).parents[2].resolve()
CSV_PATH = MAIN_DIR / "data" / "dataset_split.csv"
IMAGE_DIR = MAIN_DIR / "data"
SPLIT: Optional[str] = None  # set to "train", "val", or "test" to filter
LIMIT: Optional[int] = None  # set to an integer to cap how many tiles to preview
SHUFFLE = False  # set True to randomise order


@dataclass
class Record:
    tile_id: str
    split: str
    label: str
    path: Path


def read_records(csv_path: Path, image_dir: Path) -> List[Record]:
    records: List[Record] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tile_id = row["id"].strip()
                split = row.get("split", "").strip()
                label = row.get("EUNIS_label", "").strip()
                image_path = image_dir / f"{tile_id}.tif"
                if image_path.exists():
                    records.append(Record(tile_id=tile_id, split=split, label=label, path=image_path))
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
    return records


def filter_records(records: Iterable[Record], split: Optional[str]) -> List[Record]:
    if not split:
        return list(records)
    wanted = split.lower()
    return [rec for rec in records if rec.split.lower() == wanted]


def load_image(path: Path) -> np.ndarray:
    img = imread(path)
    # Handle channel-first images (common in remote sensing)
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        img = np.moveaxis(img, 0, -1)

    # Use only the first 3 channels for RGB
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    # Normalize float data to [0, 1] if it exceeds 1.0 (likely 0-255 range)
    if np.issubdtype(img.dtype, np.floating) and img.max() > 1.0:
        img = img / 255.0

    return img


def show(record: Record) -> bool:
    try:
        img = load_image(record.path)
    except Exception as e:
        print(f"Error loading {record.path}: {e}")
        return False

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(f"{record.path.name} \u2022 {record.label} ({record.split})")
    ax.axis("off")
    stop = {"value": False}

    def on_key(event):
        if event.key in {"q", "escape"}:
            stop["value"] = True
            plt.close(fig)
        elif event.key == " ":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return stop["value"]


def main() -> None:
    records = read_records(CSV_PATH, IMAGE_DIR)
    records = filter_records(records, SPLIT)
    if not records:
        print("No matching tiles found.")
        return

    if SHUFFLE:
        random.shuffle(records)

    subset = records[:LIMIT] if LIMIT else records

    print("Press 'space' for next image, 'q' / 'escape' to quit.")
    for record in subset:
        should_stop = show(record)
        if should_stop:
            break


if __name__ == "__main__":
    main()
