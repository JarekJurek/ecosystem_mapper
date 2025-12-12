import os
from typing import List, Optional, Sequence, Dict, Any, Union, cast
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.io import imread

from .sweco_group_of_variables import sweco_variables_dict


class EcosystemDataset(Dataset):
    """
    PyTorch Dataset to load images and associated tabular variables from a single CSV
    (default: data/dataset_split.csv). Supports:
    - Selecting variable groups by name (keys in sweco_variables_dict)
    - Selecting an explicit list of variable column names
    - Selecting 'all' variables (all known variable columns in the CSV)
    - Selecting no variables (image-only)
    - Variables-only mode without images (for ablation)

    Assumes the CSV has columns: 'id', 'split', 'EUNIS_cls', 'EUNIS_label' and many variable columns.
    Images are optional and, if used, are located by `image_dir` + f"{id}{image_ext}".
    """

    def __init__(
        self,
        csv_path: str = "data/dataset_split.csv",
        image_dir: Optional[str] = None,
        image_ext: str = ".tif",
        load_images: bool = True,
        variable_selection: Optional[Union[str, Sequence[str]]] = "all",
        image_transform: Optional[Any] = None,
        return_label: bool = True,
        subset: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.load_images = load_images
        self.image_transform = image_transform
        self.return_label = return_label

        df = pd.read_csv(csv_path)
        # Filter by subset (train/val/test) if provided; split is defined inside CSV
        if subset:
            df = df[df["split"] == subset]
        df = df.reset_index(drop=True)
        self.df = df

        # Determine which variable columns are present in CSV
        # Exclude non-variable metadata columns
        meta_cols = {"id", "x", "y", "split", "EUNIS_cls", "EUNIS_label"}
        all_var_cols = [c for c in df.columns if c not in meta_cols]

        # Resolve variable selection
        self.var_cols: List[str] = []
        if variable_selection is None:
            self.var_cols = []  # none
        elif isinstance(variable_selection, str):
            if variable_selection.lower() == "all":
                self.var_cols = all_var_cols
            elif variable_selection in sweco_variables_dict:
                wanted = sweco_variables_dict[variable_selection]
                # keep only those present in CSV
                self.var_cols = [c for c in wanted if c in df.columns]
            else:
                # single column name
                self.var_cols = (
                    [variable_selection] if variable_selection in df.columns else []
                )
        else:
            # Sequence of columns or group names
            # Support special cases like ["all"] and allow mixed group names/columns.
            seq = list(variable_selection)
            # If 'all' appears anywhere in the sequence, select all variable columns
            if any(isinstance(it, str) and it.lower() == "all" for it in seq):
                self.var_cols = all_var_cols
            else:
                cols: List[str] = []
                for item in seq:
                    if isinstance(item, str) and item in sweco_variables_dict:
                        cols.extend(sweco_variables_dict[item])
                    else:
                        # treat as a direct column name
                        cols.append(cast(str, item))
                # deduplicate and keep only present columns
                dedup: List[str] = []
                seen = set()
                for c in cols:
                    if c in df.columns and c not in seen:
                        dedup.append(c)
                        seen.add(c)
                self.var_cols = dedup

        # Variables-only check
        if not self.load_images:
            # images disabled entirely
            self.image_dir = None

        # Sanity: if images requested, image_dir must be provided
        if self.load_images and not self.image_dir:
            raise ValueError("image_dir must be provided when load_images=True")


    @classmethod
    def from_split(
        cls,
        subset: str,
        csv_path: str = "data/dataset_split.csv",
        image_dir: Optional[str] = None,
        image_ext: str = ".tif",
        load_images: bool = True,
        variable_selection: Optional[Union[str, Sequence[str]]] = "all",
        image_transform: Optional[Any] = None,
        return_label: bool = True,
    ) -> "EcosystemDataset":
        """Factory to build a dataset filtered by a CSV-defined split."""
        return cls(
            csv_path=csv_path,
            image_dir=image_dir,
            image_ext=image_ext,
            load_images=load_images,
            variable_selection=variable_selection,
            image_transform=image_transform,
            return_label=return_label,
            subset=subset,
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, sample_id: str) -> Optional[torch.Tensor]:
        if not self.load_images:
            return None
        dir_path = cast(str, self.image_dir)
        img_path = os.path.join(dir_path, f"{sample_id}{self.image_ext}")
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return None
        try:
            img = imread(img_path)
            if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim < 2:
                raise ValueError("Invalid or empty image array")
            if np.all(np.isnan(img)):
                raise ValueError("Image contains only NaNs")
            if np.nanmax(img) == 0:
                raise ValueError("Image is all zeros")

            if np.issubdtype(img.dtype, np.floating) and img.max() > 1.0:
                img = img / 255.0


            if self.image_transform is not None:
                tensor = self.image_transform(img)
            else:
                tensor = torch.from_numpy(np.array(img)).to(torch.float32)

            return tensor

        except Exception as exc:
            print(f"Failed to read/prepare image {img_path}: {exc}")
            return None

    def _load_variables(self, row: pd.Series) -> Optional[torch.Tensor]:
        if not self.var_cols:
            return None
        vals = row[self.var_cols].astype(float).values
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample_id = str(row["id"])  # id used to find image

        item: Dict[str, Any] = {
            "id": sample_id,
        }
        img = self._load_image(sample_id)
        if img is not None:
            item["image"] = img
        else:
            print(f"Image not loaded for id: {sample_id}")

        vars_t = self._load_variables(row)
        if vars_t is not None:
            item["variables"] = vars_t
        else:
            print(f"Variables not loaded for id: {sample_id}")

        if self.return_label:
            item["label"] = int(row["EUNIS_cls"])  # integer class for training
            item["label_name"] = str(row["EUNIS_label"])  # human-readable

        return item


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that supports optional images/variables.
    - Stacks available tensors; fills None where missing.
    - Keeps ids and label_names as lists.
    """
    ids = [b["id"] for b in batch]
    images = [b.get("image") for b in batch]
    if any(img is not None for img in images):
        ref_img = next(img for img in images if img is not None)
        padded_imgs = [
            img if img is not None else torch.zeros_like(ref_img)
            for img in images
        ]
        images_t = torch.stack(padded_imgs)
    else:
        images_t = None

    variables = [b.get("variables") for b in batch]
    if any(v is not None for v in variables):
        # Pad missing variable tensors to same length with zeros
        # Determine max dim
        max_d = max((v.shape[0] for v in variables if v is not None), default=0)
        padded = []
        for v in variables:
            if v is None:
                padded.append(torch.zeros(max_d, dtype=torch.float32))
            else:
                if v.shape[0] < max_d:
                    pad = torch.zeros(max_d - v.shape[0], dtype=torch.float32)
                    padded.append(torch.cat([v, pad], dim=0))
                else:
                    padded.append(v)
        variables_t = torch.stack(padded)
    else:
        variables_t = None

    labels = [b.get("label") for b in batch]
    if any(l is not None for l in labels):
        labels_t = torch.tensor(
            [l if l is not None else -1 for l in labels], dtype=torch.long
        )
    else:
        labels_t = None

    label_names = [b.get("label_name") for b in batch]

    out: Dict[str, Any] = {
        "ids": ids,
        "images": images_t,
        "variables": variables_t,
        "labels": labels_t,
        "label_names": label_names,
    }
    return out


def get_dataloaders(
    csv_path: Path = Path("data/dataset_split.csv"),
    image_dir: Optional[Path] = None,
    image_ext: str = ".tif",
    variable_selection: Optional[Union[str, Sequence[str]]] = "all",
    batch_size: int = 32,
    num_workers: int = 4,
    train_image_transform: Optional[Any] = None,
    eval_image_transform: Optional[Any] = None,
    load_images: bool = True,
    return_label: bool = True,
    collate_fn=default_collate,
) -> Dict[str, DataLoader]:
    """
    Convenience function returning dataloaders for train/val/test splits.
    """
    loaders: Dict[str, DataLoader] = {}
    # Ensure types align with EcosystemDataset (expects str paths)
    csv_str = str(csv_path)
    img_str = str(image_dir) if image_dir is not None else None
    for split in ["train", "val", "test"]:
        if split == "train":
            tfm = train_image_transform
        else:
            tfm = eval_image_transform

        ds = EcosystemDataset.from_split(
            subset=split,
            csv_path=csv_str,
            image_dir=img_str,
            image_ext=image_ext,
            load_images=load_images,
            variable_selection=variable_selection,
            image_transform=tfm,
            return_label=return_label,
        )
        # Print counts per split: number of images and variables
        var_count = len(ds.var_cols)
        img_count = 0
        if load_images and img_str is not None:
            try:
                for i in range(len(ds)):
                    sid = str(ds.df.iloc[i]["id"]) if "id" in ds.df.columns else str(i)
                    img_path = os.path.join(img_str, f"{sid}{image_ext}")
                    if os.path.exists(img_path):
                        img_count += 1
            except Exception:
                pass
        print(f"Split '{split}': images={img_count}, variables={var_count}, samples={len(ds)}")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if num_workers > 0 else None,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=True,
        )
    return loaders
