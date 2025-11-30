# ecosystem_mapper

EPFL Deep learning project for mapping Swiss ecosystems by fusing aerial imagery with SWECO25 environmental variables and evaluating how each data source affects EUNIS habitat prediction accuracy.

## Repository layout
- `code/` – main directory for all Python scripts.
- `code/dataset` – Python scripts for the dataset.
- `code/dataset/eunis_labels.py` — label lookup table mapping EUNIS class IDs to human-readable names used across notebooks and training code.
- `code/dataset/sweco_group_of_variables.py` — thematic groupings of SWECO25 variables.
- `data/` — placeholder for processed datasets and splits (kept empty via `.gitkeep`).
- `hpc_scripts/` — queue or batch scripts for running experiments on shared compute (placeholder).
- `models/` — trained checkpoints, evaluation outputs, and experiment metadata (placeholder).
- `sources/project_description.pdf` — original course brief describing objectives, data sources, and expected analyses.

## Project brief
- Main goal: predict 17 EUNIS ecosystem categories for 16,925 Swiss locations using 100×100 m RGB aerial tiles and 48 standardised SWECO25 variables.
- Goals: build and compare image-only, tabular-only, and fused models; run thematic ablations to assess variable importance; report metrics on the held-out geographic test split; visualise and interpret key findings.
- Data references: swissIMAGE orthophotos, SWECO25 environmental rasters, EUNIS habitat catalogue (see `sources/project_description.pdf`).

## Dataloaders (PyTorch)
- **Dataset**: `code/dataset/ecosystem_dataset.py` provides `EcosystemDataset` and `get_dataloaders`.
- **CSV input**: expects `data/dataset_split.csv` with columns `id,x,y,split,EUNIS_cls,EUNIS_label` plus variable columns (e.g., `bioclim_*`, `lulc_*`, `edaph_*`, etc.).
- **Images**: optional; looked up as `<image_dir>/<id><image_ext>` (default `image_ext=".png"`).
- **Variables**: selectable by group key from `sweco_group_of_variables.py`, explicit list, `'all'`, or `None`.

### Quick examples
- Images + all variables
```
python
from code.dataset.ecosystem_dataset import get_dataloaders

loaders = get_dataloaders(
	csv_path="data/dataset_split.csv",
	image_dir="/path/to/tiles",
	image_ext=".png",
	variable_selection="all",
	batch_size=32,
	num_workers=4,
	load_images=True,
)
batch = next(iter(loaders["train"]))
images, variables, labels = batch["images"], batch["variables"], batch["labels"]
```

- Images + selected group (e.g., `bioclim`)
```
python
loaders = get_dataloaders(
	csv_path="data/dataset_split.csv",
	image_dir="/path/to/tiles",
	variable_selection="bioclim",
)
```

- Variables-only (no images)
```
python
from torch.utils.data import DataLoader
from code.dataset.ecosystem_dataset import EcosystemDataset, default_collate

ds = EcosystemDataset(
	csv_path="data/dataset_split.csv",
	split="train",
	load_images=False,
	variable_selection=["bioclim", "edaph"],  # combine groups or explicit columns
)
loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=default_collate)
batch = next(iter(loader))
variables_only = batch["variables"]
labels = batch["labels"]
```

- Images-only (no variables)
```
python
from torch.utils.data import DataLoader
from code.dataset.ecosystem_dataset import EcosystemDataset

ds = EcosystemDataset(
	csv_path="data/dataset_split.csv",
	split="train",
	image_dir="/path/to/tiles",
	load_images=True,
	variable_selection=None,
)
loader = DataLoader(ds, batch_size=32, shuffle=True)
batch = next(iter(loader))
images_only = batch["images"]
```

### Minimal ResNet training loop (image-only)
```
python
import torch
import torch.nn as nn
import torchvision.models as models
from code.dataset.ecosystem_dataset import get_dataloaders

num_classes = 17
loaders = get_dataloaders(
	csv_path="data/dataset_split.csv",
	image_dir="/path/to/tiles",
	variable_selection=None,  # images only
	batch_size=32,
)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(2):
	model.train()
	for batch in loaders["train"]:
		imgs = batch["images"]
		labels = batch["labels"]
		if imgs is None:  # guard if images missing
			continue
		imgs = imgs.to(device)
		labels = labels.to(device)
		logits = model(imgs)
		loss = loss_fn(logits, labels)
		opt.zero_grad()
		loss.backward()
		opt.step()
```

### Notes
- If an image file is missing, the dataset returns `None` for that sample's image; the default collate pads stacks but your training loop should check for `None`.
- Variable columns are inferred by excluding `id,x,y,split,EUNIS_cls,EUNIS_label` from the CSV.
- Replace `/path/to/tiles` and image extension to match your storage.