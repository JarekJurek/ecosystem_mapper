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

## Run tests

```bash
python3 -m unittest src.tests.dataset_test -v
```

## Config

Configure by creating a new config file in `src/configs`, see `defaults.yaml` for reference. Run it with `python3 src/main.py --config src/configs/<your_config>.yaml`
