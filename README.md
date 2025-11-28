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