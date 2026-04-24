# Rugby Union ML Dissertation Project

**COMP3931/3932 — University of Leeds**  
**Author:** Cian Woodsy

## Overview

Machine learning analysis of Rugby Union player performance data across two seasons (2022/23 and 2023/24), covering Backs and Forwards separately.

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter lab
```

## Project Structure

```
Project/
├── data/raw/          ← CSV copies from Misc/ (read-only)
├── data/processed/    ← Cleaned, feature-engineered data
├── notebooks/         ← Jupyter notebooks for EDA & experiments
├── src/               ← Reusable Python modules
├── models/            ← Saved trained models
├── results/           ← Evaluation outputs and plots
└── reports/           ← Planning docs and dissertation drafts
```

## Notebooks (run in order)

1. `notebooks/01_data_exploration.ipynb` — EDA, distributions, correlation heatmaps
2. `notebooks/02_position_classification.ipynb` — Back vs. Forward classifier + SHAP
3. `notebooks/03_performance_prediction.ipynb` — Next-game regression models
4. `notebooks/04_player_clustering.ipynb` — Unsupervised archetype discovery

## Reproducing the dissertation experiments

From the project root:

```bash
# Reproduce every experiment reported in the dissertation (~1-2 minutes)
python main.py --task all --seed 42

# Just the cross-season evaluation (Section 4.5)
python main.py --task cross-season

# 5-fold classification comparison (Chapter 4)
python main.py --task classify

# Unsupervised clustering (Chapter 6)
python main.py --task cluster --output-dir results
```

Results are written to `results/` (CSV, JSON, `experiments.log`).

## Running the test suite

```bash
pytest                  # fast unit tests only
pytest -m slow          # end-to-end experiment smoke tests
```

CI runs the fast tests against Python 3.10 and 3.11 on every push
(see `.github/workflows/ci.yml`).

## Key Files

- `main.py` — CLI entrypoint (thin wrapper around `src/experiments.py`)
- `src/data_loader.py` — Loads all CSVs into a unified pandas DataFrame
- `src/features.py` — Feature engineering and cross-season split helpers
- `src/models.py` — Classifier / regressor / clusterer definitions
- `src/experiments.py` — Reproducible experiment runner
- `Project_Plan.md` — Full planning document with ML ideas and dissertation structure
- `requirements.txt` — Python dependencies

## Data Note

The CSV row indices (rows 2–32) map to rugby statistics that must be confirmed. See Section 2 of `Project_Plan.md` for preliminary mappings and the critical labelling step.
