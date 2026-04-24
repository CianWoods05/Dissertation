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

## Key Files

- `Project_Plan.md` — Full planning document with ML ideas and dissertation structure
- `src/data_loader.py` — Loads all CSVs into a unified pandas DataFrame
- `requirements.txt` — Python dependencies

## Data Note

The CSV row indices (rows 2–32) map to rugby statistics that must be confirmed. See Section 2 of `Project_Plan.md` for preliminary mappings and the critical labelling step.
