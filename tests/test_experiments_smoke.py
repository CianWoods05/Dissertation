"""
Smoke tests for src/experiments.py.

These run the real experiments end-to-end, with only the minimal data that
is guaranteed to exist in the repo. They are intentionally lightweight:
we check outputs exist and contain sane values, not specific numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import run_cross_season, run_classification


@pytest.mark.slow
def test_run_classification_writes_csv(tmp_path):
    df = run_classification(seed=42, output_dir=tmp_path)
    assert (tmp_path / "clf_aggregated_results.csv").exists()
    assert isinstance(df, pd.DataFrame) and not df.empty
    assert {"model", "accuracy", "f1", "roc_auc"}.issubset(df.columns)
    # Any classifier should beat the dummy baseline
    dummy_acc = df.loc[df["model"].str.contains("Dummy"), "accuracy"].max()
    best_acc  = df["accuracy"].max()
    assert best_acc > dummy_acc, "Best model failed to beat the dummy classifier"


@pytest.mark.slow
def test_run_cross_season_writes_json(tmp_path):
    summary = run_cross_season(
        seed=42,
        output_dir=tmp_path,
        train_seasons=["22_23", "23_24"],
        test_seasons=["24_25"],
    )
    assert 0.0 <= summary["best_accuracy"] <= 1.0
    assert (tmp_path / "cross_season_results.csv").exists()
    assert (tmp_path / "cross_season_summary.json").exists()

    with open(tmp_path / "cross_season_summary.json") as f:
        on_disk = json.load(f)
    assert on_disk["best_model"] == summary["best_model"]
