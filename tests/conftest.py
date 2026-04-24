"""
Pytest configuration — puts src/ on sys.path and exposes a cached dataset
fixture so the data layer is only loaded once per test session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def wide_df():
    """Session-scoped wide-format DataFrame for the whole dataset."""
    from data_loader import load_all_games, pivot_to_wide
    df_long = load_all_games(base_dir=str(DATA_DIR))
    return pivot_to_wide(df_long)


@pytest.fixture(scope="session")
def wide_df_multi_season(wide_df):
    """Same as wide_df but asserts at least two seasons are present."""
    seasons = set(wide_df["season"].unique())
    if len(seasons) < 2:
        pytest.skip(f"Need >=2 seasons for cross-season tests, found {seasons}")
    return wide_df
