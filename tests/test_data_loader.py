"""Tests for the data-loading layer."""
from __future__ import annotations

import pandas as pd
import pytest

from data_loader import (
    ROW_LABELS, load_all_games, pivot_to_wide, add_derived_features,
    ALL_RAW_FEATURES,
)


def test_row_labels_cover_1_to_30():
    """ROW_LABELS should map every row 1..30 to a non-empty stat name."""
    for i in range(1, 31):
        assert i in ROW_LABELS, f"Row {i} missing from ROW_LABELS"
        assert isinstance(ROW_LABELS[i], str) and ROW_LABELS[i]


def test_load_all_games_returns_nonempty_long_frame(wide_df):
    """pivot_to_wide on a load_all_games result should give rows and the
    expected meta columns."""
    assert not wide_df.empty
    for col in ("player", "season", "position", "game"):
        assert col in wide_df.columns, f"Missing meta column: {col}"
    # Every raw stat column should appear
    for col in ALL_RAW_FEATURES:
        assert col in wide_df.columns, f"Missing stat column: {col}"


def test_seasons_filter_limits_output():
    """The seasons=... filter should restrict which folders are read."""
    from data_loader import load_all_games, pivot_to_wide
    from tests.conftest import DATA_DIR

    df_long = load_all_games(base_dir=str(DATA_DIR), seasons=["22_23"])
    df = pivot_to_wide(df_long)
    assert set(df["season"].unique()) == {"22_23"}, \
        "seasons=['22_23'] should yield only 22_23 rows"


def test_derived_features_added(wide_df):
    df = add_derived_features(wide_df)
    for col in (
        "total_positive_carries",
        "total_positive_tackle_count",
        "carry_success_rate",
        "pass_success_rate",
        "tackle_success_rate",
        "turnover_ratio",
        "discipline_score",
    ):
        assert col in df.columns, f"Missing derived column: {col}"

    # Carry success rate must be in [0,1] wherever it is defined
    rate = df["carry_success_rate"].dropna()
    assert ((rate >= 0) & (rate <= 1)).all()
