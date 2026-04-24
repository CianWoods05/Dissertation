"""Tests for the feature-engineering layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    build_classification_matrix,
    build_clustering_matrix,
    get_cross_season_split,
)


def test_classification_matrix_shapes(wide_df):
    X, y, names, meta = build_classification_matrix(wide_df, aggregate=True, scale=True)
    assert X.shape[0] == len(y) == len(meta)
    assert X.shape[1] == len(names) > 0
    # y must be binary {0, 1}
    assert set(np.unique(y)).issubset({0, 1})
    # At least one of each class to make CV sane
    assert 0 < y.sum() < len(y)


def test_clustering_matrix_shapes(wide_df):
    X, names, meta = build_clustering_matrix(wide_df, scale=True)
    assert X.shape[0] == len(meta) > 0
    assert X.shape[1] == len(names) > 0


def test_cross_season_split_non_empty(wide_df_multi_season):
    tr, te = get_cross_season_split(
        wide_df_multi_season,
        train_seasons=["22_23", "23_24"],
        test_seasons=["24_25"],
    )
    assert not tr.empty and not te.empty
    # No overlap between seasons in train and test
    assert set(tr["season"]).isdisjoint(set(te["season"]))


def test_cross_season_split_missing_season_raises(wide_df):
    with pytest.raises(ValueError):
        get_cross_season_split(
            wide_df,
            train_seasons=["22_23"],
            test_seasons=["99_99"],  # guaranteed missing
        )
