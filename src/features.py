"""
features.py
-----------
Feature engineering for the Rugby Union ML dissertation.

Builds player-level and game-level feature matrices from the wide-format
DataFrame produced by data_loader.py. All transformations are sklearn-
compatible so they slot cleanly into Pipeline objects.

Usage:
    from src.features import build_player_features, build_game_features, RollingWindowFeatures
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

from data_loader import (
    ALL_RAW_FEATURES, ATTACK_FEATURES, DEFENCE_FEATURES,
    TURNOVER_DISCIPLINE_FEATURES, DERIVED_FEATURES, add_derived_features
)


# ── 1. Player-level season aggregate features ───────────────────────────────

def build_player_features(wide_df: pd.DataFrame,
                          include_derived: bool = True,
                          agg_funcs: list = None) -> pd.DataFrame:
    """
    Aggregate per-game wide-format data into per-player-season profiles.

    Each row = one player in one season.
    Each column = aggregated statistic (mean, std, max across all games played).

    Parameters
    ----------
    wide_df : pd.DataFrame
        Output of pivot_to_wide() (one row per player-game).
    include_derived : bool
        Whether to include derived features from add_derived_features().
    agg_funcs : list
        Aggregation functions to apply. Default: ['mean', 'std', 'max', 'sum'].

    Returns
    -------
    pd.DataFrame with multi-level columns flattened to 'stat_aggfunc'.
    """
    if agg_funcs is None:
        agg_funcs = ['mean', 'std', 'max', 'sum']

    df = wide_df.copy()
    if include_derived:
        df = add_derived_features(df)

    feature_cols = ALL_RAW_FEATURES.copy()
    if include_derived:
        feature_cols += DERIVED_FEATURES

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    grouped = df.groupby(['player', 'season', 'position'])[feature_cols].agg(agg_funcs)
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    # Add games_played as a useful meta-feature
    games_played = df.groupby(['player', 'season', 'position'])['game'].count().reset_index()
    games_played.columns = ['player', 'season', 'position', 'games_played']
    grouped = grouped.merge(games_played, on=['player', 'season', 'position'])

    return grouped


# ── 2. Game-level team aggregate features ───────────────────────────────────

def build_game_features(wide_df: pd.DataFrame,
                        position: str = None) -> pd.DataFrame:
    """
    Aggregate player stats per game to get team-level game profiles.

    Useful for studying how team performance varies across games.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Output of pivot_to_wide().
    position : str or None
        If provided, filter to 'Back' or 'Forward' only.

    Returns
    -------
    pd.DataFrame with one row per (season, game, position) combination.
    """
    df = wide_df.copy()
    if position:
        df = df[df['position'] == position]

    feature_cols = [c for c in ALL_RAW_FEATURES if c in df.columns]
    game_agg = df.groupby(['season', 'game', 'position'])[feature_cols].agg(['sum', 'mean'])
    game_agg.columns = ['_'.join(col) for col in game_agg.columns]
    return game_agg.reset_index()


# ── 3. Rolling window features for sequential prediction ────────────────────

class RollingWindowFeatures(BaseEstimator, TransformerMixin):
    """
    Construct rolling-window features for per-player game sequences.

    For each player, sorts games chronologically and computes rolling
    mean and std of the past N games for each statistic.

    Useful for: Idea 2 (performance prediction) where the model sees a
    player's recent form and predicts their next game.

    Parameters
    ----------
    window : int
        Number of past games to include in the rolling window.
    feature_cols : list
        Columns to compute rolling statistics for.
    target_cols : list
        Columns to use as prediction targets (next game's values).
    """

    def __init__(self, window: int = 3, feature_cols: list = None,
                 target_cols: list = None):
        self.window = window
        self.feature_cols = feature_cols or ALL_RAW_FEATURES
        self.target_cols = target_cols or ['total_metres_made', 'dominant_tackle',
                                           'effective_tackle', 'try', 'linebreak_made']

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Input: wide-format DataFrame (one row per player-game).
        Output: DataFrame where each row has rolling features + next-game targets.
        """
        df = X.copy().sort_values(['player', 'season', 'game'])
        records = []

        for (player, season), group in df.groupby(['player', 'season']):
            group = group.reset_index(drop=True)
            feat_cols = [c for c in self.feature_cols if c in group.columns]
            tgt_cols  = [c for c in self.target_cols if c in group.columns]

            for i in range(self.window, len(group)):
                window_data = group.loc[i - self.window: i - 1, feat_cols]
                target_data = group.loc[i, tgt_cols]

                row = {
                    'player': player,
                    'season': season,
                    'position': group.loc[i, 'position'],
                    'game': group.loc[i, 'game'],
                }
                for col in feat_cols:
                    row[f'{col}_roll_mean'] = window_data[col].mean()
                    row[f'{col}_roll_std']  = window_data[col].std()
                for col in tgt_cols:
                    row[f'target_{col}'] = target_data[col]

                records.append(row)

        return pd.DataFrame(records)


# ── 4. Classification feature matrix (Ideas 1 & 3) ─────────────────────────

def build_classification_matrix(wide_df: pd.DataFrame,
                                 aggregate: bool = True,
                                 scale: bool = True) -> tuple:
    """
    Build X, y for position classification (Back vs Forward).

    Parameters
    ----------
    wide_df : pd.DataFrame
        Output of pivot_to_wide().
    aggregate : bool
        If True, aggregate to season-level player profiles.
        If False, use per-game observations (more data, more noise).
    scale : bool
        If True, apply RobustScaler (handles outliers better than StandardScaler
        for small rugby datasets).

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)  — 0=Back, 1=Forward
    feature_names : list of column names (for SHAP interpretation)
    player_meta : pd.DataFrame with player/season/game metadata (for analysis)
    """
    df = add_derived_features(wide_df)

    if aggregate:
        df = build_player_features(wide_df, include_derived=True)
        meta_cols = ['player', 'season', 'position', 'games_played']
        feature_cols = [c for c in df.columns if c not in meta_cols]
    else:
        meta_cols = ['player', 'season', 'position', 'game']
        feature_cols = [c for c in (ALL_RAW_FEATURES + DERIVED_FEATURES)
                        if c in df.columns]

    player_meta = df[meta_cols].copy()
    X = df[feature_cols].fillna(0).values
    y = (df['position'] == 'Forward').astype(int).values

    if scale:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_cols, player_meta


# ── 5. Clustering feature matrix (Idea 3) ───────────────────────────────────

def build_clustering_matrix(wide_df: pd.DataFrame,
                             scale: bool = True) -> tuple:
    """
    Build a feature matrix for unsupervised player clustering.

    Returns season-level aggregated profiles without position labels
    (the idea is to see if clusters recover position structure without supervision).

    Returns
    -------
    X : np.ndarray  — feature matrix (no labels)
    feature_names : list
    player_meta : pd.DataFrame — player/season/position metadata for post-hoc validation
    """
    player_feats = build_player_features(wide_df, include_derived=True,
                                          agg_funcs=['mean', 'std'])
    meta_cols = ['player', 'season', 'position', 'games_played']
    feature_cols = [c for c in player_feats.columns if c not in meta_cols]

    player_meta = player_feats[meta_cols].copy()
    X = player_feats[feature_cols].fillna(0).values

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, feature_cols, player_meta


# ── 6. Season delta features (Idea 5) ───────────────────────────────────────

def build_season_delta_features(wide_df: pd.DataFrame,
                                 from_season: str = '22_23',
                                 to_season: str = '23_24') -> pd.DataFrame:
    """
    Compute the change in each player's stats between two seasons.

    Only includes players who appear in both seasons. Returns a DataFrame
    where each row is a player and each column is the delta (to_season minus from_season).

    Parameters
    ----------
    wide_df : pd.DataFrame
        Output of pivot_to_wide().
    from_season : str
        Earlier season label, e.g. '22_23' or '23_24'.
    to_season : str
        Later season label, e.g. '23_24' or '24_25'.

    Useful for: Idea 5 (season-over-season development modelling).
    """
    df = build_player_features(wide_df, include_derived=True, agg_funcs=['mean'])

    season_from = df[df['season'] == from_season].set_index('player')
    season_to   = df[df['season'] == to_season].set_index('player')

    common_players = season_from.index.intersection(season_to.index)
    if len(common_players) == 0:
        raise ValueError(
            f"No players found in both '{from_season}' and '{to_season}'. "
            "Check season labels and that data is loaded for both seasons."
        )

    feat_cols = [c for c in season_from.columns
                 if c not in ['season', 'position', 'games_played']]

    delta = season_to.loc[common_players, feat_cols] - season_from.loc[common_players, feat_cols]
    delta.columns = [f'delta_{c}' for c in delta.columns]

    # Add position (from the later season) and games played in each
    delta['position']             = season_to.loc[common_players, 'position']
    delta[f'games_{from_season}'] = season_from.loc[common_players, 'games_played']
    delta[f'games_{to_season}']   = season_to.loc[common_players, 'games_played']

    return delta.reset_index()


def get_cross_season_split(wide_df: pd.DataFrame,
                            train_seasons: list = None,
                            test_seasons: list = None):
    """
    Split data into train and test sets along season boundaries.

    Useful for evaluating how well models trained on historical seasons
    generalise to a held-out future season (e.g. train on 22_23 + 23_24,
    test on 24_25).

    Parameters
    ----------
    wide_df : pd.DataFrame
        Output of pivot_to_wide().
    train_seasons : list
        Seasons to include in training set. Default: ['22_23', '23_24'].
    test_seasons : list
        Seasons to include in test set. Default: ['24_25'].

    Returns
    -------
    df_train, df_test : pd.DataFrame
        Wide-format DataFrames for training and testing respectively.
    """
    if train_seasons is None:
        train_seasons = ['22_23', '23_24']
    if test_seasons is None:
        test_seasons = ['24_25']

    df_train = wide_df[wide_df['season'].isin(train_seasons)].copy()
    df_test  = wide_df[wide_df['season'].isin(test_seasons)].copy()

    if df_train.empty:
        raise ValueError(f"No training data found for seasons: {train_seasons}")
    if df_test.empty:
        raise ValueError(
            f"No test data found for seasons: {test_seasons}. "
            "Ensure 24/25 CSVs are present in data/raw/24_25/ and loaded."
        )

    print(f"Cross-season split:")
    print(f"  Train ({train_seasons}): {len(df_train)} player-game observations")
    print(f"  Test  ({test_seasons}):  {len(df_test)} player-game observations")
    return df_train, df_test
