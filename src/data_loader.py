"""
data_loader.py
--------------
Loads all Rugby Union game CSVs from the Dissertation/Misc/ directory
into a single unified pandas DataFrame.

Dataset structure (confirmed against AiL1 V Trinity 7_10_2023 spreadsheet):
    - 30 performance statistics per player per game
    - Statistics span: Attack (ball carrier, passing, support), Defence, Turnovers, Discipline
    - Three bold summary rows from the original spreadsheet are excluded from the CSVs
    - Forward CSVs contain only the 12 individual player columns (set-piece team columns excluded)

Usage:
    from src.data_loader import load_all_games, pivot_to_wide, add_derived_features
    df_long  = load_all_games(base_dir="../Misc")
    df_wide  = pivot_to_wide(df_long)
    df_final = add_derived_features(df_wide)
"""

import os
import pandas as pd
import numpy as np


# ── Row label mapping ───────────────────────────────────────────────────────
# Confirmed against the original spreadsheet (AiL1 V Trinity 7_10_2023).
#
# NOTE: Three bold summary rows from the original spreadsheet are EXCLUDED
# from the CSVs (they are derivable from individual rows):
#   - TOTAL POSITIVE CARRIES  = gainline_plus + gainline_zero
#   - TOTAL POSITIVE TACKLE COUNT = dominant_tackle + effective_tackle
#   - TOTAL INEFFECTIVE TACKLE COUNT = missed_tackle + unsuccessful_tackle
#
# NOTE: For Forward CSVs, the original spreadsheet also contains set-piece
# team columns (Scrum, Lineout, Maul) which are NOT in the CSV files.
# Only the 12 named individual player columns appear in the CSVs.
#
# Category breakdown:
#   Attack – Ball Carrier:  rows 1–9
#   Attack – Passing:       rows 10–13
#   Attack – Support:       rows 14–18
#   Defence:                rows 19–25
#   Turnovers:              rows 26–27
#   Discipline:             rows 28–30

ROW_LABELS = {
    # ── ATTACK: BALL CARRIER ────────────────────────────────────────────────
    1:  "gainline_plus",               # 2.Gainline▶Gainline+  (carried past gain line)
    2:  "gainline_zero",               # 2.Gainline▶Gainline 0 (held at gain line)
    3:  "unsuccessful_carry",          # Carry that lost ground
    4:  "total_metres_made",           # Total metres gained carrying (summary row, kept)
    5:  "defender_beaten",             # Beat a defender in contact or space
    6:  "linebreak_made",              # Full linebreak through defence
    7:  "linebreak_conceded",          # Linebreak allowed on defence
    8:  "try",                         # Try scored
    9:  "penalty_try",                 # Penalty try awarded

    # ── ATTACK: PASSING ─────────────────────────────────────────────────────
    10: "successful_pass",             # Pass completed to a team-mate
    11: "unsuccessful_pass",           # Pass intercepted or knocked on
    12: "successful_offload",          # Offload retained by team
    13: "unsuccessful_offload",        # Offload turned over

    # ── ATTACK: SUPPORT ─────────────────────────────────────────────────────
    14: "support_pos_attack_ruck",     # Arrived at ruck on attack — positive outcome
    15: "support_neg_attack_ruck",     # Arrived at ruck on attack — negative outcome
    16: "support_neutral_attack_ruck", # Arrived at ruck on attack — neutral outcome
    17: "in_possession_positive_support",   # In-possession support — positive
    18: "in_possession_ineffective_support",# In-possession support — ineffective

    # ── DEFENCE ─────────────────────────────────────────────────────────────
    19: "dominant_tackle",             # Tackle that put the opponent on the back foot
    20: "effective_tackle",            # Standard completed tackle
    21: "tackle_assist",               # Assisted in making a tackle
    22: "missed_tackle",               # Attempted tackle that was avoided
    23: "unsuccessful_tackle",         # Tackle that failed to bring carrier down
    24: "positive_barge",              # Barge that gained ground / beat defender
    25: "ineffective_barge",           # Barge that gained nothing

    # ── TURNOVERS ───────────────────────────────────────────────────────────
    26: "turnover_won",                # Turnover secured by this player
    27: "turnover_lost",               # Turnover conceded by this player

    # ── DISCIPLINE ──────────────────────────────────────────────────────────
    28: "pen_for",                     # Penalty won (in favour of team)
    29: "pen_against",                 # Penalty conceded
    30: "yellow_card",                 # Yellow card received
}


def load_game_csv(filepath: str, season: str, position: str, game_num: int) -> pd.DataFrame:
    """
    Load a single game CSV file and return a long-format DataFrame.

    Each row in the output represents one player-statistic observation.

    Parameters
    ----------
    filepath : str
        Absolute path to the game CSV file.
    season : str
        Season label, e.g. '22_23' or '23_24'.
    position : str
        Position group: 'Back' or 'Forward'.
    game_num : int
        Game number extracted from the filename.

    Returns
    -------
    pd.DataFrame with columns:
        player, stat, value, season, position, game
    """
    # Read the file line-by-line to handle variable column counts across rows
    # (some CSVs include set-piece aggregate columns that are absent from the header)
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [line.rstrip('\n').rstrip('\r') for line in f if line.strip()]

    if not lines:
        return pd.DataFrame()

    # Row 0 = player names — strip whitespace, drop empty trailing cells
    header_cells = [c.strip() for c in lines[0].split(',')]
    players = [c for c in header_cells if c]   # drop empty strings
    n_players = len(players)

    records = []
    for row_idx, line in enumerate(lines[1:], start=1):
        stat_name = ROW_LABELS.get(row_idx, f"stat_row_{row_idx + 1}")
        cells = line.split(',')
        # Only use the first n_players columns — ignores set-piece aggregate columns
        for col_idx in range(n_players):
            player = players[col_idx]
            try:
                value = float(cells[col_idx].strip()) if col_idx < len(cells) else np.nan
            except (ValueError, TypeError):
                value = np.nan
            records.append({
                "player":   player,
                "stat":     stat_name,
                "value":    value,
                "season":   season,
                "position": position,
                "game":     game_num,
            })

    return pd.DataFrame(records)


def load_all_games(base_dir: str = "../Misc",
                   seasons: list = None) -> pd.DataFrame:
    """
    Walk the directory structure and load all game CSVs.

    Expected structure:
        base_dir/
            Forward/game{N}.csv           ← season '22_23' (Forwards only)
            23_24/Back/game{N}.csv        ← season '23_24'
            23_24/Forward/game{N}.csv     ← season '23_24'
            24_25/Back/game{N}.csv        ← season '24_25'
            24_25/Forward/game{N}.csv     ← season '24_25'

    Parameters
    ----------
    base_dir : str
        Root directory containing the season folders.
    seasons : list or None
        If provided, only load the specified seasons, e.g. ['22_23', '23_24'].
        If None (default), all available seasons are loaded.

    Returns
    -------
    pd.DataFrame in long format (player × stat × game × season × position).
    """
    all_dfs = []

    # Full season/folder map — add new seasons here as data becomes available
    season_map = {
        "Forward":                          ("22_23", "Forward"),
        os.path.join("23_24", "Back"):     ("23_24", "Back"),
        os.path.join("23_24", "Forward"):  ("23_24", "Forward"),
        os.path.join("24_25", "Back"):     ("24_25", "Back"),
        os.path.join("24_25", "Forward"):  ("24_25", "Forward"),
    }

    # Filter to requested seasons if specified
    if seasons is not None:
        season_map = {k: v for k, v in season_map.items() if v[0] in seasons}

    for rel_path, (season, position) in season_map.items():
        folder = os.path.join(base_dir, rel_path)
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue

        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".csv"):
                continue
            try:
                game_num = int(filename.replace("game", "").replace(".csv", ""))
            except ValueError:
                print(f"[WARNING] Could not parse game number from: {filename}")
                continue

            filepath = os.path.join(folder, filename)
            df = load_game_csv(filepath, season, position, game_num)
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No CSV files found under: {base_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} player-stat observations from {len(all_dfs)} game files.")
    return combined


def pivot_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format DataFrame to wide format.

    Returns a DataFrame where each row is one player-game observation
    and each column is a statistic. Useful for ML model input.
    """
    wide = long_df.pivot_table(
        index=["player", "season", "position", "game"],
        columns="stat",
        values="value",
        aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide


def add_derived_features(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived / re-aggregated features to the wide-format DataFrame.

    These are the summary stats that the original spreadsheet calculated but
    excluded from the CSVs. Recomputing them here gives richer ML features.
    """
    df = wide_df.copy()

    # Re-derive the summary rows that were stripped from the CSVs
    df["total_positive_carries"]       = df["gainline_plus"] + df["gainline_zero"]
    df["total_positive_tackle_count"]  = df["dominant_tackle"] + df["effective_tackle"]
    df["total_ineffective_tackle_count"] = df["missed_tackle"] + df["unsuccessful_tackle"]

    # Attack efficiency ratios (avoid divide-by-zero with np.where)
    total_carries = df["total_positive_carries"] + df["unsuccessful_carry"]
    df["carry_success_rate"] = np.where(
        total_carries > 0, df["total_positive_carries"] / total_carries, np.nan
    )

    total_passes = df["successful_pass"] + df["unsuccessful_pass"]
    df["pass_success_rate"] = np.where(
        total_passes > 0, df["successful_pass"] / total_passes, np.nan
    )

    total_tackles = df["total_positive_tackle_count"] + df["total_ineffective_tackle_count"]
    df["tackle_success_rate"] = np.where(
        total_tackles > 0, df["total_positive_tackle_count"] / total_tackles, np.nan
    )

    # Turnover ratio (positive = wins more than loses)
    df["turnover_ratio"] = df["turnover_won"] - df["turnover_lost"]

    # Discipline score (negative impact: conceded penalties and yellow cards)
    df["discipline_score"] = df["pen_for"] - df["pen_against"] - (3 * df["yellow_card"])

    return df


# ── Convenience feature groups for ML ───────────────────────────────────────
ATTACK_FEATURES = [
    "gainline_plus", "gainline_zero", "unsuccessful_carry", "total_metres_made",
    "defender_beaten", "linebreak_made", "linebreak_conceded", "try", "penalty_try",
    "successful_pass", "unsuccessful_pass", "successful_offload", "unsuccessful_offload",
    "support_pos_attack_ruck", "support_neg_attack_ruck", "support_neutral_attack_ruck",
    "in_possession_positive_support", "in_possession_ineffective_support",
]

DEFENCE_FEATURES = [
    "dominant_tackle", "effective_tackle", "tackle_assist",
    "missed_tackle", "unsuccessful_tackle", "positive_barge", "ineffective_barge",
]

TURNOVER_DISCIPLINE_FEATURES = [
    "turnover_won", "turnover_lost", "pen_for", "pen_against", "yellow_card",
]

ALL_RAW_FEATURES = ATTACK_FEATURES + DEFENCE_FEATURES + TURNOVER_DISCIPLINE_FEATURES

DERIVED_FEATURES = [
    "total_positive_carries", "total_positive_tackle_count",
    "total_ineffective_tackle_count", "carry_success_rate",
    "pass_success_rate", "tackle_success_rate", "turnover_ratio", "discipline_score",
]


if __name__ == "__main__":
    # Quick test — run from Project/ directory
    base = os.path.join(os.path.dirname(__file__), "../data/raw")
    df_long = load_all_games(base_dir=base)
    df_wide = pivot_to_wide(df_long)
    df_final = add_derived_features(df_wide)
    print(f"\nWide format shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")
    print(f"\nSample (first 3 rows):")
    print(df_final.head(3).to_string())
