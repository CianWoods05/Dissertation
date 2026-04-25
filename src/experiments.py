"""
experiments.py
--------------
Reproducible experiment runner for the Rugby Union ML dissertation.

Wires together data_loader -> features -> models into a single pipeline so
that every experiment reported in the dissertation can be re-executed from
the command line with a deterministic seed.

Public entry points
-------------------
- run_classification    : 5-fold CV of all classifiers on the full pooled dataset
                          (populates results/clf_aggregated_results.csv).
- run_cross_season      : Train on historical seasons, evaluate on a held-out
                          future season. Produces the real 4.5 accuracy number
                          referenced in the dissertation.
- run_regression       : Next-game performance prediction (Chapter 5); writes
                          reg_single_target_results.csv and reg_multi_target_results.csv.
- run_clustering        : Unsupervised player archetype discovery; writes
                          cluster_algorithm_comparison.csv and cluster_k_selection.csv.
- run_all               : Convenience wrapper that runs all four above in sequence.

Usage (from the Project/ directory)
-----------------------------------
    python -m src.experiments --task classify --output-dir results
    python -m src.experiments --task cross-season
    python -m src.experiments --task all --seed 42

All runs are logged to both stdout and results/experiments.log.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Local imports — using the existing project modules unchanged
# Importing this way works both when run as a script *and* as a module,
# because the sys.path manipulation below adds src/ to the import path.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader import (  # noqa: E402
    load_all_games,
    pivot_to_wide,
    add_derived_features,
    ALL_RAW_FEATURES,
)
from features import (  # noqa: E402
    build_classification_matrix,
    build_clustering_matrix,
    get_cross_season_split,
    RollingWindowFeatures,
)
from models import (  # noqa: E402
    evaluate_classifiers,
    evaluate_clustering,
    evaluate_regressors,
    find_optimal_k,
    CLASSIFIERS,
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)


# ── 1. Environment setup ────────────────────────────────────────────────────

PROJECT_ROOT = _SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_SEED = 42


def configure_logging(output_dir: Path) -> logging.Logger:
    """Set up a root logger that writes to stdout *and* to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "experiments.log"

    logger = logging.getLogger("rugby_experiments")
    logger.setLevel(logging.INFO)
    # Clear any handlers from previous runs (important for repeated in-process calls)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Force UTF-8 on stdout so decorative Unicode characters (─, →, etc.)
    # don't crash the logger on Windows' default cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setFormatter(fmt)
    logger.addHandler(stdout_h)

    file_h = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed every RNG we might touch so results are byte-for-byte reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── 2. Data loading helper ──────────────────────────────────────────────────

@dataclass
class DatasetBundle:
    """Everything a downstream experiment needs out of the data layer."""
    long:  pd.DataFrame
    wide:  pd.DataFrame
    final: pd.DataFrame  # wide + derived features

    @property
    def n_players(self) -> int:
        return self.wide[["player", "season"]].drop_duplicates().shape[0]

    @property
    def n_games(self) -> int:
        return self.wide[["season", "game", "position"]].drop_duplicates().shape[0]


def load_dataset(seasons: Optional[List[str]] = None,
                 data_dir: Path = DATA_DIR,
                 logger: Optional[logging.Logger] = None) -> DatasetBundle:
    """Load raw CSVs and produce the three canonical DataFrames."""
    log = logger or logging.getLogger("rugby_experiments")
    log.info("Loading data from %s (seasons=%s)", data_dir, seasons or "ALL")

    df_long  = load_all_games(base_dir=str(data_dir), seasons=seasons)
    df_wide  = pivot_to_wide(df_long)
    df_final = add_derived_features(df_wide)

    log.info("  rows (long) : %d", len(df_long))
    log.info("  rows (wide) : %d", len(df_wide))
    log.info("  players x seasons: %d", df_wide[["player", "season"]].drop_duplicates().shape[0])
    return DatasetBundle(long=df_long, wide=df_wide, final=df_final)


# ── 3. Experiment: classification (Idea 1) ──────────────────────────────────

def run_classification(seed: int = DEFAULT_SEED,
                       output_dir: Path = RESULTS_DIR,
                       seasons: Optional[List[str]] = None,
                       logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    5-fold stratified CV of every classifier on the pooled dataset.

    Writes: results/clf_aggregated_results.csv
    Returns the results DataFrame.
    """
    log = logger or configure_logging(output_dir)
    set_seed(seed)
    log.info("─" * 70)
    log.info("EXPERIMENT: Position classification (5-fold CV)")
    log.info("─" * 70)

    data = load_dataset(seasons=seasons, logger=log)
    X, y, feature_names, _ = build_classification_matrix(
        data.wide, aggregate=True, scale=True
    )
    log.info("Feature matrix: X=%s, y=%s, positive class (Forward)=%d",
             X.shape, y.shape, int(y.sum()))

    t0 = time.perf_counter()
    results = evaluate_classifiers(X, y, cv_folds=5, feature_names=feature_names)
    log.info("Classification CV took %.1fs", time.perf_counter() - t0)

    out_path = output_dir / "clf_aggregated_results.csv"
    results.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    log.info("\n%s", results.to_string(index=False))
    return results


# ── 4. Experiment: cross-season evaluation (Section 4.5) ────────────────────

def run_cross_season(seed: int = DEFAULT_SEED,
                     output_dir: Path = RESULTS_DIR,
                     train_seasons: Optional[List[str]] = None,
                     test_seasons:  Optional[List[str]] = None,
                     logger: Optional[logging.Logger] = None) -> dict:
    """
    Train classifiers on *train_seasons* and evaluate on *test_seasons*.

    This is the experiment referenced in Section 4.5 of the dissertation
    (previously an estimate — this replaces it with a real measurement).

    Writes
    ------
    results/cross_season_results.csv  — per-model accuracy/F1/AUC on test set
    results/cross_season_summary.json — headline figures + metadata

    Returns the summary dict.
    """
    log = logger or configure_logging(output_dir)
    set_seed(seed)
    train_seasons = train_seasons or ["22_23", "23_24"]
    test_seasons  = test_seasons  or ["24_25"]

    log.info("─" * 70)
    log.info("EXPERIMENT: Cross-season evaluation")
    log.info("  Train seasons : %s", train_seasons)
    log.info("  Test  seasons : %s", test_seasons)
    log.info("─" * 70)

    data = load_dataset(seasons=train_seasons + test_seasons, logger=log)
    df_train, df_test = get_cross_season_split(
        data.wide, train_seasons=train_seasons, test_seasons=test_seasons
    )

    # Build *player-level* feature matrices separately on each split.
    # We fit the scaler on the train split only to avoid leakage.
    X_tr_raw, y_tr, feature_names, _ = build_classification_matrix(
        df_train, aggregate=True, scale=False
    )
    X_te_raw, y_te, feat_names_te, _ = build_classification_matrix(
        df_test,  aggregate=True, scale=False
    )
    assert feature_names == feat_names_te, (
        "Feature alignment failed between train and test splits — check "
        "that derived features are identical in both splits."
    )

    scaler = RobustScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    log.info("Train matrix: %s,  Test matrix: %s", X_tr.shape, X_te.shape)

    rows = []
    # Run every classifier, not just LR, so we can compare
    for name, clf in CLASSIFIERS.items():
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        try:
            probs = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else np.nan
        except (AttributeError, ValueError):
            auc = np.nan

        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, zero_division=0)
        row = {
            "model":    name,
            "accuracy": acc,
            "f1":       f1,
            "roc_auc":  auc,
            "n_train":  len(y_tr),
            "n_test":   len(y_te),
            "fit_time": time.perf_counter() - t0,
        }
        rows.append(row)
        log.info("  %-22s acc=%.3f  f1=%.3f  auc=%s",
                 name, acc, f1,
                 f"{auc:.3f}" if np.isfinite(auc) else "n/a")

    df_out = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    csv_path = output_dir / "cross_season_results.csv"
    df_out.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    best = df_out.iloc[0].to_dict()
    summary = {
        "train_seasons":  train_seasons,
        "test_seasons":   test_seasons,
        "n_train":        int(len(y_tr)),
        "n_test":         int(len(y_te)),
        "n_features":     int(X_tr.shape[1]),
        "best_model":     best["model"],
        "best_accuracy":  float(best["accuracy"]),
        "best_f1":        float(best["f1"]),
        "best_roc_auc":   float(best["roc_auc"]) if np.isfinite(best["roc_auc"]) else None,
        "seed":           seed,
    }
    with open(output_dir / "cross_season_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary → %s", summary)

    # Also log a full sklearn classification report for the best model,
    # for the dissertation text.
    best_clf = CLASSIFIERS[best["model"]]
    best_clf.fit(X_tr, y_tr)
    preds = best_clf.predict(X_te)
    log.info("Best model (%s) — classification report on held-out %s:",
             best["model"], test_seasons)
    log.info("\n%s", classification_report(y_te, preds,
                                            target_names=["Back", "Forward"],
                                            zero_division=0))
    cm = confusion_matrix(y_te, preds)
    log.info("Confusion matrix:\n%s", cm)

    return summary


# ── 5. Experiment: unsupervised clustering (Idea 3) ─────────────────────────

def run_clustering(seed: int = DEFAULT_SEED,
                   output_dir: Path = RESULTS_DIR,
                   k_range=range(2, 10),
                   logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """
    Run the clustering pipeline and write out silhouette/ARI numbers used in
    Chapter 6 of the dissertation.

    Writes
    ------
    results/cluster_k_selection.csv          — silhouette/inertia for k=2..9
    results/cluster_algorithm_comparison.csv — algorithm sweep
    """
    log = logger or configure_logging(output_dir)
    set_seed(seed)
    log.info("─" * 70)
    log.info("EXPERIMENT: Unsupervised clustering")
    log.info("─" * 70)

    data = load_dataset(logger=log)
    X, feature_names, meta = build_clustering_matrix(data.wide, scale=True)
    log.info("Clustering matrix: X=%s, %d players", X.shape, len(meta))

    log.info("── k-selection sweep ──")
    k_df = find_optimal_k(X, k_range=k_range)
    (output_dir / "cluster_k_selection.csv").write_text(k_df.to_csv(index=False))
    log.info("Wrote cluster_k_selection.csv")

    log.info("── algorithm comparison ──")
    pos = (meta["position"] == "Forward").astype(int).values
    algo_df = evaluate_clustering(X, true_labels=pos)
    (output_dir / "cluster_algorithm_comparison.csv").write_text(algo_df.to_csv(index=False))
    log.info("Wrote cluster_algorithm_comparison.csv")

    return {"k_selection": k_df, "algorithm_comparison": algo_df}


PERFORMANCE_TARGETS = [
    "total_metres_made",
    "effective_tackle",
    "try",
    "linebreak_made",
    "support_pos_attack_ruck",
]


def run_regression(seed: int = DEFAULT_SEED,
                   output_dir: Path = RESULTS_DIR,
                   window: int = 3,
                   targets: Optional[List[str]] = None,
                   logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """
    Next-game performance prediction (Chapter 5).

    Uses :class:`RollingWindowFeatures` to turn each player's chronologically
    ordered games into (rolling-mean + rolling-std features → next-game targets)
    rows, then runs every regressor with 5-fold KFold CV (no shuffle, so order
    is preserved roughly temporally).

    Writes
    ------
    results/reg_single_target_results.csv  — per-model, per-target RMSE/MAE/R²
    results/reg_multi_target_results.csv   — RandomForest MultiOutput per-target
    """
    log = logger or configure_logging(output_dir)
    set_seed(seed)
    targets = targets or PERFORMANCE_TARGETS

    log.info("─" * 70)
    log.info("EXPERIMENT: Performance prediction (rolling window=%d)", window)
    log.info("  Targets: %s", targets)
    log.info("─" * 70)

    data = load_dataset(logger=log)

    # Build rolling-window matrix (one row per player-game with >= window prior games)
    transformer = RollingWindowFeatures(
        window=window,
        feature_cols=ALL_RAW_FEATURES,
        target_cols=targets,
    )
    df_rolled = transformer.fit_transform(data.final)
    log.info("Rolling dataset: %s", df_rolled.shape)

    feat_cols = [c for c in df_rolled.columns
                 if c.endswith("_roll_mean") or c.endswith("_roll_std")]
    tgt_cols = [f"target_{t}" for t in targets]
    X = df_rolled[feat_cols].fillna(0).to_numpy()
    Y = df_rolled[tgt_cols].fillna(0).to_numpy()
    log.info("X=%s,  Y=%s,  features=%d", X.shape, Y.shape, len(feat_cols))

    # ── Single-target sweep ────────────────────────────────────────────────
    log.info("── Single-target regression (5-fold KFold, no shuffle) ──")
    per_target = []
    for i, tgt in enumerate(targets):
        log.info("Target: %s", tgt)
        df_t = evaluate_regressors(X, Y[:, i], cv_folds=5, target_name=tgt)
        per_target.append(df_t)
    single = pd.concat(per_target, ignore_index=True)
    single_path = output_dir / "reg_single_target_results.csv"
    single.to_csv(single_path, index=False)
    log.info("Wrote %s", single_path)

    best_rows = single.loc[single.groupby("target")["rmse"].idxmin(),
                           ["target", "model", "rmse", "r2"]]
    log.info("Best model per target:\n%s", best_rows.to_string(index=False))

    # ── Multi-target (one MultiOutputRegressor across all targets) ─────────
    log.info("── Multi-target regression (RandomForest MultiOutput) ──")
    multi_rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=seed)
    )
    cv = KFold(n_splits=5, shuffle=False)
    Y_pred = cross_val_predict(multi_rf, X, Y, cv=cv)

    multi_rows = []
    for i, tgt in enumerate(targets):
        rmse = float(np.sqrt(mean_squared_error(Y[:, i], Y_pred[:, i])))
        mae = float(mean_absolute_error(Y[:, i], Y_pred[:, i]))
        r2 = float(r2_score(Y[:, i], Y_pred[:, i]))
        multi_rows.append({
            "target": tgt,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        })
        log.info("  %-30s rmse=%.3f  mae=%.3f  r2=%.3f", tgt, rmse, mae, r2)
    multi = pd.DataFrame(multi_rows)
    multi_path = output_dir / "reg_multi_target_results.csv"
    multi.to_csv(multi_path, index=False)
    log.info("Wrote %s", multi_path)

    return {"single_target": single, "multi_target": multi}


# ── 6. run_all ──────────────────────────────────────────────────────────────

def run_all(seed: int = DEFAULT_SEED,
            output_dir: Path = RESULTS_DIR) -> dict:
    """Run classification + cross-season + regression + clustering in one go."""
    log = configure_logging(output_dir)
    log.info("▶ run_all started (seed=%d, output=%s)", seed, output_dir)
    t0 = time.perf_counter()
    out = {
        "classification":      run_classification(seed=seed, output_dir=output_dir, logger=log),
        "cross_season":        run_cross_season(seed=seed, output_dir=output_dir, logger=log),
        "regression":          run_regression(seed=seed, output_dir=output_dir, logger=log),
        "clustering":          run_clustering(seed=seed, output_dir=output_dir, logger=log),
    }
    log.info("▶ run_all finished in %.1fs", time.perf_counter() - t0)
    return out


# ── 7. argparse wrapper so this module is runnable directly ─────────────────

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="experiments",
        description="Reproducible experiment runner for the rugby ML dissertation.",
    )
    p.add_argument("--task", choices=["classify", "cross-season", "regress", "cluster", "all"],
                   default="all",
                   help="Which experiment to run (default: all).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed (default: {DEFAULT_SEED}).")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                   help=f"Where CSV/JSON outputs land (default: {RESULTS_DIR}).")
    p.add_argument("--train-seasons", nargs="+",
                   help="For --task cross-season: training seasons, space-separated.")
    p.add_argument("--test-seasons",  nargs="+",
                   help="For --task cross-season: test seasons, space-separated.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logger = configure_logging(args.output_dir)

    if args.task == "classify":
        run_classification(seed=args.seed, output_dir=args.output_dir, logger=logger)
    elif args.task == "cross-season":
        run_cross_season(
            seed=args.seed,
            output_dir=args.output_dir,
            train_seasons=args.train_seasons,
            test_seasons=args.test_seasons,
            logger=logger,
        )
    elif args.task == "regress":
        run_regression(seed=args.seed, output_dir=args.output_dir, logger=logger)
    elif args.task == "cluster":
        run_clustering(seed=args.seed, output_dir=args.output_dir, logger=logger)
    elif args.task == "all":
        run_all(seed=args.seed, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
