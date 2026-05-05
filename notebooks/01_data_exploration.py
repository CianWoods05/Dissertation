"""Exploratory Data Analysis for Rugby Union ML Dissertation (COMP3931).

This notebook performs comprehensive EDA on rugby player performance data:
  - Loads and validates data from raw CSVs
  - Generates descriptive statistics and distributions
  - Identifies outliers and data quality issues
  - Performs statistical tests for feature separability (Back vs. Forward)
  - Produces visualizations and CSV outputs for downstream analysis

Outputs saved to results/:
  - descriptive_statistics.csv: Mean/std by position & season
  - eda_stat_distributions.png: Histograms of all features
  - eda_boxplots.png: Box plots by position and season
  - eda_correlation_*.png: Correlation heatmaps
  - back_vs_forward_separability.csv: Mann-Whitney U test results (Bonferroni-corrected)
  - eda_summary.json: High-level summary statistics
  - eda_log.txt: Execution log with runtime and package versions
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Visualization (non-interactive)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Setup Paths & Logging ────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
RESULTS_DIR = PROJECT_DIR / 'results'
RAW_DIR = PROJECT_DIR / 'data' / 'raw'
MISC_DIR = PROJECT_DIR.parent / 'Misc'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
log_file = RESULTS_DIR / 'eda_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress only specific warnings (not all)
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path
sys.path.insert(0, str(PROJECT_DIR / 'src'))

# ── Import Module Functions ──────────────────────────────────
try:
    from data_loader import (
        load_all_games, pivot_to_wide, add_derived_features,
        ALL_RAW_FEATURES, ATTACK_FEATURES, DEFENCE_FEATURES,
        TURNOVER_DISCIPLINE_FEATURES
    )
    from visualisation import (
        plot_stat_distributions, plot_correlation_heatmap
    )
    logger.info("✓ Data loader and visualisation modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# ── Log Environment ─────────────────────────────────────────
logger.info("=" * 70)
logger.info("NOTEBOOK 01: Data Exploration & Preprocessing")
logger.info("=" * 70)
logger.info(f"Python: {sys.version}")
logger.info(f"pandas: {pd.__version__}")
logger.info(f"numpy: {np.__version__}")
logger.info(f"scipy: {stats.__version__ if hasattr(stats, '__version__') else 'scipy package'}")
logger.info(f"Results directory: {RESULTS_DIR}")


# ── Helper Functions ─────────────────────────────────────────
def get_data_dir(raw_dir: Path, misc_dir: Path) -> Path:
    """Locate data directory with validation.
    
    Args:
        raw_dir: Path to processed/raw data directory
        misc_dir: Path to fallback miscellaneous data directory
        
    Returns:
        Path to the directory containing CSV files
        
    Raises:
        FileNotFoundError: If no data directory is found
    """
    if raw_dir.exists() and any(raw_dir.glob('*.csv')):
        logger.info(f"Using RAW_DIR: {raw_dir}")
        return raw_dir
    elif misc_dir.exists() and any(misc_dir.glob('*.csv')):
        logger.info(f"Using MISC_DIR (fallback): {misc_dir}")
        return misc_dir
    else:
        msg = f"No CSV data found in {raw_dir} or {misc_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)


def compute_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cohen's d effect size (pooled standard deviation).
    
    Args:
        group1: First sample (e.g., Backs)
        group2: Second sample (e.g., Forwards)
        
    Returns:
        Cohen's d statistic
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    if n1 < 2 or n2 < 2:
        return 0.0
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:  # Avoid division by zero
        return 0.0
    
    return abs(group1.mean() - group2.mean()) / pooled_std


def check_data_quality(df: pd.DataFrame, feat_cols: list) -> dict:
    """Audit data quality: missing values, zero-variance, outliers.
    
    Args:
        df: Input DataFrame
        feat_cols: Feature columns to audit
        
    Returns:
        Dictionary with quality metrics
    """
    quality = {
        'missing_values': {},
        'zero_variance': [],
        'outliers': {}
    }
    
    # Missing values
    missing = df[feat_cols].isnull().sum()
    if missing.sum() > 0:
        quality['missing_values'] = missing[missing > 0].to_dict()
    
    # Zero-variance features
    for col in feat_cols:
        if df[col].std() < 1e-10:
            quality['zero_variance'].append(col)
    
    # Outliers (IQR method)
    for col in feat_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR < 1e-10:
            continue
        
        outlier_count = len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
        if outlier_count > 0:
            quality['outliers'][col] = {
                'count': outlier_count,
                'pct': round(100 * outlier_count / len(df), 2)
            }
    
    return quality


# ============================================================
# [1/9] Load Data
# ============================================================
logger.info("\n[1/9] Loading all game CSVs...")
try:
    data_dir = get_data_dir(RAW_DIR, MISC_DIR)
    
    df_long = load_all_games(base_dir=data_dir)
    df_wide = pivot_to_wide(df_long)
    df = add_derived_features(df_wide)
    
    logger.info(f"  Long format:  {df_long.shape}")
    logger.info(f"  Wide format:  {df_wide.shape}")
    logger.info(f"  Final (derived): {df.shape}")
    
except Exception as e:
    logger.error(f"Failed to load data: {e}", exc_info=True)
    sys.exit(1)


# ============================================================
# [2/9] Dataset Overview
# ============================================================
logger.info("\n[2/9] Dataset overview...")
try:
    logger.info(f"  Seasons:   {sorted(df['season'].unique())}")
    logger.info(f"  Positions: {sorted(df['position'].unique())}")
    logger.info(f"  Total unique players: {df['player'].nunique()}")
    
    breakdown = df.groupby(['season', 'position']).agg(
        games=('game', 'nunique'),
        player_game_obs=('player', 'count'),
        unique_players=('player', 'nunique')
    ).reset_index()
    logger.info(f"\n  Breakdown by season & position:\n{breakdown.to_string(index=False)}")
    
    # Cross-season player consistency
    players_22 = set(df[df['season'] == '22_23']['player'].unique())
    players_23 = set(df[df['season'] == '23_24']['player'].unique())
    both = players_22 & players_23
    
    logger.info(f"\n  Player overlap:")
    logger.info(f"    Players in 22/23 only:   {len(players_22 - players_23)}")
    logger.info(f"    Players in 23/24 only:   {len(players_23 - players_22)}")
    logger.info(f"    Players in BOTH seasons: {len(both)}")
    
except Exception as e:
    logger.error(f"Failed to generate dataset overview: {e}", exc_info=True)


# ============================================================
# [3/9] Feature Validation
# ============================================================
logger.info("\n[3/9] Feature validation...")
try:
    feat_cols = [c for c in ALL_RAW_FEATURES if c in df.columns]
    
    missing_features = set(ALL_RAW_FEATURES) - set(df.columns)
    if missing_features:
        logger.warning(f"  ⚠️  Missing expected features: {missing_features}")
    
    logger.info(f"  Using {len(feat_cols)} / {len(ALL_RAW_FEATURES)} features")
    
except Exception as e:
    logger.error(f"Failed during feature validation: {e}", exc_info=True)
    sys.exit(1)


# ============================================================
# [4/9] Data Quality Audit
# ============================================================
logger.info("\n[4/9] Data quality audit...")
try:
    quality = check_data_quality(df, feat_cols)
    
    if not quality['missing_values']:
        logger.info("  ✅ No missing values")
    else:
        logger.warning(f"  Missing values detected: {quality['missing_values']}")
    
    if quality['zero_variance']:
        logger.warning(f"  Zero-variance features: {quality['zero_variance']}")
    
    if quality['outliers']:
        logger.info(f"  Outliers detected (IQR method):")
        for feat, info in sorted(quality['outliers'].items())[:5]:
            logger.info(f"    {feat}: {info['count']} ({info['pct']}%)")
        if len(quality['outliers']) > 5:
            logger.info(f"    ... and {len(quality['outliers']) - 5} more")
    else:
        logger.info("  ✅ No significant outliers detected")
    
except Exception as e:
    logger.error(f"Failed during data quality audit: {e}", exc_info=True)


# ============================================================
# [5/9] Descriptive Statistics
# ============================================================
logger.info("\n[5/9] Descriptive statistics by position...")
try:
    desc = df.groupby('position')[feat_cols].mean().T
    logger.info(f"\n  Mean per position (first 10 stats):")
    logger.info(f"\n{desc.head(10).round(3).to_string()}")
    
    desc_full = df.groupby(['position', 'season'])[feat_cols].describe()
    out = RESULTS_DIR / 'descriptive_statistics.csv'
    desc_full.to_csv(out)
    logger.info(f"\n  ✓ Saved: {out}")
    
except Exception as e:
    logger.error(f"Failed to compute descriptive statistics: {e}", exc_info=True)


# ============================================================
# [6/9] Distribution & Outlier Visualizations
# ============================================================
logger.info("\n[6/9] Distribution plots...")
try:
    fig = plot_stat_distributions(df, feat_cols, save=False)
    out = RESULTS_DIR / 'eda_stat_distributions.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"  ✓ Saved: {out}")
except Exception as e:
    logger.error(f"Failed to create distribution plots: {e}", exc_info=True)

# Box plots by position and season
logger.info("\n  Creating box plots...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # By position
    df_plot = df[feat_cols[:15]].assign(position=df['position'])
    df_plot.boxplot(column=feat_cols[:15], by='position', ax=axes[0])
    axes[0].set_title('Feature Distributions by Position (first 15 stats)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # By season
    df_plot2 = df[feat_cols[:15]].assign(season=df['season'])
    df_plot2.boxplot(column=feat_cols[:15], by='season', ax=axes[1])
    axes[1].set_title('Feature Distributions by Season (first 15 stats)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    out = RESULTS_DIR / 'eda_boxplots.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"  ✓ Saved: {out}")
except Exception as e:
    logger.error(f"Failed to create box plots: {e}", exc_info=True)


# ============================================================
# [7/9] Correlation Analysis
# ============================================================
logger.info("\n[7/9] Correlation heatmaps...")
try:
    for label, subset in [
        ('all', df),
        ('backs', df[df['position'] == 'Back']),
        ('forwards', df[df['position'] == 'Forward'])
    ]:
        try:
            fig = plot_correlation_heatmap(
                subset, feat_cols,
                title=f'Feature Correlations — {label.title()}',
                save=False
            )
            out = RESULTS_DIR / f'eda_correlation_{label}.png'
            fig.savefig(out, bbox_inches='tight', dpi=150)
            plt.close(fig)
            logger.info(f"  ✓ Saved: {out}")
        except Exception as e:
            logger.warning(f"  Failed to generate {label} correlation heatmap: {e}")
    
    # Top correlations
    corr_matrix = df[feat_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    top_corrs = upper.stack().sort_values(ascending=False).head(10)
    
    logger.info(f"\n  Top 10 feature correlations:")
    logger.info(f"\n{top_corrs.round(3).to_string()}")
    
except Exception as e:
    logger.error(f"Failed during correlation analysis: {e}", exc_info=True)


# ============================================================
# [8/9] Back vs Forward Separability (Bonferroni-corrected)
# ============================================================
logger.info("\n[8/9] Back vs Forward separability (Mann-Whitney U test)...")
logger.info(f"  (Bonferroni-corrected α = 0.05/{len(feat_cols)} = {0.05/len(feat_cols):.6f})")

try:
    alpha_corrected = 0.05 / len(feat_cols)  # Bonferroni correction
    rows = []
    
    for feat in feat_cols:
        backs = df[df['position'] == 'Back'][feat].dropna()
        forwards = df[df['position'] == 'Forward'][feat].dropna()
        
        if len(backs) < 5 or len(forwards) < 5:
            logger.warning(f"  Skipping {feat}: insufficient samples (backs={len(backs)}, fwd={len(forwards)})")
            continue
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            backs, forwards, alternative='two-sided'
        )
        
        # Cohen's d (proper effect size)
        cohens_d = compute_cohens_d(backs, forwards)
        
        rows.append({
            'feature': feat,
            'back_mean': round(backs.mean(), 3),
            'back_std': round(backs.std(), 3),
            'forward_mean': round(forwards.mean(), 3),
            'forward_std': round(forwards.std(), 3),
            'cohens_d': round(cohens_d, 3),
            'p_value': p_value,
            'significant_bonf': p_value < alpha_corrected,
            'significant_uncorr': p_value < 0.05
        })
    
    sep_df = pd.DataFrame(rows).sort_values('cohens_d', ascending=False)
    out = RESULTS_DIR / 'back_vs_forward_separability.csv'
    sep_df.to_csv(out, index=False)
    
    logger.info(f"\n  Significant features (Bonferroni-corrected, p<{alpha_corrected:.6f}):")
    logger.info(f"    {sep_df['significant_bonf'].sum()} / {len(sep_df)} features")
    
    logger.info(f"\n  Significant features (uncorrected, p<0.05):")
    logger.info(f"    {sep_df['significant_uncorr'].sum()} / {len(sep_df)} features")
    
    logger.info(f"\n  Top 10 most separable features (by Cohen's d):")
    top_sep = sep_df.head(10)[['feature', 'back_mean', 'forward_mean', 'cohens_d', 'p_value', 'significant_bonf']]
    logger.info(f"\n{top_sep.to_string(index=False)}")
    
    logger.info(f"\n  ✓ Saved: {out}")
    
except Exception as e:
    logger.error(f"Failed during separability analysis: {e}", exc_info=True)


# ============================================================
# [9/9] Summary & Cleanup
# ============================================================
logger.info("\n[9/9] Generating summary...")
try:
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_rows': len(df),
        'total_players': int(df['player'].nunique()),
        'total_games': int(df['game'].nunique()),
        'seasons': sorted(df['season'].unique()),
        'positions': sorted(df['position'].unique()),
        'feature_count': len(feat_cols),
        'features_used': feat_cols,
        'significant_separators_bonf': int(sep_df['significant_bonf'].sum()),
        'significant_separators_uncorr': int(sep_df['significant_uncorr'].sum()),
        'data_quality': {
            'missing_values_total': int(df[feat_cols].isnull().sum().sum()),
            'zero_variance_features': quality['zero_variance'],
            'outlier_features': len(quality['outliers'])
        }
    }
    
    out = RESULTS_DIR / 'eda_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Saved: {out}")
    
except Exception as e:
    logger.error(f"Failed to generate summary: {e}", exc_info=True)

logger.info("\n" + "=" * 70)
logger.info("✅ NOTEBOOK 01 COMPLETE")
logger.info("=" * 70)
logger.info(f"All outputs saved to: {RESULTS_DIR}")
logger.info(f"Log saved to: {log_file}")
