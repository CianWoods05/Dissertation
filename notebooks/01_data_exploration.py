# ============================================================
# Notebook 01 — Data Exploration & Preprocessing
# Rugby Union ML Dissertation  |  COMP3931  |  Leeds
# ============================================================
import os, sys

# Absolute paths — works regardless of which directory you run from
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
RAW_DIR     = os.path.join(PROJECT_DIR, 'data', 'raw')
MISC_DIR    = os.path.abspath(os.path.join(PROJECT_DIR, '..', 'Misc'))

sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))
os.makedirs(RESULTS_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves files without needing a display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_loader import (load_all_games, pivot_to_wide, add_derived_features,
                          ALL_RAW_FEATURES, ATTACK_FEATURES, DEFENCE_FEATURES,
                          TURNOVER_DISCIPLINE_FEATURES)
from visualisation import (plot_stat_distributions, plot_correlation_heatmap)

print("=" * 60)
print("01 — Data Exploration & Preprocessing")
print("=" * 60)

# ── Load Data ────────────────────────────────────────────────
print("\n[1/7] Loading all game CSVs...")

data_dir = RAW_DIR if (os.path.exists(RAW_DIR) and
                        any(f for f in os.listdir(RAW_DIR) if not f.startswith('.'))) \
           else MISC_DIR
print(f"  Data source: {data_dir}")

df_long = load_all_games(base_dir=data_dir)
df_wide = pivot_to_wide(df_long)
df      = add_derived_features(df_wide)

print(f"  Long format:  {df_long.shape}")
print(f"  Wide format:  {df_wide.shape}")

# ── Dataset overview ─────────────────────────────────────────
print("\n[2/7] Dataset overview...")
print(f"  Seasons:   {sorted(df['season'].unique())}")
print(f"  Positions: {sorted(df['position'].unique())}")
print(f"  Total unique players: {df['player'].nunique()}")

breakdown = df.groupby(['season', 'position']).agg(
    games=('game', 'nunique'),
    player_game_obs=('player', 'count'),
    unique_players=('player', 'nunique')
).reset_index()
print("\n  Breakdown:\n", breakdown.to_string(index=False))

players_22 = set(df[df['season'] == '22_23']['player'].unique())
players_23 = set(df[df['season'] == '23_24']['player'].unique())
both = players_22 & players_23
print(f"\n  Players in 22/23 only:    {len(players_22 - players_23)}")
print(f"  Players in 23/24 only:    {len(players_23 - players_22)}")
print(f"  Players in BOTH seasons:  {len(both)}")

# ── Missing value audit ──────────────────────────────────────
print("\n[3/7] Missing value audit...")
feat_cols = [c for c in ALL_RAW_FEATURES if c in df.columns]
missing = df[feat_cols].isnull().sum()
if missing.sum() == 0:
    print("  ✅ No missing values.")
else:
    print(missing[missing > 0])

# ── Descriptive statistics ───────────────────────────────────
print("\n[4/7] Descriptive statistics by position...")
desc = df.groupby('position')[feat_cols].mean().T
print("\n  Mean per position (top 15 stats):")
print(desc.head(15).round(3).to_string())

desc_full = df.groupby(['position', 'season'])[feat_cols].describe()
out = os.path.join(RESULTS_DIR, 'descriptive_statistics.csv')
desc_full.to_csv(out)
print(f"\n  Saved: {out}")

# ── Distribution plots ───────────────────────────────────────
print("\n[5/7] Distribution plots...")
fig = plot_stat_distributions(df, feat_cols, save=False)
out = os.path.join(RESULTS_DIR, 'eda_stat_distributions.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: {out}")

# ── Correlation heatmaps ─────────────────────────────────────
print("\n[6/7] Correlation heatmaps...")
for label, subset in [('all', df),
                       ('backs', df[df['position']=='Back']),
                       ('forwards', df[df['position']=='Forward'])]:
    fig = plot_correlation_heatmap(subset, feat_cols,
                                    title=f'Feature Correlations — {label.title()}',
                                    save=False)
    out = os.path.join(RESULTS_DIR, f'eda_correlation_{label}.png')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {out}")

corr_matrix = df[feat_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
top_corrs = upper.stack().sort_values(ascending=False).head(10)
print("\n  Top 10 feature correlations:")
print(top_corrs.round(3).to_string())

# ── Back vs Forward separability ─────────────────────────────
print("\n[7/7] Back vs Forward separability (Mann-Whitney U)...")
from scipy import stats
rows = []
for feat in feat_cols:
    b = df[df['position']=='Back'][feat].dropna()
    f = df[df['position']=='Forward'][feat].dropna()
    if len(b) < 5 or len(f) < 5:
        continue
    _, p = stats.mannwhitneyu(b, f, alternative='two-sided')
    effect = abs(b.mean() - f.mean()) / (df[feat].std() + 1e-8)
    rows.append({'feature': feat, 'back_mean': round(b.mean(),3),
                 'forward_mean': round(f.mean(),3),
                 'effect_size': round(effect,3), 'p_value': round(p,4),
                 'significant': p < 0.05})

sep_df = pd.DataFrame(rows).sort_values('effect_size', ascending=False)
out = os.path.join(RESULTS_DIR, 'back_vs_forward_separability.csv')
sep_df.to_csv(out, index=False)
print(f"  Significant features (p<0.05): {sep_df['significant'].sum()} / {len(sep_df)}")
print("\n  Top 10 most separable features:")
print(sep_df.head(10)[['feature','back_mean','forward_mean','effect_size','p_value']].to_string(index=False))
print(f"\n  Saved: {out}")

print("\n✅ Notebook 01 complete — outputs in:", RESULTS_DIR)
