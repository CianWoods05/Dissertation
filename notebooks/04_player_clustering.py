# ============================================================
# Notebook 04 — Player Clustering & Archetype Discovery
# Rugby Union ML Dissertation  |  COMP3931  |  Leeds
# ============================================================
import os, sys

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_loader   import load_all_games, pivot_to_wide, add_derived_features, ALL_RAW_FEATURES
from features      import build_clustering_matrix, build_player_features
from models        import (find_optimal_k, evaluate_clustering,
                            fit_final_clustering, reduce_dimensions)
from visualisation import (plot_elbow_and_silhouette, plot_clusters_2d,
                            plot_cluster_profiles)

print("=" * 60)
print("04 — Player Profiling & Archetype Discovery")
print("=" * 60)

data_dir = RAW_DIR if (os.path.exists(RAW_DIR) and
                        any(f for f in os.listdir(RAW_DIR) if not f.startswith('.'))) \
           else MISC_DIR
df_wide = pivot_to_wide(load_all_games(base_dir=data_dir))
X, feat_names, player_meta = build_clustering_matrix(df_wide, scale=True)

print(f"\n  Feature matrix: {X.shape}")
print(f"  Players:  {len(player_meta)}")
print(f"  Position split: {player_meta['position'].value_counts().to_dict()}")

# ── Find optimal k ────────────────────────────────────────────
print("\n[1/5] Finding optimal k...")
k_results = find_optimal_k(X, k_range=range(2, 9))
k_results.to_csv(os.path.join(RESULTS_DIR, 'cluster_k_selection.csv'), index=False)

fig = plot_elbow_and_silhouette(k_results, save=False)
fig.savefig(os.path.join(RESULTS_DIR, 'cluster_elbow_silhouette.png'),
            bbox_inches='tight', dpi=150)
plt.close()

best_k = int(k_results.loc[k_results['silhouette'].idxmax(), 'k'])
print(f"\n  Best k by silhouette: {best_k}")

# ── Compare algorithms ────────────────────────────────────────
print("\n[2/5] Comparing clustering algorithms...")
true_labels = (player_meta['position'] == 'Forward').astype(int).values
cluster_results = evaluate_clustering(X, true_labels=true_labels)
cluster_results.to_csv(os.path.join(RESULTS_DIR, 'cluster_algorithm_comparison.csv'),
                        index=False)
print(cluster_results.to_string(index=False))

# ── Final KMeans fit ──────────────────────────────────────────
print(f"\n[3/5] Final KMeans (k={best_k})...")
labels = fit_final_clustering(X, k=best_k, algorithm='kmeans')
player_meta = player_meta.copy()
player_meta['cluster'] = labels

cross_tab = pd.crosstab(player_meta['cluster'], player_meta['position'], margins=True)
print("\n  Cluster × Position:")
print(cross_tab)
cross_tab.to_csv(os.path.join(RESULTS_DIR, 'cluster_vs_position_crosstab.csv'))

cross_season = pd.crosstab(player_meta['cluster'], player_meta['season'])
print("\n  Cluster × Season:")
print(cross_season)

# ── 2D visualisation ──────────────────────────────────────────
print("\n[4/5] 2D visualisation (PCA + t-SNE)...")
for method in ['pca', 'tsne']:
    X_2d, _ = reduce_dimensions(X, method=method, n_components=2)
    fig = plot_clusters_2d(X_2d, labels, player_meta,
                            method=method.upper(), show_position=True, save=False)
    fig.savefig(os.path.join(RESULTS_DIR, f'cluster_2d_{method}.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: cluster_2d_{method}.png")

# ── Cluster profile interpretation ────────────────────────────
print("\n[5/5] Cluster profiles...")
player_feats = build_player_features(df_wide, include_derived=False, agg_funcs=['mean'])
player_feats = player_feats.merge(
    player_meta[['player', 'season', 'cluster']], on=['player', 'season'], how='inner')

stat_cols = [f'{c}_mean' for c in ALL_RAW_FEATURES if f'{c}_mean' in player_feats.columns]

fig = plot_cluster_profiles(player_feats, player_feats['cluster'].values,
                             feature_cols=stat_cols[:12], save=False)
fig.savefig(os.path.join(RESULTS_DIR, 'cluster_profiles_radar.png'),
            bbox_inches='tight', dpi=150)
plt.close()

player_meta[['player','season','position','cluster']].sort_values(
    ['cluster','position','player']
).to_csv(os.path.join(RESULTS_DIR, 'cluster_player_memberships.csv'), index=False)

key_stats = [c for c in ['total_metres_made_mean', 'effective_tackle_mean',
                           'successful_pass_mean', 'support_pos_attack_ruck_mean',
                           'linebreak_made_mean', 'try_mean']
             if c in player_feats.columns]
print("\n  Cluster summaries (key stat means):")
print(player_feats.groupby('cluster')[key_stats].mean().round(3).to_string())

print("\n✅ Notebook 04 complete — outputs in:", RESULTS_DIR)
