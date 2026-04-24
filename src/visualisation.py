"""
visualisation.py
----------------
Reusable plotting utilities for the Rugby Union ML dissertation.

All functions save figures to results/ and return the figure object.
Call plt.show() in notebooks to display inline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ── Style configuration ──────────────────────────────────────────────────────
PALETTE = {
    'back':    '#2E75B6',   # blue
    'forward': '#C55A11',   # orange
    'season_22': '#A9D18E', # green
    'season_23': '#FFD966', # yellow
    'neutral':  '#595959',
}

RUGBY_CMAP = LinearSegmentedColormap.from_list(
    'rugby', ['#2E75B6', '#FFFFFF', '#C55A11'], N=256
)

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    print(f"Saved: {path}")
    return fig


# ── 1. EDA plots ─────────────────────────────────────────────────────────────

def plot_stat_distributions(wide_df: pd.DataFrame,
                             feature_cols: list,
                             save: bool = True) -> plt.Figure:
    """
    Grid of distribution plots (violin + strip) for each statistic,
    split by Back vs Forward.
    """
    n = len(feature_cols)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data_b = wide_df[wide_df['position'] == 'Back'][col].dropna()
        data_f = wide_df[wide_df['position'] == 'Forward'][col].dropna()

        ax.violinplot([data_b, data_f], positions=[0, 1],
                      showmedians=True, showextrema=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Back', 'Fwd'], fontsize=9)
        ax.set_title(col.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_ylabel('')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Statistic Distributions: Backs vs Forwards', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        _save(fig, 'eda_stat_distributions.png')
    return fig


def plot_correlation_heatmap(wide_df: pd.DataFrame,
                              feature_cols: list,
                              title: str = 'Feature Correlation Matrix',
                              save: bool = True) -> plt.Figure:
    """Correlation heatmap with rugby colour scheme."""
    corr = wide_df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, mask=mask, cmap=RUGBY_CMAP, center=0,
                vmin=-1, vmax=1, annot=False, fmt='.2f',
                linewidths=0.3, ax=ax,
                xticklabels=[c.replace('_', ' ') for c in feature_cols],
                yticklabels=[c.replace('_', ' ') for c in feature_cols])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    if save:
        _save(fig, 'eda_correlation_heatmap.png')
    return fig


def plot_season_comparison(player_feats: pd.DataFrame,
                            stats: list,
                            save: bool = True) -> plt.Figure:
    """
    Bar chart comparing mean stats between 22_23 and 23_24 seasons,
    faceted by position group.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, position in zip(axes, ['Back', 'Forward']):
        subset = player_feats[player_feats['position'] == position]
        stats_cols = [f'{s}_mean' for s in stats if f'{s}_mean' in subset.columns]

        mean_22 = subset[subset['season'] == '22_23'][stats_cols].mean()
        mean_23 = subset[subset['season'] == '23_24'][stats_cols].mean()

        x = np.arange(len(stats_cols))
        w = 0.35
        ax.bar(x - w/2, mean_22.values, w,
               label='22/23', color=PALETTE['season_22'], edgecolor='white')
        ax.bar(x + w/2, mean_23.values, w,
               label='23/24', color=PALETTE['season_23'], edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_mean', '').replace('_', ' ')
                            for s in stats_cols], rotation=45, ha='right', fontsize=9)
        ax.set_title(f'{position}s: Season Comparison', fontweight='bold')
        ax.legend()

    plt.suptitle('Mean Statistics by Season and Position', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    if save:
        _save(fig, 'eda_season_comparison.png')
    return fig


# ── 2. Classification plots ──────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame,
                           metric: str = 'roc_auc',
                           title: str = None,
                           save: bool = True) -> plt.Figure:
    """Horizontal bar chart comparing model performance."""
    df = results_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    colours = [PALETTE['forward'] if m != 'Dummy (majority)' else '#AAAAAA'
               for m in df['model']]
    bars = ax.barh(df['model'], df[metric], color=colours, height=0.6)

    # Error bars if std column exists
    std_col = f'{metric}_std'
    if std_col in df.columns:
        ax.errorbar(df[metric], range(len(df)),
                    xerr=df[std_col], fmt='none', color='black', capsize=3)

    ax.set_xlabel(metric.upper().replace('_', ' '), fontsize=11)
    ax.set_title(title or f'Model Comparison — {metric}', fontweight='bold', fontsize=13)
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

    # Value labels
    for bar, val in zip(bars, df[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    if save:
        _save(fig, f'clf_model_comparison_{metric}.png')
    return fig


def plot_confusion_matrix(y_true, y_pred,
                           labels: list = None,
                           save: bool = True) -> plt.Figure:
    """Annotated confusion matrix."""
    labels = labels or ['Back', 'Forward']
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, cbar=False)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=13)
    plt.tight_layout()
    if save:
        _save(fig, 'clf_confusion_matrix.png')
    return fig


def plot_shap_summary(shap_results: dict, save: bool = True) -> plt.Figure:
    """SHAP bar chart of top features by mean absolute SHAP value."""
    if not shap_results:
        print("No SHAP results to plot.")
        return None

    top = shap_results['top_features']
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = [PALETTE['back'] if i % 2 == 0 else PALETTE['forward']
               for i in range(len(top))]
    ax.barh(top['feature'].str.replace('_', ' '),
            top['mean_abs_shap'], color=PALETTE['forward'], height=0.6)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title('Feature Importance (SHAP)\nTop Predictors: Back vs Forward',
                 fontweight='bold', fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    if save:
        _save(fig, 'clf_shap_importance.png')
    return fig


# ── 3. Regression plots ──────────────────────────────────────────────────────

def plot_regression_comparison(results_df: pd.DataFrame,
                                metric: str = 'rmse',
                                save: bool = True) -> plt.Figure:
    """Bar chart comparing regression models."""
    df = results_df.sort_values(metric, ascending=(metric == 'rmse'))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df['model'], df[metric], color=PALETTE['back'], height=0.6)
    ax.set_xlabel(metric.upper(), fontsize=11)
    ax.set_title(f'Regression Model Comparison — {metric.upper()}',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    if save:
        _save(fig, f'reg_model_comparison_{metric}.png')
    return fig


def plot_prediction_vs_actual(y_true, y_pred,
                               target_name: str = 'target',
                               model_name: str = 'model',
                               save: bool = True) -> plt.Figure:
    """Scatter plot of predicted vs actual values with regression line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color=PALETTE['back'], s=40)

    lims = [min(y_true.min(), y_pred.min()) - 1,
            max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6, label='Perfect prediction')

    from numpy.polynomial.polynomial import polyfit
    b, m = polyfit(y_true, y_pred, 1)
    ax.plot(np.array(lims), b + m * np.array(lims),
            color=PALETTE['forward'], linewidth=1.5, label='Fitted line')

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    ax.set_xlabel(f'Actual {target_name.replace("_", " ")}', fontsize=11)
    ax.set_ylabel(f'Predicted {target_name.replace("_", " ")}', fontsize=11)
    ax.set_title(f'{model_name}: Predicted vs Actual\n'
                 f'RMSE={rmse:.3f}  MAE={mae:.3f}', fontweight='bold', fontsize=12)
    ax.legend()
    plt.tight_layout()
    if save:
        _save(fig, f'reg_{target_name}_{model_name.replace(" ", "_")}_scatter.png')
    return fig


# ── 4. Clustering plots ──────────────────────────────────────────────────────

def plot_elbow_and_silhouette(k_results: pd.DataFrame,
                               save: bool = True) -> plt.Figure:
    """Two-panel plot: inertia elbow curve + silhouette scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_results['k'], k_results['inertia'],
             marker='o', color=PALETTE['back'], linewidth=2)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=11)
    ax1.set_title('Elbow Curve', fontweight='bold', fontsize=13)

    ax2.plot(k_results['k'], k_results['silhouette'],
             marker='s', color=PALETTE['forward'], linewidth=2)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score (higher = better)', fontsize=11)
    ax2.set_title('Silhouette Score vs k', fontweight='bold', fontsize=13)
    best_k = k_results.loc[k_results['silhouette'].idxmax(), 'k']
    ax2.axvline(x=best_k, color='grey', linestyle='--',
                label=f'Best k={best_k}', linewidth=1)
    ax2.legend()

    plt.suptitle('Choosing Optimal Number of Clusters', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    if save:
        _save(fig, 'cluster_elbow_silhouette.png')
    return fig


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray,
                     player_meta: pd.DataFrame = None,
                     method: str = 'PCA',
                     show_position: bool = True,
                     save: bool = True) -> plt.Figure:
    """
    2D scatter plot of cluster assignments.
    Optionally overlays true position labels as marker shapes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_clusters = np.unique(labels)
    cluster_colours = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))

    for ci, cluster in enumerate(unique_clusters):
        mask = labels == cluster
        if show_position and player_meta is not None:
            for pos, marker in [('Back', 'o'), ('Forward', 's'), ('Unknown', '^')]:
                pos_mask = mask & (player_meta['position'].values == pos)
                ax.scatter(X_2d[pos_mask, 0], X_2d[pos_mask, 1],
                           c=[cluster_colours[ci]], marker=marker, s=80, alpha=0.8,
                           edgecolors='white', linewidths=0.5,
                           label=f'Cluster {cluster} – {pos}' if pos_mask.any() else None)
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[cluster_colours[ci]], s=80, alpha=0.8,
                       edgecolors='white', linewidths=0.5,
                       label=f'Cluster {cluster}')

    # Annotate player names if available
    if player_meta is not None and 'player' in player_meta.columns:
        for i, row in player_meta.reset_index(drop=True).iterrows():
            ax.annotate(row['player'].split()[0],  # First name only
                        (X_2d[i, 0], X_2d[i, 1]),
                        fontsize=6, alpha=0.6, ha='center', va='bottom')

    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, loc='best', fontsize=9, framealpha=0.8)
    ax.set_xlabel(f'{method} Component 1', fontsize=11)
    ax.set_ylabel(f'{method} Component 2', fontsize=11)
    ax.set_title(f'Player Clusters ({method})\n'
                 f'Circles = Backs, Squares = Forwards', fontweight='bold', fontsize=13)
    plt.tight_layout()
    if save:
        _save(fig, f'cluster_2d_{method.lower()}.png')
    return fig


def plot_cluster_profiles(wide_df: pd.DataFrame, labels: np.ndarray,
                           feature_cols: list,
                           player_meta: pd.DataFrame = None,
                           save: bool = True) -> plt.Figure:
    """
    Radar/spider chart showing the mean profile of each cluster.
    Useful for interpreting what each cluster represents in rugby terms.
    """
    # Aggregate mean per cluster
    df = wide_df[[c for c in feature_cols if c in wide_df.columns]].copy().fillna(0)
    df['cluster'] = labels

    cluster_means = df.groupby('cluster').mean()

    # Normalise to 0–1 range for radar chart
    normed = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min() + 1e-8
    )

    cats = list(normed.columns)
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10),
                            subplot_kw=dict(polar=True))
    colours = plt.cm.Set2(np.linspace(0, 1, len(normed)))

    for i, (cluster, row) in enumerate(normed.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2,
                color=colours[i], label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1, color=colours[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', '\n') for c in cats], fontsize=7)
    ax.set_title('Cluster Profiles (Normalised Mean Statistics)',
                 fontweight='bold', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    if save:
        _save(fig, 'cluster_profiles_radar.png')
    return fig


# ── 5. Season delta plot (Idea 5) ────────────────────────────────────────────

def plot_season_deltas(delta_df: pd.DataFrame,
                        top_n: int = 10,
                        save: bool = True) -> plt.Figure:
    """
    Bar chart showing which statistics changed most between seasons,
    split by position.
    """
    delta_cols = [c for c in delta_df.columns if c.startswith('delta_')]
    mean_deltas = delta_df.groupby('position')[delta_cols].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, position in zip(axes, ['Back', 'Forward']):
        if position not in mean_deltas.index:
            continue
        row = mean_deltas.loc[position].sort_values(key=abs, ascending=False).head(top_n)
        colours = [PALETTE['forward'] if v > 0 else PALETTE['back'] for v in row.values]
        ax.barh([c.replace('delta_', '').replace('_', ' ') for c in row.index],
                row.values, color=colours, height=0.6)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Mean Change (23/24 − 22/23)', fontsize=10)
        ax.set_title(f'{position}s: Biggest Statistical Changes Between Seasons',
                     fontweight='bold', fontsize=12)

    plt.suptitle('Season-over-Season Player Development', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    if save:
        _save(fig, 'season_delta_bars.png')
    return fig
