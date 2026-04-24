# ============================================================
# Notebook 02 — Position Classification (Back vs Forward)
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

from data_loader   import load_all_games, pivot_to_wide
from features      import build_classification_matrix
from models        import (evaluate_classifiers, get_best_classifier,
                            shap_analysis, tune_model, RF_CLASSIFIER_GRID)
from visualisation import plot_model_comparison, plot_confusion_matrix, plot_shap_summary

print("=" * 60)
print("02 — Position Classification: Back vs Forward")
print("=" * 60)

data_dir = RAW_DIR if (os.path.exists(RAW_DIR) and
                        any(f for f in os.listdir(RAW_DIR) if not f.startswith('.'))) \
           else MISC_DIR
df_wide = pivot_to_wide(load_all_games(base_dir=data_dir))

# ── Experiment A: Per-game ────────────────────────────────────
print("\n[A] Per-game classification...")
X_game, y_game, feat_names, meta_game = build_classification_matrix(
    df_wide, aggregate=False, scale=True)
print(f"  Samples: {X_game.shape[0]}  Features: {X_game.shape[1]}")
print(f"  Backs: {(y_game==0).sum()}  Forwards: {(y_game==1).sum()}")

print("\n  CV results (5-fold stratified):")
results_game = evaluate_classifiers(X_game, y_game, cv_folds=5, feature_names=feat_names)
out = os.path.join(RESULTS_DIR, 'clf_per_game_results.csv')
results_game.to_csv(out, index=False)

fig = plot_model_comparison(results_game, metric='roc_auc',
                             title='Position Classifier — Per-Game (5-fold CV)', save=False)
fig.savefig(os.path.join(RESULTS_DIR, 'clf_per_game_model_comparison.png'),
            bbox_inches='tight', dpi=150)
plt.close()

# ── Experiment B: Aggregated ──────────────────────────────────
print("\n[B] Aggregated player-season classification...")
X_agg, y_agg, feat_names_agg, meta_agg = build_classification_matrix(
    df_wide, aggregate=True, scale=True)
print(f"  Samples: {X_agg.shape[0]}  Features: {X_agg.shape[1]}")

print("\n  CV results (5-fold stratified):")
results_agg = evaluate_classifiers(X_agg, y_agg, cv_folds=5, feature_names=feat_names_agg)
out = os.path.join(RESULTS_DIR, 'clf_aggregated_results.csv')
results_agg.to_csv(out, index=False)

fig = plot_model_comparison(results_agg, metric='roc_auc',
                             title='Position Classifier — Season Aggregated (5-fold CV)',
                             save=False)
fig.savefig(os.path.join(RESULTS_DIR, 'clf_aggregated_model_comparison.png'),
            bbox_inches='tight', dpi=150)
plt.close()

# ── Experiment C: Hyperparameter tuning ──────────────────────
print("\n[C] Hyperparameter tuning (Random Forest)...")
from sklearn.ensemble import RandomForestClassifier
gs = tune_model(RandomForestClassifier(random_state=42),
                RF_CLASSIFIER_GRID, X_game, y_game, scoring='roc_auc', cv=5)
tuning_df = pd.DataFrame([{'best_roc_auc': gs.best_score_, **gs.best_params_}])
tuning_df.to_csv(os.path.join(RESULTS_DIR, 'clf_tuning_results.csv'), index=False)

# ── Experiment D: SHAP feature importance ────────────────────
print("\n[D] SHAP feature importance...")
best_rf = get_best_classifier(X_game, y_game, clf_name="Random Forest")
shap_results = shap_analysis(best_rf, X_game, feat_names, max_display=20)

if shap_results:
    shap_results['top_features'].to_csv(
        os.path.join(RESULTS_DIR, 'clf_shap_feature_importance.csv'), index=False)
    fig = plot_shap_summary(shap_results, save=False)
    if fig:
        fig.savefig(os.path.join(RESULTS_DIR, 'clf_shap_importance.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()
    print("\n  Top 10 SHAP features:")
    print(shap_results['top_features'].head(10).to_string(index=False))

# ── Experiment E: Confusion matrix ───────────────────────────
print("\n[E] Confusion matrix (cross-validated predictions)...")
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(best_rf, X_game, y_game, cv=cv)
print("\n  Classification Report:")
print(classification_report(y_game, y_pred, target_names=['Back', 'Forward']))

report_dict = classification_report(y_game, y_pred,
                                     target_names=['Back', 'Forward'],
                                     output_dict=True)
pd.DataFrame(report_dict).T.to_csv(
    os.path.join(RESULTS_DIR, 'clf_classification_report.csv'))

fig = plot_confusion_matrix(y_game, y_pred, save=False)
fig.savefig(os.path.join(RESULTS_DIR, 'clf_confusion_matrix.png'),
            bbox_inches='tight', dpi=150)
plt.close()

print("\n✅ Notebook 02 complete — outputs in:", RESULTS_DIR)
