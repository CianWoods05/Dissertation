# ============================================================
# Notebook 03 — Performance Prediction
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
from features      import RollingWindowFeatures
from models        import evaluate_regressors, tune_model
from visualisation import plot_regression_comparison, plot_prediction_vs_actual

TARGETS = ['total_metres_made', 'effective_tackle', 'try',
           'linebreak_made', 'support_pos_attack_ruck']

print("=" * 60)
print("03 — Performance Prediction (Next-Game Regression)")
print("=" * 60)

data_dir = RAW_DIR if (os.path.exists(RAW_DIR) and
                        any(f for f in os.listdir(RAW_DIR) if not f.startswith('.'))) \
           else MISC_DIR
df_wide = pivot_to_wide(load_all_games(base_dir=data_dir))
df      = add_derived_features(df_wide)

# ── Build rolling window features ────────────────────────────
print("\n[1/4] Building rolling-window features (window=3)...")
transformer = RollingWindowFeatures(window=3, feature_cols=ALL_RAW_FEATURES,
                                     target_cols=TARGETS)
df_rolled = transformer.fit_transform(df)
print(f"  Rolling dataset: {df_rolled.shape}")

feat_cols = [c for c in df_rolled.columns
             if c.endswith('_roll_mean') or c.endswith('_roll_std')]
tgt_cols  = [f'target_{t}' for t in TARGETS]
X = df_rolled[feat_cols].fillna(0).values
Y = df_rolled[tgt_cols].fillna(0).values
print(f"  X: {X.shape}   Y: {Y.shape}")

# ── Single-target regression ──────────────────────────────────
print("\n[2/4] Single-target regression per statistic...")
all_results = []
for i, tgt in enumerate(TARGETS):
    print(f"\n  Target: {tgt}")
    res = evaluate_regressors(X, Y[:, i], cv_folds=5, target_name=tgt)
    all_results.append(res)
    fig = plot_regression_comparison(res, metric='rmse', save=False)
    fig.savefig(os.path.join(RESULTS_DIR, f'reg_{tgt}_comparison.png'),
                bbox_inches='tight', dpi=150)
    plt.close()

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(os.path.join(RESULTS_DIR, 'reg_single_target_results.csv'), index=False)
print("\n  Summary (best model per target):")
print(combined.loc[combined.groupby('target')['rmse'].idxmin(),
                   ['target','model','rmse','r2']].to_string(index=False))

# ── Multi-target regression ───────────────────────────────────
print("\n[3/4] Multi-target regression (all stats simultaneously)...")
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

multi_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
cv = KFold(n_splits=5, shuffle=False)
Y_pred = cross_val_predict(multi_rf, X, Y, cv=cv)

multi_results = []
for i, tgt in enumerate(TARGETS):
    rmse = np.sqrt(mean_squared_error(Y[:, i], Y_pred[:, i]))
    mae  = mean_absolute_error(Y[:, i], Y_pred[:, i])
    r2   = r2_score(Y[:, i], Y_pred[:, i])
    multi_results.append({'target': tgt, 'rmse': round(rmse,4),
                          'mae': round(mae,4), 'r2': round(r2,4)})
    print(f"  {tgt:35s}  RMSE={rmse:.3f}  R²={r2:.3f}")

pd.DataFrame(multi_results).to_csv(
    os.path.join(RESULTS_DIR, 'reg_multi_target_results.csv'), index=False)

# ── Scatter plots for best target ────────────────────────────
print("\n[4/4] Predicted vs actual scatter plot...")
rf = RandomForestRegressor(n_estimators=200, random_state=42)
y_cv = cross_val_predict(rf, X, Y[:, 0], cv=KFold(n_splits=5, shuffle=False))
fig = plot_prediction_vs_actual(Y[:, 0], y_cv,
                                  target_name=TARGETS[0],
                                  model_name='Random Forest', save=False)
fig.savefig(os.path.join(RESULTS_DIR, f'reg_{TARGETS[0]}_scatter.png'),
            bbox_inches='tight', dpi=150)
plt.close()

print("\n✅ Notebook 03 complete — outputs in:", RESULTS_DIR)
