# main.py
# Entry point — runs the full factor timing pipeline in order
#
# Pipeline:
#   1. Load & prepare data        (data_prep.py)
#   2. Train models               (models.py)
#   3. Evaluate predictions       (evaluation.py)
#   4. Feature importance         (feature_importance.py)
#   5. Backtest with costs        (backtest.py)

from data_prep          import load_data, build_features, split_data
from models             import train_random_forest, train_xgboost, get_predictions
from evaluation         import print_model_evaluation, univariate_significance
from feature_importance import plot_feature_importance
from backtest           import run_cost_analysis


# ── 1. Data ───────────────────────────────────────────────────
df_master = load_data('master.parquet')
df_model  = build_features(df_master)

(X_train_scaled, X_test_scaled,
 y_train_ret, y_test_ret,
 y_train_bin, y_test_bin,
 train_raw, test_raw) = split_data(df_model)


# ── 2. Train Models ───────────────────────────────────────────
rf_model  = train_random_forest(X_train_scaled, y_train_bin)
xgb_model = train_xgboost(X_train_scaled, y_train_ret)

rf_signals, xgb_preds, test_results = get_predictions(
    rf_model, xgb_model, X_test_scaled, test_raw
)


# ── 3. Evaluate ───────────────────────────────────────────────
print("\n=== Model Evaluation ===")
print_model_evaluation(rf_signals, xgb_preds, y_train_ret, y_test_ret, y_test_bin)

print("\n=== Univariate Significance (Clark-West) ===")
df_uni = univariate_significance(train_raw, test_raw, y_train_ret, y_test_ret, y_train_bin)


# ── 4. Feature Importance ─────────────────────────────────────
print("\n=== Feature Importance ===")
plot_feature_importance(rf_model, X_test_scaled, y_test_bin)


# ── 5. Backtest ───────────────────────────────────────────────
run_cost_analysis(rf_signals, y_test_ret)
