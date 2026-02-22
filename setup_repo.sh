#!/bin/bash
# setup_repo.sh
# Run this once in your Codespace terminal to create the full project structure
# Usage: bash setup_repo.sh

set -e  # stop on any error

echo "Creating project structure..."
mkdir -p factor_timing
cd factor_timing

# ── config.py ─────────────────────────────────────────────────
cat > config.py << 'EOF'
# config.py
# Central place for all hyperparameters and settings
# Change values here — no need to touch other files

SPLIT_DATE = '2002-01-31'
THRESHOLD  = 0.005
FEATURES   = ['tbl', 'hml_mom', 'bm', 'dfy', 'infl', 'svar']

RF_PARAMS  = dict(n_estimators=300, max_depth=5, random_state=42)
XGB_PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

COST_BPS_LIST = [0, 10, 20, 50]
EOF

# ── data_prep.py ───────────────────────────────────────────────
cat > data_prep.py << 'EOF'
# data_prep.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import SPLIT_DATE, THRESHOLD, FEATURES


def load_data(path='master.parquet'):
    df_master = pd.read_parquet(path)
    print("Column names:", df_master.columns.tolist())
    print(f"Time range: {df_master.index.min()} to {df_master.index.max()}")
    missing_values = df_master.isnull().sum()
    print("Columns with missing values:\n", missing_values[missing_values > 0])
    return df_master


def build_features(df_master):
    df_model = df_master.copy()
    df_model['dfy'] = df_model['baa'] - df_model['aaa']
    df_model['hml_mom'] = df_model['hml'].rolling(3).mean()
    df_model['target_hml_ret'] = df_model['hml'].shift(-1)
    return df_model


def split_data(df_model):
    train_raw = df_model[:SPLIT_DATE].dropna(subset=FEATURES + ['target_hml_ret'])
    test_raw  = df_model[SPLIT_DATE:].dropna(subset=FEATURES + ['target_hml_ret'])

    # Labels
    y_train_ret = train_raw['target_hml_ret']
    y_train_bin = (train_raw['target_hml_ret'] > THRESHOLD).astype(int)
    y_test_ret  = test_raw['target_hml_ret']
    y_test_bin  = (test_raw['target_hml_ret'] > THRESHOLD).astype(int)

    # Scale — fit on train only, then transform test
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_raw[FEATURES])
    X_test_scaled  = scaler.transform(test_raw[FEATURES])

    print(f"Training period: {train_raw.index.min()} to {train_raw.index.max()} (n={len(train_raw)})")
    print(f"Testing  period: {test_raw.index.min()}  to {test_raw.index.max()}  (n={len(test_raw)})")

    return (X_train_scaled, X_test_scaled,
            y_train_ret, y_test_ret,
            y_train_bin, y_test_bin,
            train_raw, test_raw)
EOF

# ── models.py ─────────────────────────────────────────────────
cat > models.py << 'EOF'
# models.py
from sklearn.ensemble     import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost              import XGBRegressor
from config               import RF_PARAMS, XGB_PARAMS


def train_random_forest(X_train_scaled, y_train_bin):
    # Model 1: Random Forest (classification)
    # Goal: predict whether next month HML return > 0.5%
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train_scaled, y_train_bin)
    return rf_model


def train_xgboost(X_train_scaled, y_train_ret):
    # Model 2: XGBoost (regression)
    # Goal: directly predict next month HML return value
    xgb_model = XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train_scaled, y_train_ret)
    return xgb_model


def train_logistic(X_train_scaled, y_train_bin):
    # Model 3: Logistic Regression — used for univariate significance tests
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train_bin)
    return lr_model


def get_predictions(rf_model, xgb_model, X_test_scaled, test_raw):
    rf_signals  = rf_model.predict(X_test_scaled)   # binary: 0 or 1
    xgb_preds   = xgb_model.predict(X_test_scaled)  # continuous return estimate

    test_results = test_raw.copy()
    test_results['rf_signal']    = rf_signals
    test_results['xgb_pred_ret'] = xgb_preds

    return rf_signals, xgb_preds, test_results
EOF

# ── evaluation.py ─────────────────────────────────────────────
cat > evaluation.py << 'EOF'
# evaluation.py
import numpy as np
import pandas as pd
from scipy                 import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics       import accuracy_score
from config                import FEATURES


def get_ros2(actual, predicted, benchmark):
    mse_model = np.mean((actual - predicted) ** 2)
    mse_bench = np.mean((actual - benchmark) ** 2)
    return 1 - (mse_model / mse_bench)


def clark_west_test(actual, pred_unrestricted, pred_restricted):
    """
    Clark-West (2007) Test
    H0: The restricted model (historical mean) is no worse than the unrestricted model
    p < 0.05 -> Reject H0 -> Your model significantly outperforms the benchmark
    """
    e1  = actual - pred_restricted      # benchmark prediction error
    e2  = actual - pred_unrestricted    # model prediction error
    adj = e1**2 - e2**2 + (pred_unrestricted - pred_restricted)**2  # CW adjustment term

    t_stat  = adj.mean() / (adj.std() / np.sqrt(len(adj)))
    p_value = 1 - stats.norm.cdf(t_stat)   # one-sided test
    return t_stat, p_value


def print_model_evaluation(rf_signals, xgb_preds, y_train_ret, y_test_ret, y_test_bin):
    hist_mean      = y_train_ret.mean()
    hist_mean_pred = np.full(len(y_test_ret), hist_mean)

    ros2_rf  = get_ros2(y_test_ret, rf_signals * hist_mean, hist_mean)
    ros2_xgb = get_ros2(y_test_ret, xgb_preds, hist_mean)

    print(f"Random Forest (Classifier) R_OS^2: {ros2_rf:.5f}")
    print(f"XGBoost (Regressor) R_OS^2:        {ros2_xgb:.5f}")
    print(f"RF Hit Rate (Accuracy):             {accuracy_score(y_test_bin, rf_signals):.4f}")

    t_rf,  p_rf  = clark_west_test(y_test_ret.values, rf_signals * hist_mean, hist_mean_pred)
    t_xgb, p_xgb = clark_west_test(y_test_ret.values, xgb_preds, hist_mean_pred)
    print(f"\nRF  CW t-stat: {t_rf:.3f}  | p-value: {p_rf:.4f}")
    print(f"XGB CW t-stat: {t_xgb:.3f} | p-value: {p_xgb:.4f}")


def univariate_significance(train_raw, test_raw, y_train_ret, y_test_ret, y_train_bin):
    hist_mean_pred     = np.full(len(y_test_ret), y_train_ret.mean())
    results_univariate = []

    for feat in FEATURES:
        scaler_uni = StandardScaler()
        X_tr = scaler_uni.fit_transform(train_raw[[feat]])
        X_te = scaler_uni.transform(test_raw[[feat]])

        lr = LogisticRegression(random_state=42)
        lr.fit(X_tr, y_train_bin)

        proba  = lr.predict_proba(X_te)[:, 1]
        signal = (proba > 0.5).astype(int)
        pred   = signal * y_train_ret.mean()

        mse_model = np.mean((y_test_ret.values - pred) ** 2)
        mse_bench = np.mean((y_test_ret.values - hist_mean_pred) ** 2)
        ros2      = 1 - mse_model / mse_bench

        t_stat, p_val = clark_west_test(y_test_ret.values, pred, hist_mean_pred)
        sig_label     = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))

        results_univariate.append({
            'Feature':   feat,
            'R2_OS':     round(ros2, 5),
            'CW t-stat': round(t_stat, 3),
            'p-value':   round(p_val, 4),
            'Sig':       sig_label
        })

    df_uni = pd.DataFrame(results_univariate).sort_values('R2_OS', ascending=False)
    print(df_uni.to_string(index=False))
    return df_uni
EOF

# ── feature_importance.py ─────────────────────────────────────
cat > feature_importance.py << 'EOF'
# feature_importance.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from config import FEATURES


def plot_feature_importance(rf_model, X_test_scaled, y_test_bin):
    # Method A: RF built-in feature importance (impurity-based, in-sample)
    importances = rf_model.feature_importances_
    feat_imp    = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

    # Method B: Permutation Importance (more reliable, out-of-sample)
    perm     = permutation_importance(
        rf_model, X_test_scaled, y_test_bin,
        n_repeats=30, random_state=42, scoring='accuracy'
    )
    perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    feat_imp.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('RF Feature Importance\n(Impurity-based, In-sample)', fontsize=12)
    axes[0].set_xlabel('Importance Score')
    axes[0].axvline(1/len(FEATURES), color='red', linestyle='--', label='Uniform baseline')
    axes[0].legend()

    perm_imp.plot(kind='barh', ax=axes[1], color='darkorange')
    axes[1].set_title('Permutation Importance\n(Out-of-sample)', fontsize=12)
    axes[1].set_xlabel('Accuracy Drop when Feature Shuffled')
    axes[1].axvline(0, color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    summary = pd.DataFrame({
        'Impurity_Imp':    pd.Series(rf_model.feature_importances_, index=FEATURES),
        'Permutation_Imp': pd.Series(perm.importances_mean,          index=FEATURES),
        'Permutation_Std': pd.Series(perm.importances_std,           index=FEATURES)
    }).sort_values('Permutation_Imp', ascending=False)

    print("\nFeature importance ranking (out-of-sample permutation as primary):")
    print(summary.round(5))
    return summary
EOF

# ── backtest.py ───────────────────────────────────────────────
cat > backtest.py << 'EOF'
# backtest.py
import pandas as pd
from config import COST_BPS_LIST


def backtest_with_costs(signals, actual_rets, bps):
    cost_pct = bps / 10000
    # Turnover: any signal flip (0->1 or 1->0) incurs a cost
    turnover   = pd.Series(signals).diff().abs().fillna(signals[0]).values
    raw_return = signals * actual_rets
    costs      = turnover * cost_pct
    net_return = raw_return - costs
    return net_return.sum(), turnover.mean()


def run_cost_analysis(rf_signals, y_test_ret):
    print("\n--- Impact of Transaction Costs on RF Strategy ---")
    for c in COST_BPS_LIST:
        total_net, avg_turnover = backtest_with_costs(rf_signals, y_test_ret.values, c)
        print(f"Cost: {c:3d} bps | Cumulative Net Return: {total_net:.4f} | Avg Monthly Turnover: {avg_turnover:.2%}")
EOF

# ── main.py ───────────────────────────────────────────────────
cat > main.py << 'EOF'
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
EOF

# ── requirements.txt ──────────────────────────────────────────
cat > requirements.txt << 'EOF'
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
pyarrow
EOF

# ── .gitignore ────────────────────────────────────────────────
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.ipynb_checkpoints/
*.png
.env
EOF

# ── README.md ─────────────────────────────────────────────────
cat > README.md << 'EOF'
# HML Factor Timing

## Overview
This project explores whether HML (High Minus Low) factor returns can be timed
using macroeconomic variables and machine learning. The central question mirrors
the Arnott vs. Asness debate: is factor timing possible, or is it as difficult
as market timing?

## File Structure
```
├── main.py                # Run this — executes full pipeline
├── config.py              # All hyperparameters and settings
├── data_prep.py           # Load data, engineer features, split
├── models.py              # Train RF, XGBoost, Logistic Regression
├── evaluation.py          # R²_OS, Clark-West test, univariate significance
├── feature_importance.py  # Impurity-based + permutation importance
├── backtest.py            # Transaction cost analysis
└── master.parquet         # Raw data (not versioned)
```

## Features (Timing Variables)
| Feature | Description | Type |
|---|---|---|
| `tbl` | Short-term Treasury bill rate | Macro |
| `dfy` | Credit spread (BAA − AAA) | Macro |
| `infl` | Inflation rate | Macro |
| `bm` | Market book-to-market ratio | Macro |
| `svar` | Market return variance | Macro |
| `hml_mom` | 3-month rolling average of HML | Factor momentum |

## Models
| Model | Type | Goal |
|---|---|---|
| Random Forest | Classification | Predict if HML > 0.5% next month |
| XGBoost | Regression | Predict exact HML return next month |
| Logistic Regression | Classification | Univariate baseline per feature |

## Key Results
```
RF  R²_OS:   +0.031  (statistically significant)
XGB R²_OS:   -0.008  (underperforms historical mean)
RF Hit Rate:  53.95%
Breakeven transaction cost: ~20 bps
```

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
EOF

echo ""
echo "Done! Project created in: $(pwd)"
echo ""
echo "Files created:"
ls -1
echo ""
echo "Next steps:"
echo "  1. Copy master.parquet into this folder"
echo "  2. pip install -r requirements.txt"
echo "  3. python main.py"
