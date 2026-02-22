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
