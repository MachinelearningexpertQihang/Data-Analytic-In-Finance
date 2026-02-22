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
