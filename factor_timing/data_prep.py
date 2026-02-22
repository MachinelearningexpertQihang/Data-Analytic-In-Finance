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
