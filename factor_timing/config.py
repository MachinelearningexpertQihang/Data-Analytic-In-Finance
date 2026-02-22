# config.py
# Central place for all hyperparameters and settings
# Change values here — no need to touch other files

SPLIT_DATE = '2002-01-31'
THRESHOLD  = 0.005
FEATURES   = ['tbl', 'hml_mom', 'bm', 'dfy', 'infl', 'svar']

RF_PARAMS  = dict(n_estimators=300, max_depth=5, random_state=42)
XGB_PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

COST_BPS_LIST = [0, 10, 20, 50]
