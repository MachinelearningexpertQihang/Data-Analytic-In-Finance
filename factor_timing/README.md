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
