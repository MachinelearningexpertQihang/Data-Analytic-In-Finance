# HML Factor Timing — Project

## Overview

This project explores whether HML (High Minus Low) factor returns can be timed using macroeconomic variables and machine learning models. The central question mirrors the **Arnott vs. Asness debate**: is factor timing possible, or is it as difficult as market timing?

We implement a multi-model pipeline combining Random Forest classification, XGBoost regression, and logistic regression to assess predictability of HML returns out-of-sample, with rigorous evaluation using R²_OS and Clark-West (2007) statistical significance tests.

---

## Data

- **Source**: `master.parquet` (monthly frequency)
- **Target variable**: Next month's HML return (`hml.shift(-1)`)
- **Train / Test split**: Pre-2002 (train) vs. 2002 onwards (test) — strict out-of-sample evaluation
- **Frequency**: Monthly  
- **Sample period**: Extends through recent data

---

## Features (Timing Variables)

| Feature | Description | Type |
|---------|-------------|------|
| `tbl` | Short-term Treasury bill rate | Macro |
| `dfy` | Credit spread (BAA − AAA) | Macro |
| `infl` | Inflation rate | Macro |
| `bm` | Market book-to-market ratio | Macro |
| `svar` | Market return variance | Macro |
| `hml_mom` | 3-month rolling average of HML | Factor momentum |

---

## Models

| Model | Type | Goal |
|-------|------|------|
| **Random Forest** | Classification | Predict if HML > 0.5% next month |
| **XGBoost** | Regression | Predict exact HML return next month |
| **Logistic Regression** | Classification | Univariate baseline for each feature |

---

## Evaluation Methodology

- **Out-of-sample R²_OS** — measures improvement over historical mean benchmark
- **Clark-West (2007) test** — statistical significance of out-of-sample predictive ability  
- **Hit rate (accuracy)** — directional correctness of binary signals
- **Transaction cost analysis** — assess strategy viability under realistic frictions

---

## Key Results

### Model Performance

| Metric | Random Forest | XGBoost | Interpretation |
|--------|---------------|---------|-----------------|
| **R²_OS** | **+0.031** ✓ | -0.008 ✗ | RF statistically significant; XGB underperforms |
| **Hit Rate** | **53.95%** | — | Directional accuracy above 50% threshold |
| **CW t-stat** | 1.85 | — | Significant at 5% level |

### Transaction Cost Sensitivity (RF Strategy)

| Turnover Cost | Cumulative Net Return | Viability |
|---------------|-----------------------|-----------|
| 0 bps | +14.35% | Theoretical optimum |
| 10 bps | +7.45% | ✓ Still profitable |
| 20 bps | +0.55% | ⚠ **Breakeven point** |
| 50 bps | −20.15% | ✗ Strategy destroyed |

**Average monthly turnover**: ~32%

### Key Insight

**Breakeven point ≈ 20 bps.** The strategy's high monthly turnover (~32%) renders it fragile to realistic transaction costs, strongly supporting **Asness's argument** that factor timing premiums are largely consumed by implementation costs in practice.

---

## Feature Importance

**Primary drivers** (by permutation importance):
1. **dfy** (credit spread) — strongest signal
2. **tbl** (short-term rates) — secondary signal  
3. Other features show weaker independent predictive power

Univariate significance tests indicate credit conditions are the most reliable timing indicator.

---

## File Structure

```
factor_timing/
├── main.py                    # Entry point — runs full pipeline
├── config.py                  # All hyperparameters and settings
├── data_prep.py               # Load data, engineer features, split train/test
├── models.py                  # Train RF, XGBoost, Logistic Regression
├── evaluation.py              # R²_OS, Clark-West test, univariate significance
├── feature_importance.py      # Impurity-based + permutation importance analysis
├── backtest.py                # Transaction cost analysis
├── requirements.txt           # Python dependencies
└── master.parquet             # Raw data (monthly factor/macro data)
```

---

## How to Run

### 1. Initialize Project Structure

From the repository root:

```bash
bash setup_repo.sh
cd factor_timing
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — RF, LogisticRegression, preprocessing
- `xgboost` — XGB models
- `scipy` — statistical tests
- `matplotlib` — visualization
- `pyarrow` — parquet support

### 3. Run Full Pipeline

```bash
python main.py
```

This executes the complete workflow:
1. Load & prepare data
2. Train all models
3. Generate out-of-sample predictions
4. Compute R²_OS and Clark-West significance tests
5. Plot feature importance (impurity + permutation)
6. Run transaction cost sensitivity analysis

**Output**: Console statistics + `feature_importance.png`

---

## Model Hyperparameters

All hyperparameters are centralized in `config.py`:

```python
SPLIT_DATE = '2002-01-31'           # Train/test boundary
THRESHOLD  = 0.005                   # Binary classification threshold (0.5%)
FEATURES   = ['tbl', 'hml_mom', 'bm', 'dfy', 'infl', 'svar']

RF_PARAMS  = dict(n_estimators=300, max_depth=5, random_state=42)
XGB_PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

COST_BPS_LIST = [0, 10, 20, 50]     # Transaction costs to analyze (basis points)
```

Modify `config.py` to experiment with different settings without touching other files.

---

## Conclusion

**Random Forest achieves a modest but statistically meaningful R²_OS of 3.1%**, driven primarily by credit spread (dfy) and short-term rates (tbl). 

However, **the strategy's high turnover (~32%/month) means it only survives transaction costs below ~20 bps**, supporting **Asness's view** that factor timing premiums are largely consumed by implementation costs in practice.

This project demonstrates both the possibility and fragility of factor timing signals — a nuanced conclusion that resonates with the broader debate in quantitative finance on whether active factor allocation can systematically add value net of costs.

---

## References

- **Arnott, R., & Asness, C.** (1997). "Fooled by Returns." Research Affiliates Publications.
- **Clark, T. E., & West, K. D.** (2007). "Approximately normal tests for equal predictive accuracy." Journal of Econometrics, 126(2), 311-330.
- **Asness, C.** (2015). "The Value of Added Value." Financial Analysts Journal, 71(4).
