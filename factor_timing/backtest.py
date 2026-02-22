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
