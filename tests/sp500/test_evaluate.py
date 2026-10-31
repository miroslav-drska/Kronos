"""Exact metrics and leakage/missing-bar/strategy checks."""
import numpy as np
import pandas as pd
import pytest

from sp500.data import FEATURES, market_bars
from sp500.evaluate import baselines, equity_metrics, metrics, summarize, top_k_strategy, walk_forward
from sp500.forecast import ForecastConfig, Forecaster


def test_exact_metrics_baselines():
    result = metrics(np.array([100., 110.]), np.array([90., 120.]), np.array([100., 100.]),
                     np.array([95., 105.]), np.array([105., 115.]))
    assert result["MAE"] == 10 and result["RMSE"] == 10
    assert result["MAPE_pct"] == pytest.approx(9.5454545)
    assert result["directional_accuracy_pct"] == 50
    assert result["coverage_pct"] == 100
    output = baselines(pd.DataFrame({"close": [10., 12., 14.]}), 2, 2)
    np.testing.assert_equal(output["naive"], [14, 14])
    np.testing.assert_equal(output["drift"], [16, 18])
    np.testing.assert_equal(output["sma"], [13, 13])


class SentinelPredictor:
    def predict_batch(self, frames, past, future, horizon, **kwargs):
        for frame, before, after in zip(frames, past, future):
            assert before.max() < after.min()
            assert (frame.close < 1000).all()
        return [pd.DataFrame({feature: 11. for feature in FEATURES}, index=after) for after in future]


def test_walk_forward_no_leakage_and_missing_targets():
    index = market_bars("2023-12-01", "2024-01-10")
    data = pd.DataFrame({feature: 10. for feature in FEATURES}, index=index)
    data.loc[data.index >= "2024-01-02", "close"] = 1000.
    forecaster = Forecaster(ForecastConfig(lookback=5, horizon=3, samples=1, device="cpu"), SentinelPredictor())
    scores, skipped = walk_forward({"AAPL": data}, forecaster, "2024-01-02", "2024-01-05")
    assert len(scores) == 12 and skipped.empty
    assert scores.actual.eq(1000).all()
    assert set(summarize(scores).method) == {"kronos", "naive", "drift", "sma"}
    with pytest.raises(ValueError, match="No complete"):
        walk_forward({"AAPL": data.drop(pd.Timestamp("2024-01-03"))}, forecaster, "2024-01-02", "2024-01-05")


def test_strategy_uses_next_open_and_roundtrip_costs():
    scored = pd.DataFrame([{"ticker": ticker, "method": "kronos", "step": 2,
                            "first_target": pd.Timestamp("2024-01-02"), "target": pd.Timestamp("2024-01-03"),
                            "origin_close": 100., "entry_open": 120., "actual": actual, "predicted": predicted}
                           for ticker, actual, predicted in [("AAPL", 132., 150.), ("MSFT", 120., 130.)]])
    curve, report = top_k_strategy(scored, ["AAPL", "MSFT"], 2, 1, 10)
    assert curve.strategy_equity.iloc[0] == pytest.approx(1.1 * .999 / 1.001)
    assert curve.buy_hold_equity.iloc[0] == pytest.approx(1.05 * .999 / 1.001)
    assert len(report) == 2
    assert equity_metrics(np.array([1.1, .99]), 252)["max_drawdown_pct"] == pytest.approx(-10)