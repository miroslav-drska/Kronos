"""Sampling, batch alignment, artifact and CLI tests without weights."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from sp500.__main__ import build_parser, run_forecast
from sp500.data import FEATURES, market_bars
from sp500.forecast import ForecastConfig, Forecaster, save_forecast


def history(value=100.):
    return pd.DataFrame({feature: value for feature in FEATURES},
                        index=market_bars("2024-01-02", "2024-07-01"))


class Predictor:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, frames, past, future, horizon, **kwargs):
        assert kwargs["sample_count"] == 1
        self.calls += 1
        assert len({len(frame) for frame in frames}) == 1
        return [pd.DataFrame({feature: frame.close.iloc[-1] + self.calls for feature in FEATURES},
                             index=stamps) for frame, stamps in zip(frames, future)]


def test_independent_paths_alignment_and_quantiles(tmp_path):
    predictor = Predictor()
    config = ForecastConfig(lookback=10, horizon=2, samples=3, device="cpu")
    output = Forecaster(config, predictor).forecast_batch({"AAPL": history(), "MSFT": history(200)})
    assert predictor.calls == 3
    assert output["AAPL"].close.iloc[0] == 102
    assert output["MSFT"].close.iloc[0] == 202
    assert output["AAPL"].close_p10.iloc[0] == pytest.approx(101.2)
    assert output["AAPL"].close_p90.iloc[0] == pytest.approx(102.8)
    assert output["AAPL"].index[0] == pd.Timestamp("2024-07-01")
    save_forecast(tmp_path, "AAPL", history(), output["AAPL"], config)
    assert {path.suffix for path in tmp_path.iterdir()} == {".csv", ".json", ".png"}
    assert json.loads((tmp_path / "AAPL.json").read_text())["config"] == asdict(config)


def test_parser_and_invalid_config():
    args = build_parser().parse_args(["forecast", "--tickers", "AAPL", "MSFT", "--horizon", "5"])
    assert args.tickers == ["AAPL", "MSFT"] and args.horizon == 5
    with pytest.raises(ValueError):
        ForecastConfig(horizon=513)
    with pytest.raises(ValueError):
        Forecaster(ForecastConfig(lookback=400, device="cpu"), Predictor()).forecast(history())


def test_resume_does_not_reload_model(tmp_path, monkeypatch):
    monkeypatch.setattr("sp500.__main__.MarketData.fetch", lambda *args, **kwargs: history())
    monkeypatch.setattr("sp500.__main__.Forecaster", lambda config: Forecaster(config, Predictor()))
    argv = ["forecast", "--tickers", "AAPL", "MSFT", "--lookback", "10", "--samples", "2",
            "--device", "cpu", "--output", str(tmp_path)]
    run_forecast(build_parser().parse_args(argv))
    def forbidden(config):
        raise AssertionError("Completed run must not load model")
    monkeypatch.setattr("sp500.__main__.Forecaster", forbidden)
    run_forecast(build_parser().parse_args(argv + ["--resume", str(tmp_path)]))
    assert len(json.loads((tmp_path / "manifest.json").read_text())["completed"]) == 2