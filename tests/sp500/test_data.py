"""Offline data and calendar contract tests."""
import pandas as pd
import pytest

from sp500.data import MarketData, future_timestamps, market_bars, normalize_candles, normalize_ticker


def test_symbols_and_amount():
    assert normalize_ticker("BRK.B") == "BRK-B"
    with pytest.raises(ValueError):
        normalize_ticker("../AAPL")
    raw = pd.DataFrame({"Open": [10, 11], "High": [12, 13], "Low": [9, 10],
                        "Close": [11, 12], "Volume": [3, 6]},
                       index=pd.to_datetime(["2024-01-02", "2024-01-02"]))
    result = normalize_candles(raw)
    assert len(result) == 1
    assert result.amount.iloc[0] == 70


def test_holidays_early_close():
    stamps = future_timestamps(pd.Timestamp("2024-07-03"), 2)
    assert stamps.tolist() == list(pd.to_datetime(["2024-07-05", "2024-07-08"]))
    bars = market_bars("2024-11-29", "2024-11-30", "1h")
    assert len(bars) == 4
    assert bars[-1] == pd.Timestamp("2024-11-29 12:30")
    future = future_timestamps(bars[-1], 1, "1h")
    assert future.iloc[0] == pd.Timestamp("2024-12-02 09:30")


def test_incremental_cache_and_refresh(tmp_path, monkeypatch):
    calls = []
    def download(ticker, start, end, **kwargs):
        calls.append((start, end))
        stamps = market_bars(start, end)
        return pd.DataFrame({"Open": 10, "High": 12, "Low": 9, "Close": 11,
                             "Volume": 3}, index=stamps)
    monkeypatch.setattr("sp500.data.yf.download", download)
    cache = MarketData(tmp_path)
    assert len(cache.fetch("AAPL", "2024-01-02", "2024-01-06")) == 4
    cache.fetch("AAPL", "2024-01-02", "2024-01-06")
    assert len(calls) == 1
    cache.fetch("AAPL", "2024-01-02", "2024-01-10")
    assert calls[-1] == ("2024-01-06", "2024-01-10")
    cache.fetch("AAPL", "2024-01-02", "2024-01-10", refresh=True)
    assert calls[-1] == ("2024-01-02", "2024-01-10")
    cache.fetch("AAPL", "2024-01-02", "2024-01-06")
    assert pd.read_parquet(tmp_path / "AAPL_1d.parquet").index[-1] == pd.Timestamp("2024-01-09")


def test_failed_download_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("sp500.data.yf.download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        MarketData(tmp_path).fetch("AAPL", "2024-01-02", "2024-01-06")
    assert not list(tmp_path.glob("*.json"))