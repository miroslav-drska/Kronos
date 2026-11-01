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


def test_timezone_multiindex_and_nonfinite_rows():
    raw = pd.DataFrame({("Open", "AAPL"): [10, float("inf")], ("High", "AAPL"): [12, 12],
                        ("Low", "AAPL"): [9, 9], ("Close", "AAPL"): [11, 11],
                        ("Volume", "AAPL"): [3, 3]},
                       index=pd.date_range("2024-03-11 13:30", periods=2, freq="h", tz="UTC"))
    frame = normalize_candles(raw, "1h")
    assert len(frame) == 1 and frame.index[0] == pd.Timestamp("2024-03-11 09:30")


def test_constituent_mapping_cache_and_fetch_date(tmp_path, monkeypatch):
    from sp500.data import constituents
    class Response:
        text = '<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr><tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td></tr></table>'
        def raise_for_status(self):
            pass
    calls = []
    def fetch(*args, **kwargs):
        calls.append(1)
        return Response()
    monkeypatch.setattr("sp500.data.requests.get", fetch)
    assert constituents(tmp_path).ticker.tolist() == ["BRK-B"]
    assert "fetched_at" in constituents(tmp_path)
    assert len(calls) == 1
    constituents(tmp_path, refresh=True)
    assert len(calls) == 2


def test_intraday_limits_and_deleted_cache(tmp_path, monkeypatch):
    calls = []
    def download(*args, start, end, **kwargs):
        calls.append(1)
        return pd.DataFrame({"Open": 10, "High": 12, "Low": 9, "Close": 11,
                             "Volume": 3}, index=market_bars(start, end))
    monkeypatch.setattr("sp500.data.yf.download", download)
    cache = MarketData(tmp_path)
    with pytest.raises(ValueError, match="history must start"):
        cache.fetch("AAPL", "2000-01-01", "2000-01-05", "1h")
    cache.fetch("AAPL", "2024-01-02", "2024-01-06")
    (tmp_path / "AAPL_1d.parquet").unlink()
    cache.fetch("AAPL", "2024-01-02", "2024-01-06")
    assert len(calls) == 2


def test_holiday_only_fetch_is_empty_and_cached(tmp_path, monkeypatch):
    calls = []
    def download(*args, **kwargs):
        calls.append(1)
        return pd.DataFrame()
    monkeypatch.setattr("sp500.data.yf.download", download)
    cache = MarketData(tmp_path)
    assert cache.fetch("AAPL", "2024-07-04", "2024-07-05").empty
    assert cache.fetch("AAPL", "2024-07-04", "2024-07-05").empty
    assert len(calls) == 0


def test_cached_history_with_weekend_gap_skips_yahoo(tmp_path, monkeypatch):
    calls = []
    def download(ticker, start, end, **kwargs):
        stamps = market_bars(start, end)
        assert not stamps.empty, "Closed-market ranges must not contact Yahoo"
        calls.append((start, end))
        return pd.DataFrame({"Open": 10, "High": 12, "Low": 9, "Close": 11,
                             "Volume": 3}, index=stamps)
    monkeypatch.setattr("sp500.data.yf.download", download)
    cache = MarketData(tmp_path)
    original = cache.fetch("GOOG", "2024-07-05", "2024-07-06")
    extended = cache.fetch("GOOG", "2024-07-05", "2024-07-08")
    pd.testing.assert_frame_equal(original, extended)
    pd.testing.assert_frame_equal(extended, cache.fetch("GOOG", "2024-07-05", "2024-07-08"))
    assert calls == [("2024-07-05", "2024-07-06")]