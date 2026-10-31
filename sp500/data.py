"""Adjusted Yahoo candles, current constituents and NYSE bar calendars."""
from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
INTERVALS = {"1d": None, "1h": 60, "60m": 60, "30m": 30, "15m": 15,
             "5m": 5, "2m": 2, "1m": 1}
NY = "America/New_York"


def normalize_ticker(ticker: str) -> str:
    """Convert exchange symbols to Yahoo symbols and reject unsafe filenames."""
    ticker = ticker.strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z0-9^-]+", ticker):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return ticker


def constituents(cache_dir: str | Path = "data/cache", refresh: bool = False) -> pd.DataFrame:
    """Fetch/cache the current Wikipedia universe (not historical membership)."""
    path = Path(cache_dir) / "sp500_constituents.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Kronos-SP500/1.0"}, timeout=30)
    response.raise_for_status()
    table = next(table for table in pd.read_html(StringIO(response.text))
                 if "Symbol" in table and "Security" in table)
    table = table.rename(columns={"Symbol": "ticker", "Security": "name",
                                  "GICS Sector": "sector"})
    table["ticker"] = table.ticker.map(normalize_ticker)
    table["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def market_bars(start: str | pd.Timestamp, end: str | pd.Timestamp,
                interval: str = "1d") -> pd.DatetimeIndex:
    """Return NY-local naive bar starts, with end exclusive, including early closes."""
    if interval not in INTERVALS:
        raise ValueError(f"Supported intervals: {', '.join(INTERVALS)}")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        return pd.DatetimeIndex([])
    schedule = mcal.get_calendar("NYSE").schedule(start.date(), end.date())
    if interval == "1d":
        result = pd.DatetimeIndex(schedule.index).normalize()
    else:
        stamps = []
        for row in schedule.itertuples():
            stamps.extend(pd.date_range(row.market_open, row.market_close,
                                        freq=f"{INTERVALS[interval]}min", inclusive="left"))
        result = (pd.DatetimeIndex(stamps).tz_convert(NY).tz_localize(None)
                  if stamps else pd.DatetimeIndex([]))
    return result[(result >= start) & (result < end)]


def future_timestamps(last: pd.Timestamp, horizon: int, interval: str = "1d") -> pd.Series:
    """Produce exactly horizon exchange bars strictly after the final input bar."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    last = pd.Timestamp(last)
    if last.tzinfo is not None:
        last = last.tz_convert(NY).tz_localize(None)
    days = max(20, horizon * 3)
    bars = market_bars(last.normalize(), last.normalize() + pd.Timedelta(days=days), interval)
    bars = bars[bars > last]
    if len(bars) < horizon:
        raise ValueError("Not enough exchange sessions for the requested horizon")
    return pd.Series(bars[:horizon], name="timestamps")


def normalize_candles(raw: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """Standardize Yahoo single-symbol OHLCV and estimate traded dollar amount."""
    raw = raw.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = raw.columns.str.lower()
    required = FEATURES[:5]
    if not set(required).issubset(raw.columns):
        raise ValueError("Yahoo response is missing OHLCV columns")
    frame = raw[required].apply(pd.to_numeric, errors="coerce")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert(NY).tz_localize(None)
    frame.index = index.normalize() if interval == "1d" else index
    frame.index.name = "timestamps"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[(frame[required[:4]] > 0).all(axis=1) & (frame.volume >= 0)]
    frame["amount"] = (frame.high + frame.low + frame.close) / 3 * frame.volume
    return frame[FEATURES]


class MarketData:
    """Incremental per-symbol Parquet cache with explicit fetched-range metadata."""

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        """Create a cache rooted at cache_dir."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, ticker: str, start: str | pd.Timestamp, end: str | pd.Timestamp,
              interval: str = "1d", refresh: bool = False, lookback: int = 0) -> pd.DataFrame:
        """Get completed adjusted candles in [start,end); download only uncovered ranges."""
        ticker = normalize_ticker(ticker)
        if interval not in INTERVALS:
            raise ValueError(f"Supported intervals: {', '.join(INTERVALS)}")
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        now = pd.Timestamp.now(tz=NY)
        end = min(end, now.tz_localize(None).normalize() + pd.Timedelta(days=1))
        if end <= start:
            raise ValueError("end must be later than start")
        path = self.cache_dir / f"{ticker}_{interval}.parquet"
        meta = path.with_suffix(".json")
        frame = (pd.read_parquet(path) if path.exists() and not refresh else
             pd.DataFrame(columns=FEATURES, index=pd.DatetimeIndex([], name="timestamps")))
        ranges = json.loads(meta.read_text()).get("ranges", []) if meta.exists() and path.exists() and not refresh else []
        uncovered = [(start, end)]
        for lower, upper in ranges:
            lower, upper = pd.Timestamp(lower), pd.Timestamp(upper)
            remaining = []
            for left, right in uncovered:
                if upper <= left or lower >= right:
                    remaining.append((left, right))
                else:
                    if left < lower:
                        remaining.append((left, lower))
                    if upper < right:
                        remaining.append((upper, right))
            uncovered = remaining
        for left, right in uncovered:
            if interval != "1d":
                limit = 7 if interval == "1m" else 730 if interval in ("1h", "60m") else 60
                if left < now.tz_localize(None).normalize() - pd.Timedelta(days=limit - 1):
                    raise ValueError(f"Yahoo {interval} history must start within the last {limit - 1} days; use 1d or cached bars")
            raw = yf.download(ticker, start=left.strftime("%Y-%m-%d"), end=right.strftime("%Y-%m-%d"),
                              interval=interval, auto_adjust=True, progress=False, threads=False,
                              timeout=30, keepna=False)
            if raw.empty:
                if len(market_bars(left, right, interval)):
                    raise RuntimeError(f"Yahoo returned no data for {ticker} {left.date()}:{right.date()}; retry later")
                normalized = pd.DataFrame(columns=FEATURES, index=pd.DatetimeIndex([], name="timestamps"))
            else:
                normalized = normalize_candles(raw, interval)
                if normalized.empty and len(market_bars(left, right, interval)):
                    raise RuntimeError(f"Yahoo returned no valid OHLCV bars for {ticker}; cache coverage not updated")
            frame = pd.concat([frame, normalized]) if not frame.empty else normalized
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            covered_end = min(right, now.tz_localize(None).normalize())
            if covered_end > left:
                ranges.append([left.isoformat(), covered_end.isoformat()])
        if not frame.empty:
            schedule = mcal.get_calendar("NYSE").schedule(frame.index[0].date(), frame.index[-1].date())
            if interval == "1d":
                complete = pd.DatetimeIndex(schedule.index[schedule.market_close <= now.tz_convert("UTC")])
                frame = frame[frame.index.isin(complete)]
            else:
                frame = frame[frame.index + pd.Timedelta(minutes=INTERVALS[interval]) <= now.tz_localize(None)]
            temporary = path.with_suffix(".tmp.parquet")
            frame.to_parquet(temporary)
            temporary.replace(path)
        else:
            frame.to_parquet(path)
        meta.write_text(json.dumps({"ticker": ticker, "interval": interval, "auto_adjust": True,
                                    "ranges": ranges, "updated_at": now.isoformat()}, indent=2))
        result = frame.loc[(frame.index >= start) & (frame.index < end)].copy()
        if len(result) < lookback:
            warnings.warn(f"{ticker}: only {len(result)} bars, need {lookback}", stacklevel=2)
        if not result.empty:
            expected = market_bars(result.index[0], result.index[-1] + pd.Timedelta(minutes=1), interval)
            missing = expected.difference(result.index)
            if len(missing):
                warnings.warn(f"{ticker}: {len(missing)} missing NYSE bars; no prices imputed", stacklevel=2)
        return result