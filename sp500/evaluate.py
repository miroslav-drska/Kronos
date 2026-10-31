"""Strict walk-forward scoring and a next-open, cost-aware ranking experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sp500.data import MarketData, market_bars
from sp500.forecast import ForecastConfig, Forecaster, write_json


def metrics(actual: np.ndarray, predicted: np.ndarray, origin_close: np.ndarray,
            lower: np.ndarray | None = None, upper: np.ndarray | None = None) -> dict[str, float | int | None]:
    """Score price errors, origin-relative direction and optional sampled-band coverage."""
    actual, predicted, origin_close = map(np.asarray, (actual, predicted, origin_close))
    if actual.shape != predicted.shape or actual.shape != origin_close.shape:
        raise ValueError("Metric array shapes must match")
    if actual.size == 0 or np.any(actual <= 0) or not all(np.isfinite(values).all()
                                                        for values in (actual, predicted, origin_close)):
        raise ValueError("Metrics require nonempty finite arrays and positive actual prices")
    errors = predicted - actual
    return {"n": int(actual.size), "MAE": float(np.mean(np.abs(errors))),
            "MAPE_pct": float(np.mean(np.abs(errors / actual)) * 100),
            "RMSE": float(np.sqrt(np.mean(errors**2))),
            "directional_accuracy_pct": float(np.mean(np.sign(predicted - origin_close) ==
                                                       np.sign(actual - origin_close)) * 100),
            "coverage_pct": float(np.mean((actual >= lower) & (actual <= upper)) * 100)
            if lower is not None and upper is not None else None}


def baselines(history: pd.DataFrame, horizon: int, sma_window: int = 20) -> dict[str, np.ndarray]:
    """Predict using only the input history: last close, random-walk drift and SMA."""
    close = history.close.to_numpy()
    if len(close) < 2 or sma_window < 1:
        raise ValueError("Baselines require two history bars and positive SMA window")
    return {"naive": np.repeat(close[-1], horizon),
            "drift": close[-1] + np.arange(1, horizon + 1) * (close[-1] - close[0]) / (len(close) - 1),
            "sma": np.repeat(np.mean(close[-sma_window:]), horizon)}


def walk_forward(histories: dict[str, pd.DataFrame], forecaster: Forecaster,
                 start: str, end: str, stride: int | None = None, batch_size: int = 8,
                 sma_window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast pre-target windows; discard missing inputs/targets rather than shift dates."""
    config = forecaster.config
    stride = config.horizon if stride is None else stride
    if stride < 1 or batch_size < 1 or sma_window < 1:
        raise ValueError("stride, batch-size and sma-window must be positive")
    grid = market_bars(start, end, config.interval)
    jobs: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    skipped = []
    for position in range(0, max(0, len(grid) - config.horizon + 1), stride):
        target_index = grid[position:position + config.horizon]
        earlier = market_bars(target_index[0] - pd.Timedelta(days=config.lookback * 3),
                              target_index[0], config.interval)[-config.lookback:]
        for ticker, data in histories.items():
            key = f"{ticker}|{target_index[0].isoformat()}"
            if len(earlier) < config.lookback or not earlier.isin(data.index).all() or not target_index.isin(data.index).all():
                skipped.append({"ticker": ticker, "first_target": str(target_index[0]),
                                "reason": "missing exchange input/target bars or insufficient history"})
                continue
            history = data.loc[earlier].copy()
            actual = data.loc[target_index].copy()
            if history.index[-1] >= actual.index[0]:
                raise AssertionError("Walk-forward history overlaps target")
            jobs.append((key, ticker, history, actual))
    rows = []
    for offset in range(0, len(jobs), batch_size):
        chunk = jobs[offset:offset + batch_size]
        predictions = forecaster.forecast_batch({key: history for key, _, history, _ in chunk},
                                                seed=config.seed + offset * 100003)
        for key, ticker, history, actual in chunk:
            prediction = predictions[key]
            if not prediction.index.equals(actual.index):
                raise RuntimeError(f"Forecast/actual timestamp mismatch: {key}")
            paths = {"kronos": prediction.close.to_numpy(), **baselines(history, config.horizon, sma_window)}
            for method, close in paths.items():
                for step, stamp in enumerate(actual.index):
                    rows.append({"ticker": ticker, "method": method, "origin": history.index[-1],
                                 "first_target": actual.index[0], "target": stamp, "step": step + 1,
                                 "origin_close": float(history.close.iloc[-1]), "actual": float(actual.close.iloc[step]),
                                 "entry_open": float(actual.open.iloc[0]), "predicted": float(close[step]),
                                 "lower": float(prediction.close_p10.iloc[step]) if method == "kronos" else np.nan,
                                 "upper": float(prediction.close_p90.iloc[step]) if method == "kronos" else np.nan})
        print(f"Evaluation progress {min(offset + batch_size, len(jobs))}/{len(jobs)} windows", flush=True)
    if not rows:
        raise ValueError("No complete evaluation windows; expand cached history or reduce lookback/horizon")
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """Return per-ticker and pooled same-window metrics for all four predictors."""
    rows = []
    for ticker, subset in [("ALL", scored), *list(scored.groupby("ticker", sort=False))]:
        for method, group in subset.groupby("method", sort=False):
            values = metrics(group.actual.to_numpy(), group.predicted.to_numpy(), group.origin_close.to_numpy(),
                             group.lower.to_numpy() if method == "kronos" else None,
                             group.upper.to_numpy() if method == "kronos" else None)
            rows.append({"ticker": ticker, "method": method, **values})
    return pd.DataFrame(rows)


def equity_metrics(equity: np.ndarray, periods_per_year: float) -> dict[str, float]:
    """Compute cumulative return, periodic-return Sharpe (zero risk-free) and drawdown."""
    equity = np.r_[1., np.asarray(equity, dtype=float)]
    returns = equity[1:] / equity[:-1] - 1
    deviation = returns.std(ddof=1) if len(returns) > 1 else 0.
    return {"cumulative_return_pct": float((equity[-1] - 1) * 100),
            "sharpe": float(returns.mean() / deviation * np.sqrt(periods_per_year)) if deviation > 0 else 0.,
            "max_drawdown_pct": float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100)}


def top_k_strategy(scored: pd.DataFrame, tickers: list[str], horizon: int,
                   top_k: int, cost_bps: float = 10.) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank at origin close, enter next open, liquidate at horizon close; compare buy/hold."""
    if not 1 <= top_k <= len(tickers) or not 0 <= cost_bps < 10000:
        raise ValueError("strategy-top-k must fit universe; cost-bps must be in [0,10000)")
    final = scored[(scored.method == "kronos") & (scored.step == horizon)]
    periods = []
    for _, group in final.groupby("first_target", sort=True):
        if set(group.ticker) == set(tickers):
            periods.append(group.set_index("ticker").loc[tickers])
    if not periods:
        raise ValueError("Strategy requires common complete windows for the entire selected universe")
    for previous, current in zip(periods, periods[1:]):
        if current.first_target.iloc[0] <= previous.target.iloc[0]:
            raise ValueError("Strategy windows overlap; use --stride equal to horizon")
    cost = cost_bps / 10000
    initial_open = periods[0].entry_open
    equity = 1.
    records = []
    for group in periods:
        ranking = (group.predicted / group.origin_close - 1).sort_values(ascending=False, kind="stable")
        selected = ranking.head(top_k).index
        gross = float((group.loc[selected, "actual"] / group.loc[selected, "entry_open"]).mean())
        equity *= gross * (1 - cost) / (1 + cost)
        hold = float((group.actual / initial_open).mean()) / (1 + cost)
        records.append({"entry": group.first_target.iloc[0], "exit": group.target.iloc[0],
                        "selected": ",".join(selected), "strategy_equity": equity, "buy_hold_equity": hold})
    records[-1]["buy_hold_equity"] *= 1 - cost
    curve = pd.DataFrame(records)
    elapsed = len(market_bars(curve.entry.iloc[0], curve.exit.iloc[-1] + pd.Timedelta(days=1)))
    frequency = 252 * len(curve) / elapsed
    report = pd.DataFrame([{"strategy": name, **equity_metrics(curve[column].to_numpy(), frequency)}
                           for name, column in [("top_k", "strategy_equity"), ("equal_weight_buy_hold", "buy_hold_equity")]])
    return curve, report


def save_evaluation(output: Path, scored: pd.DataFrame, summary: pd.DataFrame,
                    skipped: pd.DataFrame, settings: dict[str, Any],
                    strategy: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> None:
    """Persist auditable scores, Markdown methodology, metrics and error/equity plots."""
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "scored_windows.csv", index=False)
    summary.to_csv(output / "metrics.csv", index=False)
    skipped.to_csv(output / "skipped_windows.csv", index=False)
    write_json(output / "config.json", settings)
    write_json(output / "metrics.json", json_records(summary))
    aggregate = summary[summary.ticker == "ALL"].drop(columns="ticker")
    report = "# Walk-Forward Evaluation\n\n" + aggregate.to_markdown(index=False, floatfmt=".4f")
    report += "\n\n## Method\n\n"
    report += (f"Settings: `{settings}`.\n\n"
               "Target dates use [start,end). Only complete exchange-calendar windows are scored. "
               "Each input ends before the first target; no future rows enter input normalization. "
               "Missing bars are skipped, not filled or replaced with forecasts. All baselines use "
               "exactly the same windows. Naive repeats last close, drift extrapolates the input "
               "endpoint slope, SMA repeats the trailing mean. Direction compares each target to "
               "the origin close (flat naive predictions count correct only for flat actuals). "
               "Coverage is the fraction inside sampled p10/p90, not a calibrated confidence claim. "
               "ALL pools stock/bar errors; dollar MAE/RMSE are influenced by higher-priced stocks.\n\n"
               f"Skipped ticker windows: {len(skipped)}.\n\n")
    if strategy is not None:
        curve, results = strategy
        curve.to_csv(output / "strategy_curve.csv", index=False)
        results.to_csv(output / "strategy_metrics.csv", index=False)
        report += "## Strategy\n\n" + results.to_markdown(index=False, floatfmt=".4f")
        report += ("\n\nFixed top-K equal-weight longs ranked using forecasts at the prior close. "
                   "Entry at the next adjusted open, full liquidation at horizon adjusted close. "
                   "Per-side costs apply to every entry and exit, including unchanged names. "
                   "Only common complete, nonoverlapping daily windows are traded. Missing periods "
                   "leave cash uninvested. Buy-and-hold invests equally in the same universe at the "
                   "first entry open and holds through the final exit, with entry/exit costs only. "
                   "The ranking strategy is flat overnight between periods; buy-and-hold is not. "
                   "Sharpe uses observed period returns, zero risk-free rate and annualized NYSE "
                   "period frequency. Drawdown is measured at period endpoints, not intraday.\n\n")
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(curve.exit, curve.strategy_equity, label="Top-K after costs")
        axis.plot(curve.exit, curve.buy_hold_equity, label="Equal-weight buy/hold after costs")
        axis.legend()
        axis.set(ylabel="Equity (initial = 1)")
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output / "strategy.png", dpi=150)
        plt.close(figure)
    report += ("## Limitations\n\nCurrent S&P 500 membership causes survivorship bias in historical "
               "universe tests. Yahoo adjusted data is retrospective, not point-in-time; vendor "
               "quality/rate limits and corporate actions matter. Kronos pretraining overlap with "
               "these dates is unknown, so this is not certified out-of-sample evidence. No future "
               "guarantee or financial advice. Short samples are especially unreliable.\n\n"
               "## Per Ticker\n\n" + summary[summary.ticker != "ALL"].to_markdown(index=False, floatfmt=".4f"))
    (output / "report.md").write_text(report + "\n")
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(aggregate.method, aggregate.MAPE_pct, color=["#007f73", "#555555", "#bd5252", "#c19d31"])
    axis.set(ylabel="MAPE (%)", title="Same-window pooled close-price errors")
    figure.tight_layout()
    figure.savefig(output / "errors.png", dpi=150)
    plt.close(figure)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert tabular NaNs to JSON null via pandas' structured serializer."""
    import json
    return json.loads(frame.to_json(orient="records"))


def run_evaluation(args: argparse.Namespace) -> Path:
    """Fetch pre-start history, evaluate and print/save the actual results table."""
    from sp500.__main__ import config_from_args, date_bounds, new_output, selected_tickers
    config: ForecastConfig = config_from_args(args)
    start, end = date_bounds(args)
    if pd.Timestamp(end) <= pd.Timestamp(start):
        raise ValueError("end must be after start")
    if args.strategy_top_k and (config.interval != "1d" or (args.stride or config.horizon) != config.horizon):
        raise ValueError("Strategy requires 1d and nonoverlapping --stride equal to horizon")
    cache = MarketData(args.cache_dir)
    prior = market_bars(pd.Timestamp(start) - pd.Timedelta(days=config.lookback * 3), start, config.interval)
    if len(prior) < config.lookback:
        raise ValueError("Cannot locate enough exchange history before start")
    history_start = str(prior[-config.lookback].date())
    tickers = selected_tickers(args)
    histories = {ticker: cache.fetch(ticker, history_start, end, config.interval, args.refresh, config.lookback)
                 for ticker in tickers}
    forecaster = Forecaster(config)
    scored, skipped = walk_forward(histories, forecaster, start, end, args.stride, args.batch_size, args.sma_window)
    summary = summarize(scored)
    strategy = top_k_strategy(scored, tickers, config.horizon, args.strategy_top_k, args.cost_bps) if args.strategy_top_k else None
    output = Path(args.output) if args.output else new_output()
    if (output / "report.md").exists():
        raise ValueError("Evaluation output already exists; choose a new directory")
    save_evaluation(output, scored, summary, skipped,
                    {**asdict(config), "device_used": forecaster.device, "start": start, "end": end,
                     "tickers": tickers, "stride": args.stride or config.horizon, "batch_size": args.batch_size,
                     "sma_window": args.sma_window, "strategy_top_k": args.strategy_top_k, "cost_bps": args.cost_bps}, strategy)
    print(summary[summary.ticker == "ALL"].to_string(index=False))
    print(f"Report: {output / 'report.md'}")
    return output