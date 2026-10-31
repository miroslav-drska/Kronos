"""Command-line data, forecast, evaluation and optional fine-tuning workflows."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

import pandas as pd

from sp500.data import INTERVALS, MarketData, constituents, normalize_ticker
from sp500.forecast import ForecastConfig, Forecaster, MODELS, save_forecast, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser without importing/downloading models."""
    parser = argparse.ArgumentParser(description="Kronos S&P 500 stock forecasting")
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list-tickers", help="Cached current S&P 500 constituents")
    listing.add_argument("--cache-dir", default="data/cache")
    listing.add_argument("--refresh", action="store_true")
    for command in ("fetch", "forecast", "evaluate", "finetune"):
        sub = subparsers.add_parser(command)
        universe = sub.add_mutually_exclusive_group(required=True)
        universe.add_argument("--tickers", nargs="+")
        universe.add_argument("--all", action="store_true")
        sub.add_argument("--cache-dir", default="data/cache")
        sub.add_argument("--refresh", action="store_true")
        sub.add_argument("--interval", choices=INTERVALS, default="1d")
        sub.add_argument("--start", help="Inclusive NY date (default depends on interval)")
        sub.add_argument("--end", help="Exclusive NY date (default tomorrow)")
        if command == "fetch":
            continue
        sub.add_argument("--output", help="Output directory; default outputs/<UTC date>_<run-id>")
        if command == "finetune":
            sub.add_argument("--config", default="configs/finetune_sp500.yaml")
            sub.add_argument("--export-only", action="store_true")
            sub.add_argument("--allow-cpu", action="store_true")
            continue
        sub.add_argument("--model", choices=MODELS, default="small")
        sub.add_argument("--model-path", help="Local predictor checkpoint (model architecture must match --model)")
        sub.add_argument("--tokenizer-path", help="Local matching tokenizer checkpoint")
        sub.add_argument("--lookback", type=int, default=400)
        sub.add_argument("--horizon", type=int, default=10)
        sub.add_argument("--samples", type=int, default=20)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--device", default="auto")
        sub.add_argument("--temperature", type=float, default=1.)
        sub.add_argument("--top-p", type=float, default=.9)
        sub.add_argument("--top-k", type=int, default=0)
        sub.add_argument("--batch-size", type=int, default=8)
        if command == "forecast":
            sub.add_argument("--resume", help="Existing forecast run directory; sampling settings must match")
        else:
            sub.add_argument("--stride", type=int, help="Origin spacing in bars (default horizon)")
            sub.add_argument("--sma-window", type=int, default=20)
            sub.add_argument("--strategy-top-k", type=int, default=0, help="Optional long-only strategy; 0 disables")
            sub.add_argument("--cost-bps", type=float, default=10.)
    return parser


def config_from_args(args: argparse.Namespace) -> ForecastConfig:
    """Extract model settings from a parsed forecast/evaluate command."""
    return ForecastConfig(**{key: getattr(args, key) for key in ForecastConfig.__dataclass_fields__})


def new_output() -> Path:
    """Allocate a readable collision-resistant UTC run directory."""
    return Path("outputs") / (datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "_" + uuid4().hex[:8])


def selected_tickers(args: argparse.Namespace) -> list[str]:
    """Resolve the requested universe, preserving order and dropping duplicates."""
    values = constituents(args.cache_dir).ticker.tolist() if args.all else args.tickers
    return list(dict.fromkeys(normalize_ticker(ticker) for ticker in values))


def date_bounds(args: argparse.Namespace) -> tuple[str, str]:
    """Choose Yahoo-compatible default history ranges."""
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    days = 900 if args.interval == "1d" else 729 if args.interval in ("1h", "60m") else 6 if args.interval == "1m" else 59
    return args.start or str((today - pd.Timedelta(days=days)).date()), args.end or str((today + pd.Timedelta(days=1)).date())


def run_forecast(args: argparse.Namespace) -> Path:
    """Run resumable fixed batches, atomically recording successful ticker artifacts."""
    config = config_from_args(args)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output = Path(args.resume or args.output) if args.resume or args.output else new_output()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    tickers = selected_tickers(args)
    start, end = date_bounds(args)
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != asdict(config) or manifest["tickers"] != tickers or manifest["batch_size"] != args.batch_size:
            raise ValueError("Resume requires identical universe, batch size and model/sampling settings")
        start, end = manifest["start"], manifest["end"]
    else:
        if manifest_path.exists():
            raise ValueError("Output already contains a run; use --resume")
        manifest = {"config": asdict(config), "tickers": tickers, "batch_size": args.batch_size,
                    "start": start, "end": end, "created_at": datetime.now(timezone.utc).isoformat(),
                    "completed": {}, "errors": {}, "input_hashes": {}}
        write_json(manifest_path, manifest)
    cache = MarketData(args.cache_dir)
    forecaster = None
    for offset in range(0, len(tickers), args.batch_size):
        batch_tickers = tickers[offset:offset + args.batch_size]
        if all(ticker in manifest["completed"] and all((output / f"{ticker}.{ext}").exists()
                    for ext in ("csv", "json", "png")) for ticker in batch_tickers):
            continue
        histories = {}
        for ticker in batch_tickers:
            try:
                history = cache.fetch(ticker, start, end, config.interval, args.refresh, config.lookback)
                if len(history) < config.lookback:
                    raise ValueError(f"Need {config.lookback} bars, received {len(history)}; increase history or reduce lookback")
                fingerprint = hashlib.sha256(history.iloc[-config.lookback:].to_csv().encode()).hexdigest()
                old = manifest["input_hashes"].get(ticker)
                if old is not None and old != fingerprint:
                    raise ValueError("Input changed since interrupted run; start a new output directory")
                manifest["input_hashes"][ticker] = fingerprint
                histories[ticker] = history
                manifest["errors"].pop(ticker, None)
            except Exception as error:
                manifest["errors"][ticker] = str(error)
                print(f"{ticker}: {error}", file=sys.stderr)
        write_json(manifest_path, manifest)
        if histories:
            if forecaster is None:
                forecaster = Forecaster(config)
                manifest["device"] = forecaster.device
            predictions = forecaster.forecast_batch(histories, seed=config.seed + offset * 100003)
            for ticker, prediction in predictions.items():
                save_forecast(output, ticker, histories[ticker], prediction, config,
                              {"device": forecaster.device, "batch_seed": config.seed + offset * 100003,
                               "input_sha256": manifest["input_hashes"][ticker]})
                manifest["completed"][ticker] = str(histories[ticker].index[-1])
                write_json(manifest_path, manifest)
        print(f"Forecast progress {min(offset + args.batch_size, len(tickers))}/{len(tickers)} | {len(manifest['completed'])} saved", flush=True)
    print(f"Outputs: {output}")
    if manifest["errors"]:
        raise RuntimeError(f"{len(manifest['errors'])} ticker(s) failed; inspect {manifest_path} and resume")
    return output


def main(argv: list[str] | None = None) -> int:
    """Execute a CLI command and return a concise nonzero status on failures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-tickers":
            print(constituents(args.cache_dir, args.refresh)[["ticker", "name", "sector"]].to_string(index=False))
        elif args.command == "forecast":
            run_forecast(args)
        elif args.command == "fetch":
            start, end = date_bounds(args)
            cache = MarketData(args.cache_dir)
            for ticker in selected_tickers(args):
                data = cache.fetch(ticker, start, end, args.interval, args.refresh)
                print(f"{ticker}: {len(data)} bars cached")
        elif args.command == "evaluate":
            from sp500.evaluate import run_evaluation
            run_evaluation(args)
        else:
            from sp500.finetune import run_finetune
            run_finetune(args)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())