"""Strict per-stock CSV exports and reuse of the existing CSV training loops."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from sp500.data import FEATURES, MarketData, market_bars, normalize_ticker
from sp500.forecast import seed_everything, write_json


def date_split(data: pd.DataFrame, train_end: str, val_end: str) -> dict[str, pd.DataFrame]:
    """Split globally at exclusive NY date boundaries with no shared rows."""
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if train_end >= val_end or not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("Require ordered unique timestamps and train_end < val_end")
    return {"train": data.loc[data.index < train_end].copy(),
            "val": data.loc[(data.index >= train_end) & (data.index < val_end)].copy(),
            "test": data.loc[data.index >= val_end].copy()}


def export_splits(histories: dict[str, pd.DataFrame], output: str | Path,
                  train_end: str, val_end: str, interval: str = "1d") -> dict[str, list[Path]]:
    """Export per-ticker split files; never concatenate stocks into one time series."""
    output = Path(output)
    files: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    inventory = []
    for ticker, data in histories.items():
        ticker = normalize_ticker(ticker)
        if data.empty or not np.isfinite(data[FEATURES].to_numpy()).all():
            raise ValueError(f"{ticker}: empty or nonfinite training data")
        expected = market_bars(data.index[0], data.index[-1] + pd.Timedelta(minutes=1), interval)
        if not expected.equals(pd.DatetimeIndex(data.index)):
            raise ValueError(f"{ticker}: training data has missing or off-calendar bars; repair cache, do not impute")
        for split, frame in date_split(data, train_end, val_end).items():
            folder = output / split
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{ticker}.csv"
            frame = frame[FEATURES].rename_axis("timestamps")
            frame.to_csv(path)
            files[split].append(path)
            inventory.append({"ticker": ticker, "split": split, "rows": len(frame),
                              "start": str(frame.index.min()), "end": str(frame.index.max()), "path": str(path)})
    write_json(output / "splits.json", {"train_end_exclusive": train_end, "val_end_exclusive": val_end,
                                        "interval": interval, "inventory": inventory})
    return files


class StockWindows(ConcatDataset):
    """Concatenate already-formed stock datasets, never their underlying candles."""

    def set_epoch_seed(self, epoch: int) -> None:
        """Forward upstream trainer epoch seeds to each independent stock dataset."""
        for dataset in self.datasets:
            dataset.set_epoch_seed(epoch)


def split_dataloaders(files: dict[str, list[Path]], config: Any, dataset_class: Any) -> tuple[Any, ...]:
    """Reuse CustomKlineDataset on pre-split files and provide the trainer's six-tuple."""
    datasets = {}
    window = config.lookback_window + config.predict_window + 1
    for split in ("train", "val"):
        children = []
        for path in files[split]:
            rows = len(pd.read_csv(path, usecols=["timestamps"]))
            if rows <= window:
                warnings.warn(f"Skipping {path}: {rows} rows <= window {window}", stacklevel=2)
                continue
            ratios = (1., 0., 0.) if split == "train" else (0., 1., 0.)
            children.append(dataset_class(str(path), data_type=split, lookback_window=config.lookback_window,
                                          predict_window=config.predict_window, clip=config.clip, seed=config.seed,
                                          train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2]))
        if not children:
            raise ValueError(f"No usable {split} windows; expand dates or reduce lookback/predict window")
        datasets[split] = StockWindows(children)
    train = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                       num_workers=0, drop_last=True)
    val = DataLoader(datasets["val"], batch_size=config.batch_size, shuffle=False,
                     num_workers=0, drop_last=False)
    if len(train) == 0 or len(val) == 0:
        raise ValueError("Empty train/validation loader; reduce batch size or expand data")
    return train, val, datasets["train"], datasets["val"], None, None


@contextmanager
def csv_trainers() -> Iterator[tuple[ModuleType, ModuleType, ModuleType]]:
    """Temporarily resolve upstream script-local imports without changing their files."""
    root = Path(__file__).resolve().parents[1] / "finetune_csv"
    names = ["config_loader", "finetune_base_model", "finetune_tokenizer"]
    previous = {name: sys.modules.get(name) for name in names}
    modules = []
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load upstream trainer {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules.append(module)
        yield modules[0], modules[1], modules[2]
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def train_exported(files: dict[str, list[Path]], config_path: str | Path,
                   output: str | Path, allow_cpu: bool = False) -> dict[str, Any]:
    """Train tokenizer then predictor with upstream loops and reload the best tokenizer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA unavailable. Data export succeeded; use a CUDA machine for training or explicitly --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warnings.warn("CPU fine-tuning is slow; MPS training is not certified. Use only small smoke jobs here.", stacklevel=2)
    torch.set_num_threads(4)
    with csv_trainers() as (config_module, base_trainer, tokenizer_trainer):
        config = config_module.CustomFinetuneConfig(str(config_path))
        if config.lookback_window < 2 or config.predict_window < 1 or config.lookback_window + config.predict_window > config.max_context:
            raise ValueError("Fine-tuning lookback + predict_window must fit max_context")
        if config.batch_size < 1 or config.accumulation_steps < 1 or config.batch_size % config.accumulation_steps:
            raise ValueError("batch_size must be positive and divisible by accumulation_steps")
        if config.log_interval < 1 or config.tokenizer_epochs < 1 or config.basemodel_epochs < 1:
            raise ValueError("Epoch counts and log_interval must be positive")
        seed_everything(config.seed)
        loaders = split_dataloaders(files, config, base_trainer.CustomKlineDataset)
        base_trainer.create_dataloaders = lambda unused: loaders
        tokenizer_trainer.create_dataloaders = lambda unused: loaders
        from model import Kronos, KronosTokenizer
        logger = logging.getLogger(f"sp500.finetune.{output}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(output / "training.log")
        logger.addHandler(handler)
        results: dict[str, Any] = {"device": str(device), "config": config.loader.config,
                                   "test_used_for_training": False}
        try:
            tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path).to(device)
            if config.train_tokenizer:
                loss = tokenizer_trainer.train_tokenizer(tokenizer, device, config, str(output / "tokenizer"), logger)
                if not np.isfinite(loss):
                    raise RuntimeError("Tokenizer training returned nonfinite validation loss")
                results["tokenizer_val_loss"] = loss
                tokenizer = KronosTokenizer.from_pretrained(str(output / "tokenizer/best_model")).to(device)
            tokenizer.eval()
            if config.train_basemodel:
                model = Kronos.from_pretrained(config.pretrained_predictor_path).to(device)
                loss = base_trainer.train_model(model, tokenizer, device, config, str(output / "predictor"), logger)
                if not np.isfinite(loss):
                    raise RuntimeError("Predictor training returned nonfinite validation loss")
                results["predictor_val_loss"] = loss
            write_json(output / "training_results.json", results)
        finally:
            logger.removeHandler(handler)
            handler.close()
    return results


def run_finetune(args: argparse.Namespace) -> Path:
    """Export strict date splits, then optionally train using the existing CSV trainers."""
    from sp500.__main__ import new_output, selected_tickers
    settings = yaml.safe_load(Path(args.config).read_text())
    data_settings = settings["data"]
    start = args.start or data_settings["start"]
    end = args.end or data_settings["end"]
    if not pd.Timestamp(start) < pd.Timestamp(data_settings["train_end"]) < pd.Timestamp(data_settings["val_end"]) < pd.Timestamp(end):
        raise ValueError("Require start < train_end < val_end < end")
    output = Path(args.output) if args.output else new_output()
    if (output / "data/splits.json").exists() or (output / "training_results.json").exists():
        raise ValueError("Fine-tuning output already exists; choose a new directory")
    cache = MarketData(args.cache_dir)
    histories = {ticker: cache.fetch(ticker, start, end, args.interval, args.refresh)
                 for ticker in selected_tickers(args)}
    files = export_splits(histories, output / "data", data_settings["train_end"], data_settings["val_end"], args.interval)
    print(f"CSV splits: {output / 'data'}")
    if not args.export_only:
        train_exported(files, args.config, output, args.allow_cpu)
        print(f"Checkpoints: {output}")
    return output