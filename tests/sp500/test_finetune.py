"""Strict date splits and real upstream dataset reuse, without training weights."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from sp500.data import FEATURES, market_bars
from sp500.finetune import csv_trainers, date_split, export_splits, split_dataloaders, train_exported
from sp500.forecast import ForecastConfig, Forecaster


def test_disjoint_global_splits_and_per_stock_windows(tmp_path):
    index = market_bars("2023-01-01", "2024-01-01")
    history = pd.DataFrame({feature: 10. for feature in FEATURES}, index=index)
    splits = date_split(history, "2023-05-01", "2023-09-01")
    assert splits["train"].index.max() < splits["val"].index.min() < splits["test"].index.min()
    assert sum(map(len, splits.values())) == len(history)
    files = export_splits({"AAPL": history, "MSFT": history * 2}, tmp_path, "2023-05-01", "2023-09-01")
    assert len(files["train"]) == 2
    assert list(pd.read_csv(files["train"][0]).columns) == ["timestamps", *FEATURES]
    config = SimpleNamespace(lookback_window=5, predict_window=2, clip=5, seed=42, batch_size=4)
    with csv_trainers() as (_, base, _):
        train, val, dataset, _, _, _ = split_dataloaders(files, config, base.CustomKlineDataset)
        assert len(dataset.datasets) == 2
        assert all(len(child.timestamps) < len(history) for child in dataset.datasets)
        dataset.set_epoch_seed(123)
        batch, stamps = next(iter(train))
        assert batch.shape == (4, 8, 6) and stamps.shape == (4, 8, 5)
        assert len(val) > 0


def test_missing_bars_rejected(tmp_path):
    index = market_bars("2023-01-01", "2023-03-01")
    history = pd.DataFrame({feature: 10. for feature in FEATURES}, index=index).drop(index[2])
    with pytest.raises(ValueError, match="missing"):
        export_splits({"AAPL": history}, tmp_path, "2023-02-01", "2023-02-15")


def test_no_cuda_message(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Data export succeeded"):
        train_exported({}, "configs/finetune_sp500.yaml", tmp_path)


@pytest.mark.slow
def test_real_training_checkpoint_roundtrip(tmp_path):
    index = market_bars("2023-01-01", "2024-01-01")
    history = pd.DataFrame({feature: 100. + np.arange(len(index)) / 10 for feature in FEATURES}, index=index)
    files = export_splits({"AAPL": history}, tmp_path / "data", "2023-05-01", "2023-09-01")
    settings = yaml.safe_load(open("configs/finetune_sp500.yaml"))
    settings["data"].update(lookback_window=5, predict_window=2)
    settings["training"].update(tokenizer_epochs=1, basemodel_epochs=1, batch_size=16)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    results = train_exported(files, config_path, tmp_path, allow_cpu=True)
    assert np.isfinite(results["tokenizer_val_loss"])
    assert np.isfinite(results["predictor_val_loss"])
    forecaster = Forecaster(ForecastConfig(lookback=5, horizon=2, samples=1, device="cpu",
                                          model_path=str(tmp_path / "predictor/best_model"),
                                          tokenizer_path=str(tmp_path / "tokenizer/best_model")))
    assert forecaster.forecast(history).shape == (2, 8)