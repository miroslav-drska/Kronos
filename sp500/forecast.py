"""A reusable Kronos adapter retaining independent paths for uncertainty."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sp500.data import FEATURES, INTERVALS, future_timestamps

MODELS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


@dataclass(frozen=True)
class ForecastConfig:
    """Sampling and model selection shared by forecasting and evaluation."""
    model: str = "small"
    lookback: int = 400
    horizon: int = 10
    samples: int = 20
    interval: str = "1d"
    seed: int = 42
    device: str = "auto"
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    model_path: str | None = None
    tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid windows before downloading any weights."""
        if self.model not in MODELS:
            raise ValueError(f"Unknown model {self.model!r}")
        if self.interval not in INTERVALS:
            raise ValueError(f"Unknown interval {self.interval!r}")
        context = MODELS[self.model][2]
        if not 2 <= self.lookback <= context or not 1 <= self.horizon <= context:
            raise ValueError(f"lookback must be 2..{context}; horizon must be 1..{context}")
        if self.samples < 1 or self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("Invalid sampling parameters")
        if self.device not in ("auto", "cpu", "mps", "cuda", "cuda:0"):
            raise ValueError("device must be auto, cpu, mps or cuda")


def choose_device(device: str = "auto") -> str:
    """Select CUDA, then Apple MPS, then CPU, or validate an explicit choice."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu or auto")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable; use --device cpu or auto")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch (repeatability is device/version dependent)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


class Forecaster:
    """Load weights once and forecast equal-length windows using predict_batch."""

    def __init__(self, config: ForecastConfig, predictor: Any = None) -> None:
        """Initialize a real predictor, or inject one for offline testing."""
        self.config = config
        self.device = choose_device(config.device)
        torch.set_num_threads(4)
        seed_everything(config.seed)
        if predictor is None:
            from model import Kronos, KronosPredictor, KronosTokenizer
            model_id, tokenizer_id, context = MODELS[config.model]
            tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_path or tokenizer_id).eval()
            model = Kronos.from_pretrained(config.model_path or model_id).eval()
            predictor = KronosPredictor(model, tokenizer, device=self.device, max_context=context)
        self.predictor = predictor

    def forecast(self, history: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        """Return median OHLCVA and empirical 10th/90th percentile close prices."""
        return self.forecast_batch({"single": history}, seed=seed)["single"]

    def forecast_batch(self, histories: dict[str, pd.DataFrame],
                       seed: int | None = None) -> dict[str, pd.DataFrame]:
        """Forecast all symbols in one batch per independent sample, in input order."""
        if not histories:
            return {}
        config = self.config
        windows, past, future = [], [], []
        for ticker, history in histories.items():
            if len(history) < config.lookback:
                raise ValueError(f"{ticker}: {len(history)} bars < lookback {config.lookback}")
            window = history.iloc[-config.lookback:][FEATURES]
            if not window.index.is_monotonic_increasing or window.index.has_duplicates:
                raise ValueError(f"{ticker}: timestamps must be increasing and unique")
            if not np.isfinite(window.to_numpy()).all():
                raise ValueError(f"{ticker}: nonfinite input prices")
            windows.append(window.reset_index(drop=True))
            past.append(pd.Series(window.index, name="timestamps"))
            future.append(future_timestamps(window.index[-1], config.horizon, config.interval))
        draws: list[list[np.ndarray]] = [[] for _ in histories]
        for sample in range(config.samples):
            seed_everything((config.seed if seed is None else seed) + sample)
            results = self.predictor.predict_batch(windows, past, future, config.horizon,
                                                   T=config.temperature, top_p=config.top_p,
                                                   top_k=config.top_k, sample_count=1, verbose=False)
            if len(results) != len(histories):
                raise RuntimeError("Kronos returned a misaligned batch")
            for position, result in enumerate(results):
                values = result[FEATURES].to_numpy()
                if values.shape != (config.horizon, 6) or not np.isfinite(values).all():
                    raise RuntimeError("Kronos returned invalid forecasts")
                draws[position].append(values)
        output = {}
        for position, ticker in enumerate(histories):
            paths = np.stack(draws[position])
            frame = pd.DataFrame(np.median(paths, axis=0), columns=FEATURES,
                                 index=pd.DatetimeIndex(future[position]))
            frame.index.name = "timestamps"
            frame["close_p10"] = np.quantile(paths[:, :, 3], .1, axis=0)
            frame["close_p90"] = np.quantile(paths[:, :, 3], .9, axis=0)
            output[ticker] = frame
        return output


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON metadata so interrupted runs do not corrupt manifests."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def save_forecast(output_dir: str | Path, ticker: str, history: pd.DataFrame,
                  prediction: pd.DataFrame, config: ForecastConfig,
                  metadata: dict[str, Any] | None = None) -> None:
    """Save a ticker's CSV, self-contained JSON and history/band PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_dir / f"{ticker}.csv")
    payload = {"ticker": ticker, "config": asdict(config), "as_of": history.index[-1],
               "last_close": float(history.close.iloc[-1]), "close_is": "sample median",
               "band_is": "empirical 10th to 90th percentiles, not calibrated confidence",
               "metadata": metadata or {},
               "forecast": json.loads(prediction.reset_index().to_json(orient="records", date_format="iso"))}
    write_json(output_dir / f"{ticker}.json", payload)
    figure, axis = plt.subplots(figsize=(11, 5))
    observed = history.iloc[-100:]
    axis.plot(observed.index, observed.close, color="#444444", label="Adjusted history")
    axis.plot(prediction.index, prediction.close, color="#007f73", label="Kronos median")
    axis.fill_between(prediction.index, prediction.close_p10, prediction.close_p90,
                      color="#007f73", alpha=.2, label="10-90% sampled band")
    axis.set(title=f"{ticker} | {config.model} | {config.samples} paths", ylabel="Adjusted price (USD)")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / f"{ticker}.png", dpi=150)
    plt.close(figure)