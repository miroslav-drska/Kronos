# S&P 500 Guide

Run commands from the repository root. The upstream model code is unchanged.
This tool is usable without training, API keys, Qlib, Comet or the legacy web UI.

## Installation

Use Python 3.10-3.12, preferably 3.12. On Apple Silicon with Homebrew:

```bash
PYTHON=/opt/homebrew/bin/python3.12 bash scripts/setup.sh
source .venv/bin/activate
python -m pip check
```

On Linux/Windows WSL, use `PYTHON=python3.12 bash scripts/setup.sh` instead.
On native Windows, run `py -3.12 -m venv .venv`, then
`.venv\Scripts\python -m pip install -r requirements-sp500.txt` and use that
interpreter for the commands below. Native Windows has not been tested here.

The separate pinned requirements constrain the upstream dependencies.
Do not use `pip install .`: upstream pyproject metadata has incompatible
constraints and an invalid Python specifier. Work from this checkout so
`model/` and the reused CSV trainers remain importable. NumPy 1.26/pandas 2.2.2
pins do not support Python 3.13+. This environment was tested on macOS ARM64
Python 3.12.16/Torch 2.6.0. Linux CUDA is supported by the adapter but was not
available for local certification; Intel Macs need a different Torch pin and
are not certified by this requirements set.

Public Hugging Face weights download on first inference. Expect roughly
100-150 MB for the small model/tokenizer plus package downloads. Model weights
use the normal Hugging Face disk cache; market caches are separate.

## Quickstart

After installation, this small CPU-safe job is intended to produce a real
forecast in under five minutes on ordinary broadband and a modern laptop.
Download/network delays can exceed that target; no synthetic data is substituted.

```bash
python -m sp500 forecast --tickers AAPL --lookback 128 --horizon 10 --samples 5
```

The printed directory contains AAPL.csv, AAPL.json and AAPL.png. Forecast dates
are NYSE sessions, not calendar days. `close` is the sample median, and
`close_p10`/`close_p90` form an empirical sampled band. With only five paths,
quantiles are coarse. The default production/research starting configuration
uses small, 400 input bars, 10 future bars and 20 paths, not the shortened
quickstart configuration.

```bash
python -m sp500 forecast --tickers AAPL --horizon 10
python -m sp500 forecast --tickers AAPL MSFT NVDA --horizon 10
python -m sp500 list-tickers
python -m sp500 fetch --tickers AAPL MSFT --start 2023-01-01 --end 2025-01-01
python -m sp500 evaluate --tickers AAPL MSFT NVDA --start 2024-01-01 --end 2025-12-31 --horizon 5
make test
make evaluate-demo
```

Long evaluation ranges take substantially longer than the quickstart. The last
evaluation date above is exclusive. The small measured ten-stock demo used
January-March 2024; see [the results](EVALUATION_DEMO.md).

## Commands And Flags

`python -m sp500 --help` and `python -m sp500 COMMAND --help` show the parser.
There are five commands: list-tickers, fetch, forecast, evaluate and finetune.

Shared by fetch, forecast, evaluate and finetune:

| Flag | Meaning / Default |
| --- | --- |
| --tickers SYMBOL ... | Explicit ordered symbols; duplicates removed; BRK.B becomes BRK-B |
| --all | Current cached S&P 500 universe; mutually exclusive with --tickers |
| --cache-dir | Market cache root, default data/cache |
| --refresh | Redownload the requested candle range; not a historical membership selector |
| --interval | 1d (default), 1h, 60m, 30m, 15m, 5m, 2m, 1m |
| --start | Inclusive NY date; fetch/forecast/evaluate default to recent history |
| --end | Exclusive NY date; default tomorrow, with unfinished candles filtered |

Default recent history is 900 calendar days for daily bars, 729 for hourly,
59 for other intraday and 6 for one-minute. Evaluation interprets start/end as
target dates and automatically fetches the needed pre-start lookback. Fine-tuning
instead takes default start/end from its YAML. Yahoo intraday restrictions apply
to the age of the earliest newly downloaded bar, not just the requested span:
hourly approximately 730 days, other intraday approximately 60, one-minute
approximately seven. The implementation keeps a one-day safety margin and
rejects older uncached requests clearly. Already cached older bars can be reused.

list-tickers only accepts --cache-dir and --refresh; its refresh updates the
constituent table. The cache stores fetched_at, name, sector and Yahoo symbols.
Prices are downloaded with yfinance `auto_adjust=True`. The amount feature is
estimated as `(high + low + close) / 3 * volume`, not an independently measured
trade-notional series. Parquet caches are per ticker/interval, with JSON fetched
range metadata; normal fetches request only uncovered date ranges. Today's
boundary is rechecked rather than permanently caching an unfinished session.
Empty failed trading-day downloads do not count as fetched coverage.

Daily timestamps are NY session dates at midnight; intraday timestamps are
NY-local naive bar starts. These wall-clock features match the upstream `.dt`
API. UTC inputs are converted to New York before removing timezone information.
NYSE schedules handle weekends, holidays, early closes and DST. Unfinished
hourly early-close bars are conservatively retained only once a full nominal
bar duration has elapsed. NaNs/nonfinite/invalid price rows and duplicates are
removed; missing exchange bars are warned about, never price-filled.

Shared by forecast and evaluate:

| Flag | Meaning / Default |
| --- | --- |
| --model | mini, small (default), base |
| --model-path | Optional local predictor save_pretrained checkpoint directory |
| --tokenizer-path | Optional matching local tokenizer checkpoint directory |
| --lookback | Input bars, default 400; minimum 2, at most model context |
| --horizon | Future bars, default 10; positive, at most model context |
| --samples | Independent sampled paths, default 20; one path gives zero-width bands |
| --seed | Python/NumPy/Torch initial seed, default 42 |
| --device | auto (CUDA, then MPS, then CPU), cpu, mps, cuda, cuda:0 |
| --temperature | Positive sampling temperature, default 1.0 |
| --top-p | Nucleus probability, default 0.9 |
| --top-k | Token sampling cap, default 0; 1 makes paths effectively deterministic |
| --batch-size | Equal-length windows per batch, default 8; reduce for low memory |
| --output | Explicit run directory; default outputs/<UTC date>_<time>_<run-id> |

Model identifiers and contexts:

| Choice | Predictor | Tokenizer | Context |
| --- | --- | --- | --- |
| mini | NeoQuasar/Kronos-mini | NeoQuasar/Kronos-Tokenizer-2k | 2048 |
| small | NeoQuasar/Kronos-small | NeoQuasar/Kronos-Tokenizer-base | 512 |
| base | NeoQuasar/Kronos-base | NeoQuasar/Kronos-Tokenizer-base | 512 |

Local checkpoints must match --model and the tokenizer vocabulary. We do not
claim reproducibility across different devices, Torch versions, updated hub
weights or revised Yahoo histories. Keep the artifacts, package environment,
weights, seed, universe ordering and batch size to reproduce a run. Separate
`sample_count=1` calls preserve full paths; upstream sample_count>1 averages
paths and cannot produce valid empirical quantiles.

forecast adds --resume EXISTING_RUN_DIRECTORY. It checks universe, batch size
and all model/sampling flags; dates are frozen to the existing manifest.
Completed batches with all artifacts are skipped without model loading.
An interrupted partial batch is regenerated with its stable seed and verified
input hashes, then successful ticker files are recorded atomically. Do not change
the cache input history during resume. Network/insufficient-history failures
are recorded by ticker and make the CLI return nonzero; retry with --resume.

```bash
python -m sp500 forecast --all --samples 5 --batch-size 8 --output outputs/universe-run
python -m sp500 forecast --all --samples 5 --batch-size 8 --resume outputs/universe-run
```

The current 500-company universe includes multiple share classes. Entire-universe
runs are expensive and can hit Yahoo rate limits. Batched inference is real
predict_batch reuse, not independent model reloads. Insufficient stock histories
are explicitly reported, not padded with invented candles. No full-universe live
run was performed during local acceptance.

evaluate adds:

| Flag | Meaning / Default |
| --- | --- |
| --stride | Origin spacing in exchange bars, default horizon |
| --sma-window | Trailing mean length, default 20; shorter histories use available input |
| --strategy-top-k | Optional long-only selection count, default 0 disables |
| --cost-bps | Per-side trade cost in basis points, default 10 |

## Python API And Configuration

```python
from sp500.data import MarketData
from sp500.forecast import ForecastConfig, Forecaster, save_forecast
from sp500.evaluate import walk_forward, summarize

cache = MarketData()
histories = {
    ticker: cache.fetch(ticker, "2022-01-01", "2025-01-01")
    for ticker in ["AAPL", "MSFT"]
}
config = ForecastConfig(horizon=5, samples=5, device="cpu")
forecaster = Forecaster(config)
predictions = forecaster.forecast_batch(histories)
for ticker, prediction in predictions.items():
    save_forecast("outputs/python-demo", ticker, histories[ticker], prediction, config)
scored, skipped = walk_forward(histories, forecaster, "2024-01-01", "2024-04-01")
print(summarize(scored))
```

ForecastConfig is a validated immutable dataclass; one Forecaster retains weights
for successive calls. `forecast(history)` is the single-stock equivalent.
Provide an increasing, unique DatetimeIndex and six features in the normalized
cache schema. CLI flags configure inference; per-run JSON saves their resolved
values. Fine-tuning has its own editable YAML described below.

## Evaluation Method

For each origin, exactly lookback consecutive NYSE input bars end strictly
before the first of horizon target bars. Only these inputs enter upstream
normalization. Missing inputs/targets skip the whole ticker window, rather than
shifting target dates or substituting predictions for actual prices.
Every method is scored on identical actual closes:

- Kronos: median across independent sampled paths.
- Naive: the final observed close at every horizon step.
- Drift: last close plus step times `(last - first) / (lookback - 1)`.
- SMA: fixed trailing average of the input closes.

Reports contain MAE and RMSE in adjusted USD, MAPE in percent, directional
accuracy relative to the origin close, and sampled p10-p90 coverage. Flat naive
forecasts count as directionally correct only for truly flat actual returns;
this definition explains their near-zero directional accuracy. Band coverage
is not available for the three point baselines. Aggregate metrics pool all
stock/target errors; higher-price stocks dominate dollar errors. The raw scores
support alternative equal-ticker aggregation and horizon-specific analyses.

Optional strategy requires daily bars and stride=horizon (no overlapping
positions). Rank predicted final returns at the observed origin close, enter
equal-weight top-K longs at the next open, and liquidate at the final horizon
close. Entry and exit each incur cost-bps; unchanged names still incur a full
round trip. There is no leverage, shorting or next-open information in ranking.
Trade only windows complete for every selected ticker; sit in cash in skipped
periods. Equal-weight buy-and-hold uses the same universe, first entry open,
final exit close and one entry/exit cost. Buy-and-hold retains overnight exposure,
while the periodic strategy does not. Sharpe uses period returns, zero risk-free
rate and NYSE-derived annualization; drawdown is at period endpoints only.
Short reports cannot establish a durable trading edge.

## Fine-Tuning

The optional adapter reuses CustomKlineDataset, train_tokenizer and train_model
from finetune_csv without modifying them or invoking their fragile script mains.
It concatenates datasets of already-formed windows, never raw ticker candles.
Global date cuts are disjoint: train before train_end, validation from train_end
to val_end, test from val_end onward. Test rows do not enter either trainer.
Training windows must fit completely inside their per-stock split; missing
calendar bars fail export rather than being silently imputed.

```bash
python -m sp500 finetune --tickers AAPL MSFT NVDA --export-only --output outputs/csv-export
python -m sp500 finetune --tickers AAPL MSFT NVDA --config configs/finetune_sp500.yaml --output outputs/tuned
python -m sp500 forecast --tickers AAPL --model small --model-path outputs/tuned/predictor/best_model --tokenizer-path outputs/tuned/tokenizer/best_model
```

The first command is safe on CPU. Without CUDA, the second exports data, prints
a clear CUDA-unavailable message and returns nonzero; it does not quietly start
a large training run. --allow-cpu explicitly opts into slow CPU training.
MPS training is not certified. Export-only and --allow-cpu are fine-tune-only
flags. Fine-tuning also accepts --output and --config (default shown above).

Edit configs/finetune_sp500.yaml before a serious experiment:

| Section | Keys / Behavior |
| --- | --- |
| data | start/end default fetch dates; train_end/val_end exclusive split dates; lookback_window=90, predict_window=10, max_context=512, clip=5 |
| training | tokenizer_epochs=3, basemodel_epochs=5, batch_size=16, seed=42, log_interval=50; tokenizer/predictor learning rates and Adam settings |
| training.accumulation_steps | Tokenizer microbatch split; must divide batch_size; predictor upstream loop does not accumulate |
| training.num_workers | Adapter deliberately uses 0 for safe scoped script imports and portable workers |
| model_paths | Public or local pretrained_tokenizer/pretrained_predictor; output path comes from CLI, not upstream path templates |
| experiment | train_tokenizer/train_basemodel stage switches; use_comet=false |
| device | Legacy YAML keys retained for compatibility; adapter chooses CUDA or explicitly opted-in CPU |

Default train/val/test periods are 2015-2021 / 2022-2023 / 2024-2025. These are
examples, not a guarantee against original pretraining overlap. If evaluating
the local model, use dates strictly after its training and model-selection
period; do not optimize settings on the reported test set. Upstream training
uses teacher-forced whole-window normalization within each split, whereas
forecasting normalizes input history only; this inherited mismatch deserves
measurement. Best tokenizer weights are reloaded before predictor training.

Planning estimates, **not measured GPU benchmarks**: small with batch16 and
101-token windows may need roughly 4-8 GB VRAM; base with long 512-token windows
can require 12-24+ GB. A few stocks/few epochs may take minutes to tens of minutes
on a recent GPU; hundreds of stocks and long histories can take hours. Profile
your actual hardware and start with mini/small, low batch size and one epoch.
Locally, only a tiny one-epoch CPU tokenizer/predictor checkpoint round-trip was
certified; no full S&P 500 GPU fine-tuning was performed.

## Output Formats

Forecast runs have manifest.json plus one CSV, JSON and PNG per ticker. CSV
columns: timestamps, open, high, low, close, volume, amount, close_p10, close_p90.
OHLCVA fields are marginal sample medians, not one internally consistent candle
path. JSON adds configuration, origin, last observed close, input hash/device/
batch seed and timestamped records. PNG shows recent adjusted history, median
and the sampled band. Model outputs are not arbitrarily smoothed or clipped
into plausible prices; inspect implausible forecasts before any downstream use.

Evaluation runs include report.md, config.json, metrics.csv/json,
scored_windows.csv, skipped_windows.csv and errors.png. Optional strategy adds
strategy_curve.csv, strategy_metrics.csv and strategy.png. Scores retain ticker,
method, origin, first target, target, step, actual, predicted, origin close,
entry open and band endpoints. Empty skipped output has no rows. Fine-tuning
adds data/{train,val,test}/SYMBOL.csv, data/splits.json, training.log,
training_results.json and predictor/tokenizer/best_model save_pretrained folders.
Generated outputs, market cache, virtualenv and checkpoint binaries are Git-ignored.

## Testing And Convenience

```bash
make setup PYTHON=python3.12
make test
make test-slow
make smoke
make forecast-demo
make evaluate-demo
```

`pytest -m "not slow"` runs mocked, network-blocked tests. Slow tests are
explicit real Yahoo/model forecast integration, a tiny trainer/checkpoint
round-trip, and the four preserved pinned upstream CPU regressions. Run all
with `python -m pytest -q`. Local acceptance passed all 24 tests, verified
single CPU and three-stock MPS inference, NYSE future dates, CSV/JSON/PNG export,
resume behavior, and the ten-stock actual evaluation. Full-universe inference,
CUDA and full training remain untested.

## Troubleshooting

- Wrong interpreter: use `.venv/bin/python`; macOS system Python 3.9 is too old.
- Python 3.13+: use 3.12 for this pinned set, not a NumPy source build workaround.
- Yahoo empty response/rate limit: retry later; inspect manifest errors and resume.
  Only cached valid ranges are reused. --refresh rechecks vendor corrections.
- Intraday history limit: request recent start/end, use daily bars or archived
  caches. Even a short range several years ago is unavailable from Yahoo intraday.
- Insufficient bars: expand daily history or reduce --lookback; newly listed
  stocks cannot be invented into long histories.
- Out of memory: reduce --batch-size/--lookback, use --model mini or --device cpu.
- MPS operator/runtime issue: rerun with --device cpu; CPU inference is certified.
- Cannot download weights: check public Hugging Face connectivity or provide
  previously downloaded matching local --model-path and --tokenizer-path.
- Resume mismatch/input change: preserve original flags/cache or start a new
  output directory. Do not overwrite existing runs as a workaround.
- Upstream plotting warnings: Matplotlib 3.9.3 emits Pyparsing deprecations in
  this environment; plots and tests still pass.

## Limitations

Forecasts are probabilistic and noisy. The measured default-small demo was
materially worse than simple baselines and its bands were badly undercalibrated.
Past evaluation does not guarantee future performance. Current-constituent
universe tests suffer survivorship bias; we do not claim historical membership.
Yahoo quality/rate limits, missing candles, retrospective split/dividend
adjustments and ticker changes are material. Adjusted data is not a point-in-time
archive. Original Kronos pretraining overlap is unknown, so a calendar split
alone cannot certify out-of-sample evidence. Costs omit spreads, market impact,
taxes, execution failures and intraday drawdowns. This is not financial advice.