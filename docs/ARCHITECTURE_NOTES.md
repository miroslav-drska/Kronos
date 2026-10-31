# Architecture Notes

## Upstream Review (Before Implementation)

Reviewed README.md, all Python in model/, examples/ (including yuce/),
finetune_csv/, finetune/config.py, tests/, and requirements.txt. Inspected the
bundled JSON reports, PNG charts, CSV schemas, row counts and time boundaries.
The regression input has 2,500 five-minute candles; the Hong Kong CSV has
93,912 candles from November 2019 through September 2025.

- model/kronos.py owns the hub-loadable tokenizer, decoder and predictor.
  model/module.py implements spherical binary quantization, hierarchical tokens,
  causal RoPE attention and temporal embeddings. model/__init__.py exports them.
- Predictors normalize six features per input window, clip normalized values,
  sample autoregressively and denormalize. Timestamp inputs must be pandas
  Series (`.dt` is used). Batch inputs need equal lengths, not equal price scales.
  Models must be put in eval mode by the caller. Device detection already uses
  CUDA, then MPS, then CPU. Small/base context is 512; mini is 2048.
- sample_count averages paths before returning. Separate sample_count=1 calls
  are necessary for genuine empirical close quantiles. Horizon must not exceed
  context: the final decode only retains max_context tokens.
- The short examples show the correct API but reference an absent stock CSV;
  the batch example uses local checkpoints. Other examples target Chinese
  markets, contain Windows paths/CUDA assumptions, synthetic network fallbacks,
  hand-maintained holidays and hardcoded forecast adjustments. The example
  backtests sometimes substitute predictions for missing actuals. None of those
  behaviors should be inherited by this tool.
- finetune_csv expects timestamps/open/high/low/close/volume/amount. Its
  CustomKlineDataset sorts and splits by row ratios, normalizes each window,
  and creates lookback+predict+1 windows. Tokenizer training minimizes
  reconstruction plus quantization loss; predictor training uses next-token
  cross entropy. SequentialTrainer calls these two reusable training functions.
  Supply per-ticker, pre-split datasets through their dataloader boundary to
  reuse the trainers without mixing tickers or leaking across date splits.
  Avoid their script main functions (local imports and an os scoping bug).
- finetune/config.py is a Qlib/DDP demonstration with overlapping date ranges,
  placeholder model paths and optional Comet credentials. It is not required.
- Existing tests are real Hugging Face CPU regression tests at pinned revisions,
  not offline unit tests. Preserve them and classify them as slow in test
  collection. New unit tests must block external traffic by default.
- requirements.txt has overlapping pandas declarations and unpinned torch/numpy.
  pyproject.toml has impossible/conflicting package constraints. Install via the
  separate pinned requirements, not `pip install .`; preserve upstream files.

## New Boundaries

All runtime additions live in sp500/: data and NYSE calendar, a model adapter,
CLI/output persistence, walk-forward evaluation, and CSV trainer integration.
No model/ changes are needed. Generated market caches/checkpoints/outputs and
the virtual environment are excluded from Git. Adjusted Yahoo data is not a
point-in-time archive; current constituents and unknown pretraining overlap
must be disclosed in historical reports.