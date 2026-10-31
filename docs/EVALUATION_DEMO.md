# Measured Ten-Stock Demo

Actual local run on macOS ARM64, Python 3.12.16, Torch 2.6.0, CPU.
Yahoo adjusted daily data was fetched on 2026-10-31. No synthetic or
forecast-substituted actuals were used. Seed 42, small public weights,
lookback400, horizon5, samples5, stride5, batch8, temperature1, top_p0.9,
top_k0, SMA20. Targets: [2024-01-01,2024-04-01), 12 complete five-session
windows per stock, 600 scored closes per method. Zero windows skipped;
the final unused session does not form a complete five-bar window.

```bash
python -m sp500 evaluate --tickers AAPL MSFT NVDA AMZN GOOGL META JPM XOM JNJ WMT --start 2024-01-01 --end 2024-04-01 --horizon 5 --samples 5 --device cpu --strategy-top-k 3
```

| Method | MAE (USD) | MAPE (%) | RMSE (USD) | Direction (%) | p10-p90 Coverage (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kronos median | 25.0938 | 10.8112 | 54.8521 | 35.1667 | 21.0000 |
| Naive last close | 3.9033 | 2.1027 | 6.8928 | 0.1667 | N/A |
| Random-walk drift | 3.7279 | 2.0261 | 6.5605 | 63.6667 | N/A |
| Trailing SMA | 8.1686 | 4.3808 | 14.2747 | 40.3333 | N/A |

| Ticker | Kronos MAPE (%) | Drift MAPE (%) | Kronos Coverage (%) |
| --- | ---: | ---: | ---: |
| AAPL | 2.1522 | 2.1720 | 46.6667 |
| MSFT | 8.1867 | 1.4640 | 5.0000 |
| NVDA | 33.5136 | 4.4541 | 6.6667 |
| AMZN | 8.1675 | 1.8237 | 5.0000 |
| GOOGL | 2.7364 | 2.7207 | 43.3333 |
| META | 28.7958 | 2.6356 | 11.6667 |
| JPM | 15.4072 | 1.2189 | 1.6667 |
| XOM | 2.2698 | 1.7316 | 33.3333 |
| JNJ | 1.5789 | 1.0504 | 43.3333 |
| WMT | 5.3036 | 0.9894 | 13.3333 |

Long-only top3, next-open entry/five-session-close exit, 10bps per side:

| Strategy | Cumulative Return (%) | Sharpe | Max Drawdown (%) |
| --- | ---: | ---: | ---: |
| Forecast-ranked top3 | 4.6258 | 1.5222 | -6.1699 |
| Same-universe equal-weight buy-and-hold | 20.6163 | 11.2569 | -0.1441 |

Kronos did not beat the pooled price baselines or the buy-and-hold comparator.
Five sampled paths yield coarse quantiles and 21% empirical coverage is far
below the nominal 80%. No tuning was performed to make this report look better.
Direction is measured against origin close, so flat naive forecasts are usually
wrong under that definition. Sharpe/drawdown use only 12 period endpoints;
the high buy-and-hold Sharpe is not evidence of a stable investment opportunity.

Revised Yahoo adjustments, updated hub weights, changed ordering/batch size or
hardware can change a rerun. Current constituents cause survivorship bias,
pretraining overlap is unknown, adjusted data is not point-in-time and no future
guarantee is implied. Full local raw scores and plots are in the generated
outputs/evaluation-demo directory (ignored by Git); this file preserves the
measured summary for fresh checkouts. See [the methodology](SP500_GUIDE.md).