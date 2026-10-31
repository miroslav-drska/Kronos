PYTHON ?= python3.12
PY := .venv/bin/python

.PHONY: setup test test-slow smoke forecast-demo evaluate-demo
setup:
	PYTHON="$(PYTHON)" bash scripts/setup.sh

test:
	$(PY) -m pytest -m "not slow" -q

test-slow:
	$(PY) -m pytest -m slow -q

smoke:
	$(PY) -m sp500.smoke

forecast-demo:
	$(PY) -m sp500 forecast --tickers AAPL MSFT NVDA --lookback 128 --horizon 10 --samples 5

evaluate-demo:
	$(PY) -m sp500 evaluate --tickers AAPL MSFT NVDA AMZN GOOGL META JPM XOM JNJ WMT \
		--start 2024-01-01 --end 2024-04-01 --horizon 5 --samples 5 \
		--device cpu --strategy-top-k 3