#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), "Use Python 3.10-3.13 (3.12 recommended)"'
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-sp500.txt
.venv/bin/python -m pip check
printf '\nReady. Run: source .venv/bin/activate\n'