# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`metalib` is a Python algorithmic trading library for MetaTrader 5 (MT5). Each strategy (`metaga`, `metago`, `metane`, `metafvg`, `metaob`, `metamlp`, `metamtou`, ...) is a class in `metalib/meta*.py` inheriting from the base class in `metastrategy.py`. `metaworker.py` holds the `strategy_registry` dict mapping strategy-type strings to their classes, and `run_strategy_loop()` drives execution. Entry-point scripts live in `metalib/mains/main_<strategy>.py`, one per strategy, each loading a YAML config from `metalib/config/{dev,prod}/<strategy>.yaml`.

## Environment

Always run Python via the `adonys` conda environment: `C:\ProgramData\miniconda3\envs\adonys\python.exe`. Do not use a system Python or other env — dependencies (MetaTrader5, vectorbt, numba, xgboost, etc., from `requirements.txt`) are only installed there.

BLAS/LAPACK in this env crashes on some linear-algebra calls (e.g. `numpy.linalg`/`scipy.linalg` paths used by OLS). Where this has been hit, the workaround is a manual OLS implementation instead of the library call — don't "fix" hand-rolled OLS code by reverting it to `statsmodels`/`scipy.linalg` without checking this first.

## Running strategies

Production processes are orchestrated by PM2 via `metalib/ecosystem.config.js`, which pins the `adonys` interpreter and sets `PYTHONPATH=..` plus conda env vars for every app. Useful commands (run from `metalib/`):
```
pm2 start ecosystem.config.js --only <strategy>   # e.g. metamtou
pm2 logs <strategy>
pm2 restart <strategy>
```
To run a single strategy script directly for testing, invoke `mains/main_<strategy>.py` with the adonys interpreter and `PYTHONPATH` set to the parent of `metalib/` (mirrors the PM2 env) rather than launching it bare.

## Testing and linting

There is no test suite (no `tests/`, no pytest config) — verify changes by running the relevant strategy script or notebook, not by writing/expecting unit tests unless asked. CI (`.github/workflows/pylint.yml`) runs `pylint` over all tracked `.py` files on Python 3.8–3.10; there is no local lint config file, so match pylint's defaults when in doubt.

## Config and data layout

- `metalib/config/{dev,prod}/<strategy>.yaml` — per-strategy instance configs (symbols, timeframe, sizing, strategy-specific params). `prod/` configs often define multiple tagged instances of the same strategy across symbol/timeframe combinations.
- `metalib/data/` — historical OHLC exports (MT5 `.txt`/`.csv`).
- `metalib/store/` — HDF5 persistence for price/signal data.
- `metalib/logs/` — runtime logs per strategy (gitignored).
- `metalib/notebooks/` — research notebooks (not production code; don't apply production code-quality bars there).
- `metalib/metadash/` — separate Flask dashboard subproject with its own README.
