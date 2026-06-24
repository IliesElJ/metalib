---
name: run-strategy
description: Run or restart a metalib trading strategy (e.g. metaga, metago, metamtou) for manual testing or to verify a code change, since there is no test suite. Use when asked to "run", "test", "restart", or "check" a strategy.
---

This repo has no unit tests — the way to verify a change to a strategy is to actually run it (or its dry-run/notebook equivalent) with the correct interpreter and environment.

## Running a single strategy script directly

From the `metalib/` directory, invoke the strategy's main script with the `adonys` conda interpreter and `PYTHONPATH` set to the parent of `metalib/` (mirroring what PM2 sets in `ecosystem.config.js`):

```
cd metalib
PYTHONPATH=.. "C:\ProgramData\miniconda3\envs\adonys\python.exe" mains/main_<strategy>.py
```

On Windows PowerShell:
```
cd metalib
$env:PYTHONPATH = ".."
& "C:\ProgramData\miniconda3\envs\adonys\python.exe" mains\main_<strategy>.py
```

Replace `<strategy>` with the script name under `mains/` (e.g. `main_metamtou.py`, `main_metaga.py`, `main_metagomano.py` for `metago`). Check `metalib/ecosystem.config.js` for the exact script-name-to-strategy mapping if unsure.

## Running via PM2 (closer to production)

```
cd metalib
pm2 start ecosystem.config.js --only <strategy>   # e.g. metamtou, metaga, metago
pm2 logs <strategy>
pm2 restart <strategy>
pm2 stop <strategy>
```

Use `pm2 logs <strategy>` immediately after starting/restarting to confirm the strategy initialized without errors (e.g. MT5 connection, config load) before declaring a change verified.

## Notes

- If the run fails with a linear-algebra crash (BLAS/LAPACK), see the gotcha in CLAUDE.md — check whether the code path uses a manual OLS implementation as it should, rather than `numpy.linalg`/`scipy.linalg`/`statsmodels`.
- Each strategy loads its config from `metalib/config/{dev,prod}/<strategy>.yaml` — when testing a change, confirm which config (dev vs prod) the main script actually points to before running against prod symbols/sizing.
