# metalib/research/

Ad-hoc backtest/A-B-testing research for strategies in `metalib/` — not
production code (see top-level `CLAUDE.md` for that). Currently covers
MetaFVG (trend-filter A/B testing: ADX/regression/Lasso/decision-tree gates,
ATR trailing stop) and MetaOB (trend-filter fix validation).

## Layout

- `scripts/` — sweep runners and PDF report generators. Flat, sibling-import
  namespace (not a package) — MetaOB's scripts import `metafvg_ab_universe`
  directly, so everything here must stay in one directory.
- `reports/` — generated PDF tearsheets. Regeneratable from `data/`, but
  kept in git since they're small and useful without re-running anything.
- `data/` — pickled sweep-result caches (can be hundreds of MB). Gitignored.
- `logs/` — raw stdout logs from parallel-shard sweep runs. Gitignored.
- `notebooks/` — reserved for future interactive/Jupyter exploration.

## Running

All scripts assume PYTHONPATH is the repo root (same convention as
`metalib/mains/`) and the `adonys` conda interpreter:

```
PYTHONPATH=. "<adonys python>" metalib/research/scripts/metafvg_ab_sweep.py
PYTHONPATH=. "<adonys python>" metalib/research/scripts/generate_metafvg_ab_report.py
```

Sweep scripts support `--shard i/N` (split the symbol universe across N
parallel processes, each writing its own `*_shard{i}.pkl` into `data/`) and
`--tag` (suffix the output filename, for testing one config without
clobbering the main cache). Merge shards with `metafvg_ab_sweep_merge.py
--slug <slug>`.

Report scripts read `METAFVG_LTF`/`METAFVG_HTF` env vars to pick which
timeframe pair's cache to load (default H4/1 Week).
