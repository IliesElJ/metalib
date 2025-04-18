# Metalib

Should add:
- Mapping for each strategy in the following manner: mt5.TF => Stategy params (short.sma = .., lookback = .., ...)
- Send telegram plots for Metagomano
- MetaDO: add an arg, either -1 (mean-reversion) or 1 (trend following) and add to config a column for that arg.
- Use average returns for each strategy to guess risk weights that distribute downside in an equilly wighted manner.
- Compute sensitivities of (smoothed) returns to some features that we compute. For that, need to do the following:
- Save indicators and last return to daily HDF5 files. One per strategy instance.
