# Metalib

Should add:
- Event Driven Architecture transform:

Phase 1: Add Event Bus

Implement basic event system
Keep existing code but add event publishing

Phase 2: Extract Components

Move risk management to separate component
Extract execution logic
Add event subscriptions

Phase 3: Full Migration

Convert all strategies to event-driven
Remove direct method calls between components
Add advanced features (retries, circuit breakers, monitoring)


Nearly Finished:

- Mapping for each strategy in the following manner: mt5.TF => Stategy params (short.sma = .., lookback = .., ...)
- Send telegram plots for Metagomano
- MetaDO: add an arg, either -1 (mean-reversion) or 1 (trend following) and add to config a column for that arg.
- Use average returns for each strategy to guess risk weights that distribute downside in an equilly wighted manner.
- Compute sensitivities of (smoothed) returns to some features that we compute. For that, need to do the following:
- Save indicators and last return to daily HDF5 files. One per strategy instance.
- Once we have the return saved at each run, take first price close from mt5 and rebuild the price time series to see if we match the price at each timestamp.
