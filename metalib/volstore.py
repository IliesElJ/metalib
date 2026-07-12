"""
SQLite-backed store for volatility predictions, shared across strategy
processes.

Writers: MetaHAR saves one row per model bar from _process_predictions()
(INSERT OR REPLACE keyed on (tag, symbol, ts), so re-runs within the same bar
just refresh the row).

Readers: other strategies use get_latest()/get_all_latest()/get_history(),
either directly or via MetaStrategy.get_vol_prediction(). SQLite in WAL mode
handles concurrent PM2 processes (single writer at a time, readers never
blocked); every call opens and closes its own connection, so there is no
shared state to corrupt.

Conventions:
- ts is the right-edge label of the feature bar (MT5 server time, naive);
  the prediction applies to the NEXT model bar, i.e. (ts, ts + horizon_min].
- Staleness checks use created_at (UTC wall clock at write time), because MT5
  server time is broker-dependent (often UTC+2/3) and would misjudge age.
"""

import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "store", "vol_predictions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vol_predictions (
    tag TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    horizon_min INTEGER NOT NULL,
    prediction REAL NOT NULL,
    realized_prev_diff REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tag, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_vol_pred_symbol_ts
    ON vol_predictions (symbol, ts);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=5.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.executescript(_SCHEMA)
    return con


def save_prediction(tag, symbol, ts, horizon_min, prediction, realized_prev_diff=None):
    """Insert or refresh the prediction row for (tag, symbol, ts)."""
    con = _connect()
    try:
        con.execute(
            "INSERT OR REPLACE INTO vol_predictions "
            "(tag, symbol, ts, horizon_min, prediction, realized_prev_diff, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                tag,
                symbol,
                str(pd.Timestamp(ts)),
                int(horizon_min),
                float(prediction),
                None if realized_prev_diff is None else float(realized_prev_diff),
                datetime.utcnow().isoformat(),
            ),
        )
        con.commit()
    finally:
        con.close()


def get_latest(symbol, tag=None, max_age_minutes=None):
    """Most recent prediction for a symbol as a dict, or None.

    max_age_minutes filters on created_at (UTC): stale rows — e.g. a stopped
    metahar instance — return None rather than a misleading old forecast.
    """
    query = "SELECT * FROM vol_predictions WHERE symbol = ?"
    params = [symbol]
    if tag is not None:
        query += " AND tag = ?"
        params.append(tag)
    query += " ORDER BY ts DESC LIMIT 1"

    con = _connect()
    try:
        row = con.execute(query, params).fetchone()
    finally:
        con.close()

    if row is None:
        return None
    out = dict(row)
    if max_age_minutes is not None:
        age = datetime.utcnow() - datetime.fromisoformat(out["created_at"])
        if age > timedelta(minutes=max_age_minutes):
            return None
    return out


def get_all_latest(max_age_minutes=None):
    """Latest prediction per symbol as a DataFrame (empty if none)."""
    con = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT p.* FROM vol_predictions p "
            "JOIN (SELECT symbol, MAX(ts) AS max_ts FROM vol_predictions GROUP BY symbol) m "
            "ON p.symbol = m.symbol AND p.ts = m.max_ts",
            con,
        )
    finally:
        con.close()

    if max_age_minutes is not None and not df.empty:
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        df = df[pd.to_datetime(df["created_at"]) >= cutoff]
    return df.reset_index(drop=True)


def get_history(symbol=None, tag=None, start=None, end=None):
    """Prediction history as a DataFrame, optionally filtered."""
    query = "SELECT * FROM vol_predictions WHERE 1=1"
    params = []
    if symbol is not None:
        query += " AND symbol = ?"
        params.append(symbol)
    if tag is not None:
        query += " AND tag = ?"
        params.append(tag)
    if start is not None:
        query += " AND ts >= ?"
        params.append(str(pd.Timestamp(start)))
    if end is not None:
        query += " AND ts <= ?"
        params.append(str(pd.Timestamp(end)))
    query += " ORDER BY ts"

    con = _connect()
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()
