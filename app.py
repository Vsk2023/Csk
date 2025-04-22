import os
import glob
import tempfile
import sqlite3
import zipfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ─── PARAMETERS ───────────────────────────────────────
POSITION_USD = 3000
TP_PCT       = 0.005   # 0.5%
SL_PCT       = 0.0075  # 0.75%
BAR_HOURS    = 4       # signals come from 4h bars

# ─── GLOBAL CACHE ─────────────────────────────────────
_tick_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

def load_tick_array(csv_path: str):
    if csv_path in _tick_cache and len(_tick_cache[csv_path][0]) > 0:
        return _tick_cache[csv_path]
    pq_path = Path(csv_path).with_suffix(".parquet")
    if not pq_path.exists():
        df0 = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
        df0.to_parquet(pq_path)
    df = pd.read_parquet(pq_path)
    ts = df.index.values.astype("datetime64[ns]").astype("int64")
    pr = df["price"].values
    _tick_cache[csv_path] = (ts, pr)
    return ts, pr

def simulate_tick_trades(args):
    table, bars, (ts, pr) = args
    trades = []
    for i in range(len(bars) - 1):
        row = bars.iloc[i]
        if not (row["buy"] or row["sell"]):
            continue

        nb = bars.iloc[i + 1]
        entry = nb["open"]
        units = POSITION_USD / entry
        is_long = bool(row["buy"])
        tp_price = entry * (1 + TP_PCT) if is_long else entry * (1 - TP_PCT)
        sl_price = entry * (1 - SL_PCT) if is_long else entry * (1 + SL_PCT)

        start_ns = nb["timestamp"].value
        end_ns   = (nb["timestamp"] + pd.Timedelta(hours=BAR_HOURS)).value
        i0 = np.searchsorted(ts, start_ns, side="left")
        i1 = np.searchsorted(ts, end_ns,   side="right")
        window = pr[i0:i1]
        if window.size == 0:
            continue

        if is_long:
            tp_hits = np.where(window >= tp_price)[0]
            sl_hits = np.where(window <= sl_price)[0]
        else:
            tp_hits = np.where(window <= tp_price)[0]
            sl_hits = np.where(window >= sl_price)[0]

        if tp_hits.size and (not sl_hits.size or tp_hits[0] < sl_hits[0]):
            idx_hit, exit_type = tp_hits[0], "TP Hit"
        elif sl_hits.size:
            idx_hit, exit_type = sl_hits[0], "SL Hit"
        else:
            idx_hit, exit_type = window.size - 1, "Close"

        fill_price = window[idx_hit]
        pnl = ((fill_price - entry) if is_long else (entry - fill_price)) * units
        exit_time = pd.to_datetime(ts[i0 + idx_hit])

        trades.append({
            "Pair":       table,
            "EntryBar":   row["timestamp"],
            "EntryPrice": entry,
            "TP_Price":   tp_price,
            "SL_Price":   sl_price,
            "ExitTime":   exit_time,
            "TradeType":  "Long" if is_long else "Short",
            "ExitType":   exit_type,
            "PnL":        pnl
        })
    return trades


def main():
    st.set_page_config(page_title="🔥 Fast Tick Backtest", layout="wide")
    st.title("🔥 Fast Tick‑Level Backtest on 4h Signals")

    # --- Upload your SQLite DB ---
    uploaded_db = st.sidebar.file_uploader(
        "Upload your SQLite DB of 4H bars + signals", type=["db", "sqlite"]
    )
    if not uploaded_db:
        st.sidebar.info("Please upload a 4H‑bars SQLite DB (.db or .sqlite).")
        st.stop()

    # Save DB locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tf:
        tf.write(uploaded_db.getbuffer())
        db_path = tf.name

    # --- Upload tick CSV files ---
    uploaded_ticks = st.sidebar.file_uploader(
        "Upload tick CSV(s)",
        type="csv",
        accept_multiple_files=True
    )
    if not uploaded_ticks:
        st.sidebar.info("Please upload one or more tick CSV files.")
        st.stop()

    # Save tick files into a temporary directory
    tick_root = tempfile.mkdtemp()
    for up in uploaded_ticks:
        dest = os.path.join(tick_root, up.name)
        with open(dest, "wb") as f:
            f.write(up.getbuffer())

    # Discover tables in DB
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    conn.close()

    # Load bars and map tick CSVs
    table_data, tick_map = {}, {}
    for tbl in tables:
        conn = sqlite3.connect(db_path)
        bars = pd.read_sql(f"SELECT * FROM `{tbl}`", conn)
        conn.close()

        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        for src, dst in [("buy_signal", "buy"), ("sell_signal", "sell")]:
            if src in bars:
                bars[dst] = bars[src].fillna(0).astype(int)
                bars.drop(src, axis=1, inplace=True)
        for col in ["buy", "sell"]:
            bars[col] = bars.get(col, 0).astype(int)

        bars.sort_values("timestamp", inplace=True)
        bars.reset_index(drop=True, inplace=True)
        table_data[tbl] = bars

        symbol = tbl.split("_")[-1]
        # find a matching CSV in the uploaded batch
        matches = glob.glob(os.path.join(tick_root, f"{symbol}*_ticks*.csv"))
        tick_map[tbl] = matches[0] if matches else None

    st.header("Per‑Pair Backtest Results")
    all_trades = []

    for tbl, bars in table_data.items():
        csv = tick_map.get(tbl)
        if not csv:
            st.warning(f"No tick file found for {tbl}, skipping.")
            continue

        ts, pr = load_tick_array(csv)
        with st.spinner(f"Running backtest for {tbl}…"):
            trades = simulate_tick_trades((tbl, bars, (ts, pr)))

        if trades:
            df_pair = pd.DataFrame(trades)
            st.subheader(f"{tbl} — {len(df_pair)} trades")
            st.dataframe(df_pair)
            all_trades.extend(trades)
        else:
            st.warning(f"{tbl}: no trades generated.")

    # Aggregate download
    if all_trades:
        df_all = pd.DataFrame(all_trades)
        csv_bytes = df_all.to_csv(index=False).encode()
        st.download_button(
            "Download all pairs results",
            csv_bytes,
            "tick_backtest_all.csv",
            "text/csv"
        )
    else:
        st.info("No trades across any pairs.")

if __name__ == "__main__":
    main()
