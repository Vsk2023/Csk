# app.py
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
CHUNK_SIZE   = 10**6   # number of rows per CSV chunk

# ─── UTILITIES ────────────────────────────────────────

def load_tick_window(csv_path: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
    """
    Read only the portion of the tick CSV between start_time and end_time, in chunks.
    Returns two np.ndarrays: timestamps (ns) and prices.
    """
    times_list = []
    prices_list = []
    # iterate in chunks to avoid loading the entire file
    for chunk in pd.read_csv(
        csv_path,
        parse_dates=["time"],
        usecols=["time", "price"],
        chunksize=CHUNK_SIZE
    ):
        # filter rows by timestamp window
        mask = (chunk["time"] >= start_time) & (chunk["time"] < end_time)
        sel = chunk.loc[mask]
        if not sel.empty:
            ts = sel["time"].values.astype("datetime64[ns]").astype("int64")
            pr = sel["price"].values
            times_list.append(ts)
            prices_list.append(pr)
    if times_list:
        ts_all = np.concatenate(times_list)
        pr_all = np.concatenate(prices_list)
    else:
        ts_all = np.array([], dtype="int64")
        pr_all = np.array([], dtype="float64")
    return ts_all, pr_all


def simulate_tick_trades(args):
    table, bars, csv_path = args
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

        start_time = nb["timestamp"]
        end_time = start_time + pd.Timedelta(hours=BAR_HOURS)
        # load only the relevant tick window
        ts_window, pr_window = load_tick_window(csv_path, start_time, end_time)
        if ts_window.size == 0:
            continue

        # determine hit indices
        if is_long:
            tp_hits = np.where(pr_window >= tp_price)[0]
            sl_hits = np.where(pr_window <= sl_price)[0]
        else:
            tp_hits = np.where(pr_window <= tp_price)[0]
            sl_hits = np.where(pr_window >= sl_price)[0]

        if tp_hits.size and (not sl_hits.size or tp_hits[0] < sl_hits[0]):
            idx_hit, exit_type = tp_hits[0], "TP Hit"
        elif sl_hits.size:
            idx_hit, exit_type = sl_hits[0], "SL Hit"
        else:
            idx_hit, exit_type = len(pr_window) - 1, "Close"

        fill_price = pr_window[idx_hit]
        pnl = ((fill_price - entry) if is_long else (entry - fill_price)) * units
        exit_time = pd.to_datetime(ts_window[idx_hit])

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tf:
        tf.write(uploaded_db.getbuffer())
        db_path = tf.name

    # --- Upload tick CSV files (can be large) ---
    uploaded_ticks = st.sidebar.file_uploader(
        "Upload tick CSV(s)",
        type="csv",
        accept_multiple_files=True
    )
    if not uploaded_ticks:
        st.sidebar.info("Please upload your tick CSV files (they can be large).")
        st.stop()

    # Save tick CSVs into a temp directory
    tick_root = tempfile.mkdtemp()
    for up in uploaded_ticks:
        dest = os.path.join(tick_root, up.name)
        with open(dest, "wb") as f:
            f.write(up.getbuffer())

    # Discover tables in the DB
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    conn.close()

    # Load bars & map each table to its CSV
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
        # find the uploaded CSV matching this symbol
        matches = glob.glob(os.path.join(tick_root, f"{symbol}*_ticks*.csv"))
        tick_map[tbl] = matches[0] if matches else None

    st.header("Per‑Pair Backtest Results")
    all_trades = []

    for tbl, bars in table_data.items():
        csv_path = tick_map.get(tbl)
        if not csv_path:
            st.warning(f"No tick file found for {tbl}, skipping.")
            continue

        with st.spinner(f"Running backtest for {tbl}…"):
            trades = simulate_tick_trades((tbl, bars, csv_path))

        if trades:
            df_pair = pd.DataFrame(trades)
            st.subheader(f"{tbl} — {len(df_pair)} trades")
            st.dataframe(df_pair)
            all_trades.extend(trades)
        else:
            st.warning(f"{tbl}: no trades generated.")

    # Download aggregated results
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
