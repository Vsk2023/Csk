# app.py
import os
import glob
import tempfile
import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import fsspec

# ─── PARAMETERS ───────────────────────────────────────
POSITION_USD = 3000
TP_PCT       = 0.005   # 0.5%
SL_PCT       = 0.0075  # 0.75%
BAR_HOURS    = 4       # signals come from 4h bars
CHUNK_SIZE   = 10**6   # rows per chunk when streaming

# ─── UTILITIES ────────────────────────────────────────
def load_tick_window(path: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
    """
    Stream tick data in chunks (local or S3) between start_time and end_time.
    Returns timestamps (int64 ns) and prices.
    """
    times_list = []
    prices_list = []
    reader = None
    # choose file handle for local or S3
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        file_obj = fs.open(path, mode="rb")
        reader = pd.read_csv(
            file_obj,
            parse_dates=["time"],
            usecols=["time", "price"],
            chunksize=CHUNK_SIZE
        )
    else:
        reader = pd.read_csv(
            path,
            parse_dates=["time"],
            usecols=["time", "price"],
            chunksize=CHUNK_SIZE
        )
    # iterate chunks
    for chunk in reader:
        mask = (chunk["time"] >= start_time) & (chunk["time"] < end_time)
        sel = chunk.loc[mask]
        if not sel.empty:
            ts = sel["time"].values.astype("datetime64[ns]").astype("int64")
            pr = sel["price"].values
            times_list.append(ts)
            prices_list.append(pr)
    if times_list:
        return np.concatenate(times_list), np.concatenate(prices_list)
    return np.array([], dtype="int64"), np.array([], dtype="float64")


def simulate_tick_trades(args):
    table, bars, path = args
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
        ts_win, pr_win = load_tick_window(path, start_time, end_time)
        if ts_win.size == 0:
            continue
        if is_long:
            tp_hits = np.where(pr_win >= tp_price)[0]
            sl_hits = np.where(pr_win <= sl_price)[0]
        else:
            tp_hits = np.where(pr_win <= tp_price)[0]
            sl_hits = np.where(pr_win >= sl_price)[0]
        if tp_hits.size and (not sl_hits.size or tp_hits[0] < sl_hits[0]):
            idx, exit_type = tp_hits[0], "TP Hit"
        elif sl_hits.size:
            idx, exit_type = sl_hits[0], "SL Hit"
        else:
            idx, exit_type = len(pr_win) - 1, "Close"
        fill = pr_win[idx]
        pnl = ((fill - entry) if is_long else (entry - fill)) * units
        exit_time = pd.to_datetime(ts_win[idx])
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
    # --- Database upload ---
    uploaded_db = st.sidebar.file_uploader(
        "Upload 4h‑bars SQLite DB", type=["db", "sqlite"]
    )
    if not uploaded_db:
        st.sidebar.info("Upload your SQLite DB (.db/.sqlite)")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tf:
        tf.write(uploaded_db.getbuffer())
        db_path = tf.name
    # --- Remote tick data prefix ---
    tick_root = st.sidebar.text_input(
        "Tick data root (e.g. s3://my-bucket/ticks)", ""
    )
    if not tick_root:
        st.sidebar.info("Enter your tick data root path (S3 or local)")
        st.stop()
    # --- Discover tables ---
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()
    conn.close()
    # --- Load bars & map to paths ---
    table_data, tick_map = {}, {}
    for tbl in tables:
        conn = sqlite3.connect(db_path)
        bars = pd.read_sql(f"SELECT * FROM `{tbl}`", conn)
        conn.close()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        for src, dst in [("buy_signal","buy"),("sell_signal","sell")]:
            if src in bars:
                bars[dst] = bars[src].fillna(0).astype(int)
                bars.drop(src, axis=1, inplace=True)
        for col in ["buy","sell"]:
            bars[col] = bars.get(col,0).astype(int)
        bars.sort_values("timestamp", inplace=True)
        bars.reset_index(drop=True, inplace=True)
        table_data[tbl] = bars
        sym = tbl.split("_")[-1]
        if tick_root.startswith("s3://"):
            fs = fsspec.filesystem("s3")
            found = fs.glob(f"{tick_root}/{sym}*_ticks*.csv")
            tick_map[tbl] = found[0] if found else None
        else:
            cand = glob.glob(os.path.join(tick_root, f"{sym}*_ticks*.csv"))
            tick_map[tbl] = cand[0] if cand else None
    st.header("Per‑Pair Backtest Results")
    all_trades = []
    for tbl, bars in table_data.items():
        path = tick_map.get(tbl)
        if not path:
            st.warning(f"No tick file for {tbl}, skipping.")
            continue
        with st.spinner(f"Backtesting {tbl}…"):
            trades = simulate_tick_trades((tbl, bars, path))
        if trades:
            df = pd.DataFrame(trades)
            st.subheader(f"{tbl}: {len(df)} trades")
            st.dataframe(df)
            all_trades.extend(trades)
        else:
            st.warning(f"{tbl}: no trades generated.")
    if all_trades:
        df_all = pd.DataFrame(all_trades)
        buf = df_all.to_csv(index=False).encode()
        st.download_button("Download CSV", buf, "all_trades.csv","text/csv")
    else:
        st.info("No trades across any pairs.")

if __name__ == "__main__":
    main()
