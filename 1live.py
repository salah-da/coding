# path_match_tool.py
# Live path-matching tool (NOT a trading bot).
# Run:  streamlit run path_match_tool.py

import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ------------------------------- Utils -------------------------------

BINANCE_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000
}

def to_binance_symbol(sym: str) -> str:
    s = sym.strip().upper().replace("/", "")
    return s

def fetch_last_n_klines(symbol: str, interval: str, n: int, futures: bool=True, rate_sleep: float=0.2) -> pd.DataFrame:
    """Fetch last n klines from Binance Futures/Spot, paginating 1500-limit."""
    base = "https://fapi.binance.com/fapi/v1/klines" if futures else "https://api.binance.com/api/v3/klines"
    sym = to_binance_symbol(symbol)
    tf_ms = BINANCE_INTERVAL_MS[interval]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - n * tf_ms

    rows = []
    cur = start_ms
    remain = n
    limit = 1500
    while remain > 0:
        this_limit = min(limit, remain)
        params = {"symbol": sym, "interval": interval, "startTime": cur, "limit": this_limit}
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_close = int(batch[-1][6])  # close time
        next_start = last_close + 1
        if next_start <= cur:
            break
        got = len(batch)
        remain -= got
        cur = next_start
        time.sleep(rate_sleep)

    if not rows:
        raise RuntimeError("No klines fetched.")
    K = np.array(rows, dtype=object)
    open_ts = K[:,0].astype(np.int64)
    open_, high_, low_, close_, vol_ = map(lambda c: K[:,c].astype(float), [1,2,3,4,5])
    dt = pd.to_datetime(open_ts, unit="ms", utc=True)
    df = pd.DataFrame({"open":open_, "high":high_, "low":low_, "close":close_, "volume":vol_}, index=dt)
    df.index.name = "time"
    df = df[~df.index.duplicated(keep="last")]
    return df

def mean_center(x: np.ndarray) -> np.ndarray:
    m = np.mean(x)
    return x - m

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def velocities(series: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ΔP":
        v = np.diff(series, prepend=series[0])
    else:  # ΔlogP
        s = np.where(series<=0, np.nan, series)
        v = np.diff(np.log(s), prepend=np.log(s[0]))
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v

def std_ratio(win_a: np.ndarray, win_b: np.ndarray) -> float:
    sa = np.std(win_a)
    sb = np.std(win_b)
    if sb == 0:
        return np.inf if sa>0 else 1.0
    return float(sa / sb)

def swing_similarity_from_ratio(r: float, kappa: float=0.25) -> float:
    # 0..100% where 100% at r=1, decays with |ln r|
    return float(100.0 * math.exp(-abs(math.log(max(r, 1e-12))) / max(kappa, 1e-6)))

def ema(x, span: int):
    a = 2.0 / (span + 1.0)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a*x[i] + (1-a)*y[i-1]
    return y

# ----- DTW helpers -----

def huber(x: float, delta: float=1.5) -> float:
    ax = abs(x)
    return 0.5*ax*ax if ax <= delta else delta*(ax - 0.5*delta)

def dtw_banded_similarity(a: np.ndarray, b: np.ndarray, band: int=2,
                          loss: str="sq", delta: float=1.5) -> float:
    """
    Banded DTW on equal-length series with average-path normalization.
    Returns a bounded similarity in (0,1], where 1 == perfect match.
    Inputs should already be mean-centered (for price) or standardized.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    W = len(a)
    INF = 1e18
    D = np.full((W+1, W+1), INF, dtype=float)
    D[0,0] = 0.0

    if loss == "sq":
        def lc(i,j):
            d = a[i]-b[j]
            return d*d
    else:
        def lc(i,j):
            return huber(a[i]-b[j], delta)

    # dynamic programming within Sakoe–Chiba band
    for i in range(1, W+1):
        j_min, j_max = max(1, i-band), min(W, i+band)
        for j in range(j_min, j_max+1):
            c = lc(i-1, j-1)
            D[i,j] = c + min(D[i-1,j], D[i,j-1], D[i-1,j-1])

    # estimate path length by simple backtrack (good enough)
    i, j, steps = W, W, 0
    while i>0 and j>0 and steps < 4*W:
        steps += 1
        candidates = [(D[i-1,j], i-1, j), (D[i,j-1], i, j-1), (D[i-1,j-1], i-1, j-1)]
        _, ni, nj = min(candidates, key=lambda t: t[0])
        i, j = ni, nj
    path_len = max(1, steps)

    avg_cost = D[W,W] / path_len
    sim = 1.0 / (1.0 + avg_cost)   # in (0,1]
    return float(sim)

# ------------------------------- UI -------------------------------

st.set_page_config(page_title="Path Match Tool", layout="wide")

with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Symbol", value="SOL/USDT", help="Format like SOL/USDT or BTC/USDT")
    interval = st.selectbox("Timeframe", options=list(BINANCE_INTERVAL_MS.keys()), index=0,
                            help="Candle interval")
    futures = st.toggle("Use Binance Futures (USDT-M)", value=True, help="If off, use Binance Spot")
    N_CANDLES = st.number_input("N_CANDLES (fetch)", min_value=500, max_value=100000, value=5000, step=500,
                                help="How many most-recent candles to pull (auto-paginated)")
    W = st.number_input("W (path length, bars)", min_value=10, max_value=1000, value=60, step=5,
                        help="Bars in each path window")
    H = st.number_input("H (horizon, bars for labeling winners)", min_value=5, max_value=5000, value=120, step=5,
                        help="Only for labeling winners; outcomes must be known at t+H")
    WIN_THRESH = st.number_input("WIN_THRESH (abs price change)", min_value=0.0, value=0.5, step=0.1,
                                 help="Absolute price move over H to call a winner (>= +THRESH long, <= -THRESH short)")

    st.divider()
    USE_VEL = st.toggle("Use velocity channel", value=True, help="Match price AND velocity shapes")
    vel_kind = st.selectbox("Velocity type", options=["ΔP","ΔlogP"], index=0,
                            help="ΔP = price differences; ΔlogP = log returns")
    THETA_PRICE = st.slider("THETA_PRICE (price similarity gate)", min_value=0.0, max_value=1.0, value=0.90, step=0.01,
                            help="For Cosine: gate on cosine [-1..1] with same threshold; For DTW: gate on [0..1]")
    THETA_VEL = st.slider("THETA_VEL (velocity similarity gate)", min_value=0.0, max_value=1.0, value=0.75, step=0.01,
                          help="For Cosine: gate on cosine [-1..1] with same threshold; For DTW: gate on [0..1]")
    STD_MODE = st.selectbox("Std-ratio domain", options=["price","returns"], index=0,
                            help="Compare std of prices over W or std of returns over W")
    R_MIN = st.number_input("R_MIN (std-ratio lower)", min_value=0.1, value=0.85, step=0.05)
    R_MAX = st.number_input("R_MAX (std-ratio upper)", min_value=0.2, value=1.20, step=0.05)
    K_MAX = st.number_input("K_MAX winners cap", min_value=10, max_value=10000, value=1000, step=10,
                            help="If more winners exist, keep top-K by |forward move|")
    KAPPA = st.slider("Swing similarity κ", min_value=0.05, max_value=1.0, value=0.25, step=0.05,
                      help="Controls how strict the price-swing similarity % is from std-ratio")

    st.divider()
    st.subheader("Similarity methods")
    PRICE_METHOD = st.selectbox("Price similarity", options=["Cosine","DTW"], index=0,
                                help="Cosine uses mean-centered prices; DTW allows tiny time warps")
    VEL_METHOD   = st.selectbox("Velocity similarity", options=["Cosine","DTW"], index=0,
                                help="Cosine uses mean-centered velocities; DTW is more timing-tolerant")

    st.caption("DTW parameters")
    DTW_BAND  = st.slider("DTW band (±bars)", min_value=1, max_value=5, value=2, step=1,
                          help="Sakoe–Chiba band width; small = tighter warping")
    DTW_LOSS  = st.selectbox("DTW local loss", options=["sq","huber"], index=0,
                             help="Squared error or Huber loss for robustness")
    DTW_DELTA = st.number_input("Huber δ (if used)", min_value=0.1, value=1.5, step=0.1)

    st.caption("Note: THETA_PRICE/THETA_VEL gate Cosine ([-1..1]) and DTW ([0..1]) similarities.")

    st.divider()
    # NEW: how many overlays / table rows
    TOP_N_PRICE = st.number_input("TOP_N_PRICE overlays (price)", min_value=1, max_value=20, value=5, step=1)
    TOP_N_VEL   = st.number_input("TOP_N_VEL overlays (velocity)", min_value=1, max_value=20, value=5, step=1)
    MAX_TABLE_ROWS = st.number_input("MAX rows in table", min_value=20, max_value=5000, value=500, step=20)

    st.divider()
    if st.button("Refresh data / Rebuild library", type="primary", use_container_width=True):
        st.session_state.get("refresh_key", 0)
        st.session_state["refresh_key"] = st.session_state.get("refresh_key", 0) + 1

# ------------------------------- Data -------------------------------

status = st.empty()
try:
    status.info("Fetching data from Binance…")
    df = fetch_last_n_klines(symbol, interval, int(N_CANDLES), futures=futures)
    status.success(f"Got {len(df)} candles from {df.index[0]} to {df.index[-1]}")
except Exception as e:
    status.error(f"Data fetch failed: {e}")
    st.stop()

close = df["close"].values.astype(float)
vel_raw  = velocities(close, vel_kind)
vel_full = ema(vel_raw, span=5)   # smooth velocity

# --------------------------- Build Winners --------------------------

if len(df) < W + H + 5:
    st.warning("Not enough data for chosen W and H.")
    st.stop()

winners = []
for t in range(W-1, len(df)-H):
    win_slice = slice(t-W+1, t+1)    # W bars ending at t
    P_win = close[win_slice]
    V_win = vel_full[win_slice]

    # Label by absolute move over H (signed by direction)
    delta = close[t+H] - close[t]
    side = 0
    if delta >= WIN_THRESH:
        side = +1
    elif delta <= -WIN_THRESH:
        side = -1
    if side == 0:
        continue

    Px = mean_center(P_win)
    Vx = mean_center(V_win)

    winners.append({
        "t_idx": t,
        "time": df.index[t],
        "side": side,
        "delta": float(delta),
        "Px": Px,
        "Vx": Vx,
        "P_raw": P_win.copy(),
        "V_raw": V_win.copy()
    })

if not winners:
    st.warning("No winners found. Try lowering WIN_THRESH or increasing H / N_CANDLES.")
    st.stop()

# Cap to K_MAX by absolute forward move
if len(winners) > K_MAX:
    winners = sorted(winners, key=lambda d: abs(d["delta"]), reverse=True)[:int(K_MAX)]

# --------------------------- Current Path ---------------------------

cur_P = close[-W:]
cur_V = vel_full[-W:]

cur_Px = mean_center(cur_P)
cur_Vx = mean_center(cur_V)

# ----------------------------- Matching -----------------------------

def std_gate(cur_raw, tpl_raw, cur_raw_ret, tpl_raw_ret) -> float:
    if STD_MODE == "price":
        return std_ratio(cur_raw, tpl_raw)
    else:  # returns (simple ΔP here; swap to log returns if preferred)
        return std_ratio(cur_raw_ret, tpl_raw_ret)

matches = []
for tpl in winners:

    # ----- PRICE similarity -----
    if PRICE_METHOD == "Cosine":
        s_price = cosine(cur_Px, tpl["Px"])  # [-1,1]
        if np.isnan(s_price) or s_price < THETA_PRICE:
            continue
        s_price_percent = abs(s_price) * 100.0
    else:  # DTW
        s_price = dtw_banded_similarity(cur_Px, tpl["Px"],
                                        band=int(DTW_BAND), loss=DTW_LOSS, delta=float(DTW_DELTA))  # (0,1]
        if s_price < THETA_PRICE:
            continue
        s_price_percent = s_price * 100.0

    # ----- VELOCITY similarity (optional) -----
    s_vel = None; s_vel_percent = None
    if USE_VEL:
        if VEL_METHOD == "Cosine":
            s_vel = cosine(cur_Vx, tpl["Vx"])  # [-1,1]
            if np.isnan(s_vel) or s_vel < THETA_VEL:
                continue
            s_vel_percent = abs(s_vel) * 100.0
        else:  # DTW
            s_vel = dtw_banded_similarity(cur_Vx, tpl["Vx"],
                                          band=int(DTW_BAND), loss=DTW_LOSS, delta=float(DTW_DELTA))  # (0,1]
            if s_vel < THETA_VEL:
                continue
            s_vel_percent = s_vel * 100.0

    # ----- std-ratio in chosen domain -----
    cur_ret_win = np.diff(cur_P, prepend=cur_P[0])          # NOTE: change to log returns if you prefer returns-mode to be log
    tpl_ret_win = np.diff(tpl["P_raw"], prepend=tpl["P_raw"][0])
    r = std_gate(cur_P, tpl["P_raw"], cur_ret_win, tpl_ret_win)
    if not (R_MIN <= r <= R_MAX):
        continue

    matches.append({
        "time": tpl["time"],
        "side": "LONG" if tpl["side"]==1 else "SHORT",
        "delta": tpl["delta"],
        "s_price": s_price,
        "s_price_%": s_price_percent,
        "s_vel": s_vel if USE_VEL else None,
        "s_vel_%": s_vel_percent,
        "std_ratio": r,
        "swing_%": swing_similarity_from_ratio(r, KAPPA),
        "P_tpl": tpl["P_raw"],
        "V_tpl": tpl["V_raw"]
    })

# --------------------------- Reporting UI ---------------------------

col_top = st.columns(2)
with col_top[0]:
    st.subheader("Current path (last W bars)")
    st.line_chart(pd.DataFrame({"price": cur_P}, index=df.index[-W:]))

with col_top[1]:
    st.subheader("Velocity (if enabled)")
    if USE_VEL:
        st.line_chart(pd.DataFrame({"velocity": cur_V}, index=df.index[-W:]))
    else:
        st.info("Velocity channel disabled")

st.caption(f"Price similarity: {PRICE_METHOD} | Velocity similarity: {'OFF' if not USE_VEL else VEL_METHOD}")

st.divider()

if not matches:
    st.warning("No matches at current bar with the chosen thresholds.\n"
               "Tips: lower THETA_PRICE/THETA_VEL, widen [R_MIN, R_MAX], increase N_CANDLES or reduce WIN_THRESH.")
    st.stop()

m = pd.DataFrame(matches)
m_long = m[m["side"]=="LONG"]
m_short= m[m["side"]=="SHORT"]

def agg_side(df_side: pd.DataFrame):
    if df_side.empty:
        return {"count":0,"avg_delta":np.nan,"med_delta":np.nan,"avg_s_price%":np.nan,"avg_swing%":np.nan}
    return {
        "count": int(len(df_side)),
        "avg_delta": float(df_side["delta"].mean()),
        "med_delta": float(df_side["delta"].median()),
        "avg_s_price%": float(df_side["s_price_%"].mean()),
        "avg_swing%": float(df_side["swing_%"].mean())
    }

stats_long  = agg_side(m_long)
stats_short = agg_side(m_short)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matched LONG", stats_long["count"])
c2.metric("Avg profit (LONG)", f'{stats_long["avg_delta"]:.4f}')
c3.metric("Avg shape % (LONG)", f'{stats_long["avg_s_price%"]:.1f}%')
c4.metric("Avg swing % (LONG)", f'{stats_long["avg_swing%"]:.1f}%')

d1, d2, d3, d4 = st.columns(4)
d1.metric("Matched SHORT", stats_short["count"])
d2.metric("Avg profit (SHORT)", f'{stats_short["avg_delta"]:.4f}')
d3.metric("Avg shape % (SHORT)", f'{stats_short["avg_s_price%"]:.1f}%')
d4.metric("Avg swing % (SHORT)", f'{stats_short["avg_swing%"]:.1f}%')

st.divider()

# ---------- Top matches (overlay) — PRICE ----------
st.subheader("Top matches (overlay) — Price (mean-centered)")
ov_cols = st.columns(2)
for idx, (title, df_side) in enumerate([("LONG", m_long), ("SHORT", m_short)]):
    with ov_cols[idx]:
        n_here = int(min(TOP_N_PRICE, max(0, len(df_side))))
        st.caption(f"Top {n_here} {title} overlays (mean-centered price)")
        overlays = {"Current (centered)": mean_center(cur_P)}
        for _, row in df_side.sort_values("s_price", ascending=False).head(n_here).iterrows():
            overlays[f"{row['time']} (shape {row['s_price_%']:.0f}%)"] = mean_center(row["P_tpl"])
        idx_w = np.arange(len(overlays["Current (centered)"]))
        df_ov = pd.DataFrame({"idx": idx_w}).set_index("idx")
        for k, v in overlays.items():
            df_ov[k] = v
        st.line_chart(df_ov)

st.divider()

# ---------- Top matches (overlay) — VELOCITY ----------
st.subheader("Top matches (overlay) — Velocity (mean-centered)")
if not USE_VEL:
    st.info("Velocity channel is disabled.")
else:
    cur_V_centered = mean_center(cur_V)
    ov_cols_v = st.columns(2)
    for idx, (title, df_side) in enumerate([("LONG", m_long), ("SHORT", m_short)]):
        with ov_cols_v[idx]:
            n_here = int(min(TOP_N_VEL, max(0, len(df_side))))
            st.caption(f"Top {n_here} {title} overlays (mean-centered velocity)")
            overlays_v = {"Current vel (centered)": cur_V_centered}
            # Prefer ranking by velocity similarity if available, else price similarity
            rank_key = "s_vel" if ("s_vel" in df_side.columns and df_side["s_vel"].notna().any()) else "s_price"
            sorted_side = df_side.sort_values(rank_key, ascending=False).head(n_here)
            for _, row in sorted_side.iterrows():
                tpl_vel_centered = mean_center(row["V_tpl"])
                disp = row["s_vel_%"] if (rank_key=="s_vel" and row.get("s_vel_%") is not None) else row["s_price_%"]
                overlays_v[f"{row['time']} ({'vel' if rank_key=='s_vel' else 'price'} {disp:.0f}%)"] = tpl_vel_centered
            idx_w = np.arange(len(cur_V_centered))
            df_ov_v = pd.DataFrame({"idx": idx_w}).set_index("idx")
            for k, v in overlays_v.items():
                df_ov_v[k] = v
            st.line_chart(df_ov_v)

st.divider()
st.subheader("Matched winners (table)")
show_cols = ["time","side","delta","s_price","s_price_%","s_vel","s_vel_%","std_ratio","swing_%"]
tbl = m[show_cols].sort_values(["side","s_price"], ascending=[True,False])
if len(tbl) > int(MAX_TABLE_ROWS):
    st.caption(f"Showing first {int(MAX_TABLE_ROWS)} of {len(tbl)} rows")
    tbl = tbl.head(int(MAX_TABLE_ROWS))
st.dataframe(tbl, use_container_width=True)

