"""
app.py

Single-file prototype for:
- Binance tick ingestion (via websocket)
- SQLite tick storage + in-memory recent buffer
- Resampling and analytics (OLS hedge, spread, z-score, ADF, rolling corr)
- Dash frontend with interactive charts, alerts, CSV export

Run: python app.py
"""

import asyncio
import json
import threading
import time
import math
import sqlite3
from collections import deque, defaultdict
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

# Websockets for Binance
import websockets

# ----------------------------
# Config / Globals
# ----------------------------

SQLITE_DB = "ticks.db"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # default symbols to connect
IN_MEMORY_MAX_TICKS = 20000  # keep recent ticks per symbol
BINANCE_WSS = "wss://stream.binance.com:9443/ws"

# Per-symbol ring buffer for low-latency analytics
recent_ticks = defaultdict(lambda: deque(maxlen=IN_MEMORY_MAX_TICKS))

# SQLite connection (use same DB in threads, so use check_same_thread=False)
conn = sqlite3.connect(SQLITE_DB, check_same_thread=False)
cursor = conn.cursor()

# Create table if missing
cursor.execute("""
CREATE TABLE IF NOT EXISTS ticks (
    ts INTEGER,      -- epoch ms
    symbol TEXT,
    price REAL,
    qty REAL
)
""")
conn.commit()

# Utility: insert tick
def save_tick_sqlite(ts_ms:int, symbol:str, price:float, qty:float):
    cursor.execute("INSERT INTO ticks (ts, symbol, price, qty) VALUES (?, ?, ?, ?)",
                   (int(ts_ms), symbol, float(price), float(qty)))
    conn.commit()

# ----------------------------
# Ingestion: Binance websocket client
# ----------------------------

async def subscribe_and_listen(symbols):
    """
    Subscribe to combined trade streams on Binance (aggTrade or trade)
    We'll use trade streams: <symbol>@trade
    """
    # Construct combined stream url path
    stream_names = "/".join([f"{s.lower()}@trade" for s in symbols])
    url = f"wss://stream.binance.com:9443/stream?streams={stream_names}"
    print(f"[ingest] connecting to {url}")
    async with websockets.connect(url, ping_interval=20) as ws:
        print("[ingest] connected")
        async for message in ws:
            try:
                data = json.loads(message)
                # https://binance-docs.github.io/apidocs/spot/en/#trade-streams
                # For combined streams payload has 'data' key
                payload = data.get("data", {})
                # fields: 'E' Event time (ms), 's' symbol, 'p' price, 'q' qty
                ts = int(payload.get("E", time.time()*1000))
                sym = payload.get("s")
                price = float(payload.get("p", 0.0))
                qty = float(payload.get("q", 0.0))

                # Save in-memory
                recent_ticks[sym].append((ts, price, qty))
                # Persist (non-blocking acceptable for small demo)
                save_tick_sqlite(ts, sym, price, qty)
            except Exception as e:
                print("[ingest] error parsing message:", e)

def start_ingestion(symbols):
    """
    Start the asyncio event loop and run subscribe_and_listen in a background thread.
    """
    async def runner():
        while True:
            try:
                await subscribe_and_listen(symbols)
            except Exception as e:
                print("[ingest] connection failed, reconnecting in 2s:", e)
                await asyncio.sleep(2)

    def thread_target():
        asyncio.run(runner())

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()
    print("[ingest] ingestion thread started")

# ----------------------------
# Analytics Functions
# ----------------------------

def ticks_to_df(ticks):
    """ticks: list of (ts_ms, price, qty)"""
    if not ticks:
        return pd.DataFrame(columns=["ts", "price", "qty"]).set_index("ts")
    df = pd.DataFrame(ticks, columns=["ts", "price", "qty"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").sort_index()
    return df

def resample_ticks(symbol, timeframe='1s'):
    """Return OHLCV style frame for timeframe like '1s','1min','5min'"""
    ticks = list(recent_ticks[symbol])
    df = ticks_to_df(ticks)
    if df.empty:
        # attempt to pull last 1000 ticks from SQLite for this symbol
        q = f"SELECT ts, price, qty FROM ticks WHERE symbol = ? ORDER BY ts DESC LIMIT 2000"
        rows = cursor.execute(q, (symbol,)).fetchall()
        df = ticks_to_df([(r[0], r[1], r[2]) for r in reversed(rows)])
        if df.empty:
            return df
    # Resample
    ohlc = df['price'].resample(timeframe).ohlc()
    vol = df['qty'].resample(timeframe).sum()
    ohlc['volume'] = vol
    ohlc = ohlc.dropna()
    return ohlc

def compute_hedge_ratio(series_x, series_y):
    """
    Linear OLS hedge ratio: regress Y ~ X (no intercept or with intercept? We'll use intercept optional).
    Returns beta (hedge ratio), residuals (spread) as array, and R2.
    """
    # align
    df = pd.concat([series_x, series_y], axis=1).dropna()
    if df.shape[0] < 5:
        return None, None, None
    X = df.iloc[:,0].values.reshape(-1,1)
    Y = df.iloc[:,1].values
    X1 = np.hstack([np.ones_like(X), X])  # intercept
    beta_full, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    intercept, slope = float(beta_full[0]), float(beta_full[1])
    pred = intercept + slope * X.flatten()
    resid = Y - pred
    # R^2
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
    return (intercept, slope), resid, r2

def compute_zscore(spread):
    s = pd.Series(spread)
    return (s - s.mean()) / (s.std() if s.std()!=0 else 1.0)

def compute_adf(series):
    try:
        res = adfuller(series, autolag='AIC', maxlag=10)
        # return p-value and statistic
        return {"adf_stat": float(res[0]), "pvalue": float(res[1]), "usedlag": int(res[2])}
    except Exception as e:
        return {"error": str(e)}

def rolling_corr(series_x, series_y, window):
    df = pd.concat([series_x, series_y], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[:,0].rolling(window).corr(df.iloc[:,1])

# ----------------------------
# Dash App
# ----------------------------

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Quant Dev — Live Tick Analytics Demo"),
    html.Div([
        html.Label("Symbols (comma)"),
        dcc.Input(id="symbols-in", type="text", value="BTCUSDT,ETHUSDT", style={"width":"40%"}),
        html.Button("Reconnect", id="reconnect-btn"),
    ]),
    html.Div([
        html.Label("Timeframe (resample)"),
        dcc.Dropdown(id="timeframe", options=[
            {"label":"1 second","value":"1s"},
            {"label":"1 minute","value":"1min"},
            {"label":"5 minutes","value":"5min"},
        ], value="1s", style={"width":"200px"}),
        html.Label("Rolling Window (points)"),
        dcc.Input(id="rolling-window", type="number", value=30, min=1, style={"width":"120px"}),
    ], style={"display":"flex", "gap":"20px", "align-items":"center", "padding":"10px 0"}),
    html.Div([
        html.Label("Symbol A"),
        dcc.Dropdown(id="sym-a", options=[{"label":s,"value":s} for s in SYMBOLS], value=SYMBOLS[0]),
        html.Label("Symbol B"),
        dcc.Dropdown(id="sym-b", options=[{"label":s,"value":s} for s in SYMBOLS], value=SYMBOLS[1]),
    ], style={"display":"flex", "gap":"10px", "width":"100%"}),
    html.Div([
        html.Button("Compute ADF (current resample)", id="adf-btn"),
        html.Button("Export resampled CSV", id="export-btn"),
        dcc.Download(id="download-dataframe-csv"),
    ], style={"padding":"8px 0"}),
    html.Hr(),
    # graphs
    dcc.Tabs(id="tabs", value="tab-prices", children=[
        dcc.Tab(label="Prices", value="tab-prices"),
        dcc.Tab(label="Spread & Z-score", value="tab-spread"),
        dcc.Tab(label="Rolling Correlation", value="tab-corr"),
        dcc.Tab(label="Summary Stats", value="tab-stats"),
    ]),
    html.Div(id="tab-content"),
    # alerts
    html.Hr(),
    html.Div([
        html.H4("Alerts (simple z-score)"),
        html.Label("z-score threshold (abs)"),
        dcc.Input(id="alert-thresh", type="number", value=2.0, step=0.1),
        html.Button("Add alert", id="add-alert"),
        html.Ul(id="alerts-list")
    ]),
    # hidden interval for live updates
    dcc.Interval(id="live-interval", interval=500, n_intervals=0),
    dcc.Store(id="adf-result"),
    dcc.Store(id="last-export"),
    html.Div(id="debug", style={"display":"none"})
], style={"max-width":"1200px", "margin":"auto", "padding":"20px"})

# Simple persistent alert list in server memory for demo
alert_rules = []

# Callback: reconnect / restart ingestion
@app.callback(Output("sym-a","options"),
              Output("sym-b","options"),
              Input("reconnect-btn","n_clicks"),
              State("symbols-in","value"),
              prevent_initial_call=True)
def on_reconnect(n_clicks, symbols_text):
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    if symbols:
        # clear in-memory buffers for new symbols
        for k in list(recent_ticks.keys()):
            recent_ticks.pop(k, None)
        for s in symbols:
            recent_ticks[s]  # initialize
        # start ingestion thread for new symbols
        start_ingestion(symbols)
    options = [{"label":s,"value":s} for s in (symbols or SYMBOLS)]
    return options, options

# Initial start ingestion
start_ingestion(SYMBOLS)

# Tab content renderer + live update interval
@app.callback(Output("tab-content","children"),
              Input("tabs","value"),
              Input("live-interval","n_intervals"),
              State("sym-a","value"),
              State("sym-b","value"),
              State("timeframe","value"),
              State("rolling-window","value"))
def render_content(tab, n_intervals, sym_a, sym_b, timeframe, rolling_window):
    if not sym_a or not sym_b:
        raise PreventUpdate
    # build resampled frames
    meta = {}
    a_df = resample_ticks(sym_a, timeframe)
    b_df = resample_ticks(sym_b, timeframe)

    price_a = a_df['close'] if 'close' in a_df else pd.Series(dtype=float)
    price_b = b_df['close'] if 'close' in b_df else pd.Series(dtype=float)

    # Compute hedge ratio (Y ~ X) where Y is B, X is A
    hedge, resid, r2 = compute_hedge_ratio(price_a, price_b)
    if hedge is not None:
        intercept, slope = hedge
        spread_series = pd.Series(resid, index=price_b.dropna().index[:len(resid)])
        z = compute_zscore(spread_series)
    else:
        intercept, slope, spread_series, z = (None, None, pd.Series(dtype=float), pd.Series(dtype=float))

    # rolling corr
    rolling_corr_series = rolling_corr(price_a, price_b, rolling_window)

    # Tabs
    if tab == "tab-prices":
        fig = go.Figure()
        if not price_a.empty:
            fig.add_trace(go.Scatter(x=price_a.index, y=price_a.values, name=sym_a))
        if not price_b.empty:
            fig.add_trace(go.Scatter(x=price_b.index, y=price_b.values, name=sym_b))
        fig.update_layout(title=f"Prices: {sym_a} & {sym_b}")
        return dcc.Graph(figure=fig, id="prices-graph")

    elif tab == "tab-spread":
        fig = go.Figure()
        if not spread_series.empty:
            fig.add_trace(go.Scatter(x=spread_series.index, y=spread_series.values, name="spread"))
        if not z.empty:
            fig.add_trace(go.Scatter(x=z.index, y=z.values, name="z-score", yaxis="y2"))
        fig.update_layout(
            title=f"Spread & z-score (hedge slope={slope:.4f} intercept={intercept:.4f})" if hedge else "Spread & z-score",
            yaxis=dict(title="spread"),
            yaxis2=dict(title="z-score", overlaying="y", side="right")
        )
        return dcc.Graph(figure=fig, id="spread-graph")

    elif tab == "tab-corr":
        fig = go.Figure()
        if not rolling_corr_series.empty:
            fig.add_trace(go.Scatter(x=rolling_corr_series.index, y=rolling_corr_series.values, name="rolling_corr"))
        fig.update_layout(title=f"Rolling correlation (window={rolling_window})")
        return dcc.Graph(figure=fig, id="corr-graph")

    elif tab == "tab-stats":
        rows = []
        rows.append(html.Div(f"Hedge intercept: {intercept}"))
        rows.append(html.Div(f"Hedge slope: {slope}"))
        rows.append(html.Div(f"R^2: {r2}"))
        rows.append(html.Div(f"Latest z-score (last): {float(z.iloc[-1]) if len(z)>0 else 'N/A'}"))
        rows.append(html.Div(f"Available resampled points A: {len(price_a)}, B: {len(price_b)}"))
        return html.Div(rows)

# ADF compute
@app.callback(Output("adf-result","data"),
              Input("adf-btn","n_clicks"),
              State("sym-a","value"),
              State("sym-b","value"),
              State("timeframe","value"),
              prevent_initial_call=True)
def do_adf(n_clicks, sym_a, sym_b, timeframe):
    a_df = resample_ticks(sym_a, timeframe)
    b_df = resample_ticks(sym_b, timeframe)
    price_a = a_df['close'] if 'close' in a_df else pd.Series(dtype=float)
    price_b = b_df['close'] if 'close' in b_df else pd.Series(dtype=float)
    hedge, resid, r2 = compute_hedge_ratio(price_a, price_b)
    if resid is None:
        return {"error":"not enough data"}
    adf_res = compute_adf(resid)
    return adf_res

@app.callback(Output("debug","children"),
              Input("adf-result","data"))
def show_adf(data):
    # show temporary debug if needed
    return json.dumps(data)

# Alerts: add alert
@app.callback(Output("alerts-list","children"),
              Input("add-alert","n_clicks"),
              State("alert-thresh","value"),
              State("alerts-list","children"),
              State("sym-a","value"),
              State("sym-b","value"),
              prevent_initial_call=True)
def add_alert(n_clicks, thresh, current_list, sym_a, sym_b):
    # for demo save a rule: if abs(z) > thresh then trigger
    rule = {"type":"zscore", "threshold":float(thresh), "sym_a":sym_a, "sym_b":sym_b}
    alert_rules.append(rule)
    # render list items
    items = []
    for i, r in enumerate(alert_rules):
        items.append(html.Li(f"{i+1}. zscore > {r['threshold']} for {r['sym_a']} / {r['sym_b']}"))
    return items

# Periodic check for alerts and trigger via debug (demo)
@app.callback(Output("last-export","data"),
              Input("live-interval","n_intervals"),
              State("sym-a","value"),
              State("sym-b","value"),
              State("timeframe","value"),
              prevent_initial_call=False)
def alert_checker(n_intervals, sym_a, sym_b, timeframe):
    # compute latest z
    a_df = resample_ticks(sym_a, timeframe)
    b_df = resample_ticks(sym_b, timeframe)
    price_a = a_df['close'] if 'close' in a_df else pd.Series(dtype=float)
    price_b = b_df['close'] if 'close' in b_df else pd.Series(dtype=float)
    hedge, resid, r2 = compute_hedge_ratio(price_a, price_b)
    if resid is None:
        return {}
    z = compute_zscore(pd.Series(resid))
    last_z = float(z.iloc[-1]) if len(z)>0 else None
    fired = []
    for r in alert_rules:
        if r['sym_a']==sym_a and r['sym_b']==sym_b and last_z is not None:
            if abs(last_z) > r['threshold']:
                fired.append({"rule": r, "z": last_z, "at": datetime.utcnow().isoformat()})
    if fired:
        # For the demo we simply print; in prod you may send webhook/email
        print("[alert] triggered:", fired)
    return {"last_z": last_z, "fired": fired}

# Export CSV
@app.callback(Output("download-dataframe-csv","data"),
              Input("export-btn","n_clicks"),
              State("sym-a","value"),
              State("sym-b","value"),
              State("timeframe","value"),
              prevent_initial_call=True)
def export_csv(n_clicks, sym_a, sym_b, timeframe):
    a_df = resample_ticks(sym_a, timeframe)
    b_df = resample_ticks(sym_b, timeframe)
    # join and produce CSV
    left = a_df[['open','high','low','close','volume']].rename(columns=lambda x: f"{sym_a}_{x}")
    right = b_df[['open','high','low','close','volume']].rename(columns=lambda x: f"{sym_b}_{x}")
    df = pd.concat([left, right], axis=1)
    csv_str = df.to_csv()
    return dcc.send_string(csv_str, filename=f"resampled_{sym_a}_{sym_b}_{timeframe}.csv")

if __name__ == "__main__":
    print("Starting Dash app on http://127.0.0.1:8050")
    app.run(debug=True)
