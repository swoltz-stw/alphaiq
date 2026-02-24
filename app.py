"""
AlphaIQ — Stock Evaluation Platform
Streamlit UI · powered by yfinance · no API key required
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import scorer ─────────────────────────────────────────────────────────────
from scorer import (
    fetch_and_score, fetch_and_score_batch, backtest_ticker,
    run_scenario, optimize_strategy, fetch_diagnostics,
    WEIGHTS, DATA_POINT_LABELS, _signal_label,
)

# ── Index ticker lists ────────────────────────────────────────────────────────
DOW = ["AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","HD","MRK","CVX","MCD",
       "BA","CAT","GS","AXP","IBM","MMM","HON","DIS","NKE","AMGN","CRM","TRV",
       "VZ","CSCO","DOW","KO","INTC","WBA"]
NASDAQ = ["AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AVGO","COST","NFLX",
          "AMD","ADBE","QCOM","PYPL","INTC","CMCSA","TXN","INTU","SBUX","MDLZ",
          "REGN","GILD","FISV","MU","LRCX","KLAC","SNPS","CDNS","MELI","ISRG"]
SP500 = list(set(DOW + NASDAQ + [
    "BRK-B","LLY","XOM","ORCL","ABBV","BAC","PFE","COP","RTX","T","DE","GE",
    "SPGI","BLK","SLB","EOG","PLD","AMT","EQIX","CB","MO","DUK","SO","NEE",
    "AEP","EXC","D","ETR","SPY","QQQ",
]))
INDICES = {"DOW (30)": DOW, "NASDAQ-100": NASDAQ, "S&P 500 Sample": SP500}
HORIZONS = ["1D","1W","1M","1Q","1Y"]
HORIZON_LABELS = {"1D":"1 Day","1W":"1 Week","1M":"1 Month","1Q":"1 Quarter","1Y":"1 Year"}

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Playfair+Display:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f0f4f9;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] h1 {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    color: #0f6fff;
}

/* Main area */
.main { background: #f4f6f9; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}

/* Score pill colors */
.sig-strong-buy  { background:#edfaf4; color:#0cad6b; border:1px solid #a7f0c8; padding:3px 10px; border-radius:20px; font-family:'IBM Plex Mono'; font-size:11px; font-weight:600; letter-spacing:1px; }
.sig-buy         { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; padding:3px 10px; border-radius:20px; font-family:'IBM Plex Mono'; font-size:11px; font-weight:600; letter-spacing:1px; }
.sig-neutral     { background:#fffbeb; color:#d97706; border:1px solid #fde68a; padding:3px 10px; border-radius:20px; font-family:'IBM Plex Mono'; font-size:11px; font-weight:600; letter-spacing:1px; }
.sig-sell        { background:#fff7ed; color:#ea6f1a; border:1px solid #fed7aa; padding:3px 10px; border-radius:20px; font-family:'IBM Plex Mono'; font-size:11px; font-weight:600; letter-spacing:1px; }
.sig-strong-sell { background:#fef2f4; color:#e8344a; border:1px solid #fecaca; padding:3px 10px; border-radius:20px; font-family:'IBM Plex Mono'; font-size:11px; font-weight:600; letter-spacing:1px; }

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 800;
    color: #1a2332;
    margin-bottom: 4px;
}
.section-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #7a90a8;
    letter-spacing: 1px;
    margin-bottom: 20px;
}
.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
    margin-bottom: 16px;
}
.mono { font-family: 'IBM Plex Mono', monospace; }
.disclaimer {
    text-align: center;
    color: #b0bece;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    margin-top: 40px;
    padding: 16px;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def signal_html(signal):
    cls = {
        "STRONG BUY": "sig-strong-buy", "BUY": "sig-buy",
        "NEUTRAL": "sig-neutral", "SELL": "sig-sell",
        "STRONG SELL": "sig-strong-sell",
    }.get(signal, "sig-neutral")
    return f'<span class="{cls}">{signal}</span>'

def score_color(score):
    if score >= 75: return "#0cad6b"
    if score >= 60: return "#16a34a"
    if score >= 45: return "#d97706"
    if score >= 30: return "#ea6f1a"
    return "#e8344a"

def brier_label(b):
    if b is None: return "—"
    if b <= 0.08: return f"{b} ✦ Excellent"
    if b <= 0.13: return f"{b} ✦ Good"
    if b <= 0.18: return f"{b} ✦ Fair"
    return f"{b} ✦ Low"

def pct_color(val):
    if val is None: return "—"
    color = "#0cad6b" if val > 0 else ("#e8344a" if val < 0 else "#7a90a8")
    prefix = "+" if val > 0 else ""
    return f'<span style="color:{color};font-family:IBM Plex Mono;font-weight:600">{prefix}{val}%</span>'

def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="AlphaIQ Results")
    return buf.getvalue()

# ── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 AlphaIQ")
    st.markdown("*Stock Evaluation Platform*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["◎  Evaluator", "⊞  Screener", "↺  Backtest", "⚙  Optimizer", "⬡  Methodology", "🔬  Diagnostics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: Yahoo Finance (live)\nNo API key required\nRefreshes on each run")
    st.caption("⚠️ For educational use only.\nNot financial advice.")

page = page.strip().split("  ")[-1]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

if page == "Evaluator":
    st.markdown('<div class="section-title">Stock Evaluator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">LIVE SCORING · PRICE TARGETS · ALL TIME HORIZONS</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            ticker_input = st.text_input("Ticker symbol(s)", value="AAPL", placeholder="AAPL, MSFT, TSLA, ...")
        with col2:
            index_sel = st.selectbox("Or select an index", ["— individual tickers —"] + list(INDICES.keys()))
        with col3:
            st.write("")
            st.write("")
            run_btn = st.button("▶  Evaluate", type="primary", use_container_width=True)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_dp = st.toggle("Show individual data point scores", value=False)
    with col_opt2:
        horizon = st.select_slider("Time horizon", options=HORIZONS,
                                    format_func=lambda h: HORIZON_LABELS[h], value="1M")

    if run_btn:
        if index_sel != "— individual tickers —":
            tickers = INDICES[index_sel]
        else:
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.stop()

        # ── Single ticker: full horizon matrix ──
        if len(tickers) == 1:
            ticker = tickers[0]
            st.divider()
            st.markdown(f"#### Scoring `{ticker}` across all horizons...")

            cols = st.columns(5)
            results = {}
            for i, h in enumerate(HORIZONS):
                with cols[i]:
                    with st.spinner(HORIZON_LABELS[h]):
                        r = fetch_and_score(ticker, h)
                        results[h] = r

            st.divider()

            # Score cards row
            cols = st.columns(5)
            for i, h in enumerate(HORIZONS):
                r = results[h]
                clr = score_color(r["overall"])
                with cols[i]:
                    st.markdown(f"""
                    <div class="card" style="text-align:center;border-top:3px solid {clr};">
                        <div style="font-family:'IBM Plex Mono';font-size:10px;color:#7a90a8;letter-spacing:2px;margin-bottom:8px">{HORIZON_LABELS[h].upper()}</div>
                        <div style="font-family:'Playfair Display';font-size:48px;color:{clr};line-height:1">{r['overall']}</div>
                        <div style="font-size:10px;color:#7a90a8;margin:4px 0">/100</div>
                        {signal_html(r['signal'])}
                        <div style="margin-top:10px;font-family:'IBM Plex Mono';font-size:13px;color:#1a2332">${r['target']}</div>
                        <div style="font-size:11px;color:{'#0cad6b' if r['target'] and r['price'] and r['target']>r['price'] else '#e8344a'}">
                            {'▲' if r['target'] and r['price'] and r['target']>r['price'] else '▼'}
                            {abs(round((r['target']/r['price']-1)*100,1)) if r['price'] and r['target'] else '—'}%
                        </div>
                        <div style="margin-top:8px;font-family:'IBM Plex Mono';font-size:10px;color:#7a90a8">
                            Confidence: {r['confidence']}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Data points breakdown
            if show_dp:
                st.divider()
                st.markdown("**Data Point Breakdown** · 1 Month")
                r1m = results["1M"]
                dp_data = {
                    DATA_POINT_LABELS[k]: v
                    for k, v in r1m["data_scores"].items()
                }
                dp_df = pd.DataFrame(dp_data.items(), columns=["Data Point", "Score"])
                dp_df = dp_df.sort_values("Score", ascending=False)

                cols = st.columns(4)
                for idx, row in dp_df.iterrows():
                    c = cols[idx % 4]
                    clr = score_color(row["Score"])
                    c.markdown(f"""
                    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin-bottom:8px">
                        <div style="font-size:10px;color:#7a90a8;font-family:'IBM Plex Mono';margin-bottom:4px">{row['Data Point']}</div>
                        <div style="font-family:'Playfair Display';font-size:22px;color:{clr}">{row['Score']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Export
            export_rows = []
            for h in HORIZONS:
                r = results[h]
                row = {"Ticker": ticker, "Horizon": HORIZON_LABELS[h],
                       "Score": r["overall"], "Signal": r["signal"],
                       "Price": r.get("price"), "Target": r.get("target"),
                       "Confidence %": r["confidence"]}
                if show_dp:
                    for k, v in r["data_scores"].items():
                        row[DATA_POINT_LABELS[k]] = v
                export_rows.append(row)
            df_exp = pd.DataFrame(export_rows)
            st.download_button("⬇ Export to Excel", to_excel_bytes(df_exp),
                               f"alphaiq_{ticker}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── Multiple tickers: table view ──
        else:
            st.divider()
            prog = st.progress(0, text="Fetching live data...")
            results = []
            for i, t in enumerate(tickers[:50]):
                r = fetch_and_score(t, horizon)
                results.append(r)
                prog.progress((i + 1) / min(len(tickers), 50),
                              text=f"Scoring {t}… ({i+1}/{min(len(tickers),50)})")
            prog.empty()

            df = pd.DataFrame([{
                "Ticker": r["ticker"],
                "Name": r.get("name", ""),
                "Sector": r.get("sector", ""),
                "Score": r["overall"],
                "Signal": r["signal"],
                "Price": f"${r['price']}" if r.get("price") else "—",
                "Target": f"${r['target']}" if r.get("target") else "—",
                "Upside %": round((r["target"]/r["price"]-1)*100, 1) if r.get("price") and r.get("target") else None,
                "Confidence %": r["confidence"],
            } for r in results])

            st.markdown(f"**{len(df)} stocks scored** · {HORIZON_LABELS[horizon]} · Live data")

            # Colour-coded display
            st.dataframe(
                df,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                    "Upside %": st.column_config.NumberColumn("Upside %", format="%.1f%%"),
                    "Confidence %": st.column_config.NumberColumn("Confidence %", format="%d%%"),
                },
                hide_index=True,
                use_container_width=True,
            )

            # Export
            if show_dp:
                all_results = results
                export_rows = []
                for r in all_results:
                    row = {"Ticker": r["ticker"], "Score": r["overall"],
                           "Signal": r["signal"], "Horizon": HORIZON_LABELS[horizon],
                           "Price": r.get("price"), "Target": r.get("target"),
                           "Upside %": round((r["target"]/r["price"]-1)*100,1) if r.get("price") and r.get("target") else None}
                    for k, v in r["data_scores"].items():
                        row[DATA_POINT_LABELS[k]] = v
                    export_rows.append(row)
                df_exp = pd.DataFrame(export_rows)
            else:
                df_exp = df

            st.download_button("⬇ Export to Excel", to_excel_bytes(df_exp),
                               f"alphaiq_batch_{horizon}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Screener":
    st.markdown('<div class="section-title">Stock Screener</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">FILTER ANY INDEX BY SCORE RANGE AND HORIZON</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        idx = st.selectbox("Index", list(INDICES.keys()))
        h   = st.select_slider("Horizon", options=HORIZONS,
                                format_func=lambda x: HORIZON_LABELS[x], value="1M")
    with col2:
        min_score, max_score = st.slider("Score filter range", 0, 100, (0, 100), step=5)
        show_dist = st.toggle("Show score distribution chart", value=True)

    run_scr = st.button("▶  Scan Index", type="primary")

    if run_scr:
        tickers = INDICES[idx]
        prog    = st.progress(0, text="Scanning stocks...")
        all_results = []

        for i, ticker in enumerate(tickers):
            r = fetch_and_score(ticker, h)
            all_results.append(r)
            prog.progress((i + 1) / len(tickers), text=f"Scoring {ticker}… ({i+1}/{len(tickers)})")
        prog.empty()

        # Build full dataframe first
        all_df = pd.DataFrame([{
            "Ticker":       r["ticker"],
            "Name":         r.get("name", ""),
            "Sector":       r.get("sector", ""),
            "Score":        r["overall"],
            "Raw Score":    r.get("raw_score", r["overall"]),
            "Signal":       r["signal"],
            "Price":        r.get("price"),
            "Target":       r.get("target"),
            "Upside %":     round((r["target"]/r["price"]-1)*100, 1)
                            if r.get("price") and r.get("target") else None,
            "Confidence %": r["confidence"],
        } for r in sorted(all_results, key=lambda x: -x["overall"])])

        # Score distribution chart
        if show_dist and not all_df.empty:
            st.markdown("#### Score Distribution — All Scanned Stocks")
            bins   = list(range(0, 105, 5))
            labels = [f"{b}-{b+4}" for b in bins[:-1]]
            counts = pd.cut(all_df["Score"], bins=bins, labels=labels, right=False).value_counts().sort_index()
            dist_df = pd.DataFrame({"Score Range": counts.index, "# Stocks": counts.values})

            # Colour bars by zone
            bar_colors = []
            for label in dist_df["Score Range"]:
                lo = int(label.split("-")[0])
                if lo >= 65:   bar_colors.append("#0cad6b")
                elif lo >= 45: bar_colors.append("#d97706")
                else:          bar_colors.append("#e8344a")

            # Use st.bar_chart (native, no plotly needed)
            st.bar_chart(dist_df.set_index("Score Range"), use_container_width=True, height=200)

            # Quick stats row
            sc = all_df["Score"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Avg Score",   round(sc.mean(), 1))
            c2.metric("Median",      round(sc.median(), 1))
            c3.metric("Lowest",      int(sc.min()))
            c4.metric("Highest",     int(sc.max()))
            c5.metric("Std Dev",     round(sc.std(), 1))
            st.divider()

        # Apply filter
        filtered_df = all_df[(all_df["Score"] >= min_score) & (all_df["Score"] <= max_score)]

        if filtered_df.empty:
            # Helpful diagnostic instead of silent failure
            st.warning(f"No stocks matched score range **{min_score}–{max_score}**.")
            st.markdown(f"Score range of all {len(all_df)} scanned stocks: "
                        f"**{int(all_df['Score'].min())}** – **{int(all_df['Score'].max())}** "
                        f"(avg: {round(all_df['Score'].mean(),1)})")
            st.markdown("Try widening your score filter, or check the distribution chart above.")
            st.dataframe(all_df.head(10), hide_index=True, use_container_width=True)
        else:
            st.success(f"**{len(filtered_df)} of {len(all_df)} stocks** matched score range "
                       f"{min_score}–{max_score} · {idx} · {HORIZON_LABELS[h]}")
            st.dataframe(
                filtered_df,
                column_config={
                    "Score":        st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                    "Raw Score":    st.column_config.NumberColumn("Raw Score", format="%.1f",
                                    help="Pre-stretch composite score. Shows true separation before curve is applied."),
                    "Upside %":     st.column_config.NumberColumn("Upside %", format="%.1f%%"),
                    "Price":        st.column_config.NumberColumn("Price",  format="$%.2f"),
                    "Target":       st.column_config.NumberColumn("Target", format="$%.2f"),
                    "Confidence %": st.column_config.NumberColumn("Confidence %", format="%d%%"),
                },
                hide_index=True,
                use_container_width=True,
            )
            st.download_button("⬇ Export to Excel", to_excel_bytes(filtered_df),
                               f"screener_{idx.replace(' ','_')}_{h}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Backtest":
    st.markdown('<div class="section-title">Backtesting</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">VALIDATE PREDICTIONS · RUN TRADING SCENARIOS</div>', unsafe_allow_html=True)

    bt_tab, sc_tab = st.tabs(["📊 Single Stock Backtest", "💰 Scenario Simulator"])

    # ── Single backtest ──
    with bt_tab:
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            bt_ticker = st.text_input("Ticker", value="QQQ", key="bt_ticker")
        with col2:
            bt_date = st.date_input("Start Date", value=datetime(2024, 1, 1),
                                    min_value=datetime(2020, 1, 1),
                                    max_value=datetime.now() - timedelta(days=30))
        with col3:
            bt_end = st.date_input("End Date (optional)",
                                   value=datetime.now().date(),
                                   min_value=datetime(2020, 1, 2),
                                   max_value=datetime.now().date(),
                                   help="Caps how far forward actual returns are measured. Leave at today to use full horizon periods.")
        with col4:
            st.write("")
            st.write("")
            run_bt = st.button("▶  Run Backtest", type="primary", key="run_bt")

        st.caption("Note: Technical signals (momentum, RSI, MACD) are reconstructed from historical price data. Fundamental scores use today's values applied retroactively — a known limitation of free data sources.")

        if run_bt:
            end_date_str = str(bt_end) if bt_end and str(bt_end) != str(datetime.now().date()) else None
            with st.spinner(f"Running backtest for {bt_ticker.upper()} from {bt_date}…"):
                bt_res = backtest_ticker(bt_ticker.upper(), str(bt_date), end_date_str)

            if bt_res.get("error"):
                st.error(f"Error: {bt_res['error']}")
            else:
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ticker", bt_res["ticker"])
                col2.metric("Start Date", bt_res["date"])
                col3.metric("End Date", bt_res.get("end_date") or "Full horizons")
                col4.metric("Price at Start", f"${bt_res['base_price']}")

                df_bt = pd.DataFrame(bt_res["results"])
                correct = df_bt["Direction Correct"].str.contains("YES").sum()
                total   = len(df_bt[df_bt["Direction Correct"] != "—"])
                st.markdown(f"**Direction accuracy: {correct}/{total} horizons correct**")

                st.dataframe(
                    df_bt,
                    column_config={
                        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                        "Actual Return %": st.column_config.NumberColumn("Actual Return %", format="%.2f%%"),
                        "Pred Target": st.column_config.NumberColumn("Pred Target", format="$%.2f"),
                        "Actual Price": st.column_config.NumberColumn("Actual Price", format="$%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
                st.download_button("⬇ Export to Excel", to_excel_bytes(df_bt),
                                   f"backtest_{bt_ticker}_{bt_date}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── Scenario simulator ──
    with sc_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            sc_ticker  = st.text_input("Ticker", value="SPY", key="sc_ticker")
            sc_dollars = st.number_input("Starting Amount ($)", value=10000, step=1000)
        with col2:
            sc_start = st.date_input("Start Date", value=datetime(2023, 1, 1),
                                     min_value=datetime(2018, 1, 1),
                                     max_value=datetime.now() - timedelta(days=60),
                                     key="sc_date")
            sc_end = st.date_input("End Date", value=datetime.now().date(),
                                   min_value=datetime(2018, 1, 2),
                                   max_value=datetime.now().date(),
                                   key="sc_end",
                                   help="Simulation stops on this date.")
            sc_interval = st.selectbox("Trade Interval", ["1D","1W","1M"],
                                       format_func=lambda x: HORIZON_LABELS[x])
        with col3:
            buy_rating  = st.slider("Buy when score ≥",  50, 100, 65)
            sell_rating = st.slider("Sell when score ≤",  0,  50, 35)

        run_sc = st.button("▶  Run Scenario", type="primary", key="run_sc")

        if run_sc:
            if sc_end <= sc_start:
                st.error("End date must be after start date.")
            else:
                with st.spinner(f"Simulating {sc_ticker.upper()} {sc_start} → {sc_end}…"):
                    df_sc, summary = run_scenario(
                        sc_ticker.upper(), str(sc_start), str(sc_end),
                        sc_dollars, sc_interval, buy_rating, sell_rating
                    )

                if df_sc is None:
                    st.error(f"Error: {summary}")
                else:
                    st.divider()
                    c1, c2, c3, c4, c5 = st.columns(5)
                    alpha = summary["Alpha vs B&H %"]
                    c1.metric("Final Portfolio",    f"${summary['Final Portfolio']:,.2f}")
                    c2.metric("Total Return",       f"{summary['Total Return %']}%",
                              delta=f"{summary['Total Return %']}%")
                    c3.metric("Buy & Hold Return",  f"{summary['Buy & Hold Return %']}%")
                    c4.metric("Alpha vs B&H",       f"{alpha}%",
                              delta=f"{alpha}%", delta_color="normal")
                    c5.metric("Total Trades",       summary["Total Trades"])

                    st.line_chart(df_sc.set_index("Date")[["Portfolio Value"]])

                    st.dataframe(
                        df_sc,
                        column_config={
                            "Score":           st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                            "Portfolio Value": st.column_config.NumberColumn("Portfolio Value", format="$%.2f"),
                            "Cash":            st.column_config.NumberColumn("Cash",            format="$%.2f"),
                            "Price":           st.column_config.NumberColumn("Price",           format="$%.2f"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.download_button("⬇ Export to Excel", to_excel_bytes(df_sc),
                                       f"scenario_{sc_ticker}_{sc_start}_to_{sc_end}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Optimizer":
    st.markdown('<div class="section-title">Strategy Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">FIND OPTIMAL BUY/SELL THRESHOLDS · WALK-FORWARD VALIDATED</div>', unsafe_allow_html=True)

    st.info("""
**How this works:** The optimizer splits your date range into a **Training window (70%)** and a **Test window (30%)**.
It grid-searches hundreds of buy/sell/interval combinations on the training data, then validates the top 5 on
the unseen test window. This guards against curve-fitting — if a strategy only worked in the past because it was
tuned to that specific period, the test window will expose it.

⚠️ Even validated strategies are not guaranteed to work in the future. Use this as a research tool, not a trading signal.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        opt_ticker  = st.text_input("Ticker", value="TSLA", key="opt_ticker")
        opt_dollars = st.number_input("Starting Amount ($)", value=10000, step=1000, key="opt_dollars")
    with col2:
        opt_start = st.date_input("Start Date", value=datetime(2023, 1, 1),
                                  min_value=datetime(2018, 1, 1),
                                  max_value=datetime.now() - timedelta(days=90),
                                  key="opt_start")
        opt_end   = st.date_input("End Date", value=datetime.now().date(),
                                  min_value=datetime(2018, 4, 1),
                                  max_value=datetime.now().date(),
                                  key="opt_end")
    with col3:
        st.markdown("**What gets tested:**")
        st.caption("• Buy thresholds: 52 → 79 (step 3)")
        st.caption("• Sell thresholds: 25 → 49 (step 3)")
        st.caption("• Intervals: Daily, Weekly, Monthly")
        st.caption("• ~270 combinations total")

    run_opt = st.button("⚙  Run Optimizer", type="primary", key="run_opt")

    if run_opt:
        if opt_end <= opt_start:
            st.error("End date must be after start date.")
        elif (opt_end - opt_start).days < 90:
            st.error("Need at least 90 days of history to optimize.")
        else:
            total_days = (opt_end - opt_start).days
            split_day  = int(total_days * 0.70)
            train_end  = opt_start + timedelta(days=split_day)

            st.markdown(f"""
            **Running optimizer for `{opt_ticker.upper()}`**
            - 🏋️ Training window: `{opt_start}` → `{train_end.strftime('%Y-%m-%d')}` ({split_day} days)
            - 🧪 Test window: `{(train_end + timedelta(days=1)).strftime('%Y-%m-%d')}` → `{opt_end}` ({total_days - split_day} days)
            """)

            with st.spinner("Testing ~270 parameter combinations… this takes 2–4 minutes ⏳"):
                result = optimize_strategy(
                    opt_ticker.upper(), str(opt_start), str(opt_end), float(opt_dollars)
                )

            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                best = result["best_params"]
                st.divider()

                # ── Best strategy callout ──────────────────────────────
                overfit_warn = best.get("overfit_flag", False)
                box_color    = "#fff8e6" if overfit_warn else "#f0fdf4"
                border_color = "#f5c518" if overfit_warn else "#0cad6b"
                warn_text    = "⚠️ Possible overfit — in-sample performance significantly better than out-of-sample." if overfit_warn else "✅ Validated — out-of-sample performance is consistent with training."

                st.markdown(f"""
                <div style="background:{box_color};border:2px solid {border_color};border-radius:12px;padding:20px 24px;margin-bottom:20px">
                    <div style="font-family:'Playfair Display';font-size:22px;font-weight:800;margin-bottom:12px">
                        🏆 Best Validated Strategy
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;font-family:'IBM Plex Mono';font-size:13px">
                        <div><div style="color:#7a90a8;font-size:10px;letter-spacing:1px">INTERVAL</div><div style="font-size:20px;font-weight:700">{best['interval']}</div></div>
                        <div><div style="color:#7a90a8;font-size:10px;letter-spacing:1px">BUY WHEN SCORE ≥</div><div style="font-size:20px;font-weight:700;color:#0cad6b">{best['buy_rating']}</div></div>
                        <div><div style="color:#7a90a8;font-size:10px;letter-spacing:1px">SELL WHEN SCORE ≤</div><div style="font-size:20px;font-weight:700;color:#e8344a">{best['sell_rating']}</div></div>
                    </div>
                    <div style="margin-top:16px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px;font-family:'IBM Plex Mono';font-size:12px">
                        <div><div style="color:#7a90a8;font-size:10px">TRAIN RETURN</div><div style="color:#0cad6b;font-weight:600">+{best['total_return']}%</div></div>
                        <div><div style="color:#7a90a8;font-size:10px">TRAIN ALPHA</div><div style="color:#0cad6b;font-weight:600">+{best['alpha']}%</div></div>
                        <div><div style="color:#7a90a8;font-size:10px">TEST RETURN</div><div style="color:{'#0cad6b' if (best['oos_return'] or 0)>0 else '#e8344a'};font-weight:600">{'+' if (best['oos_return'] or 0)>0 else ''}{best['oos_return']}%</div></div>
                        <div><div style="color:#7a90a8;font-size:10px">TEST ALPHA</div><div style="color:{'#0cad6b' if (best['oos_alpha'] or 0)>0 else '#e8344a'};font-weight:600">{'+' if (best['oos_alpha'] or 0)>0 else ''}{best['oos_alpha']}%</div></div>
                    </div>
                    <div style="margin-top:12px;font-size:11px;color:#7a90a8">{warn_text}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Top 5 comparison table ─────────────────────────────
                st.markdown("#### Top 5 Strategies — Training vs. Test Performance")
                st.caption(f"Tested {result['total_combos']} combinations · Sorted by out-of-sample alpha")

                rows = []
                for i, r in enumerate(result["top5"]):
                    rows.append({
                        "Rank":          i + 1,
                        "Interval":      r["interval"],
                        "Buy ≥":         r["buy_rating"],
                        "Sell ≤":        r["sell_rating"],
                        "Train Return %":r["total_return"],
                        "Train Alpha %": r["alpha"],
                        "Train Trades":  r["trades"],
                        "Test Return %": r["oos_return"],
                        "Test Alpha %":  r["oos_alpha"],
                        "Overfit?":      "⚠️ Yes" if r["overfit_flag"] else "✅ No",
                    })
                df_opt = pd.DataFrame(rows)
                st.dataframe(df_opt, hide_index=True, use_container_width=True,
                             column_config={
                                 "Train Return %": st.column_config.NumberColumn(format="%.1f%%"),
                                 "Train Alpha %":  st.column_config.NumberColumn(format="%.1f%%"),
                                 "Test Return %":  st.column_config.NumberColumn(format="%.1f%%"),
                                 "Test Alpha %":   st.column_config.NumberColumn(format="%.1f%%"),
                             })

                # ── Interpretation guide ───────────────────────────────
                st.divider()
                with st.expander("📖 How to read these results"):
                    st.markdown(f"""
**Training window** (`{opt_start}` → `{train_end.strftime('%Y-%m-%d')}`): The period used to find the best parameters.
Always looks good — it was optimized on this data.

**Test window** (`{(train_end + timedelta(days=1)).strftime('%Y-%m-%d')}` → `{opt_end}`): Unseen data.
This is what matters. If the test alpha is positive and close to the train alpha, the strategy has
*some* generalizability. If test performance collapses vs training, it's overfit.

**Alpha** = Strategy return minus buy-and-hold return. Positive alpha means the strategy beat doing nothing.

**The honest truth:** Even a positive test alpha doesn't guarantee future performance. Markets change.
Use these settings as a starting point in the Scenario Simulator to explore further.
""")

                # ── Export ────────────────────────────────────────────
                st.download_button("⬇ Export Results to Excel", to_excel_bytes(df_opt),
                                   f"optimizer_{opt_ticker}_{opt_start}_{opt_end}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Methodology":  # Page 5
    st.markdown('<div class="section-title">Scoring Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">20 DATA POINTS · HORIZON-CALIBRATED WEIGHTS · BRIER VALIDATION</div>', unsafe_allow_html=True)

    with st.expander("📐 How the Score Works", expanded=True):
        st.markdown("""
**The AlphaIQ score (0–100)** is a weighted composite of 20 individually scored data points, 
all sourced live from Yahoo Finance.

| Score Range | Signal | Interpretation |
|---|---|---|
| **75–100** | STRONG BUY | High probability of upward movement |
| **60–74** | BUY | Moderate bullish signal |
| **45–59** | NEUTRAL | High probability of sideways movement |
| **30–44** | SELL | Moderate bearish signal |
| **0–29** | STRONG SELL | High probability of downward movement |

**Key design principle:** Weights are calibrated by time horizon. Short horizons (1D, 1W) rely 
more on technical signals like momentum, RSI, and MACD. Long horizons (1Q, 1Y) emphasize 
fundamentals like revenue growth, FCF yield, and EPS growth.

**Brier Score** measures prediction calibration on a 0–1 scale. Lower = better.  
- **0.00** = Perfect calibration  
- **0.08** = Excellent (our target)  
- **0.25** = Equivalent to random guessing  
""")

    with st.expander("🏦 Data Sources"):
        sources = [
            ("Yahoo Finance (yfinance)", "Price, volume, fundamentals, analyst consensus, options data"),
            ("SEC EDGAR (via yfinance)", "10-K/10-Q filings: EPS, revenue, margins, FCF"),
            ("FINRA (via short data)", "Short interest, days-to-cover"),
            ("Federal Reserve (proxied via VIX)", "Macro environment scoring"),
            ("CBOE VIX Index", "Market volatility / fear gauge for macro score"),
            ("Sector ETFs (XLK, XLV, etc.)", "Sector momentum via ETF 1-month return"),
            ("Earnings Whispers (via timestamps)", "Earnings proximity catalyst detection"),
        ]
        for name, desc in sources:
            st.markdown(f"**{name}** — {desc}")

    with st.expander("⚖️ Weight Table by Horizon"):
        w_df = pd.DataFrame(WEIGHTS).T
        w_df.index.name = "Horizon"
        w_df.columns = [DATA_POINT_LABELS.get(c, c) for c in w_df.columns]
        st.dataframe(w_df, use_container_width=True)
        st.caption("Weights are on a 1–10 scale per horizon. The composite score is a weighted average.")

    with st.expander("📊 20 Data Points — Ranked by Long-Term Importance"):
        dp_rows = []
        for k, label in DATA_POINT_LABELS.items():
            avg_w = round(sum(WEIGHTS[h][k] for h in HORIZONS) / len(HORIZONS), 1)
            dp_rows.append({"Data Point": label, "Key": k, "Avg Weight": avg_w,
                            "1D": WEIGHTS["1D"][k], "1W": WEIGHTS["1W"][k],
                            "1M": WEIGHTS["1M"][k], "1Q": WEIGHTS["1Q"][k],
                            "1Y": WEIGHTS["1Y"][k]})
        dp_df = pd.DataFrame(dp_rows).sort_values("1Y", ascending=False).drop(columns=["Key"])
        st.dataframe(dp_df, hide_index=True, use_container_width=True,
                     column_config={
                         "Avg Weight": st.column_config.ProgressColumn("Avg Weight", min_value=0, max_value=10, format="%.1f"),
                     })

    with st.expander("⚠️ Limitations & Roadmap"):
        st.markdown("""
**Current limitations (free tier):**
- Fundamental scores (P/E, revenue, margins) reflect *today's* values, not historical snapshots — affects backtest accuracy
- Sentiment scoring uses analyst consensus, not NLP on news text
- No options flow data
- Rate-limited to ~30 tickers/min on Yahoo Finance free tier

**Roadmap to improve accuracy:**
1. Add a paid financial data API (Polygon.io ~$29/mo or Tiingo ~$10/mo) for point-in-time historical fundamentals
2. Add news NLP scoring via NewsAPI or FinBERT
3. Run rolling Brier score tracking to continuously recalibrate weights
4. Add machine learning layer trained on 5+ years of return data
""")

# ── Footer ────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

if page == "Diagnostics":
    st.markdown('<div class="section-title">Data Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">TEST WHAT YAHOO FINANCE IS ACTUALLY RETURNING</div>', unsafe_allow_html=True)

    st.info("""
Use this page if scores are showing as 50 across the board. It shows exactly what
data Yahoo Finance returned for a ticker — so you can see whether the problem is
a rate limit, a bad ticker, or missing fields.
""")

    diag_ticker = st.text_input("Ticker to diagnose", value="AAPL")
    run_diag    = st.button("🔬 Run Diagnostics", type="primary")

    if run_diag:
        with st.spinner(f"Fetching raw data for {diag_ticker.upper()}…"):
            d = fetch_diagnostics(diag_ticker.upper())

        st.divider()

        # Status indicators
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Info Keys Returned",  d["info_keys_returned"])
        c2.metric("Price History Rows",  d["hist_rows"])
        c3.metric("Fetch Failed?",       "❌ YES" if d["info_fetch_failed"] else "✅ NO")
        c4.metric("Using Fallback Data?", "⚠️ YES" if d["info_is_fallback"] else "✅ NO")

        if d["info_fetch_failed"] or d["info_keys_returned"] < 5:
            st.error("""
**Yahoo Finance returned no/empty data.** This is the root cause of all-50 scores.

Common causes:
- **Rate limiting**: Too many requests in a short window. Wait 30–60 seconds and try again.
- **Invalid ticker**: Check the ticker symbol is correct (e.g. `BRK-B` not `BRK.B`)
- **Yahoo Finance outage**: Try again in a few minutes
- **Streamlit Cloud IP blocked**: Yahoo occasionally blocks cloud server IPs

**Immediate workaround**: Try scoring 1–2 tickers manually in the Evaluator tab first,
then run the screener. This "warms up" the connection.
""")
        elif d["info_is_fallback"]:
            st.warning("⚠️ Using fallback data (fast_info only). Fundamental scores will be limited — technical scores should still work.")
        else:
            st.success(f"✅ Data fetched successfully. {d['info_keys_returned']} info fields returned.")

        # Key fields table
        st.markdown("#### Key Fields Retrieved")
        fields = {
            "Company Name":        d["shortName"],
            "Sector":              d["sector"],
            "Current Price":       d["price"],
            "Trailing P/E":        d["trailingPE"],
            "Forward P/E":         d["forwardPE"],
            "Revenue Growth":      f"{round(d['revenueGrowth']*100,1)}%" if d["revenueGrowth"] else "❌ Missing",
            "Profit Margin":       f"{round(d['profitMargins']*100,1)}%" if d["profitMargins"] else "❌ Missing",
            "Analyst Rec. Mean":   d["recommendationMean"],
            "Analyst Price Target":d["targetMeanPrice"],
            "Trailing EPS":        d["trailingEps"],
            "Free Cash Flow":      f"${d['freeCashflow']:,.0f}" if d["freeCashflow"] else "❌ Missing",
            "Short % Float":       f"{round(d['shortPercentOfFloat']*100,1)}%" if d["shortPercentOfFloat"] else "❌ Missing",
            "Price History Rows":  d["hist_rows"],
            "Latest Price Date":   d["hist_latest"],
        }
        df_diag = pd.DataFrame(fields.items(), columns=["Field", "Value"])
        st.dataframe(df_diag, hide_index=True, use_container_width=True)

        # Also run a live score to show what happens
        st.divider()
        st.markdown("#### Live Score (using data above)")
        with st.spinner("Scoring…"):
            r = fetch_and_score(diag_ticker.upper(), "1M")

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Score",  r["overall"])
        col2.metric("Raw Score",    r.get("raw_score", "—"))
        col3.metric("Confidence %", r["confidence"])

        if r["overall"] == 50 and r["confidence"] == 0:
            st.error("Score is 50 with 0% confidence — data is not loading. See diagnostic above.")
        elif r["confidence"] < 40:
            st.warning(f"Low confidence ({r['confidence']}%) — many fields missing. Scores will be approximate.")
        else:
            st.success(f"Score looks valid. {r['confidence']}% of data fields populated.")

        if r.get("data_scores"):
            st.markdown("**Individual data point scores:**")
            dp_df = pd.DataFrame([
                {"Data Point": DATA_POINT_LABELS.get(k, k), "Score": v,
                 "Status": "🟢" if v > 60 else ("🔴" if v < 40 else "🟡")}
                for k, v in r["data_scores"].items()
            ]).sort_values("Score", ascending=False)
            st.dataframe(dp_df, hide_index=True, use_container_width=True,
                         column_config={"Score": st.column_config.ProgressColumn(
                             "Score", min_value=0, max_value=100, format="%d")})

st.markdown("""
<div class="disclaimer">
ALPHAIQ · DATA VIA YAHOO FINANCE · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE<br>
PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS
</div>
""", unsafe_allow_html=True)
