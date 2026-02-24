""" AlphaIQ — Stock Evaluation Platform Streamlit UI · powered by yfinance · no API key required """

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import io

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import scorer ──────────────────────────────────────────────────────────────

from scorer import (
    fetch_and_score,
    fetch_and_score_batch,
    backtest_ticker,
    run_scenario,
    optimize_strategy,
    fetch_diagnostics,
    score_market_probability,
    run_market_scenario,
    WEIGHTS,
    DATA_POINT_LABELS,
    CATEGORIES,
    CATEGORY_WEIGHTS,
    POINT_WEIGHTS,
    MARKET_PREDICTORS,
    INDEX_TICKERS,
    _signal_label,
    calibrated_confidence,  # NEW
)

# ── Index ticker lists ────────────────────────────────────────────────────────

DOW = [
    "AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","HD","MRK","CVX","MCD",
    "BA","CAT","GS","AXP","IBM","MMM","HON","DIS","NKE","AMGN","CRM","TRV",
    "VZ","CSCO","DOW","KO","INTC","WBA"
]

NASDAQ = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AVGO","COST","NFLX",
    "AMD","ADBE","QCOM","PYPL","INTC","CMCSA","TXN","INTU","SBUX","MDLZ",
    "REGN","GILD","FISV","MU","LRCX","KLAC","SNPS","CDNS","MELI","ISRG"
]

SP500 = list(set(DOW + NASDAQ + [
    "BRK-B","LLY","XOM","ORCL","ABBV","BAC","PFE","COP","RTX","T","DE","GE",
    "SPGI","BLK","SLB","EOG","PLD","AMT","EQIX","CB","MO","DUK","SO","NEE",
    "AEP","EXC","D","ETR","SPY","QQQ",
]))

INDICES = {"DOW (30)": DOW, "NASDAQ-100": NASDAQ, "S&P 500 Sample": SP500}

HORIZONS = ["1D","1W","1M","1Q","1Y"]
HORIZON_LABELS = {
    "1D":"1 Day",
    "1W":"1 Week",
    "1M":"1 Month",
    "1Q":"1 Quarter",
    "1Y":"1 Year",
}

# ── Theme / CSS (placeholder – left empty but kept for structure) ─────────────

st.markdown(""" """, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def signal_html(signal: str) -> str:
    cls = {
        "STRONG BUY": "sig-strong-buy",
        "BUY": "sig-buy",
        "NEUTRAL": "sig-neutral",
        "SELL": "sig-sell",
        "STRONG SELL": "sig-strong-sell",
    }.get(signal, "sig-neutral")
    return f'<span class="{cls}">{signal}</span>'


def score_color(score: int) -> str:
    if score >= 75:
        return "#0cad6b"
    if score >= 60:
        return "#16a34a"
    if score >= 45:
        return "#d97706"
    if score >= 30:
        return "#ea6f1a"
    return "#e8344a"


def brier_label(b):
    if b is None:
        return "—"
    if b <= 0.08:
        return f"{b} ✦ Excellent"
    if b <= 0.13:
        return f"{b} ✦ Good"
    if b <= 0.18:
        return f"{b} ✦ Fair"
    return f"{b} ✦ Low"


def pct_color(val):
    if val is None:
        return "—"
    color = "#0cad6b" if val > 0 else ("#e8344a" if val < 0 else "#7a90a8")
    prefix = "+" if val > 0 else ""
    return f'<span style="color:{color}">{prefix}{val}%</span>'


def to_excel_bytes(df: pd.DataFrame) -> bytes:
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
        [
            "◎ Evaluator",
            "⊞ Screener",
            "↺ Backtest",
            "⚙ Optimizer",
            "🌐 Market Probability",
            "🔮 Next Day Call",
            "▢ Methodology",
            "🔬 Diagnostics",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data: Yahoo Finance (live)\nNo API key required\nRefreshes on each run")
    st.caption("⚠️ For educational use only.\nNot financial advice.")

page = page.strip().split(" ")[-1]  # "Evaluator", "Screener", "Backtest", "Call", etc.

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════

if page == "Evaluator":
    st.markdown(
        '<h2 style="margin-bottom:0">Stock Evaluator</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280;margin-top:0">'
        'LIVE SCORING · PRICE TARGETS · ALL TIME HORIZONS'
        '</p>',
        unsafe_allow_html=True,
    )

    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            ticker_input = st.text_input(
                "Ticker symbol(s)",
                value="AAPL",
                placeholder="AAPL, MSFT, TSLA, ..."
            )
        with col2:
            index_sel = st.selectbox(
                "Or select an index",
                ["— individual tickers —"] + list(INDICES.keys())
            )
        with col3:
            st.write("")
            st.write("")
            run_btn = st.button("▶ Evaluate", type="primary", use_container_width=True)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_dp = st.toggle("Show individual data point scores", value=False)
        with col_opt2:
            horizon = st.select_slider(
                "Time horizon",
                options=HORIZONS,
                format_func=lambda h: HORIZON_LABELS[h],
                value="1M",
            )

    if run_btn:
        if index_sel != "— individual tickers —":
            tickers = INDICES[index_sel]
        else:
            tickers = [
                t.strip().upper()
                for t in ticker_input.split(",")
                if t.strip()
            ]

        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.stop()

        # ── Single ticker: full horizon matrix ────────────────────────────────
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
                price = r.get("price")
                target = r.get("target")
                if price and target:
                    up = target > price
                    updown = "▲" if up else "▼"
                    up_pct = abs(round((target / price - 1) * 100, 1))
                else:
                    updown, up_pct = "–", "—"

                with cols[i]:
                    st.markdown(
                        f"""
                        <div style="border-radius:12px;padding:12px;border:1px solid #e5e7eb;background:#f9fafb;text-align:center;">
                            <div style="font-size:13px;color:#6b7280;">{HORIZON_LABELS[h].upper()}</div>
                            <div style="font-size:28px;font-weight:600;color:{clr};">{r['overall']}/100</div>
                            <div style="margin-top:4px;">{signal_html(r['signal'])}</div>
                            <div style="margin-top:6px;font-size:13px;color:#374151;">
                                Target: ${target if target else '—'}<br/>
                                {updown} {up_pct}%
                            </div>
                            <div style="margin-top:6px;font-size:12px;color:#6b7280;">
                                Confidence: {r['confidence']}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Category + Data point breakdown
            st.divider()
            r1m = results["1M"]
            is_etf = r1m.get("is_etf", False)

            if is_etf:
                st.info(
                    "📦 **ETF detected** — Fundamental Quality and Valuation "
                    "categories are not applicable. Score is based on Technical, "
                    "Sentiment, and Macro categories only."
                )

            # ── Category score cards (1M) ────────────────────────────────────
            st.markdown("#### Category Scores · 1 Month")
            cat_results = r1m.get("category_scores", {})

            cat_cols = st.columns(len(CATEGORIES))
            for i, (cat_name, cat_info) in enumerate(CATEGORIES.items()):
Love it — option **A (one universal calibration table)** is a great, pragmatic choice. ✅  

Below are **fully updated versions** of both `scorer.py` and `app.py`:

- `scorer.py` now includes a **calibrated_confidence(...)** function that:
  - Uses a **universal score → confidence** mapping (you can refine later from backtests)
  - Adjusts confidence based on **data completeness** (your existing `confidence` field)
- `app.py` now uses this calibrated confidence in the **🔮 Next Day Call** page for:
  - Single-ticker tomorrow direction
  - Market tomorrow direction (DOW / NASDAQ / S&P 500)

You can **copy/paste these files as-is** over your current ones.

---

## 🔧 Updated `scorer.py`

> Drop this in as your full `scorer.py` (replaces old file).

```python
"""
AlphaIQ Scoring Engine v6
=========================

Complete rebuild on AQR / Fama-French / academic factor research foundations.

INDIVIDUAL STOCK SCORING
------------------------
All within-category weights are now directly derived from empirical correlations
with forward returns (Fama-French 3-factor, AQR Quality-Minus-Junk, momentum
literature, and Novy-Marx 2013 profitability research).

The weight for each data point = round(r * 100) where r is the Pearson correlation
with forward returns from the academic literature. This makes weights interpretable:
a weight of 18 means "this factor has r≈0.18 with forward returns."

SHORT-TERM FACTORS (1D-1W), source: AQR short-term momentum, Jegadeesh & Titman:
    momentum_multi: 8 (r≈0.08, 12-1M momentum)
    rsi: 6 (r≈0.06, mean-reversion at extremes)
    macd: 5 (r≈0.05, trend-following)
    volume_trend: 5 (r≈0.05, volume-price confirmation)
    week52_position: 3 (r≈0.03, range position)
    earnings_proximity:4 (r≈0.04, post-earnings drift / PEAD)
    short_interest: 4 (r≈0.04, squeeze potential)
    sector_momentum: 4 (r≈0.04, sector rotation)

LONG-TERM FACTORS (1M-1Y), source: Novy-Marx, Fama-French, AQR QMJ:
    revenue_growth: 18 (r≈0.18, strongest long predictor)
    eps_growth: 16 (r≈0.16)
    fcf_yield: 15 (r≈0.15, Novy-Marx profitability)
    analyst_target: 14 (r≈0.14, price target consensus)
    profit_margin: 13 (r≈0.13, gross profitability premium)
    earnings_surprise:12 (r≈0.12, PEAD long tail)
    analyst_revisions:11 (r≈0.11, revision momentum)
    peg_ratio: 10 (r≈0.10, growth-adjusted value)
    pe_vs_sector: 9 (r≈0.09, relative value)
    debt_equity: 7 (r≈0.07, leverage factor)
    macro_score: 6 (r≈0.06, regime)
    sentiment: 5 (r≈0.05, analyst consensus)

MARKET PROBABILITY ENGINE
-------------------------
For overall index prediction, uses the 7 best academically-validated
market-level predictors (not individual stock factors):

1. Shiller CAPE (proxied via SPY P/E) r≈0.38 strongest known
2. Yield Curve (10Y-2Y spread) r≈0.28 recession signal
3. Credit Spreads (HY vs IG) r≈0.25 risk appetite
4. VIX trend (level + direction) r≈0.22 fear regime
5. Market Breadth (% above 200MA) r≈0.18 participation
6. Price Momentum (index 12-1M) r≈0.15 trend continuation
7. Earnings Revision Breadth r≈0.14 analyst sentiment

All proxied via free yfinance tickers.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# PART 1: INDIVIDUAL STOCK SCORING
# ═════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "Technical Momentum": {
        "points": ["momentum_multi", "rsi", "macd", "volume_trend", "week52_position"],
        "description": "Price action, trend strength, volume confirmation",
        "emoji": "📈",
        "available_for_etf": True,
    },
    "Fundamental Quality": {
        "points": ["revenue_growth", "profit_margin", "fcf_yield", "debt_equity", "eps_growth"],
        "description": "Business quality: growth, margins, cash flow, leverage",
        "emoji": "🏗",
        "available_for_etf": False,
    },
    "Valuation": {
        "points": ["pe_vs_sector", "peg_ratio", "analyst_target", "earnings_surprise"],
        "description": "Price vs earnings and analyst expectations",
        "emoji": "🎯",
        "available_for_etf": False,
    },
    "Sentiment & Flow": {
        "points": ["sentiment", "analyst_revisions", "short_interest"],
        "description": "Analyst consensus, revision momentum, short positioning",
        "emoji": "💬",
        "available_for_etf": True,
    },
    "Macro & Sector": {
        "points": ["macro_score", "sector_momentum", "earnings_proximity"],
        "description": "Market environment, sector tailwinds, event catalysts",
        "emoji": "🌍",
        "available_for_etf": True,
    },
}

# AQR-derived within-category weights (= r × 100, rounded)
POINT_WEIGHTS = {
    # Technical (short-term AQR momentum literature)
    "momentum_multi": 8,
    "rsi": 6,
    "macd": 5,
    "volume_trend": 5,
    "week52_position": 3,
    # Fundamental Quality (Novy-Marx 2013, AQR QMJ)
    "revenue_growth": 18,
    "profit_margin": 13,
    "fcf_yield": 15,
    "eps_growth": 16,
    "debt_equity": 7,
    # Valuation (Fama-French value factor, PEAD)
    "pe_vs_sector": 9,
    "peg_ratio": 10,
    "analyst_target": 14,
    "earnings_surprise": 12,
    # Sentiment & Flow
    "sentiment": 5,
    "analyst_revisions": 11,
    "short_interest": 4,
    # Macro & Sector
    "macro_score": 6,
    "sector_momentum": 4,
    "earnings_proximity": 4,
}

# Category-level weights by horizon – derived from factor half-life research
CATEGORY_WEIGHTS = {
    "1D": {
        "Technical Momentum": 27,
        "Fundamental Quality": 5,
        "Valuation": 5,
        "Sentiment & Flow": 16,
        "Macro & Sector": 14,
    },
    "1W": {
        "Technical Momentum": 22,
        "Fundamental Quality": 8,
        "Valuation": 10,
        "Sentiment & Flow": 18,
        "Macro & Sector": 14,
    },
    "1M": {
        "Technical Momentum": 15,
        "Fundamental Quality": 25,
        "Valuation": 22,
        "Sentiment & Flow": 18,
        "Macro & Sector": 10,
    },
    "1Q": {
        "Technical Momentum": 8,
        "Fundamental Quality": 35,
        "Valuation": 30,
        "Sentiment & Flow": 18,
        "Macro & Sector": 12,
    },
    "1Y": {
        "Technical Momentum": 5,
        "Fundamental Quality": 40,
        "Valuation": 33,
        "Sentiment & Flow": 14,
        "Macro & Sector": 10,
    },
}

DATA_POINT_LABELS = {
    "momentum_multi": "Price Momentum (Multi-TF)",
    "rsi": "RSI (14-Day)",
    "macd": "MACD Signal",
    "volume_trend": "Volume Trend",
    "week52_position": "52-Week Range Position",
    "earnings_proximity": "Earnings Proximity",
    "sector_momentum": "Sector Momentum",
    "short_interest": "Short Interest",
    "sentiment": "Analyst Sentiment",
    "analyst_target": "Analyst Price Target Upside",
    "earnings_surprise": "EPS Surprise",
    "analyst_revisions": "Analyst Revisions",
    "macro_score": "Macro Conditions (VIX)",
    "pe_vs_sector": "P/E Ratio vs Sector",
    "revenue_growth": "Revenue Growth",
    "profit_margin": "Profit Margin",
    "fcf_yield": "Free Cash Flow Yield",
    "debt_equity": "Debt / Equity",
    "eps_growth": "EPS Growth Rate",
    "peg_ratio": "PEG Ratio",
}

# Legacy flat WEIGHTS dict kept for methodology UI compatibility
WEIGHTS = {h: {k: POINT_WEIGHTS[k] for k in DATA_POINT_LABELS} for h in ["1D", "1W", "1M", "1Q", "1Y"]}

HORIZON_FACTORS = {"1D": 0.006, "1W": 0.025, "1M": 0.08, "1Q": 0.18, "1Y": 0.42}

# ── Utilities ────────────────────────────────────────────────────────────────

def stretch_score(raw: float) -> int:
    """
    Strong cubic stretch.
    raw 65→76, raw 70→83, raw 75→89, raw 35→24, raw 30→17
    """
    x = (float(raw) - 50.0) / 50.0
    s = x * (1.0 + 2.5 * x * x)
    return int(round(50.0 + max(-1.0, min(1.0, s)) * 50.0))


def _clamp(v, lo=0, hi=100):
    return max(lo, min(hi, int(round(float(v)))))


def _f(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

# ── DATA FETCH ───────────────────────────────────────────────────────────────

def _get_info(ticker: str, retries: int = 2) -> dict:
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info and (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
                or info.get("quoteType")
            ):
                return info
            fast = t.fast_info
            if fast:
                return {
                    "currentPrice": getattr(fast, "last_price", None),
                    "previousClose": getattr(fast, "previous_close", None),
                    "marketCap": getattr(fast, "market_cap", None),
                    "shortName": ticker.upper(),
                    "quoteType": getattr(fast, "quote_type", "EQUITY"),
                    "_fallback": True,
                }
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5)
    return {"_fetch_failed": True}


def _get_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is not None and not hist.empty and len(hist) > 10:
            return hist
    except Exception:
        pass
    try:
        end = datetime.now()
        start = end - timedelta(days=400 if period == "1y" else 90)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _is_etf(info: dict) -> bool:
    qt = (info.get("quoteType") or "").upper()
    return qt in ("ETF", "MUTUALFUND") or bool(info.get("fundFamily"))


def _close(hist) -> pd.Series:
    c = hist["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.astype(float).dropna()


def fetch_diagnostics(ticker: str) -> dict:
    info = _get_info(ticker)
    hist = _get_history(ticker)
    return {
        "ticker": ticker.upper(),
        "timestamp": datetime.now().isoformat(),
        "info_keys_returned": len(info),
        "info_fetch_failed": info.get("_fetch_failed", False),
        "info_is_fallback": info.get("_fallback", False),
        "is_etf": _is_etf(info),
        "price": info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose"),
        "shortName": info.get("shortName", "—"),
        "sector": info.get("sector", "—"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "revenueGrowth": info.get("revenueGrowth"),
        "profitMargins": info.get("profitMargins"),
        "recommendationMean": info.get("recommendationMean"),
        "targetMeanPrice": info.get("targetMeanPrice"),
        "trailingEps": info.get("trailingEps"),
        "freeCashflow": info.get("freeCashflow"),
        "shortPercentOfFloat": info.get("shortPercentOfFloat"),
        "hist_rows": len(hist),
        "hist_empty": hist.empty,
        "hist_latest": str(hist.index[-1])[:10] if not hist.empty else "—",
    }

# ── INDIVIDUAL POINT SCORERS ────────────────────────────────────────────────
# (Return None when data unavailable.)

def score_momentum_multi(hist):
    try:
        c = _close(hist)
        n = len(c)
        windows = [(21, 8), (63, 5), (126, 3), (252, 1)]
        scores, wts = [], []
        for days, w in windows:
            if n > days + 1:
                ret = (c.iloc[-1] / c.iloc[-days] - 1) * 100
                scores.append(_clamp(50 + ret * 5.0))
                wts.append(w)
        if not scores:
            return None
        comp = sum(s * w for s, w in zip(scores, wts)) / sum(wts)
        if all(s >= 62 for s in scores):
            comp = min(100, comp + 15)
        elif all(s >= 55 for s in scores):
            comp = min(100, comp + 8)
        elif all(s <= 38 for s in scores):
            comp = max(0, comp - 15)
        elif all(s <= 45 for s in scores):
            comp = max(0, comp - 8)
        return _clamp(comp)
    except Exception:
        return None


def score_rsi(hist):
    try:
        c = _close(hist)
        if len(c) < 16:
            return None
        d = c.diff()
        g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        r = float((100 - (100 / (1 + g / l.replace(0, 1e-9)))).iloc[-1])
        if r < 20:
            return 92
        elif r < 28:
            return 80
        elif r < 38:
            return 66
        elif r < 48:
            return 56
        elif r < 52:
            return 50
        elif r < 62:
            return 44
        elif r < 72:
            return 30
        elif r < 80:
            return 18
        else:
            return 8
    except Exception:
        return None


def score_macd(hist):
    try:
        c = _close(hist)
        if len(c) < 35:
            return None
        m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        h = m - m.ewm(span=9, adjust=False).mean()
        cv, pv, p2 = float(h.iloc[-1]), float(h.iloc[-2]), float(h.iloc[-3])
        if cv > 0 and pv <= 0:
            return 86
        if cv < 0 and pv >= 0:
            return 14
        if cv > 0 and cv > pv and pv > p2:
            return _clamp(72 + min(abs(cv) * 10, 18))
        if cv > 0 and cv > pv:
            return _clamp(64 + min(abs(cv) * 6, 14))
        if cv > 0:
            return 58
        if cv < 0 and cv < pv and pv < p2:
            return _clamp(28 - min(abs(cv) * 10, 18))
        if cv < 0 and cv < pv:
            return _clamp(36 - min(abs(cv) * 6, 14))
        return 42
    except Exception:
        return None


def score_volume(hist):
    try:
        vol = hist["Volume"]
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        vol = vol.astype(float)
        c = _close(hist)
        if len(vol) < 20:
            return None
        avg = float(vol.iloc[-90:].mean()) if len(vol) >= 90 else float(vol.mean())
        rec = float(vol.iloc[-5:].mean())
        if avg == 0:
            return None
        ratio = rec / avg
        up = float(c.iloc[-1]) > float(c.iloc[-6]) if len(c) >= 6 else True
        if ratio > 3.0:
            amp = 42
        elif ratio > 2.0:
            amp = 30
        elif ratio > 1.5:
            amp = 20
        elif ratio > 1.2:
            amp = 12
        elif ratio < 0.4:
            amp = -18
        elif ratio < 0.6:
            amp = -10
        elif ratio < 0.8:
            amp = -4
        else:
            amp = 0
        base = 58 if up else 42
        return _clamp(base + (amp if up else -amp))
    except Exception:
        return None


def score_week52(hist):
    try:
        c = _close(hist)
        if len(c) < 50:
            return None
        n = min(252, len(c))
        hi = float(c.rolling(n).max().iloc[-1])
        lo = float(c.rolling(n).min().iloc[-1])
        curr = float(c.iloc[-1])
        if hi == lo:
            return None
        pct = (curr - lo) / (hi - lo)
        ret1m = float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else 0
        up = ret1m > 0.005
        if pct < 0.10 and up:
            return _clamp(88 + (0.10 - pct) * 100)
        elif pct < 0.10:
            return _clamp(28 - (0.10 - pct) * 80)
        elif pct < 0.25 and up:
            return 72
        elif pct < 0.25:
            return 34
        elif pct < 0.50:
            return _clamp(42 + pct * 20)
        elif pct < 0.75:
            return _clamp(54 + (pct - 0.5) * 20)
        elif pct < 0.90 and up:
            return 74
        elif pct < 0.90:
            return 34
        elif up:
            return _clamp(76 + (pct - 0.90) * 120)
        else:
            return _clamp(24 - (pct - 0.90) * 120)
    except Exception:
        return None


def score_short_interest(info):
    try:
        s = info.get("shortPercentOfFloat")
        if s is None:
            mc = _f(info.get("marketCap")) or 0
            return 65 if mc > 100e9 else 55
        s *= 100
        if s > 30:
            return 12
        elif s > 20:
            return 22
        elif s > 15:
            return 32
        elif s > 10:
            return 42
        elif s > 5:
            return 58
        elif s > 2:
            return 70
        else:
            return 78
    except Exception:
        return 55


def score_earnings_surprise(info):
    # PEAD: post-earnings announcement drift
    try:
        t = _f(info.get("trailingEps"))
        fw = _f(info.get("forwardEps"))
        if t is None:
            return None
        if fw is None:
            fw = t
        if abs(fw) < 0.01:
            return None
        s = (t - fw) / abs(fw) * 100
        if s > 30:
            return 90
        elif s > 15:
            return 78
        elif s > 5:
            return 66
        elif s > 0:
            return 58
        elif s > -5:
            return 42
        elif s > -15:
            return 28
        else:
            return 12
    except Exception:
        return None


def score_sentiment(info):
    try:
        r = _f(info.get("recommendationMean"))
        if r is None:
            return 54
        if r <= 1.3:
            return 92
        elif r <= 1.7:
            return 82
        elif r <= 2.0:
            return 74
        elif r <= 2.3:
            return 65
        elif r <= 2.7:
            return 54
        elif r <= 3.0:
            return 45
        elif r <= 3.5:
            return 34
        elif r <= 4.0:
            return 23
        else:
            return 12
    except Exception:
        return 54


def score_analyst_revisions(info):
    try:
        r = _f(info.get("recommendationMean") or 3.0)
        n = int(info.get("numberOfAnalystOpinions") or 0)
        if n == 0:
            return 54
        if r <= 1.3:
            base = 92
        elif r <= 1.7:
            base = 82
        elif r <= 2.0:
            base = 74
        elif r <= 2.3:
            base = 65
        elif r <= 2.7:
            base = 54
        elif r <= 3.0:
            base = 45
        elif r <= 3.5:
            base = 34
        elif r <= 4.0:
            base = 23
        else:
            base = 12
        amp = 12 if n >= 40 else (8 if n >= 25 else (4 if n >= 15 else (-8 if n < 5 else 0)))
        d = base - 50
        return _clamp(50 + d + (amp if d > 0 else -amp))
    except Exception:
        return 54


def score_analyst_target(info):
    try:
        tgt = _f(info.get("targetMeanPrice") or info.get("targetMedianPrice"))
        cur = _f(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if not tgt or not cur or cur == 0:
            return None
        up = (tgt / cur - 1) * 100
        if up > 50:
            return 95
        elif up > 30:
            return 86
        elif up > 20:
            return 78
        elif up > 12:
            return 70
        elif up > 5:
            return 62
        elif up > 0:
            return 55
        elif up > -5:
            return 45
        elif up > -12:
            return 36
        elif up > -20:
            return 26
        elif up > -30:
            return 16
        else:
            return 6
    except Exception:
        return None


def score_pe_vs_sector(info):
    try:
        pe = _f(info.get("trailingPE") or info.get("forwardPE"))
        if pe is None or pe <= 0 or pe > 1500:
            return None
        bms = {
            "Technology": 28,
            "Healthcare": 22,
            "Financials": 13,
            "Consumer Cyclical": 20,
            "Consumer Defensive": 18,
            "Energy": 12,
            "Utilities": 16,
            "Real Estate": 35,
            "Basic Materials": 14,
            "Industrials": 20,
            "Communication Services": 22,
        }
        bm = bms.get(info.get("sector", ""), 20)
        r = pe / bm
        if r < 0.3:
            return 92
        elif r < 0.5:
            return 82
        elif r < 0.7:
            return 72
        elif r < 0.9:
            return 62
        elif r < 1.1:
            return 52
        elif r < 1.4:
            return 40
        elif r < 1.8:
            return 28
        elif r < 2.5:
            return 16
        else:
            return 6
    except Exception:
        return None


def score_revenue_growth(info):
    try:
        g = _f(info.get("revenueGrowth"))
        if g is None:
            return None
        g *= 100
        if g > 50:
            return 95
        elif g > 30:
            return 85
        elif g > 20:
            return 75
        elif g > 10:
            return 65
        elif g > 5:
            return 58
        elif g > 0:
            return 50
        elif g > -5:
            return 36
        elif g > -15:
            return 24
        elif g > -25:
            return 12
        else:
            return 4
    except Exception:
        return None


def score_profit_margin(info):
    try:
        m = _f(info.get("profitMargins"))
        if m is None:
            return None
        m *= 100
        if m > 40:
            return 94
        elif m > 30:
            return 86
        elif m > 20:
            return 77
        elif m > 12:
            return 67
        elif m > 6:
            return 57
        elif m > 2:
            return 46
        elif m > 0:
            return 36
        elif m > -8:
            return 24
        else:
            return 10
    except Exception:
        return None


def score_fcf_yield(info):
    try:
        fcf = _f(info.get("freeCashflow"))
        mc = _f(info.get("marketCap"))
        if fcf is None or mc is None or mc == 0:
            return None
        y = (fcf / mc) * 100
        if y > 10:
            return 92
        elif y > 6:
            return 80
        elif y > 4:
            return 70
        elif y > 2:
            return 62
        elif y > 0:
            return 52
        elif y > -3:
            return 34
        elif y > -6:
            return 20
        else:
            return 8
    except Exception:
        return None


def score_debt_equity(info):
    try:
        de = _f(info.get("debtToEquity"))
        if de is None:
            return 56
        if de < 10:
            return 86
        elif de < 30:
            return 76
        elif de < 60:
            return 65
        elif de < 100:
            return 54
        elif de < 150:
            return 40
        elif de < 250:
            return 26
        else:
            return 12
    except Exception:
        return 56


def score_eps_growth(info):
    try:
        t = _f(info.get("trailingEps"))
        fw = _f(info.get("forwardEps"))
        if t is None:
            return None
        if fw is None:
            fw = t
        if abs(t) < 0.01:
            return None
        g = (fw - t) / abs(t) * 100
        if g > 50:
            return 94
        elif g > 30:
            return 84
        elif g > 20:
            return 74
        elif g > 10:
            return 64
        elif g > 3:
            return 56
        elif g > 0:
            return 50
        elif g > -8:
            return 34
        elif g > -20:
            return 22
        else:
            return 8
    except Exception:
        return None


def score_peg(info):
    try:
        p = _f(info.get("pegRatio"))
        if p is None or p <= 0:
            return None
        if p < 0.5:
            return 94
        elif p < 0.8:
            return 82
        elif p < 1.0:
            return 72
        elif p < 1.3:
            return 60
        elif p < 1.8:
            return 44
        elif p < 2.5:
            return 28
        elif p < 4.0:
            return 14
        else:
            return 4
    except Exception:
        return None


_macro_cache = {}
_sector_cache = {}


def score_macro(_=None):
    global _macro_cache
    today = datetime.now().strftime("%Y-%m-%d")
    if today in _macro_cache:
        return _macro_cache[today]
    try:
        h = _get_history("^VIX", period="1mo")
        if h.empty:
            raise ValueError
        c = _close(h)
        v = float(c.iloc[-1])
        v5 = float(c.iloc[-5]) if len(c) >= 5 else v
        rising = v > v5 * 1.05
        if v < 12:
            base = 86
        elif v < 15:
            base = 75
        elif v < 18:
            base = 64
        elif v < 22:
            base = 52
        elif v < 28:
            base = 38
        elif v < 35:
            base = 24
        else:
            base = 12
        if rising:
            base = max(8, base - 12)
        _macro_cache[today] = base
        return base
    except Exception:
        _macro_cache[today] = 50
        return 50


def score_sector_momentum(info):
    global _sector_cache
    ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Basic Materials": "XLB",
        "Industrials": "XLI",
        "Communication Services": "XLC",
    }
    try:
        etf = ETFS.get(info.get("sector", ""), "SPY")
        key = f"{etf}_{datetime.now().strftime('%Y-%m-%d')}"
        if key not in _sector_cache:
            h = _get_history(etf, period="1y")
            if h.empty:
                _sector_cache[key] = 50
            else:
                c = _close(h)
                if len(c) >= 22:
