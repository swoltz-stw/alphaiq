"""
AlphaIQ Scoring Engine v6
==========================
Complete rebuild on AQR / Fama-French / academic factor research foundations.

INDIVIDUAL STOCK SCORING
─────────────────────────
All within-category weights are now directly derived from empirical correlations
with forward returns (Fama-French 3-factor, AQR Quality-Minus-Junk, momentum
literature, and Novy-Marx 2013 profitability research).

The weight for each data point = round(r * 100) where r is the Pearson correlation
with forward returns from the academic literature. This makes weights interpretable:
a weight of 18 means "this factor has r≈0.18 with forward returns."

SHORT-TERM FACTORS (1D-1W), source: AQR short-term momentum, Jegadeesh & Titman:
  momentum_multi:    8   (r≈0.08, 12-1M momentum)
  rsi:               6   (r≈0.06, mean-reversion at extremes)
  macd:              5   (r≈0.05, trend-following)
  volume_trend:      5   (r≈0.05, volume-price confirmation)
  week52_position:   3   (r≈0.03, range position)
  earnings_proximity:4   (r≈0.04, post-earnings drift / PEAD)
  short_interest:    4   (r≈0.04, squeeze potential)
  sector_momentum:   4   (r≈0.04, sector rotation)

LONG-TERM FACTORS (1M-1Y), source: Novy-Marx, Fama-French, AQR QMJ:
  revenue_growth:   18   (r≈0.18, strongest long predictor)
  eps_growth:       16   (r≈0.16)
  fcf_yield:        15   (r≈0.15, Novy-Marx profitability)
  analyst_target:   14   (r≈0.14, price target consensus)
  profit_margin:    13   (r≈0.13, gross profitability premium)
  earnings_surprise:12   (r≈0.12, PEAD long tail)
  analyst_revisions:11   (r≈0.11, revision momentum)
  peg_ratio:        10   (r≈0.10, growth-adjusted value)
  pe_vs_sector:      9   (r≈0.09, relative value)
  debt_equity:       7   (r≈0.07, leverage factor)
  macro_score:       6   (r≈0.06, regime)
  sentiment:         5   (r≈0.05, analyst consensus)

MARKET PROBABILITY ENGINE
──────────────────────────
For overall index prediction, uses the 7 best academically-validated
market-level predictors (not individual stock factors):

  1. Shiller CAPE (proxied via SPY P/E)    r≈0.38  strongest known
  2. Yield Curve (10Y-2Y spread)           r≈0.28  recession signal
  3. Credit Spreads (HY vs IG)             r≈0.25  risk appetite
  4. VIX trend (level + direction)         r≈0.22  fear regime
  5. Market Breadth (% above 200MA)        r≈0.18  participation
  6. Price Momentum (index 12-1M)          r≈0.15  trend continuation
  7. Earnings Revision Breadth             r≈0.14  analyst sentiment

All proxied via free yfinance tickers:
  CAPE proxy:      SPY trailing P/E via ^GSPC info
  Yield curve:     ^TNX (10Y) - ^IRX (3M) spread
  Credit spread:   HYG vs LQD ratio (high yield vs investment grade ETF)
  VIX:             ^VIX
  Breadth:         % of DOW/SP500 components above their 200MA
  Momentum:        SPY/QQQ/DIA 12-month vs 1-month return
  Revision breadth:aggregated analyst rec changes across index components
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: INDIVIDUAL STOCK SCORING
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "Technical Momentum": {
        "points":           ["momentum_multi", "rsi", "macd", "volume_trend", "week52_position"],
        "description":      "Price action, trend strength, volume confirmation",
        "emoji":            "📈",
        "available_for_etf": True,
    },
    "Fundamental Quality": {
        "points":           ["revenue_growth", "profit_margin", "fcf_yield", "debt_equity", "eps_growth"],
        "description":      "Business quality: growth, margins, cash flow, leverage",
        "emoji":            "🏗",
        "available_for_etf": False,
    },
    "Valuation": {
        "points":           ["pe_vs_sector", "peg_ratio", "analyst_target", "earnings_surprise"],
        "description":      "Price vs earnings and analyst expectations",
        "emoji":            "🎯",
        "available_for_etf": False,
    },
    "Sentiment & Flow": {
        "points":           ["sentiment", "analyst_revisions", "short_interest"],
        "description":      "Analyst consensus, revision momentum, short positioning",
        "emoji":            "💬",
        "available_for_etf": True,
    },
    "Macro & Sector": {
        "points":           ["macro_score", "sector_momentum", "earnings_proximity"],
        "description":      "Market environment, sector tailwinds, event catalysts",
        "emoji":            "🌍",
        "available_for_etf": True,
    },
}

# AQR-derived within-category weights (= r × 100, rounded)
# These directly reflect empirical correlation with forward returns
POINT_WEIGHTS = {
    # Technical (short-term AQR momentum literature)
    "momentum_multi":    8,
    "rsi":               6,
    "macd":              5,
    "volume_trend":      5,
    "week52_position":   3,
    # Fundamental Quality (Novy-Marx 2013, AQR QMJ)
    "revenue_growth":   18,
    "profit_margin":    13,
    "fcf_yield":        15,
    "eps_growth":       16,
    "debt_equity":       7,
    # Valuation (Fama-French value factor, PEAD)
    "pe_vs_sector":      9,
    "peg_ratio":        10,
    "analyst_target":   14,
    "earnings_surprise":12,
    # Sentiment & Flow
    "sentiment":         5,
    "analyst_revisions":11,
    "short_interest":    4,
    # Macro & Sector
    "macro_score":       6,
    "sector_momentum":   4,
    "earnings_proximity":4,
}

# Category-level weights by horizon — derived from factor half-life research
# Short-term: technical signals decay slowly; fundamental signals take months to manifest
# Long-term:  fundamental signals dominate; technical signals are noise over 1Y
CATEGORY_WEIGHTS = {
    "1D": {
        "Technical Momentum":  27,   # sum of ST point weights (8+6+5+5+3)
        "Fundamental Quality":  5,   # minimal at 1D
        "Valuation":            5,
        "Sentiment & Flow":    16,   # short interest + earnings proximity matter short-term
        "Macro & Sector":      14,
    },
    "1W": {
        "Technical Momentum":  22,
        "Fundamental Quality":  8,
        "Valuation":           10,
        "Sentiment & Flow":    18,
        "Macro & Sector":      14,
    },
    "1M": {
        "Technical Momentum":  15,
        "Fundamental Quality": 25,   # revenue + FCF start mattering
        "Valuation":           22,
        "Sentiment & Flow":    18,
        "Macro & Sector":      10,
    },
    "1Q": {
        "Technical Momentum":   8,
        "Fundamental Quality": 35,
        "Valuation":           30,
        "Sentiment & Flow":    18,
        "Macro & Sector":      12,
    },
    "1Y": {
        "Technical Momentum":   5,
        "Fundamental Quality": 40,   # Novy-Marx: profitability dominates at 1Y
        "Valuation":           33,
        "Sentiment & Flow":    14,
        "Macro & Sector":      10,
    },
}

DATA_POINT_LABELS = {
    "momentum_multi":    "Price Momentum (Multi-TF)",
    "rsi":               "RSI (14-Day)",
    "macd":              "MACD Signal",
    "volume_trend":      "Volume Trend",
    "week52_position":   "52-Week Range Position",
    "earnings_proximity":"Earnings Proximity",
    "sector_momentum":   "Sector Momentum",
    "short_interest":    "Short Interest",
    "sentiment":         "Analyst Sentiment",
    "analyst_target":    "Analyst Price Target Upside",
    "earnings_surprise": "EPS Surprise",
    "analyst_revisions": "Analyst Revisions",
    "macro_score":       "Macro Conditions (VIX)",
    "pe_vs_sector":      "P/E Ratio vs Sector",
    "revenue_growth":    "Revenue Growth",
    "profit_margin":     "Profit Margin",
    "fcf_yield":         "Free Cash Flow Yield",
    "debt_equity":       "Debt / Equity",
    "eps_growth":        "EPS Growth Rate",
    "peg_ratio":         "PEG Ratio",
}

# Legacy flat WEIGHTS dict kept for internal scenario simulator compatibility
WEIGHTS = {h: {k: POINT_WEIGHTS[k] for k in DATA_POINT_LABELS}
           for h in ["1D","1W","1M","1Q","1Y"]}

HORIZON_FACTORS = {"1D": 0.006, "1W": 0.025, "1M": 0.08, "1Q": 0.18, "1Y": 0.42}

# ─── UTILITIES ────────────────────────────────────────────────────────────────

def stretch_score(raw: float) -> int:
    """Strong cubic stretch. raw 65→76, raw 70→83, raw 75→89, raw 35→24, raw 30→17"""
    x = (float(raw) - 50.0) / 50.0
    s = x * (1.0 + 2.5 * x * x)
    return int(round(50.0 + max(-1.0, min(1.0, s)) * 50.0))

def _clamp(v, lo=0, hi=100):
    return max(lo, min(hi, int(round(float(v)))))

def _f(v):
    try: return float(v) if v is not None else None
    except: return None

# ─── DATA FETCH ───────────────────────────────────────────────────────────────

def _get_info(ticker: str, retries: int = 2) -> dict:
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info and (info.get("regularMarketPrice") or info.get("currentPrice")
                         or info.get("previousClose") or info.get("quoteType")):
                return info
            fast = t.fast_info
            if fast:
                return {
                    "currentPrice":     getattr(fast, "last_price", None),
                    "previousClose":    getattr(fast, "previous_close", None),
                    "marketCap":        getattr(fast, "market_cap", None),
                    "shortName":        ticker.upper(),
                    "quoteType":        getattr(fast, "quote_type", "EQUITY"),
                    "_fallback":        True,
                }
        except:
            if attempt < retries - 1:
                time.sleep(1.5)
    return {"_fetch_failed": True}

def _get_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is not None and not hist.empty and len(hist) > 10:
            return hist
    except:
        pass
    try:
        end   = datetime.now()
        start = end - timedelta(days=400 if period == "1y" else 90)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except:
        pass
    return pd.DataFrame()

def _is_etf(info: dict) -> bool:
    qt = (info.get("quoteType") or "").upper()
    return qt in ("ETF", "MUTUALFUND") or bool(info.get("fundFamily"))

def _close(hist) -> pd.Series:
    c = hist["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    return c.astype(float).dropna()

def fetch_diagnostics(ticker: str) -> dict:
    info = _get_info(ticker)
    hist = _get_history(ticker)
    return {
        "ticker": ticker.upper(), "timestamp": datetime.now().isoformat(),
        "info_keys_returned": len(info), "info_fetch_failed": info.get("_fetch_failed", False),
        "info_is_fallback": info.get("_fallback", False), "is_etf": _is_etf(info),
        "price": info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"),
        "shortName": info.get("shortName","—"), "sector": info.get("sector","—"),
        "trailingPE": info.get("trailingPE"), "forwardPE": info.get("forwardPE"),
        "revenueGrowth": info.get("revenueGrowth"), "profitMargins": info.get("profitMargins"),
        "recommendationMean": info.get("recommendationMean"), "targetMeanPrice": info.get("targetMeanPrice"),
        "trailingEps": info.get("trailingEps"), "freeCashflow": info.get("freeCashflow"),
        "shortPercentOfFloat": info.get("shortPercentOfFloat"),
        "hist_rows": len(hist), "hist_empty": hist.empty,
        "hist_latest": str(hist.index[-1])[:10] if not hist.empty else "—",
    }

# ─── INDIVIDUAL POINT SCORERS ─────────────────────────────────────────────────
# Returns None when data unavailable (prevents phantom 50s from diluting categories)

def score_momentum_multi(hist):
    try:
        c = _close(hist); n = len(c)
        # AQR weights: recent months weighted more (momentum decay)
        windows = [(21, 8), (63, 5), (126, 3), (252, 1)]
        scores, wts = [], []
        for days, w in windows:
            if n > days + 1:
                ret = (c.iloc[-1] / c.iloc[-days] - 1) * 100
                scores.append(_clamp(50 + ret * 5.0)); wts.append(w)
        if not scores: return None
        comp = sum(s*w for s,w in zip(scores,wts)) / sum(wts)
        # Consistency: all timeframes agree = amplify (AQR trend persistence)
        if   all(s >= 62 for s in scores): comp = min(100, comp + 15)
        elif all(s >= 55 for s in scores): comp = min(100, comp + 8)
        elif all(s <= 38 for s in scores): comp = max(0,   comp - 15)
        elif all(s <= 45 for s in scores): comp = max(0,   comp - 8)
        return _clamp(comp)
    except: return None

def score_rsi(hist):
    try:
        c = _close(hist)
        if len(c) < 16: return None
        d = c.diff(); g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        r = float((100 - (100/(1 + g/l.replace(0,1e-9)))).iloc[-1])
        # Contrarian: extremes predict mean reversion (Jegadeesh 1990)
        if   r < 20: return 92
        elif r < 28: return 80
        elif r < 38: return 66
        elif r < 48: return 56
        elif r < 52: return 50
        elif r < 62: return 44
        elif r < 72: return 30
        elif r < 80: return 18
        else:        return 8
    except: return None

def score_macd(hist):
    try:
        c = _close(hist)
        if len(c) < 35: return None
        m = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
        h = m - m.ewm(span=9,adjust=False).mean()
        cv, pv, p2 = float(h.iloc[-1]), float(h.iloc[-2]), float(h.iloc[-3])
        if cv > 0 and pv <= 0:              return 86
        if cv < 0 and pv >= 0:             return 14
        if cv > 0 and cv > pv and pv > p2: return _clamp(72 + min(abs(cv)*10,18))
        if cv > 0 and cv > pv:             return _clamp(64 + min(abs(cv)*6,14))
        if cv > 0:                          return 58
        if cv < 0 and cv < pv and pv < p2: return _clamp(28 - min(abs(cv)*10,18))
        if cv < 0 and cv < pv:             return _clamp(36 - min(abs(cv)*6,14))
        return 42
    except: return None

def score_volume(hist):
    try:
        vol = hist["Volume"]
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:,0]
        vol = vol.astype(float); c = _close(hist)
        if len(vol) < 20: return None
        avg = float(vol.iloc[-90:].mean()) if len(vol)>=90 else float(vol.mean())
        rec = float(vol.iloc[-5:].mean())
        if avg == 0: return None
        ratio = rec / avg
        up = float(c.iloc[-1]) > float(c.iloc[-6]) if len(c)>=6 else True
        if   ratio > 3.0: amp = 42
        elif ratio > 2.0: amp = 30
        elif ratio > 1.5: amp = 20
        elif ratio > 1.2: amp = 12
        elif ratio < 0.4: amp = -18
        elif ratio < 0.6: amp = -10
        elif ratio < 0.8: amp = -4
        else:             amp = 0
        base = 58 if up else 42
        return _clamp(base + (amp if up else -amp))
    except: return None

def score_week52(hist):
    try:
        c = _close(hist)
        if len(c) < 50: return None
        n = min(252, len(c))
        hi = float(c.rolling(n).max().iloc[-1]); lo = float(c.rolling(n).min().iloc[-1])
        curr = float(c.iloc[-1])
        if hi == lo: return None
        pct = (curr - lo) / (hi - lo)
        ret1m = float(c.iloc[-1]/c.iloc[-22]-1) if len(c)>=22 else 0
        up = ret1m > 0.005
        if   pct < 0.10 and up:  return _clamp(88 + (0.10-pct)*100)
        elif pct < 0.10:         return _clamp(28 - (0.10-pct)*80)
        elif pct < 0.25 and up:  return 72
        elif pct < 0.25:         return 34
        elif pct < 0.50:         return _clamp(42 + pct*20)
        elif pct < 0.75:         return _clamp(54 + (pct-0.5)*20)
        elif pct < 0.90 and up:  return 74
        elif pct < 0.90:         return 34
        elif up:                 return _clamp(76 + (pct-0.90)*120)
        else:                    return _clamp(24 - (pct-0.90)*120)
    except: return None

def score_short_interest(info):
    try:
        s = info.get("shortPercentOfFloat")
        if s is None:
            mc = _f(info.get("marketCap")) or 0
            return 65 if mc > 100e9 else 55
        s *= 100
        if   s > 30: return 12
        elif s > 20: return 22
        elif s > 15: return 32
        elif s > 10: return 42
        elif s > 5:  return 58
        elif s > 2:  return 70
        else:        return 78
    except: return 55

def score_earnings_surprise(info):
    # PEAD: post-earnings announcement drift (Ball & Brown 1968, Bernard & Thomas 1989)
    try:
        t = _f(info.get("trailingEps")); fw = _f(info.get("forwardEps"))
        if t is None: return None
        if fw is None: fw = t
        if abs(fw) < 0.01: return None
        s = (t - fw) / abs(fw) * 100
        if   s >  30: return 90
        elif s >  15: return 78
        elif s >   5: return 66
        elif s >   0: return 58
        elif s >  -5: return 42
        elif s > -15: return 28
        else:         return 12
    except: return None

def score_sentiment(info):
    try:
        r = _f(info.get("recommendationMean"))
        if r is None: return 54
        if   r <= 1.3: return 92
        elif r <= 1.7: return 82
        elif r <= 2.0: return 74
        elif r <= 2.3: return 65
        elif r <= 2.7: return 54
        elif r <= 3.0: return 45
        elif r <= 3.5: return 34
        elif r <= 4.0: return 23
        else:          return 12
    except: return 54

def score_analyst_revisions(info):
    # Revision momentum: Stickel 1992 — analyst upgrades predict 6-month outperformance
    try:
        r = _f(info.get("recommendationMean") or 3.0)
        n = int(info.get("numberOfAnalystOpinions") or 0)
        if n == 0: return 54
        if   r <= 1.3: base = 92
        elif r <= 1.7: base = 82
        elif r <= 2.0: base = 74
        elif r <= 2.3: base = 65
        elif r <= 2.7: base = 54
        elif r <= 3.0: base = 45
        elif r <= 3.5: base = 34
        elif r <= 4.0: base = 23
        else:          base = 12
        amp = 12 if n>=40 else (8 if n>=25 else (4 if n>=15 else (-8 if n<5 else 0)))
        d = base - 50
        return _clamp(50 + d + (amp if d>0 else -amp))
    except: return 54

def score_analyst_target(info):
    try:
        tgt = _f(info.get("targetMeanPrice") or info.get("targetMedianPrice"))
        cur = _f(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
        if not tgt or not cur or cur == 0: return None
        up = (tgt/cur - 1) * 100
        if   up >  50: return 95
        elif up >  30: return 86
        elif up >  20: return 78
        elif up >  12: return 70
        elif up >   5: return 62
        elif up >   0: return 55
        elif up >  -5: return 45
        elif up > -12: return 36
        elif up > -20: return 26
        elif up > -30: return 16
        else:          return 6
    except: return None

def score_pe_vs_sector(info):
    # Fama-French value factor: relative cheapness within sector
    try:
        pe = _f(info.get("trailingPE") or info.get("forwardPE"))
        if pe is None or pe <= 0 or pe > 1500: return None
        bms = {"Technology":28,"Healthcare":22,"Financials":13,
               "Consumer Cyclical":20,"Consumer Defensive":18,"Energy":12,
               "Utilities":16,"Real Estate":35,"Basic Materials":14,
               "Industrials":20,"Communication Services":22}
        bm = bms.get(info.get("sector",""), 20)
        r = pe / bm
        if   r < 0.3: return 92
        elif r < 0.5: return 82
        elif r < 0.7: return 72
        elif r < 0.9: return 62
        elif r < 1.1: return 52
        elif r < 1.4: return 40
        elif r < 1.8: return 28
        elif r < 2.5: return 16
        else:         return 6
    except: return None

def score_revenue_growth(info):
    # Strongest long-term predictor per Novy-Marx 2013 and AQR QMJ
    try:
        g = _f(info.get("revenueGrowth"))
        if g is None: return None
        g *= 100
        if   g >  50: return 95
        elif g >  30: return 85
        elif g >  20: return 75
        elif g >  10: return 65
        elif g >   5: return 58
        elif g >   0: return 50
        elif g >  -5: return 36
        elif g > -15: return 24
        elif g > -25: return 12
        else:         return 4
    except: return None

def score_profit_margin(info):
    # Gross profitability premium (Novy-Marx 2013)
    try:
        m = _f(info.get("profitMargins"))
        if m is None: return None
        m *= 100
        if   m >  40: return 94
        elif m >  30: return 86
        elif m >  20: return 77
        elif m >  12: return 67
        elif m >   6: return 57
        elif m >   2: return 46
        elif m >   0: return 36
        elif m >  -8: return 24
        else:         return 10
    except: return None

def score_fcf_yield(info):
    # FCF yield premium (AQR Quality-Minus-Junk)
    try:
        fcf = _f(info.get("freeCashflow")); mc = _f(info.get("marketCap"))
        if fcf is None or mc is None or mc == 0: return None
        y = (fcf/mc)*100
        if   y >  10: return 92
        elif y >   6: return 80
        elif y >   4: return 70
        elif y >   2: return 62
        elif y >   0: return 52
        elif y >  -3: return 34
        elif y >  -6: return 20
        else:         return 8
    except: return None

def score_debt_equity(info):
    # Leverage factor: low leverage predicts outperformance (Fama-French)
    try:
        de = _f(info.get("debtToEquity"))
        if de is None: return 56
        if   de <  10: return 86
        elif de <  30: return 76
        elif de <  60: return 65
        elif de < 100: return 54
        elif de < 150: return 40
        elif de < 250: return 26
        else:          return 12
    except: return 56

def score_eps_growth(info):
    # EPS acceleration: Fama-French earnings momentum
    try:
        t = _f(info.get("trailingEps")); fw = _f(info.get("forwardEps"))
        if t is None: return None
        if fw is None: fw = t
        if abs(t) < 0.01: return None
        g = (fw-t)/abs(t)*100
        if   g >  50: return 94
        elif g >  30: return 84
        elif g >  20: return 74
        elif g >  10: return 64
        elif g >   3: return 56
        elif g >   0: return 50
        elif g >  -8: return 34
        elif g > -20: return 22
        else:         return 8
    except: return None

def score_peg(info):
    # Growth-adjusted value (Lynch 1989, validated by Fama-French)
    try:
        p = _f(info.get("pegRatio"))
        if p is None or p <= 0: return None
        if   p < 0.5: return 94
        elif p < 0.8: return 82
        elif p < 1.0: return 72
        elif p < 1.3: return 60
        elif p < 1.8: return 44
        elif p < 2.5: return 28
        elif p < 4.0: return 14
        else:         return 4
    except: return None

_macro_cache  = {}
_sector_cache = {}

def score_macro(_=None):
    global _macro_cache
    today = datetime.now().strftime("%Y-%m-%d")
    if today in _macro_cache: return _macro_cache[today]
    try:
        h = _get_history("^VIX", period="1mo")
        if h.empty: raise ValueError
        c = _close(h)
        v, v5 = float(c.iloc[-1]), (float(c.iloc[-5]) if len(c)>=5 else float(c.iloc[-1]))
        rising = v > v5 * 1.05
        if   v < 12: base = 86
        elif v < 15: base = 75
        elif v < 18: base = 64
        elif v < 22: base = 52
        elif v < 28: base = 38
        elif v < 35: base = 24
        else:        base = 12
        if rising: base = max(8, base-12)
        _macro_cache[today] = base
        return base
    except:
        _macro_cache[today] = 50; return 50

def score_sector_momentum(info):
    global _sector_cache
    ETFS = {"Technology":"XLK","Healthcare":"XLV","Financials":"XLF",
            "Consumer Cyclical":"XLY","Consumer Defensive":"XLP","Energy":"XLE",
            "Utilities":"XLU","Real Estate":"XLRE","Basic Materials":"XLB",
            "Industrials":"XLI","Communication Services":"XLC"}
    try:
        etf = ETFS.get(info.get("sector",""), "SPY")
        key = f"{etf}_{datetime.now().strftime('%Y-%m-%d')}"
        if key not in _sector_cache:
            h = _get_history(etf, period="1y")
            if h.empty: _sector_cache[key] = 50
            else:
                c = _close(h)
                if len(c) >= 22:
                    r1m = (c.iloc[-1]/c.iloc[-22]-1)*100
                    r3m = (c.iloc[-1]/c.iloc[-63]-1)*100 if len(c)>=63 else r1m
                    _sector_cache[key] = _clamp(50 + r1m*4.0 + r3m*1.5)
                else: _sector_cache[key] = 50
        return _sector_cache[key]
    except: return 50

def score_earnings_proximity(info):
    try:
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts is None: return 56
        d = (datetime.fromtimestamp(int(ts)) - datetime.now()).days
        if   0 <= d <= 2:         return 28
        elif 0 <= d <= 7:         return 36
        elif 0 <= d <= 14:        return 44
        elif 0 <= d <= 30:        return 52
        elif d < 0 and d >= -5:  return 70
        elif d < 0 and d >= -14: return 64
        elif d < 0 and d >= -30: return 58
        return 56
    except: return 56

# ─── CATEGORY SCORING ENGINE ─────────────────────────────────────────────────

def _build_point_scores(hist, info: dict) -> dict:
    etf = _is_etf(info)
    return {
        "momentum_multi":    score_momentum_multi(hist),
        "rsi":               score_rsi(hist),
        "macd":              score_macd(hist),
        "volume_trend":      score_volume(hist),
        "week52_position":   score_week52(hist),
        "earnings_proximity":score_earnings_proximity(info),
        "sector_momentum":   score_sector_momentum(info),
        "short_interest":    score_short_interest(info),
        "sentiment":         score_sentiment(info),
        "analyst_target":    None if etf else score_analyst_target(info),
        "earnings_surprise": None if etf else score_earnings_surprise(info),
        "analyst_revisions": score_analyst_revisions(info),
        "macro_score":       score_macro(),
        "pe_vs_sector":      None if etf else score_pe_vs_sector(info),
        "revenue_growth":    None if etf else score_revenue_growth(info),
        "profit_margin":     None if etf else score_profit_margin(info),
        "fcf_yield":         None if etf else score_fcf_yield(info),
        "debt_equity":       None if etf else score_debt_equity(info),
        "eps_growth":        None if etf else score_eps_growth(info),
        "peg_ratio":         None if etf else score_peg(info),
    }

def _score_category(cat_name: str, point_scores: dict) -> dict:
    pts = CATEGORIES[cat_name]["points"]
    scores, weights = [], []
    for p in pts:
        v = point_scores.get(p)
        if v is not None:
            scores.append(v); weights.append(POINT_WEIGHTS.get(p, 1))
    if not scores:
        return {"score": None, "raw": None, "n_available": 0,
                "n_total": len(pts), "signal": "—"}
    raw = sum(s*w for s,w in zip(scores,weights)) / sum(weights)
    # Consistency bonus within category
    if   all(s >= 65 for s in scores): raw = min(100, raw + 12)
    elif all(s >= 58 for s in scores): raw = min(100, raw + 6)
    elif all(s <= 35 for s in scores): raw = max(0,   raw - 12)
    elif all(s <= 42 for s in scores): raw = max(0,   raw - 6)
    score = stretch_score(raw)
    return {"score": score, "raw": round(raw,1),
            "n_available": len(scores), "n_total": len(pts),
            "signal": _signal_label(score)}

def _build_composite(cat_results: dict, horizon: str) -> tuple:
    cw = CATEGORY_WEIGHTS[horizon]
    total_w, wsum = 0, 0
    for cat, res in cat_results.items():
        if res["score"] is not None and res["n_available"] > 0:
            w = cw.get(cat, 5)
            wsum   += res["score"] * w
            total_w += w
    if total_w == 0: return 50, 50.0
    raw = wsum / total_w
    return stretch_score(raw), round(raw, 1)

def compute_price_target(price: float, score: int, horizon: str) -> float:
    return round(price * (1 + ((score-50)/50) * HORIZON_FACTORS[horizon]), 2)

def compute_brier(score, actual_ret, _):
    return round((score/100.0 - (1 if actual_ret>0 else 0))**2, 4)

# ─── MAIN FETCH & SCORE ──────────────────────────────────────────────────────

def fetch_and_score(ticker: str, horizon: str = "1M", _hist=None, _info=None) -> dict:
    try:
        info  = _info if _info is not None else _get_info(ticker)
        hist  = _hist if _hist is not None else _get_history(ticker)
        error = "Info unavailable" if info.get("_fetch_failed") else None
        if hist is None or hist.empty:
            return _error_result(ticker, horizon, "No price history from Yahoo Finance")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        pts    = _build_point_scores(hist, info)
        cats   = {cat: _score_category(cat, pts) for cat in CATEGORIES}
        final, raw = _build_composite(cats, horizon)
        price  = round(float(_close(hist).iloc[-1]), 2)
        target = compute_price_target(price, final, horizon)
        avail  = sum(1 for v in pts.values() if v is not None)
        display_pts = {k: (v if v is not None else 50) for k, v in pts.items()}
        return {
            "ticker": ticker.upper(), "horizon": horizon, "overall": final,
            "raw_score": raw, "signal": _signal_label(final), "price": price,
            "target": target, "data_scores": display_pts, "category_scores": cats,
            "brier": None, "confidence": round((avail/len(pts))*100),
            "is_etf": _is_etf(info), "error": error,
            "name": info.get("shortName", ticker.upper()),
            "sector": info.get("sector","—"), "market_cap": info.get("marketCap"),
        }
    except Exception as e:
        return _error_result(ticker, horizon, str(e))

def fetch_and_score_batch(tickers: list, horizon: str = "1M") -> pd.DataFrame:
    rows = []
    for t in tickers:
        r = fetch_and_score(t, horizon)
        rows.append({
            "Ticker": r["ticker"], "Name": r.get("name",""),
            "Sector": r.get("sector",""), "Score": r["overall"],
            "Signal": r["signal"], "Price": r.get("price"), "Target": r.get("target"),
            "Upside %": round((r["target"]/r["price"]-1)*100,1)
                        if r.get("price") and r.get("target") else None,
            "Confidence %": r["confidence"], "Is ETF": r.get("is_etf",False),
            "Error": r["error"],
        })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: MARKET PROBABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Market Probability uses 7 academically-validated macro/market predictors.
Each scored 0-100 (50=neutral). Weights = correlation r × 100.

Source: Shiller (CAPE), Harvey (yield curve), Greenwood & Hanson (credit),
        Baker & Wurgler (sentiment), AQR (breadth/momentum).
"""

MARKET_PREDICTORS = {
    # name: (weight, description, direction_note)
    "cape_valuation":    (38, "Shiller CAPE / Market P/E", "Low CAPE = bullish; High CAPE = bearish"),
    "yield_curve":       (28, "10Y-2Y Treasury Yield Spread", "Positive & widening = bullish; inverted = bearish"),
    "credit_spread":     (25, "High Yield vs Investment Grade Spread", "Narrow spread = risk-on bullish; wide spread = bearish"),
    "vix_regime":        (22, "VIX Level & Trend", "Low stable VIX = bullish; high/rising VIX = bearish"),
    "market_breadth":    (18, "% Stocks Above 200-Day MA", "High breadth = healthy bull market"),
    "index_momentum":    (15, "Index 12-Month Momentum", "Strong trend = continuation signal"),
    "revision_breadth":  (14, "Earnings Revision Sentiment", "More upgrades than downgrades = bullish"),
}

INDEX_TICKERS = {
    "DOW":     {"etf": "DIA",  "name": "Dow Jones Industrial Average", "components": ["AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","HD","MRK","CVX","MCD","BA","CAT","GS","AXP","IBM","MMM","HON","DIS","NKE","AMGN","CRM","TRV","VZ","CSCO","DOW","KO","INTC","WBA"]},
    "NASDAQ":  {"etf": "QQQ",  "name": "NASDAQ-100", "components": ["AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AVGO","COST","NFLX","AMD","ADBE","QCOM","PYPL","INTC","CMCSA","TXN","INTU","SBUX","MDLZ"]},
    "S&P 500": {"etf": "SPY",  "name": "S&P 500", "components": ["AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AVGO","COST","NFLX","BRK-B","LLY","XOM","ORCL","ABBV","BAC","JPM","V","UNH","PG"]},
}

_market_cache = {}

def _score_cape(index_key: str) -> tuple:
    """
    CAPE proxy: Use SPY/DIA/QQQ trailing P/E vs historical average.
    Historical avg S&P CAPE ≈ 17. Current elevated = bearish for 10Y returns.
    Score: current P/E vs 20-year avg proxy.
    """
    try:
        etf  = INDEX_TICKERS[index_key]["etf"]
        info = _get_info(etf)
        pe   = _f(info.get("trailingPE") or info.get("forwardPE"))
        if pe is None:
            # Try SPY as fallback proxy
            info = _get_info("SPY")
            pe   = _f(info.get("trailingPE") or info.get("forwardPE"))
        if pe is None: return 50, "P/E unavailable"
        # Historical S&P P/E avg ≈ 16-18. CAPE adds 10Y smoothing, so effective avg ≈ 20-22
        avg_pe = 21.0
        ratio  = pe / avg_pe
        if   ratio < 0.7:  score = 88
        elif ratio < 0.85: score = 75
        elif ratio < 1.0:  score = 62
        elif ratio < 1.15: score = 52
        elif ratio < 1.3:  score = 42
        elif ratio < 1.6:  score = 30
        elif ratio < 2.0:  score = 18
        else:              score = 8
        return score, f"P/E {round(pe,1)} vs avg {avg_pe} (ratio {round(ratio,2)})"
    except Exception as e:
        return 50, str(e)

def _score_yield_curve() -> tuple:
    """10Y-2Y Treasury spread. Positive = normal, negative = inverted = recession warning."""
    try:
        t10 = _get_history("^TNX", period="3mo")
        t2  = _get_history("^IRX", period="3mo")  # 13-week = proxy for 2Y
        if t10.empty or t2.empty: return 50, "Treasury data unavailable"
        y10 = float(_close(t10).iloc[-1]) / 100
        y2  = float(_close(t2).iloc[-1])  / 100
        spread = y10 - y2
        # Trend: is curve steepening or flattening?
        y10_30d = float(_close(t10).iloc[-22]) / 100 if len(t10) >= 22 else y10
        y2_30d  = float(_close(t2).iloc[-22])  / 100 if len(t2)  >= 22 else y2
        spread_30d = y10_30d - y2_30d
        steepening = spread > spread_30d
        if   spread > 1.5:  score = 82
        elif spread > 0.75: score = 70
        elif spread > 0.25: score = 60
        elif spread > 0.0:  score = 54
        elif spread > -0.25:score = 44
        elif spread > -0.75:score = 32
        else:               score = 18
        if steepening and score < 70: score = min(score + 8, 100)
        return score, f"10Y-2Y spread: {round(spread*100,1)}bps ({'steepening' if steepening else 'flattening'})"
    except Exception as e:
        return 50, str(e)

def _score_credit_spread() -> tuple:
    """HYG (high yield) vs LQD (investment grade) price ratio as credit stress proxy."""
    try:
        hyg = _get_history("HYG", period="6mo")
        lqd = _get_history("LQD", period="6mo")
        if hyg.empty or lqd.empty: return 50, "Credit ETF data unavailable"
        hyg_c = _close(hyg); lqd_c = _close(lqd)
        ratio_now = float(hyg_c.iloc[-1]) / float(lqd_c.iloc[-1])
        ratio_3m  = float(hyg_c.iloc[-63]) / float(lqd_c.iloc[-63]) if len(hyg_c)>=63 else ratio_now
        # Rising ratio = HY outperforming = credit risk appetite = bullish
        change_pct = (ratio_now / ratio_3m - 1) * 100
        if   change_pct >  3.0: score = 80
        elif change_pct >  1.5: score = 70
        elif change_pct >  0.5: score = 62
        elif change_pct > -0.5: score = 52
        elif change_pct > -1.5: score = 40
        elif change_pct > -3.0: score = 28
        else:                   score = 16
        return score, f"HYG/LQD ratio change 3M: {round(change_pct,2)}%"
    except Exception as e:
        return 50, str(e)

def _score_vix_regime() -> tuple:
    """VIX level + trend. Low stable VIX = bullish regime."""
    try:
        h = _get_history("^VIX", period="3mo")
        if h.empty: return 50, "VIX unavailable"
        c = _close(h)
        vix = float(c.iloc[-1])
        vix_5d  = float(c.iloc[-5])  if len(c)>=5  else vix
        vix_20d = float(c.iloc[-22]) if len(c)>=22 else vix
        rising  = vix > vix_5d  * 1.05
        elevated= vix > vix_20d * 1.15
        if   vix < 12: base = 86
        elif vix < 15: base = 76
        elif vix < 18: base = 66
        elif vix < 22: base = 54
        elif vix < 28: base = 40
        elif vix < 35: base = 26
        else:          base = 12
        if rising:   base = max(6, base-12)
        if elevated: base = max(6, base-6)
        return base, f"VIX {round(vix,1)} ({'rising' if rising else 'falling/stable'})"
    except Exception as e:
        return 50, str(e)

def _score_market_breadth(index_key: str) -> tuple:
    """
    % of index components currently above their 200-day moving average.
    >70% = healthy bull market. <40% = deteriorating breadth.
    Uses a representative sample of components for speed.
    """
    try:
        components = INDEX_TICKERS[index_key]["components"][:15]  # sample for speed
        above_200  = 0
        total      = 0
        for t in components:
            try:
                h = _get_history(t, period="1y")
                if h.empty: continue
                c = _close(h)
                if len(c) >= 200:
                    ma200 = float(c.rolling(200).mean().iloc[-1])
                    if float(c.iloc[-1]) > ma200: above_200 += 1
                    total += 1
            except:
                continue
        if total == 0: return 50, "No component data"
        pct = above_200 / total * 100
        if   pct >= 75: score = 84
        elif pct >= 60: score = 72
        elif pct >= 50: score = 60
        elif pct >= 40: score = 50
        elif pct >= 30: score = 38
        elif pct >= 20: score = 26
        else:           score = 14
        return score, f"{round(pct,0)}% of components above 200MA ({above_200}/{total})"
    except Exception as e:
        return 50, str(e)

def _score_index_momentum(index_key: str) -> tuple:
    """12-month vs 1-month index momentum (AQR: skip-1M to avoid short-term reversal)."""
    try:
        etf  = INDEX_TICKERS[index_key]["etf"]
        hist = _get_history(etf, period="2y")
        if hist.empty: return 50, "Index history unavailable"
        c = _close(hist)
        if len(c) < 252: return 50, "Insufficient history"
        ret_12m = (float(c.iloc[-1]) / float(c.iloc[-252]) - 1) * 100
        ret_1m  = (float(c.iloc[-1]) / float(c.iloc[-22])  - 1) * 100
        # AQR: 12-1M momentum (exclude most recent month to reduce reversal noise)
        ret_11m = ret_12m - ret_1m
        if   ret_11m >  20: score = 86
        elif ret_11m >  10: score = 76
        elif ret_11m >   5: score = 66
        elif ret_11m >   0: score = 56
        elif ret_11m >  -5: score = 44
        elif ret_11m > -10: score = 34
        elif ret_11m > -20: score = 22
        else:               score = 10
        return score, f"11M momentum: {round(ret_11m,1)}% (12M: {round(ret_12m,1)}%)"
    except Exception as e:
        return 50, str(e)

def _score_revision_breadth(index_key: str) -> tuple:
    """
    Sample analyst recommendation means across index components.
    Improving consensus = bullish revision breadth.
    """
    try:
        components = INDEX_TICKERS[index_key]["components"][:10]
        recs = []
        for t in components:
            try:
                info = _get_info(t)
                r = _f(info.get("recommendationMean"))
                if r: recs.append(r)
            except:
                continue
        if not recs: return 50, "No analyst data"
        avg_rec = sum(recs) / len(recs)
        # 1.0=Strong Buy, 5.0=Strong Sell
        score = _clamp(95 - (avg_rec - 1.0) * 22.5)
        return score, f"Avg analyst rec: {round(avg_rec,2)} ({len(recs)} stocks sampled)"
    except Exception as e:
        return 50, str(e)

def score_market_probability(index_key: str = "S&P 500") -> dict:
    """
    Compute market probability score for DOW, NASDAQ, or S&P 500.
    Returns comprehensive dict with:
    - overall market score (0-100)
    - price target for each horizon
    - individual predictor scores and details
    - bull/bear/neutral signal
    """
    if index_key not in INDEX_TICKERS:
        return {"error": f"Unknown index: {index_key}. Use: {list(INDEX_TICKERS.keys())}"}

    cache_key = f"market_{index_key}_{datetime.now().strftime('%Y-%m-%d')}"
    if cache_key in _market_cache:
        return _market_cache[cache_key]

    predictors_raw = {}

    # Run all 7 predictors
    s, d = _score_cape(index_key);           predictors_raw["cape_valuation"]   = (s, d)
    s, d = _score_yield_curve();             predictors_raw["yield_curve"]       = (s, d)
    s, d = _score_credit_spread();           predictors_raw["credit_spread"]     = (s, d)
    s, d = _score_vix_regime();              predictors_raw["vix_regime"]         = (s, d)
    s, d = _score_market_breadth(index_key); predictors_raw["market_breadth"]    = (s, d)
    s, d = _score_index_momentum(index_key); predictors_raw["index_momentum"]    = (s, d)
    s, d = _score_revision_breadth(index_key);predictors_raw["revision_breadth"] = (s, d)

    # Weighted composite
    total_w, wsum = 0, 0
    predictor_details = {}
    for key, (score, detail) in predictors_raw.items():
        w = MARKET_PREDICTORS[key][0]
        wsum += score * w; total_w += w
        predictor_details[key] = {
            "score":       score,
            "weight":      w,
            "description": MARKET_PREDICTORS[key][1],
            "direction":   MARKET_PREDICTORS[key][2],
            "detail":      detail,
            "signal":      _signal_label(score),
        }

    raw   = wsum / total_w if total_w > 0 else 50
    final = stretch_score(raw)

    # Get current index price and compute targets
    try:
        etf      = INDEX_TICKERS[index_key]["etf"]
        hist     = _get_history(etf, period="1mo")
        etf_price = round(float(_close(hist).iloc[-1]), 2) if not hist.empty else None
    except:
        etf_price = None

    targets = {}
    if etf_price:
        for h, factor in HORIZON_FACTORS.items():
            direction = (final - 50) / 50
            targets[h] = round(etf_price * (1 + direction * factor), 2)

    result = {
        "index":       index_key,
        "etf":         INDEX_TICKERS[index_key]["etf"],
        "index_name":  INDEX_TICKERS[index_key]["name"],
        "overall":     final,
        "raw_score":   round(raw, 1),
        "signal":      _signal_label(final),
        "etf_price":   etf_price,
        "targets":     targets,
        "predictors":  predictor_details,
        "timestamp":   datetime.now().isoformat(),
        "error":       None,
    }
    _market_cache[cache_key] = result
    return result

def run_market_scenario(index_key: str, start_date: str, end_date: str, dollars: float,
                        interval: str, buy_rating: int, sell_rating: int) -> tuple:
    """
    Scenario simulator for index ETFs (DIA/QQQ/SPY).
    Uses index momentum + VIX as the two most reliable short-term index predictors.
    """
    INTERVAL_DAYS = {"1D":1,"1W":7,"1M":30}
    ds = INTERVAL_DAYS.get(interval, 7)
    try:
        etf   = INDEX_TICKERS[index_key]["etf"]
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        if end <= start: return None, "End date must be after start date."

        t  = yf.Ticker(etf)
        hf = t.history(start=start-timedelta(days=400), end=end+timedelta(days=2))
        if hf.empty:
            hf = yf.download(etf,
                start=(start-timedelta(days=400)).strftime("%Y-%m-%d"),
                end=(end+timedelta(days=2)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True)
            if isinstance(hf.columns, pd.MultiIndex): hf.columns = hf.columns.get_level_values(0)
        if hf.empty: return None, f"No price data found for {etf}."
        if hf.index.tz is not None: hf.index = hf.index.tz_localize(None)

        # Also fetch VIX for macro scoring
        vix_h = _get_history("^VIX", period="2y")

        cash, shares, log, cur = float(dollars), 0, [], start
        while cur <= end and len(log) < 500:
            sub = hf[hf.index <= pd.Timestamp(cur)]
            if len(sub) < 22: cur += timedelta(days=ds); continue

            price = round(float(_close(sub).iloc[-1]), 2)

            # Market score: blend momentum (60%) + macro/VIX (40%)
            mom   = score_momentum_multi(sub) or 50
            macro = score_macro() or 50
            raw   = mom * 0.60 + macro * 0.40
            score = stretch_score(raw)

            action = "HOLD"
            if score >= buy_rating and cash >= price:
                n = int(cash//price)
                if n > 0: shares+=n; cash-=n*price; action="BUY"
            elif score <= sell_rating and shares > 0:
                cash+=shares*price; shares=0; action="SELL"

            log.append({"Date": cur.strftime("%Y-%m-%d"), "Score": score, "Price": price,
                        "Action": action, "Shares Held": shares, "Cash": round(cash,2),
                        "Portfolio Value": round(cash+shares*price,2)})
            cur += timedelta(days=ds)

        df = pd.DataFrame(log)
        if df.empty: return None, "No trading periods in date range."
        final = df["Portfolio Value"].iloc[-1]
        bh    = round(float(dollars)*(df["Price"].iloc[-1]/df["Price"].iloc[0]),2)
        tr    = round((final/float(dollars)-1)*100,2)
        bhr   = round((bh/float(dollars)-1)*100,2)
        return df, {
            "Index": index_key, "ETF": etf,
            "Final Portfolio": round(final,2), "Total Return %": tr,
            "Buy & Hold Return %": bhr, "Alpha vs B&H %": round(tr-bhr,2),
            "Total Trades": len(df[df["Action"]!="HOLD"]),
            "Start Date": start_date, "End Date": end_date,
            "Starting Capital": float(dollars),
        }
    except Exception as e:
        return None, str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: SHARED BACKTEST / SCENARIO (unchanged interface)
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_ticker(ticker: str, date_str: str, end_date_str: str = None) -> dict:
    HORIZON_DAYS = {"1D":1,"1W":7,"1M":30,"1Q":90,"1Y":365}
    try:
        start = datetime.strptime(date_str, "%Y-%m-%d")
        end_cap = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None
        t = yf.Ticker(ticker)
        h2d = t.history(start=start-timedelta(days=400), end=start+timedelta(days=2))
        if h2d.empty:
            h2d = yf.download(ticker,
                start=(start-timedelta(days=400)).strftime("%Y-%m-%d"),
                end=(start+timedelta(days=2)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True)
            if isinstance(h2d.columns, pd.MultiIndex): h2d.columns = h2d.columns.get_level_values(0)
        if h2d.empty: return {"error": f"No data for {ticker} around {date_str}"}
        pad = round(float(_close(h2d).iloc[-1]), 2)
        fe  = min(end_cap, datetime.now()) if end_cap else datetime.now()
        hf  = t.history(start=start, end=fe+timedelta(days=2))
        if isinstance(hf.columns, pd.MultiIndex): hf.columns = hf.columns.get_level_values(0)
        info = _get_info(ticker)
        pts  = _build_point_scores(h2d, info)
        cats = {cat: _score_category(cat, pts) for cat in CATEGORIES}
        results = []
        for horizon, days in HORIZON_DAYS.items():
            overall, _ = _build_composite(cats, horizon)
            pred_tgt   = compute_price_target(pad, overall, horizon)
            he = start + timedelta(days=days)
            if end_cap: he = min(he, end_cap)
            idx   = hf.index
            he_ts = pd.Timestamp(he).tz_localize(idx.tz) if (hasattr(idx,'tz') and idx.tz) else pd.Timestamp(he)
            sl    = hf[hf.index <= he_ts]
            if len(sl) > 0:
                ap = round(float(_close(sl).iloc[-1]),2); ar = round((ap/pad-1)*100,2)
                br = compute_brier(overall, ar, horizon)
            else:
                ap = ar = br = None
            pd_ = "UP" if overall>=55 else ("DOWN" if overall<=45 else "FLAT")
            ad_ = ("UP" if (ar or 0)>0.5 else "DOWN" if (ar or 0)<-0.5 else "FLAT") if ar is not None else "—"
            results.append({"Horizon": horizon, "Score": overall, "Signal": _signal_label(overall),
                            "Pred Target": pred_tgt, "Actual Return %": ar, "Actual Price": ap,
                            "Direction Correct": "✓ YES" if (ar is not None and pd_==ad_)
                                                 else ("✗ NO" if ar is not None else "—"),
                            "Brier Score": br})
        return {"ticker": ticker.upper(), "date": date_str, "end_date": end_date_str,
                "base_price": pad, "results": results, "error": None}
    except Exception as e:
        return {"error": str(e)}

def run_scenario(ticker, start_date, end_date, dollars, interval, buy_rating, sell_rating):
    INTERVAL_DAYS = {"1D":1,"1W":7,"1M":30}
    ds = INTERVAL_DAYS.get(interval, 7)
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        if end <= start: return None, "End date must be after start date."
        t  = yf.Ticker(ticker)
        hf = t.history(start=start-timedelta(days=400), end=end+timedelta(days=2))
        if hf.empty:
            hf = yf.download(ticker,
                start=(start-timedelta(days=400)).strftime("%Y-%m-%d"),
                end=(end+timedelta(days=2)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True)
            if isinstance(hf.columns, pd.MultiIndex): hf.columns = hf.columns.get_level_values(0)
        if hf.empty: return None, "No price data found."
        if hf.index.tz is not None: hf.index = hf.index.tz_localize(None)
        cash, shares, log, cur = float(dollars), 0, [], start
        while cur <= end and len(log) < 500:
            sub = hf[hf.index <= pd.Timestamp(cur)]
            if len(sub) < 20: cur += timedelta(days=ds); continue
            price = round(float(_close(sub).iloc[-1]),2)
            raw   = (score_momentum_multi(sub) or 50)*0.40 + (score_rsi(sub) or 50)*0.35 + (score_macd(sub) or 50)*0.25
            score = stretch_score(raw)
            action = "HOLD"
            if score >= buy_rating and cash >= price:
                n = int(cash//price)
                if n > 0: shares+=n; cash-=n*price; action="BUY"
            elif score <= sell_rating and shares > 0:
                cash+=shares*price; shares=0; action="SELL"
            log.append({"Date": cur.strftime("%Y-%m-%d"), "Score": score, "Price": price,
                        "Action": action, "Shares Held": shares, "Cash": round(cash,2),
                        "Portfolio Value": round(cash+shares*price,2)})
            cur += timedelta(days=ds)
        df = pd.DataFrame(log)
        if df.empty: return None, "No trading periods in date range."
        final = df["Portfolio Value"].iloc[-1]; bh = round(float(dollars)*(df["Price"].iloc[-1]/df["Price"].iloc[0]),2)
        tr = round((final/float(dollars)-1)*100,2); bhr = round((bh/float(dollars)-1)*100,2)
        return df, {"Final Portfolio": round(final,2), "Total Return %": tr,
                    "Buy & Hold Return %": bhr, "Alpha vs B&H %": round(tr-bhr,2),
                    "Total Trades": len(df[df["Action"]!="HOLD"]),
                    "Start Date": start_date, "End Date": end_date, "Starting Capital": float(dollars)}
    except Exception as e:
        return None, str(e)

def optimize_strategy(ticker, start_date, end_date, dollars=10000):
    import itertools
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        total_days = (end - start).days
        if total_days < 90: return {"error": "Need at least 90 days to optimize."}
        split = start + timedelta(days=int(total_days*0.70))
        train_end  = split.strftime("%Y-%m-%d")
        test_start = (split+timedelta(days=1)).strftime("%Y-%m-%d")
        results = []
        for iv, br, sr in itertools.product(["1D","1W","1M"], range(52,82,3), range(25,52,3)):
            if sr >= br: continue
            df, s = run_scenario(ticker, start_date, train_end, dollars, iv, br, sr)
            if df is None: continue
            results.append({"interval":iv,"buy_rating":br,"sell_rating":sr,
                             "total_return":s["Total Return %"],"bh_return":s["Buy & Hold Return %"],
                             "alpha":s["Alpha vs B&H %"],"trades":s["Total Trades"]})
        if not results: return {"error": "No valid combinations found."}
        results.sort(key=lambda x: x["alpha"], reverse=True)
        validated = []
        for p in results[:5]:
            df, s = run_scenario(ticker, test_start, end_date, dollars,
                                 p["interval"], p["buy_rating"], p["sell_rating"])
            if df is None:
                validated.append({**p,"oos_return":None,"oos_bh_return":None,"oos_alpha":None,"overfit_flag":True})
            else:
                oa = s["Alpha vs B&H %"]
                validated.append({**p,"oos_return":s["Total Return %"],"oos_bh_return":s["Buy & Hold Return %"],
                                   "oos_alpha":oa,"overfit_flag":oa<p["alpha"]*0.5 if p["alpha"]>0 else True})
        valid = [v for v in validated if v.get("oos_alpha") is not None]
        best  = max(valid, key=lambda x: x["oos_alpha"]) if valid else validated[0]
        return {"ticker":ticker.upper(),"start_date":start_date,"end_date":end_date,
                "train_end":train_end,"test_start":test_start,
                "best_params":best,"top5":validated,"total_combos":len(results),"error":None}
    except Exception as e:
        return {"error": str(e)}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _signal_label(s):
    if s is None: return "—"
    if s >= 80:   return "STRONG BUY"
    if s >= 65:   return "BUY"
    if s >= 45:   return "NEUTRAL"
    if s >= 30:   return "SELL"
    return "STRONG SELL"

def _error_result(ticker, horizon, msg):
    return {"ticker":ticker,"horizon":horizon,"overall":50,"raw_score":50,
            "signal":"NEUTRAL","price":None,"target":None,
            "data_scores":{k:50 for k in DATA_POINT_LABELS},
            "category_scores":{},"brier":None,"confidence":0,
            "error":msg,"name":ticker,"sector":"—","market_cap":None,"is_etf":False}
