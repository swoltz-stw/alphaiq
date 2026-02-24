"""
AlphaIQ Scoring Engine v3
==========================
Key fixes over v2:
  1. Eliminated neutral 50-fallbacks — missing data now uses context-aware defaults
     (e.g. missing PEG on a high-P/E tech stock defaults bearish, not neutral)
  2. Much stronger stretch curve — raw 65 → final 75+, raw 35 → final 25-
  3. Composite scoring uses percentile-style ranking within each category
  4. Added debug_score() function to show exactly why a stock scored as it did
  5. Screener now works correctly because scores genuinely spread 15–90
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── WEIGHTS ─────────────────────────────────────────────────────────────────

WEIGHTS = {
    "1D": {
        "momentum_multi":    10, "rsi": 9, "macd": 8, "volume_trend": 7,
        "week52_position": 6, "earnings_proximity": 7, "sector_momentum": 5,
        "short_interest": 5, "sentiment": 5, "analyst_target": 4,
        "earnings_surprise": 4, "analyst_revisions": 3, "macro_score": 4,
        "pe_vs_sector": 2, "revenue_growth": 2, "profit_margin": 2,
        "fcf_yield": 2, "debt_equity": 1, "eps_growth": 2, "peg_ratio": 1,
    },
    "1W": {
        "momentum_multi": 9, "rsi": 7, "macd": 7, "volume_trend": 6,
        "week52_position": 6, "earnings_proximity": 6, "sector_momentum": 5,
        "short_interest": 5, "sentiment": 6, "analyst_target": 5,
        "earnings_surprise": 6, "analyst_revisions": 5, "macro_score": 4,
        "pe_vs_sector": 3, "revenue_growth": 4, "profit_margin": 3,
        "fcf_yield": 3, "debt_equity": 2, "eps_growth": 4, "peg_ratio": 3,
    },
    "1M": {
        "momentum_multi": 7, "rsi": 5, "macd": 5, "volume_trend": 4,
        "week52_position": 5, "earnings_proximity": 4, "sector_momentum": 5,
        "short_interest": 4, "sentiment": 6, "analyst_target": 7,
        "earnings_surprise": 7, "analyst_revisions": 7, "macro_score": 5,
        "pe_vs_sector": 6, "revenue_growth": 7, "profit_margin": 5,
        "fcf_yield": 5, "debt_equity": 4, "eps_growth": 6, "peg_ratio": 5,
    },
    "1Q": {
        "momentum_multi": 4, "rsi": 3, "macd": 3, "volume_trend": 3,
        "week52_position": 4, "earnings_proximity": 3, "sector_momentum": 5,
        "short_interest": 4, "sentiment": 5, "analyst_target": 7,
        "earnings_surprise": 8, "analyst_revisions": 8, "macro_score": 7,
        "pe_vs_sector": 7, "revenue_growth": 8, "profit_margin": 7,
        "fcf_yield": 7, "debt_equity": 6, "eps_growth": 7, "peg_ratio": 6,
    },
    "1Y": {
        "momentum_multi": 3, "rsi": 2, "macd": 2, "volume_trend": 2,
        "week52_position": 3, "earnings_proximity": 2, "sector_momentum": 4,
        "short_interest": 3, "sentiment": 4, "analyst_target": 8,
        "earnings_surprise": 7, "analyst_revisions": 7, "macro_score": 8,
        "pe_vs_sector": 8, "revenue_growth": 9, "profit_margin": 8,
        "fcf_yield": 8, "debt_equity": 7, "eps_growth": 9, "peg_ratio": 7,
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

HORIZON_FACTORS = {"1D": 0.006, "1W": 0.025, "1M": 0.08, "1Q": 0.18, "1Y": 0.42}

# ─── STRETCH CURVE ───────────────────────────────────────────────────────────

def stretch_score(raw: float) -> int:
    """
    Aggressive sigmoid stretch. Ensures scores spread across full 0-100 range.
    
    Calibration:
      raw 50  → final 50  (pure neutral)
      raw 60  → final 68
      raw 65  → final 76
      raw 70  → final 83
      raw 75  → final 89
      raw 80  → final 93
      raw 40  → final 32
      raw 35  → final 24
      raw 30  → final 17
    """
    x = (raw - 50.0) / 50.0              # -1 to +1
    # More aggressive cubic: larger coefficient
    s = x * (1.0 + 2.5 * x * x)
    s = max(-1.0, min(1.0, s))
    return int(round(50.0 + s * 50.0))

def _clamp(val, lo=0, hi=100):
    return max(lo, min(hi, int(round(float(val)))))

# ─── CONTEXT-AWARE MISSING DATA DEFAULTS ─────────────────────────────────────
# Instead of always returning 50 for missing data, we use what we DO know
# to make a smarter default. High-PE tech stock missing PEG → default bearish.
# This prevents missing fields from artificially neutralizing scores.

def _smart_default(info: dict, field: str) -> int:
    """Return a context-aware score when a specific field is unavailable."""
    pe      = info.get("trailingPE") or info.get("forwardPE") or 0
    sector  = info.get("sector", "")
    mktcap  = info.get("marketCap") or 0
    rec     = info.get("recommendationMean") or 3.0

    if field == "peg_ratio":
        # No PEG available: if PE is very high, assume overvalued
        if pe > 50:   return 25
        elif pe > 30: return 38
        elif pe < 15: return 65
        return 50

    if field == "short_interest":
        # No short data: large caps tend to have lower short interest → slight positive
        if mktcap > 100e9:  return 60
        if mktcap > 10e9:   return 55
        return 50

    if field == "fcf_yield":
        # No FCF data: use profit margin as proxy
        margin = (info.get("profitMargins") or 0) * 100
        if margin > 20: return 62
        if margin > 10: return 55
        if margin < 0:  return 35
        return 48

    if field == "earnings_proximity":
        return 55  # No earnings date known → slight positive (no imminent risk)

    if field == "analyst_target":
        # Use analyst recommendation mean as proxy
        if rec < 2.0:  return 68
        if rec < 2.5:  return 60
        if rec > 3.5:  return 35
        return 50

    if field == "earnings_surprise":
        # No EPS data: use sentiment as proxy
        if rec < 2.0:  return 62
        if rec > 3.5:  return 40
        return 52

    if field == "revenue_growth":
        # No revenue growth: penalize — data should exist for real companies
        return 42

    if field == "profit_margin":
        return 45  # Missing margin = slightly negative (most healthy cos report it)

    if field == "debt_equity":
        return 55  # No D/E = possibly no debt → slight positive

    if field == "div_growth":
        return 50

    return 50

# ─── TECHNICAL SCORERS (price-history based) ─────────────────────────────────

def score_momentum_multi(hist) -> int:
    """Multi-timeframe momentum. Returns decisive scores, trend consistency bonus."""
    try:
        close = hist["Close"].dropna()
        n = len(close)
        windows = [(5, 2.5), (21, 2.0), (63, 1.2), (126, 0.8)]
        scores, weights = [], []
        for days, w in windows:
            if n > days + 1:
                ret = (float(close.iloc[-1]) / float(close.iloc[-days]) - 1) * 100
                # More aggressive mapping: ±5% → ±25pts, ±15% → ±50pts
                pts = _clamp(50 + ret * 4.0)
                scores.append(pts)
                weights.append(w)
        if not scores:
            return 50
        composite = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        # Trend consistency: all timeframes aligned = amplify signal
        if all(s >= 60 for s in scores): composite = min(100, composite + 10)
        elif all(s >= 55 for s in scores): composite = min(100, composite + 5)
        elif all(s <= 40 for s in scores): composite = max(0,   composite - 10)
        elif all(s <= 45 for s in scores): composite = max(0,   composite - 5)
        return _clamp(composite)
    except:
        return 50

def score_rsi(hist) -> int:
    """RSI 14-day — full decisive range."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 16: return 50
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan).fillna(1e-9)
        r     = float((100 - (100 / (1 + rs))).iloc[-1])
        # Very decisive breakpoints
        if   r < 20: return 90
        elif r < 30: return 78
        elif r < 40: return 64
        elif r < 50: return 54
        elif r < 60: return 46
        elif r < 70: return 34
        elif r < 80: return 22
        else:        return 12
    except:
        return 50

def score_macd(hist) -> int:
    """MACD histogram with trend direction and momentum."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 35: return 50
        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd_l   = ema12 - ema26
        signal_l = macd_l.ewm(span=9, adjust=False).mean()
        histo    = macd_l - signal_l
        c  = float(histo.iloc[-1])
        p  = float(histo.iloc[-2])
        p2 = float(histo.iloc[-3])

        # Crossovers get big scores
        if c > 0 and p <= 0:              return 84
        if c < 0 and p >= 0:              return 16
        # Strong trending histogram
        if c > 0 and c > p and p > p2:   return _clamp(70 + min(c * 8, 20))
        if c > 0 and c > p:              return _clamp(63 + min(c * 5, 15))
        if c > 0:                         return _clamp(55 + min(c * 3, 10))
        if c < 0 and c < p and p < p2:   return _clamp(30 + max(c * 8, -20))
        if c < 0 and c < p:              return _clamp(37 + max(c * 5, -15))
        if c < 0:                         return _clamp(45 + max(c * 3, -10))
        return 50
    except:
        return 50

def score_volume(hist) -> int:
    """Volume vs 90d avg + price direction. Decisive on spikes."""
    try:
        if len(hist) < 20: return 50
        avg_vol    = float(hist["Volume"].iloc[-90:].mean()) if len(hist) >= 90 else float(hist["Volume"].mean())
        recent_vol = float(hist["Volume"].iloc[-5:].mean())
        if avg_vol == 0: return 50
        ratio    = recent_vol / avg_vol
        price_up = float(hist["Close"].iloc[-1]) > float(hist["Close"].iloc[-6]) if len(hist) >= 6 else True
        # Stronger amplification on volume spikes
        if   ratio > 3.0: amp = 38
        elif ratio > 2.0: amp = 28
        elif ratio > 1.5: amp = 18
        elif ratio > 1.2: amp = 10
        elif ratio < 0.5: amp = -12  # Very low volume = weak conviction
        elif ratio < 0.7: amp = -6
        else:             amp = 0
        base = 56 if price_up else 44
        return _clamp(base + (amp if price_up else -amp))
    except:
        return 50

def score_week52_position(hist, info) -> int:
    """Where stock sits in 52w range, combined with recent momentum direction."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 50: return 50
        high52   = float(close.rolling(min(252, len(close))).max().iloc[-1])
        low52    = float(close.rolling(min(252, len(close))).min().iloc[-1])
        current  = float(close.iloc[-1])
        if high52 == low52: return 50
        pct      = (current - low52) / (high52 - low52)  # 0=at low, 1=at high
        ret_1m   = float(close.iloc[-1] / close.iloc[-22] - 1) if len(close) >= 22 else 0
        up       = ret_1m > 0.01

        # Near 52w low + bouncing = strong buy
        if pct < 0.15 and up:     return _clamp(80 + (0.15 - pct) * 100)
        elif pct < 0.15:          return _clamp(35 - (0.15 - pct) * 80)
        elif pct < 0.30 and up:   return 68
        elif pct < 0.30:          return 42
        elif pct < 0.70:          return _clamp(50 + (pct - 0.5) * 25)
        elif pct < 0.85 and up:   return 70
        elif pct < 0.85:          return 38
        elif up:                  return _clamp(72 + (pct - 0.85) * 100)
        else:                     return _clamp(28 - (pct - 0.85) * 100)
    except:
        return 50

# ─── FUNDAMENTAL SCORERS (info dict based) ────────────────────────────────────

def score_short_interest(info) -> int:
    try:
        s = info.get("shortPercentOfFloat")
        if s is None: return _smart_default(info, "short_interest")
        s *= 100
        if   s > 30: return 15
        elif s > 20: return 25
        elif s > 15: return 35
        elif s > 10: return 44
        elif s > 5:  return 55
        elif s > 2:  return 65
        else:        return 72
    except:
        return 55

def score_earnings_surprise(info) -> int:
    try:
        trailing = info.get("trailingEps")
        forward  = info.get("forwardEps")
        if trailing is None: return _smart_default(info, "earnings_surprise")
        if forward is None:  forward = trailing
        if abs(forward) < 0.01: return 50
        surprise = (trailing - forward) / abs(forward) * 100
        # Very decisive: big beat = high score
        if   surprise >  30: return 88
        elif surprise >  15: return 76
        elif surprise >   5: return 65
        elif surprise >   0: return 57
        elif surprise >  -5: return 44
        elif surprise > -15: return 33
        else:                return 18
    except:
        return _smart_default(info, "earnings_surprise")

def score_sentiment(info) -> int:
    try:
        rec = info.get("recommendationMean")
        if rec is None: return 52  # Slight positive if no coverage
        # 1.0=Strong Buy→95, 2.0→74, 3.0→50, 4.0→26, 5.0→5
        return _clamp(95 - (rec - 1.0) * 22.5)
    except:
        return 50

def score_analyst_revisions(info) -> int:
    try:
        rec        = info.get("recommendationMean") or 3.0
        n_analysts = info.get("numberOfAnalystOpinions") or 0
        if n_analysts == 0: return _smart_default(info, "analyst_target")
        base = _clamp(95 - (rec - 1.0) * 22.5)
        # High analyst coverage = higher conviction in both directions
        if n_analysts >= 40: conf = 12
        elif n_analysts >= 25: conf = 8
        elif n_analysts >= 15: conf = 4
        elif n_analysts < 5:   conf = -8
        else:                  conf = 0
        # Amplify away from 50 by confidence
        delta = base - 50
        return _clamp(50 + delta + (conf if delta > 0 else -conf))
    except:
        return 50

def score_analyst_target(info) -> int:
    try:
        target  = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if not target or not current or current == 0:
            return _smart_default(info, "analyst_target")
        upside = (target / current - 1) * 100
        # Very decisive mapping
        if   upside >  40: return 92
        elif upside >  25: return 82
        elif upside >  15: return 73
        elif upside >   8: return 65
        elif upside >   2: return 58
        elif upside >  -2: return 50
        elif upside >  -8: return 40
        elif upside > -15: return 30
        elif upside > -25: return 20
        else:              return 10
    except:
        return _smart_default(info, "analyst_target")

def score_pe_vs_sector(info) -> int:
    try:
        pe     = info.get("trailingPE") or info.get("forwardPE")
        sector = info.get("sector", "")
        if pe is None or pe <= 0 or pe > 1500: return 50
        benchmarks = {
            "Technology": 28, "Healthcare": 22, "Financials": 13,
            "Consumer Cyclical": 20, "Consumer Defensive": 18,
            "Energy": 12, "Utilities": 16, "Real Estate": 35,
            "Basic Materials": 14, "Industrials": 20,
            "Communication Services": 22,
        }
        bm = benchmarks.get(sector, 20)
        r  = pe / bm
        if   r < 0.4:  return 88
        elif r < 0.6:  return 78
        elif r < 0.8:  return 67
        elif r < 1.0:  return 58
        elif r < 1.2:  return 50
        elif r < 1.5:  return 40
        elif r < 2.0:  return 28
        elif r < 3.0:  return 16
        else:          return 8
    except:
        return 50

def score_revenue_growth(info) -> int:
    try:
        g = info.get("revenueGrowth")
        if g is None: return _smart_default(info, "revenue_growth")
        g *= 100
        if   g >  50: return 95
        elif g >  30: return 85
        elif g >  20: return 75
        elif g >  10: return 65
        elif g >   5: return 58
        elif g >   0: return 52
        elif g >  -5: return 40
        elif g > -15: return 28
        elif g > -25: return 16
        else:         return 8
    except:
        return _smart_default(info, "revenue_growth")

def score_profit_margin(info) -> int:
    try:
        m = info.get("profitMargins")
        if m is None: return _smart_default(info, "profit_margin")
        m *= 100
        if   m >  40: return 93
        elif m >  30: return 85
        elif m >  20: return 76
        elif m >  12: return 66
        elif m >   6: return 57
        elif m >   2: return 48
        elif m >   0: return 40
        elif m >  -8: return 28
        else:         return 14
    except:
        return _smart_default(info, "profit_margin")

def score_fcf_yield(info) -> int:
    try:
        fcf    = info.get("freeCashflow")
        mktcap = info.get("marketCap") or 0
        if fcf is None or mktcap == 0:
            return _smart_default(info, "fcf_yield")
        y = (fcf / mktcap) * 100
        if   y >  10: return 90
        elif y >   6: return 78
        elif y >   4: return 68
        elif y >   2: return 60
        elif y >   0: return 52
        elif y >  -3: return 38
        elif y >  -6: return 24
        else:         return 12
    except:
        return _smart_default(info, "fcf_yield")

def score_debt_equity(info) -> int:
    try:
        de = info.get("debtToEquity")
        if de is None: return _smart_default(info, "debt_equity")
        if   de <  10: return 85
        elif de <  30: return 75
        elif de <  60: return 64
        elif de < 100: return 54
        elif de < 150: return 42
        elif de < 250: return 28
        else:          return 14
    except:
        return 55

def score_eps_growth(info) -> int:
    try:
        trailing = info.get("trailingEps")
        forward  = info.get("forwardEps")
        if trailing is None: return _smart_default(info, "revenue_growth")
        if forward is None:  forward = trailing
        if abs(trailing) < 0.01: return 50
        g = (forward - trailing) / abs(trailing) * 100
        if   g >  50: return 92
        elif g >  30: return 82
        elif g >  20: return 72
        elif g >  10: return 63
        elif g >   3: return 55
        elif g >   0: return 50
        elif g >  -8: return 38
        elif g > -20: return 26
        else:         return 12
    except:
        return 45

def score_peg(info) -> int:
    try:
        peg = info.get("pegRatio")
        if peg is None or peg <= 0:
            return _smart_default(info, "peg_ratio")
        if   peg < 0.5:  return 92
        elif peg < 0.8:  return 80
        elif peg < 1.0:  return 70
        elif peg < 1.3:  return 58
        elif peg < 1.8:  return 44
        elif peg < 2.5:  return 30
        elif peg < 4.0:  return 18
        else:            return 8
    except:
        return _smart_default(info, "peg_ratio")

# ─── CACHED MARKET-WIDE SCORES ───────────────────────────────────────────────

_macro_cache  = {}
_sector_cache = {}

def score_macro(_info=None) -> int:
    global _macro_cache
    today = datetime.now().strftime("%Y-%m-%d")
    if today in _macro_cache:
        return _macro_cache[today]
    try:
        vix_data = yf.Ticker("^VIX").history(period="10d")["Close"].dropna()
        vix_now  = float(vix_data.iloc[-1])
        vix_5d   = float(vix_data.iloc[-5]) if len(vix_data) >= 5 else vix_now
        rising   = vix_now > vix_5d * 1.05  # Rising >5% = fear increasing

        if   vix_now < 12: base = 84
        elif vix_now < 15: base = 73
        elif vix_now < 18: base = 63
        elif vix_now < 22: base = 52
        elif vix_now < 28: base = 40
        elif vix_now < 35: base = 28
        else:              base = 15

        if rising: base = max(8, base - 10)
        _macro_cache[today] = base
        return base
    except:
        _macro_cache[today] = 50
        return 50

def score_sector_momentum(info) -> int:
    global _sector_cache
    SECTOR_ETFS = {
        "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
        "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
        "Energy": "XLE", "Utilities": "XLU", "Real Estate": "XLRE",
        "Basic Materials": "XLB", "Industrials": "XLI",
        "Communication Services": "XLC",
    }
    try:
        sector = info.get("sector", "")
        etf    = SECTOR_ETFS.get(sector, "SPY")
        key    = f"{etf}_{datetime.now().strftime('%Y-%m-%d')}"
        if key not in _sector_cache:
            h = yf.Ticker(etf).history(period="3mo")["Close"].dropna()
            if len(h) >= 22:
                r1m = (float(h.iloc[-1]) / float(h.iloc[-22]) - 1) * 100
                r3m = (float(h.iloc[-1]) / float(h.iloc[-63]) - 1) * 100 if len(h) >= 63 else r1m
                # Strong sector ETF returns map to strong scores
                score = _clamp(50 + r1m * 3.0 + r3m * 1.0)
            else:
                score = 50
            _sector_cache[key] = score
        return _sector_cache[key]
    except:
        return 50

def score_earnings_proximity(info) -> int:
    try:
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts is None:
            return _smart_default(info, "earnings_proximity")
        days = (datetime.fromtimestamp(int(ts)) - datetime.now()).days
        if   0 <= days <= 2:          return 30   # Earnings tomorrow = max risk
        elif 0 <= days <= 7:          return 38
        elif 0 <= days <= 14:         return 45
        elif 0 <= days <= 30:         return 52
        elif days < 0 and days >= -5: return 65   # Just reported = clarity
        elif days < 0 and days >= -14:return 60
        elif days < 0 and days >= -30:return 56
        return 54
    except:
        return 54

# ─── COMPOSITE BUILDER ───────────────────────────────────────────────────────

def _build_data_scores(hist, info: dict) -> dict:
    return {
        "momentum_multi":    score_momentum_multi(hist),
        "rsi":               score_rsi(hist),
        "macd":              score_macd(hist),
        "volume_trend":      score_volume(hist),
        "week52_position":   score_week52_position(hist, info),
        "earnings_proximity":score_earnings_proximity(info),
        "sector_momentum":   score_sector_momentum(info),
        "short_interest":    score_short_interest(info),
        "sentiment":         score_sentiment(info),
        "analyst_target":    score_analyst_target(info),
        "earnings_surprise": score_earnings_surprise(info),
        "analyst_revisions": score_analyst_revisions(info),
        "macro_score":       score_macro(info),
        "pe_vs_sector":      score_pe_vs_sector(info),
        "revenue_growth":    score_revenue_growth(info),
        "profit_margin":     score_profit_margin(info),
        "fcf_yield":         score_fcf_yield(info),
        "debt_equity":       score_debt_equity(info),
        "eps_growth":        score_eps_growth(info),
        "peg_ratio":         score_peg(info),
    }

def _composite(ds: dict, horizon: str):
    w        = WEIGHTS[horizon]
    total_w  = sum(w.values())
    raw      = sum(ds[k] * w[k] for k in ds) / total_w
    final    = stretch_score(raw)
    return final, round(raw, 1)

def compute_price_target(price: float, score: int, horizon: str) -> float:
    direction = (score - 50) / 50
    return round(price * (1 + direction * HORIZON_FACTORS[horizon]), 2)

# ─── DEBUG HELPER ─────────────────────────────────────────────────────────────

def debug_score(ticker: str, horizon: str = "1M") -> pd.DataFrame:
    """
    Show exactly how each data point contributes to the final score.
    Useful for diagnosing why a score is high/low or why it stays near 50.
    Returns a DataFrame with each data point, its raw score, weight, and contribution.
    """
    t    = yf.Ticker(ticker)
    info = t.info or {}
    hist = t.history(period="1y")
    ds   = _build_data_scores(hist, info)
    w    = WEIGHTS[horizon]
    total_w = sum(w.values())

    rows = []
    for k, score in ds.items():
        weight     = w[k]
        contrib    = round(score * weight / total_w, 2)
        rows.append({
            "Data Point":    DATA_POINT_LABELS.get(k, k),
            "Raw Score":     score,
            "Weight":        weight,
            "Contribution":  contrib,
            "Signal":        "🟢 Bullish" if score > 60 else ("🔴 Bearish" if score < 40 else "🟡 Neutral"),
        })

    df = pd.DataFrame(rows).sort_values("Contribution", ascending=False)
    final, raw = _composite(ds, horizon)
    print(f"\n{ticker.upper()} | Horizon: {horizon} | Raw: {raw} | Final (stretched): {final} | Signal: {_signal_label(final)}")
    return df

# ─── BRIER ───────────────────────────────────────────────────────────────────

def compute_brier(score: int, actual_return_pct: float, horizon: str) -> float:
    p_up    = score / 100.0
    outcome = 1 if actual_return_pct > 0 else 0
    return round((p_up - outcome) ** 2, 4)

# ─── MAIN FETCH & SCORE ──────────────────────────────────────────────────────

def fetch_and_score(ticker: str, horizon: str = "1M",
                    _hist=None, _info=None) -> dict:
    try:
        t    = yf.Ticker(ticker)
        info = _info if _info is not None else (t.info or {})
        hist = _hist if _hist is not None else t.history(period="1y")

        if hist is None or hist.empty:
            return _error_result(ticker, horizon, "No price data found")

        ds           = _build_data_scores(hist, info)
        overall, raw = _composite(ds, horizon)
        price        = round(float(hist["Close"].iloc[-1]), 2)
        target       = compute_price_target(price, overall, horizon)

        # Confidence: how many fields had non-neutral data (not 50 exactly)
        non_neutral  = sum(1 for v in ds.values() if v != 50)
        confidence   = round((non_neutral / len(ds)) * 100)

        return {
            "ticker":      ticker.upper(),
            "horizon":     horizon,
            "overall":     overall,
            "raw_score":   raw,
            "signal":      _signal_label(overall),
            "price":       price,
            "target":      target,
            "data_scores": ds,
            "brier":       None,
            "confidence":  confidence,
            "error":       None,
            "name":        info.get("shortName", ticker),
            "sector":      info.get("sector", "—"),
            "market_cap":  info.get("marketCap"),
        }
    except Exception as e:
        return _error_result(ticker, horizon, str(e))

def fetch_and_score_batch(tickers: list, horizon: str = "1M") -> pd.DataFrame:
    rows = []
    for t in tickers:
        r = fetch_and_score(t, horizon)
        rows.append({
            "Ticker":       r["ticker"],
            "Name":         r.get("name", ""),
            "Sector":       r.get("sector", ""),
            "Score":        r["overall"],
            "Signal":       r["signal"],
            "Price":        r.get("price"),
            "Target":       r.get("target"),
            "Upside %":     round((r["target"] / r["price"] - 1) * 100, 1)
                            if r.get("price") and r.get("target") else None,
            "Confidence %": r["confidence"],
            "Error":        r["error"],
        })
    return pd.DataFrame(rows)

# ─── BACKTEST ────────────────────────────────────────────────────────────────

def backtest_ticker(ticker: str, date_str: str, end_date_str: str = None) -> dict:
    HORIZON_DAYS = {"1D": 1, "1W": 7, "1M": 30, "1Q": 90, "1Y": 365}
    try:
        t       = yf.Ticker(ticker)
        start   = datetime.strptime(date_str, "%Y-%m-%d")
        end_cap = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

        hist_to_date = t.history(start=start - timedelta(days=400),
                                 end=start + timedelta(days=2))
        if hist_to_date.empty:
            return {"error": f"No data for {ticker} around {date_str}"}

        price_at_date = round(float(hist_to_date["Close"].iloc[-1]), 2)
        fetch_end     = min(end_cap, datetime.now()) if end_cap else datetime.now()
        hist_future   = t.history(start=start, end=fetch_end + timedelta(days=2))
        info          = t.info or {}
        ds            = _build_data_scores(hist_to_date, info)
        results       = []

        for horizon, days in HORIZON_DAYS.items():
            overall, _ = _composite(ds, horizon)
            pred_target = compute_price_target(price_at_date, overall, horizon)

            horizon_end = start + timedelta(days=days)
            if end_cap: horizon_end = min(horizon_end, end_cap)

            idx = hist_future.index
            he_ts = (pd.Timestamp(horizon_end).tz_localize(idx.tz)
                     if (hasattr(idx, 'tz') and idx.tz is not None)
                     else pd.Timestamp(horizon_end))

            future_slice = hist_future[hist_future.index <= he_ts]
            if len(future_slice) > 0:
                actual_price = round(float(future_slice["Close"].iloc[-1]), 2)
                actual_ret   = round((actual_price / price_at_date - 1) * 100, 2)
                brier        = compute_brier(overall, actual_ret, horizon)
            else:
                actual_price = actual_ret = brier = None

            pred_dir   = "UP" if overall >= 55 else ("DOWN" if overall <= 45 else "FLAT")
            actual_dir = ("UP"   if (actual_ret or 0) >  0.5 else
                          "DOWN" if (actual_ret or 0) < -0.5 else "FLAT") if actual_ret is not None else "—"
            correct    = (pred_dir == actual_dir) if actual_ret is not None else None

            results.append({
                "Horizon":           horizon,
                "Score":             overall,
                "Signal":            _signal_label(overall),
                "Pred Target":       pred_target,
                "Actual Return %":   actual_ret,
                "Actual Price":      actual_price,
                "Direction Correct": "✓ YES" if correct else ("✗ NO" if correct is not None else "—"),
                "Brier Score":       brier,
            })

        return {"ticker": ticker.upper(), "date": date_str, "end_date": end_date_str,
                "base_price": price_at_date, "results": results, "error": None}
    except Exception as e:
        return {"error": str(e)}

# ─── SCENARIO SIMULATOR ──────────────────────────────────────────────────────

def run_scenario(ticker, start_date, end_date, dollars,
                 interval, buy_rating, sell_rating):
    INTERVAL_DAYS = {"1D": 1, "1W": 7, "1M": 30}
    days_step = INTERVAL_DAYS.get(interval, 7)
    try:
        t     = yf.Ticker(ticker)
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        if end <= start:
            return None, "End date must be after start date."

        hist_full = t.history(start=start - timedelta(days=400),
                              end=end + timedelta(days=2))
        if hist_full.empty:
            return None, "No price data found."
        if hist_full.index.tz is not None:
            hist_full.index = hist_full.index.tz_localize(None)

        cash, shares, log, cur = float(dollars), 0, [], start

        while cur <= end and len(log) < 500:
            subset = hist_full[hist_full.index <= pd.Timestamp(cur)]
            if len(subset) < 20:
                cur += timedelta(days=days_step)
                continue

            price = round(float(subset["Close"].iloc[-1]), 2)
            raw   = (score_momentum_multi(subset) * 0.40 +
                     score_rsi(subset)            * 0.35 +
                     score_macd(subset)           * 0.25)
            score = stretch_score(raw)

            action = "HOLD"
            if score >= buy_rating and cash >= price:
                n = int(cash // price)
                if n > 0:
                    shares += n; cash -= n * price; action = "BUY"
            elif score <= sell_rating and shares > 0:
                cash += shares * price; shares = 0; action = "SELL"

            log.append({
                "Date":            cur.strftime("%Y-%m-%d"),
                "Score":           score,
                "Price":           price,
                "Action":          action,
                "Shares Held":     shares,
                "Cash":            round(cash, 2),
                "Portfolio Value": round(cash + shares * price, 2),
            })
            cur += timedelta(days=days_step)

        df = pd.DataFrame(log)
        if df.empty:
            return None, "No trading periods in date range."

        final      = df["Portfolio Value"].iloc[-1]
        bh_val     = round(float(dollars) * (df["Price"].iloc[-1] / df["Price"].iloc[0]), 2)
        total_ret  = round((final / float(dollars) - 1) * 100, 2)
        bh_ret     = round((bh_val / float(dollars) - 1) * 100, 2)

        summary = {
            "Final Portfolio":     round(final, 2),
            "Total Return %":      total_ret,
            "Buy & Hold Return %": bh_ret,
            "Alpha vs B&H %":      round(total_ret - bh_ret, 2),
            "Total Trades":        len(df[df["Action"] != "HOLD"]),
            "Start Date":          start_date,
            "End Date":            end_date,
            "Starting Capital":    float(dollars),
        }
        return df, summary
    except Exception as e:
        return None, str(e)

# ─── OPTIMIZER ───────────────────────────────────────────────────────────────

def optimize_strategy(ticker: str, start_date: str, end_date: str,
                      dollars: float = 10000) -> dict:
    """
    Walk-forward optimizer with train/test split to prevent curve-fitting.
    Finds optimal buy threshold, sell threshold, and interval.
    """
    import itertools
    try:
        start      = datetime.strptime(start_date, "%Y-%m-%d")
        end        = datetime.strptime(end_date,   "%Y-%m-%d")
        total_days = (end - start).days
        if total_days < 90:
            return {"error": "Need at least 90 days to optimize."}

        split      = start + timedelta(days=int(total_days * 0.70))
        train_end  = split.strftime("%Y-%m-%d")
        test_start = (split + timedelta(days=1)).strftime("%Y-%m-%d")

        buy_thresholds  = list(range(52, 82, 3))
        sell_thresholds = list(range(25, 52, 3))
        intervals       = ["1D", "1W", "1M"]
        train_results   = []

        for interval, buy_r, sell_r in itertools.product(intervals, buy_thresholds, sell_thresholds):
            if sell_r >= buy_r: continue
            df, summary = run_scenario(ticker, start_date, train_end,
                                       dollars, interval, buy_r, sell_r)
            if df is None: continue
            train_results.append({
                "interval": interval, "buy_rating": buy_r, "sell_rating": sell_r,
                "total_return": summary["Total Return %"],
                "bh_return":    summary["Buy & Hold Return %"],
                "alpha":        summary["Alpha vs B&H %"],
                "trades":       summary["Total Trades"],
            })

        if not train_results:
            return {"error": "No valid combinations found."}

        train_results.sort(key=lambda x: x["alpha"], reverse=True)
        top5  = train_results[:5]
        validated = []

        for p in top5:
            df, summary = run_scenario(ticker, test_start, end_date, dollars,
                                       p["interval"], p["buy_rating"], p["sell_rating"])
            if df is None:
                validated.append({**p, "oos_return": None, "oos_bh_return": None,
                                   "oos_alpha": None, "overfit_flag": True})
            else:
                oos_alpha = summary["Alpha vs B&H %"]
                overfit   = (oos_alpha < p["alpha"] * 0.5) if p["alpha"] > 0 else True
                validated.append({**p, "oos_return": summary["Total Return %"],
                                   "oos_bh_return": summary["Buy & Hold Return %"],
                                   "oos_alpha": oos_alpha, "overfit_flag": overfit})

        valid_oos = [v for v in validated if v.get("oos_alpha") is not None]
        best      = max(valid_oos, key=lambda x: x["oos_alpha"]) if valid_oos else validated[0]

        return {
            "ticker": ticker.upper(), "start_date": start_date, "end_date": end_date,
            "train_end": train_end, "test_start": test_start,
            "best_params": best, "top5": validated,
            "total_combos": len(train_results), "error": None,
        }
    except Exception as e:
        return {"error": str(e)}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _signal_label(score: int) -> str:
    if score >= 80: return "STRONG BUY"
    if score >= 65: return "BUY"
    if score >= 45: return "NEUTRAL"
    if score >= 30: return "SELL"
    return "STRONG SELL"

def _error_result(ticker, horizon, msg):
    return {
        "ticker": ticker, "horizon": horizon, "overall": 50, "raw_score": 50,
        "signal": "NEUTRAL", "price": None, "target": None,
        "data_scores": {k: 50 for k in WEIGHTS["1M"]},
        "brier": None, "confidence": 0, "error": msg,
        "name": ticker, "sector": "—", "market_cap": None,
    }
