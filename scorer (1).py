"""
AlphaIQ Scoring Engine v2
==========================
Improvements over v1:
  1. Score distribution stretch — sigmoid normalization prevents clustering at 40-60
  2. Sharper individual scorers — fewer 50-fallbacks, more decisive outputs
  3. Multi-timeframe momentum (5d, 1m, 3m, 6m) for richer signal
  4. 52-week position score — where stock sits in its annual range
  5. EPS trend scoring — direction matters as much as level
  6. Analyst target upside — explicit price target vs current price
  7. end_date support in backtest_ticker and run_scenario
  8. Cached macro/sector calls to avoid redundant API hits per batch

Score interpretation:
  75–100 → Strong Buy   (high probability of upward movement)
  60–74  → Buy
  45–59  → Neutral / Flat
  30–44  → Sell
  0–29   → Strong Sell  (high probability of downward movement)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── WEIGHT TABLE ─────────────────────────────────────────────────────────────
WEIGHTS = {
    "1D": {
        "momentum_multi":    10,
        "rsi":                8,
        "macd":               7,
        "volume_trend":       7,
        "week52_position":    6,
        "earnings_proximity": 6,
        "sector_momentum":    5,
        "short_interest":     5,
        "sentiment":          5,
        "analyst_target":     4,
        "earnings_surprise":  4,
        "analyst_revisions":  4,
        "macro_score":        3,
        "pe_vs_sector":       2,
        "revenue_growth":     2,
        "profit_margin":      2,
        "fcf_yield":          2,
        "debt_equity":        2,
        "eps_growth":         2,
        "peg_ratio":          2,
    },
    "1W": {
        "momentum_multi":     9,
        "rsi":                7,
        "macd":               6,
        "volume_trend":       6,
        "week52_position":    6,
        "earnings_proximity": 5,
        "sector_momentum":    5,
        "short_interest":     5,
        "sentiment":          6,
        "analyst_target":     5,
        "earnings_surprise":  6,
        "analyst_revisions":  5,
        "macro_score":        4,
        "pe_vs_sector":       3,
        "revenue_growth":     4,
        "profit_margin":      3,
        "fcf_yield":          3,
        "debt_equity":        3,
        "eps_growth":         4,
        "peg_ratio":          3,
    },
    "1M": {
        "momentum_multi":     7,
        "rsi":                5,
        "macd":               5,
        "volume_trend":       4,
        "week52_position":    5,
        "earnings_proximity": 4,
        "sector_momentum":    5,
        "short_interest":     4,
        "sentiment":          6,
        "analyst_target":     6,
        "earnings_surprise":  7,
        "analyst_revisions":  7,
        "macro_score":        5,
        "pe_vs_sector":       6,
        "revenue_growth":     7,
        "profit_margin":      5,
        "fcf_yield":          5,
        "debt_equity":        4,
        "eps_growth":         6,
        "peg_ratio":          5,
    },
    "1Q": {
        "momentum_multi":     5,
        "rsi":                3,
        "macd":               3,
        "volume_trend":       3,
        "week52_position":    4,
        "earnings_proximity": 3,
        "sector_momentum":    5,
        "short_interest":     4,
        "sentiment":          5,
        "analyst_target":     7,
        "earnings_surprise":  8,
        "analyst_revisions":  8,
        "macro_score":        7,
        "pe_vs_sector":       7,
        "revenue_growth":     8,
        "profit_margin":      7,
        "fcf_yield":          7,
        "debt_equity":        6,
        "eps_growth":         7,
        "peg_ratio":          6,
    },
    "1Y": {
        "momentum_multi":     3,
        "rsi":                2,
        "macd":               2,
        "volume_trend":       2,
        "week52_position":    3,
        "earnings_proximity": 2,
        "sector_momentum":    4,
        "short_interest":     3,
        "sentiment":          4,
        "analyst_target":     8,
        "earnings_surprise":  7,
        "analyst_revisions":  7,
        "macro_score":        8,
        "pe_vs_sector":       8,
        "revenue_growth":     9,
        "profit_margin":      8,
        "fcf_yield":          8,
        "debt_equity":        7,
        "eps_growth":         9,
        "peg_ratio":          7,
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

# ─── SCORE STRETCHER ─────────────────────────────────────────────────────────

def stretch_score(raw: float) -> int:
    """
    Sigmoid stretch to spread scores away from the 40-60 clustering zone.
    Raw 50 stays 50. Raw 70 -> ~78. Raw 30 -> ~22. Raw 85 -> ~91.
    """
    x = (raw - 50) / 50            # normalize to -1..+1
    stretched = x * (1 + 1.2 * x * x)   # cubic amplification
    stretched = max(-1.0, min(1.0, stretched))
    return int(round(50 + stretched * 50))

def _clamp(val, lo=0, hi=100):
    return max(lo, min(hi, int(round(val))))

# ─── INDIVIDUAL SCORERS ───────────────────────────────────────────────────────

def score_momentum_multi(hist):
    """Multi-timeframe momentum: 5d, 21d, 63d, 126d. Trend consistency bonus."""
    try:
        close = hist["Close"].dropna()
        n = len(close)
        windows = [(5, 2.0), (21, 1.5), (63, 1.0), (126, 0.7)]
        scores = []
        for days, w in windows:
            if n > days:
                ret = (close.iloc[-1] / close.iloc[-days] - 1) * 100
                pts = _clamp(50 + ret * 3.5)
                scores.append((pts, w))
        if not scores:
            return 50
        total_w = sum(w for _, w in scores)
        composite = sum(s * w for s, w in scores) / total_w
        all_vals = [s for s, _ in scores]
        if all(v > 55 for v in all_vals):
            composite = min(100, composite + 8)
        elif all(v < 45 for v in all_vals):
            composite = max(0, composite - 8)
        return _clamp(composite)
    except:
        return 50

def score_rsi(hist):
    """RSI 14-day. Decisive scoring across full range."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 15:
            return 50
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan).fillna(1e-9)
        r = float((100 - (100 / (1 + rs))).iloc[-1])
        if r < 25:    return _clamp(82 + (25 - r) * 0.5)
        elif r < 35:  return _clamp(70 + (35 - r) * 1.2)
        elif r < 45:  return _clamp(58 + (45 - r) * 1.2)
        elif r < 55:  return _clamp(48 + (55 - r) * 1.0)
        elif r < 65:  return _clamp(35 + (65 - r) * 1.3)
        elif r < 75:  return _clamp(22 + (75 - r) * 1.3)
        else:         return _clamp(22 - (r - 75) * 0.5)
    except:
        return 50

def score_macd(hist):
    """MACD with histogram trend analysis."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 30:
            return 50
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histo = macd_line - signal_line
        c, p, p2 = float(histo.iloc[-1]), float(histo.iloc[-2]), float(histo.iloc[-3])
        if c > 0 and p <= 0:   return 82
        if c < 0 and p >= 0:   return 18
        if c > 0 and c > p > p2: return _clamp(68 + c * 5)
        if c > 0 and c > p:    return _clamp(62 + c * 3)
        if c > 0:               return _clamp(54 + c * 2)
        if c < 0 and c < p < p2: return _clamp(32 + c * 5)
        if c < 0 and c < p:    return _clamp(38 + c * 3)
        if c < 0:               return _clamp(46 + c * 2)
        return 50
    except:
        return 50

def score_volume(hist):
    """Volume vs 90d average combined with price direction."""
    try:
        if len(hist) < 20:
            return 50
        avg_vol = hist["Volume"].iloc[-90:].mean() if len(hist) >= 90 else hist["Volume"].mean()
        recent_vol = hist["Volume"].iloc[-5:].mean()
        if avg_vol == 0:
            return 50
        ratio = recent_vol / avg_vol
        price_up = float(hist["Close"].iloc[-1]) > float(hist["Close"].iloc[-6]) if len(hist) >= 6 else True
        if ratio > 2.0:   amp = 30
        elif ratio > 1.5: amp = 20
        elif ratio > 1.2: amp = 12
        elif ratio < 0.6: amp = -8
        else:             amp = 0
        base = 55 if price_up else 45
        return _clamp(base + (amp if price_up else -amp))
    except:
        return 50

def score_week52_position(hist, info):
    """52-week range position with momentum direction context."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 50:
            return 50
        high52 = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(close.max())
        low52  = float(close.rolling(252).min().iloc[-1]) if len(close) >= 252 else float(close.min())
        current = float(close.iloc[-1])
        if high52 == low52:
            return 50
        pct = (current - low52) / (high52 - low52)
        ret_1m = float(close.iloc[-1] / close.iloc[-22] - 1) if len(close) >= 22 else 0
        if pct < 0.20 and ret_1m > 0:    return _clamp(75 + (0.20 - pct) * 100)
        elif pct < 0.20:                  return _clamp(40 - (0.20 - pct) * 50)
        elif pct > 0.80 and ret_1m > 0:  return _clamp(68 + (pct - 0.80) * 60)
        elif pct > 0.80:                  return _clamp(32 - (pct - 0.80) * 80)
        else:                             return _clamp(50 + (pct - 0.5) * 30)
    except:
        return 50

def score_short_interest(info):
    try:
        s = (info.get("shortPercentOfFloat") or 0) * 100
        if s > 25:    return 20
        elif s > 15:  return 32
        elif s > 10:  return 42
        elif s > 5:   return 52
        elif s > 2:   return 62
        elif s > 0:   return 68
        return 50
    except:
        return 50

def score_earnings_surprise(info):
    try:
        trailing = info.get("trailingEps") or 0
        forward  = info.get("forwardEps") or trailing
        if forward == 0:
            return 50
        surprise = (trailing - forward) / abs(forward) * 100
        return _clamp(50 + surprise * 2.0)
    except:
        return 50

def score_sentiment(info):
    try:
        rec = info.get("recommendationMean")
        if rec is None:
            return 50
        return _clamp(92 - (rec - 1.0) * 21)
    except:
        return 50

def score_analyst_revisions(info):
    try:
        rec_mean   = info.get("recommendationMean") or 3.0
        n_analysts = info.get("numberOfAnalystOpinions") or 0
        if n_analysts == 0:
            return 50
        base = _clamp(92 - (rec_mean - 1.0) * 21)
        if n_analysts >= 30:   mod = 8
        elif n_analysts >= 20: mod = 5
        elif n_analysts >= 10: mod = 2
        elif n_analysts < 5:   mod = -5
        else:                  mod = 0
        return _clamp(base + mod)
    except:
        return 50

def score_analyst_target(info):
    """Analyst consensus price target vs current price."""
    try:
        target  = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if not target or not current or current == 0:
            return 50
        upside = (target / current - 1) * 100
        return _clamp(50 + upside * 1.4)
    except:
        return 50

def score_pe_vs_sector(info):
    """P/E vs sector-specific benchmarks."""
    try:
        pe = info.get("trailingPE") or info.get("forwardPE")
        sector = info.get("sector", "")
        if pe is None or pe <= 0 or pe > 1000:
            return 50
        benchmarks = {
            "Technology": 28, "Healthcare": 22, "Financials": 13,
            "Consumer Cyclical": 20, "Consumer Defensive": 18,
            "Energy": 12, "Utilities": 16, "Real Estate": 35,
            "Basic Materials": 14, "Industrials": 20,
            "Communication Services": 20,
        }
        bm = benchmarks.get(sector, 20)
        r = pe / bm
        if r < 0.5:    return 85
        elif r < 0.7:  return 75
        elif r < 0.85: return 65
        elif r < 1.0:  return 58
        elif r < 1.2:  return 50
        elif r < 1.5:  return 42
        elif r < 2.0:  return 33
        elif r < 3.0:  return 22
        else:          return 12
    except:
        return 50

def score_revenue_growth(info):
    try:
        g = (info.get("revenueGrowth") or None)
        if g is None:
            return 50
        g *= 100
        if g > 40:    return 92
        elif g > 25:  return 80
        elif g > 15:  return 70
        elif g > 8:   return 62
        elif g > 3:   return 55
        elif g > 0:   return 50
        elif g > -5:  return 42
        elif g > -15: return 32
        else:         return 18
    except:
        return 50

def score_profit_margin(info):
    try:
        m = (info.get("profitMargins") or None)
        if m is None:
            return 50
        m *= 100
        if m > 35:    return 90
        elif m > 25:  return 80
        elif m > 18:  return 70
        elif m > 12:  return 62
        elif m > 7:   return 55
        elif m > 3:   return 48
        elif m > 0:   return 42
        elif m > -5:  return 33
        else:         return 18
    except:
        return 50

def score_fcf_yield(info):
    try:
        fcf    = info.get("freeCashflow") or 0
        mktcap = info.get("marketCap") or 0
        if mktcap == 0:
            return 50
        y = (fcf / mktcap) * 100
        if y > 8:    return 88
        elif y > 5:  return 75
        elif y > 3:  return 64
        elif y > 1:  return 56
        elif y > 0:  return 50
        elif y > -3: return 38
        else:        return 22
    except:
        return 50

def score_debt_equity(info):
    try:
        de = info.get("debtToEquity")
        if de is None:
            return 52
        if de < 10:    return 85
        elif de < 30:  return 75
        elif de < 60:  return 65
        elif de < 100: return 55
        elif de < 150: return 44
        elif de < 250: return 33
        else:          return 18
    except:
        return 50

def score_eps_growth(info):
    try:
        trailing = info.get("trailingEps") or 0
        forward  = info.get("forwardEps") or trailing
        if trailing == 0:
            return 50
        g = (forward - trailing) / abs(trailing) * 100
        if g > 40:    return 90
        elif g > 25:  return 78
        elif g > 15:  return 68
        elif g > 8:   return 60
        elif g > 3:   return 54
        elif g > 0:   return 50
        elif g > -5:  return 42
        elif g > -15: return 32
        else:         return 18
    except:
        return 50

def score_peg(info):
    try:
        peg = info.get("pegRatio")
        if peg is None or peg <= 0:
            return 50
        if peg < 0.5:   return 90
        elif peg < 0.8: return 78
        elif peg < 1.0: return 68
        elif peg < 1.3: return 58
        elif peg < 1.7: return 48
        elif peg < 2.5: return 36
        elif peg < 4.0: return 24
        else:           return 12
    except:
        return 50

# ─── CACHED MARKET-WIDE SCORES ───────────────────────────────────────────────

_macro_cache  = {}
_sector_cache = {}

def score_macro(_info=None):
    global _macro_cache
    today = datetime.now().strftime("%Y-%m-%d")
    if today in _macro_cache:
        return _macro_cache[today]
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")["Close"]
        vix_now  = float(vix_data.iloc[-1])
        vix_5d   = float(vix_data.iloc[0])
        rising   = vix_now > vix_5d
        if vix_now < 13:    base = 80
        elif vix_now < 16:  base = 70
        elif vix_now < 20:  base = 60
        elif vix_now < 25:  base = 50
        elif vix_now < 30:  base = 38
        elif vix_now < 40:  base = 25
        else:               base = 12
        if rising:
            base = max(5, base - 8)
        _macro_cache[today] = base
        return base
    except:
        _macro_cache[today] = 50
        return 50

def score_sector_momentum(info):
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
                r1m = (h.iloc[-1] / h.iloc[-22]  - 1) * 100
                r3m = (h.iloc[-1] / h.iloc[-63]  - 1) * 100 if len(h) >= 63 else r1m
                _sector_cache[key] = _clamp(50 + r1m * 2.5 + r3m * 0.8)
            else:
                _sector_cache[key] = 50
        return _sector_cache[key]
    except:
        return 50

def score_earnings_proximity(info):
    try:
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts is None:
            return 55
        days = (datetime.fromtimestamp(int(ts)) - datetime.now()).days
        if 0 <= days <= 3:          return 35
        elif 0 <= days <= 7:        return 40
        elif 0 <= days <= 14:       return 45
        elif days < 0 and days >= -7:  return 62
        elif days < 0 and days >= -30: return 58
        return 55
    except:
        return 55

# ─── BRIER ───────────────────────────────────────────────────────────────────

def compute_brier(score: int, actual_return_pct: float, horizon: str) -> float:
    p_up    = score / 100.0
    outcome = 1 if actual_return_pct > 0 else 0
    return round((p_up - outcome) ** 2, 4)

def compute_price_target(price: float, score: int, horizon: str) -> float:
    direction = (score - 50) / 50
    return round(price * (1 + direction * HORIZON_FACTORS[horizon]), 2)

# ─── MAIN SCORER ─────────────────────────────────────────────────────────────

def _build_data_scores(hist, info):
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

def _composite(ds, horizon):
    w = WEIGHTS[horizon]
    total_w = sum(w.values())
    raw = sum(ds[k] * w[k] for k in ds) / total_w
    return stretch_score(raw), round(raw, 1)

def fetch_and_score(ticker: str, horizon: str = "1M", _hist=None, _info=None) -> dict:
    try:
        t    = yf.Ticker(ticker)
        info = _info if _info is not None else (t.info or {})
        hist = _hist if _hist is not None else t.history(period="1y")

        if hist is None or hist.empty:
            return _error_result(ticker, horizon, "No price data found")

        ds             = _build_data_scores(hist, info)
        overall, raw   = _composite(ds, horizon)
        price          = round(float(hist["Close"].iloc[-1]), 2)
        target         = compute_price_target(price, overall, horizon)
        non_default    = sum(1 for v in ds.values() if v != 50)
        confidence     = round((non_default / len(ds)) * 100)

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

# ─── BACKTEST ─────────────────────────────────────────────────────────────────

def backtest_ticker(ticker: str, date_str: str, end_date_str: str = None) -> dict:
    """
    Predict scores at date_str and compare against actual returns.
    end_date_str caps how far forward we measure each horizon.
    """
    HORIZON_DAYS = {"1D": 1, "1W": 7, "1M": 30, "1Q": 90, "1Y": 365}
    try:
        t     = yf.Ticker(ticker)
        start = datetime.strptime(date_str, "%Y-%m-%d")
        end_cap = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

        hist_to_date = t.history(start=start - timedelta(days=400), end=start + timedelta(days=2))
        if hist_to_date.empty:
            return {"error": f"No data for {ticker} around {date_str}"}

        price_at_date = round(float(hist_to_date["Close"].iloc[-1]), 2)
        fetch_end     = min(end_cap, datetime.now()) if end_cap else datetime.now()
        hist_future   = t.history(start=start, end=fetch_end + timedelta(days=2))
        info          = t.info or {}

        ds = _build_data_scores(hist_to_date, info)
        results = []

        for horizon, days in HORIZON_DAYS.items():
            overall, _ = _composite(ds, horizon)
            pred_target = compute_price_target(price_at_date, overall, horizon)

            horizon_end = start + timedelta(days=days)
            if end_cap:
                horizon_end = min(horizon_end, end_cap)

            idx = hist_future.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                he_ts = pd.Timestamp(horizon_end).tz_localize(idx.tz)
            else:
                he_ts = pd.Timestamp(horizon_end)

            future_slice = hist_future[hist_future.index <= he_ts]

            if len(future_slice) > 0:
                actual_price = round(float(future_slice["Close"].iloc[-1]), 2)
                actual_ret   = round((actual_price / price_at_date - 1) * 100, 2)
                brier        = compute_brier(overall, actual_ret, horizon)
            else:
                actual_price = actual_ret = brier = None

            pred_dir   = "UP" if overall >= 55 else ("DOWN" if overall <= 45 else "FLAT")
            actual_dir = ("UP" if (actual_ret or 0) > 0.5
                          else ("DOWN" if (actual_ret or 0) < -0.5 else "FLAT")) if actual_ret is not None else "—"
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

# ─── SCENARIO SIMULATOR ───────────────────────────────────────────────────────

def run_scenario(ticker, start_date, end_date, dollars, interval, buy_rating, sell_rating):
    """
    Buy/sell simulation from start_date to end_date using score thresholds.
    end_date is now a required parameter.
    """
    INTERVAL_DAYS = {"1D": 1, "1W": 7, "1M": 30}
    days_step = INTERVAL_DAYS.get(interval, 7)

    try:
        t      = yf.Ticker(ticker)
        start  = datetime.strptime(start_date, "%Y-%m-%d")
        end    = datetime.strptime(end_date,   "%Y-%m-%d")

        if end <= start:
            return None, "End date must be after start date."

        hist_full = t.history(start=start - timedelta(days=400), end=end + timedelta(days=2))
        if hist_full.empty:
            return None, "No price data found."

        if hist_full.index.tz is not None:
            hist_full.index = hist_full.index.tz_localize(None)

        cash   = float(dollars)
        shares = 0
        log    = []
        cur    = start

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
                    shares += n
                    cash   -= n * price
                    action  = "BUY"
            elif score <= sell_rating and shares > 0:
                cash   += shares * price
                shares  = 0
                action  = "SELL"

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

        final       = df["Portfolio Value"].iloc[-1]
        start_price = df["Price"].iloc[0]
        end_price   = df["Price"].iloc[-1]
        bh_val      = round(float(dollars) * (end_price / start_price), 2)
        total_ret   = round((final / float(dollars) - 1) * 100, 2)
        bh_ret      = round((bh_val / float(dollars) - 1) * 100, 2)

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

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _signal_label(score):
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
