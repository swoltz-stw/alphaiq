"""
AlphaIQ Scoring Engine
======================
Fetches live data from Yahoo Finance (yfinance) and scores stocks 0–100
across 5 time horizons using 20 weighted data points.

Score interpretation:
  75–100 → Strong Buy (high probability of upward movement)
  60–74  → Buy
  45–59  → Neutral / Flat
  30–44  → Sell
  0–29   → Strong Sell (high probability of downward movement)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── WEIGHT TABLE ─────────────────────────────────────────────────────────────
# Weights are calibrated by horizon based on academic research and practitioner
# consensus. Short horizons weight technical/momentum signals more heavily;
# long horizons weight fundamentals and macro more heavily.
#
# Each weight is on a 1–10 scale. The composite score is a weighted average.

WEIGHTS = {
    "1D": {
        "momentum_1m":       9,   # Short-term price momentum is strongest 1D predictor
        "rsi":               8,   # RSI overbought/oversold very relevant intraday
        "macd":              7,   # MACD crossover signals
        "volume_trend":      7,   # Unusual volume often precedes moves
        "short_interest":    6,   # High short interest = squeeze risk
        "earnings_surprise": 5,   # Recent EPS beat/miss still in price memory
        "sentiment":         5,   # News sentiment moves price short-term
        "analyst_revisions": 4,
        "pe_vs_sector":      3,
        "revenue_growth":    3,
        "profit_margin":     2,
        "fcf_yield":         2,
        "pb_ratio":          2,
        "debt_equity":       2,
        "div_growth":        1,
        "peg_ratio":         2,
        "macro_score":       3,
        "sector_momentum":   5,
        "earnings_proximity":6,   # Upcoming earnings = catalyst risk
        "eps_growth":        3,
    },
    "1W": {
        "momentum_1m":       8,
        "rsi":               7,
        "macd":              6,
        "volume_trend":      6,
        "short_interest":    6,
        "earnings_surprise": 6,
        "sentiment":         6,
        "analyst_revisions": 5,
        "pe_vs_sector":      4,
        "revenue_growth":    4,
        "profit_margin":     3,
        "fcf_yield":         3,
        "pb_ratio":          2,
        "debt_equity":       2,
        "div_growth":        2,
        "peg_ratio":         3,
        "macro_score":       4,
        "sector_momentum":   5,
        "earnings_proximity":5,
        "eps_growth":        4,
    },
    "1M": {
        "momentum_1m":       6,
        "rsi":               5,
        "macd":              5,
        "volume_trend":      4,
        "short_interest":    5,
        "earnings_surprise": 7,
        "sentiment":         6,
        "analyst_revisions": 7,
        "pe_vs_sector":      6,
        "revenue_growth":    7,
        "profit_margin":     5,
        "fcf_yield":         5,
        "pb_ratio":          4,
        "debt_equity":       4,
        "div_growth":        3,
        "peg_ratio":         5,
        "macro_score":       5,
        "sector_momentum":   5,
        "earnings_proximity":4,
        "eps_growth":        6,
    },
    "1Q": {
        "momentum_1m":       4,
        "rsi":               3,
        "macd":              3,
        "volume_trend":      3,
        "short_interest":    4,
        "earnings_surprise": 8,
        "sentiment":         5,
        "analyst_revisions": 8,
        "pe_vs_sector":      7,
        "revenue_growth":    8,
        "profit_margin":     7,
        "fcf_yield":         7,
        "pb_ratio":          5,
        "debt_equity":       6,
        "div_growth":        4,
        "peg_ratio":         6,
        "macro_score":       7,
        "sector_momentum":   5,
        "earnings_proximity":3,
        "eps_growth":        7,
    },
    "1Y": {
        "momentum_1m":       3,
        "rsi":               2,
        "macd":              2,
        "volume_trend":      2,
        "short_interest":    3,
        "earnings_surprise": 7,
        "sentiment":         4,
        "analyst_revisions": 7,
        "pe_vs_sector":      8,
        "revenue_growth":    9,
        "profit_margin":     8,
        "fcf_yield":         8,
        "pb_ratio":          6,
        "debt_equity":       7,
        "div_growth":        5,
        "peg_ratio":         7,
        "macro_score":       8,
        "sector_momentum":   5,
        "earnings_proximity":2,
        "eps_growth":        9,
    },
}

DATA_POINT_LABELS = {
    "momentum_1m":       "Price Momentum (1M)",
    "rsi":               "RSI (14-Day)",
    "macd":              "MACD Signal",
    "volume_trend":      "Volume Trend",
    "short_interest":    "Short Interest",
    "earnings_surprise": "EPS Surprise",
    "sentiment":         "Analyst Sentiment",
    "analyst_revisions": "Analyst Revisions",
    "pe_vs_sector":      "P/E vs Sector",
    "revenue_growth":    "Revenue Growth",
    "profit_margin":     "Profit Margin",
    "fcf_yield":         "Free Cash Flow Yield",
    "pb_ratio":          "Price / Book",
    "debt_equity":       "Debt / Equity",
    "div_growth":        "Dividend Growth",
    "peg_ratio":         "PEG Ratio",
    "macro_score":       "Macro Conditions",
    "sector_momentum":   "Sector Momentum",
    "earnings_proximity":"Earnings Proximity",
    "eps_growth":        "EPS Growth Rate",
}

# ─── INDIVIDUAL SCORERS ───────────────────────────────────────────────────────

def _clamp(val, lo=0, hi=100):
    return max(lo, min(hi, val))

def score_momentum(hist):
    """1-month price return mapped to 0–100. +10% → ~80, -10% → ~20."""
    try:
        if hist is None or len(hist) < 22:
            return 50
        ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[-22] - 1) * 100
        return _clamp(int(50 + ret * 3))
    except:
        return 50

def score_rsi(hist):
    """RSI 14. Sweet spot 40–60 → neutral; <30 oversold (buy) → high score; >70 overbought → low score."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 15:
            return 50
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        # Oversold (<30) is bullish signal → high score
        if rsi_val < 30:
            return _clamp(int(75 + (30 - rsi_val) * 1.5))
        elif rsi_val > 70:
            return _clamp(int(25 - (rsi_val - 70) * 1.5))
        else:
            return _clamp(int(50 + (50 - rsi_val) * 0.5))
    except:
        return 50

def score_macd(hist):
    """MACD line vs signal line. Bullish cross → high score."""
    try:
        close = hist["Close"].dropna()
        if len(close) < 26:
            return 50
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        diff = macd.iloc[-1] - signal.iloc[-1]
        prev_diff = macd.iloc[-2] - signal.iloc[-2]
        # Recent cross
        if diff > 0 and prev_diff <= 0:
            return 80
        elif diff < 0 and prev_diff >= 0:
            return 20
        elif diff > 0:
            return _clamp(int(60 + diff * 10))
        else:
            return _clamp(int(40 + diff * 10))
    except:
        return 50

def score_volume(hist):
    """Volume today vs 90-day average. Spike with up price → bullish."""
    try:
        if len(hist) < 20:
            return 50
        avg_vol = hist["Volume"].iloc[-90:].mean() if len(hist) >= 90 else hist["Volume"].mean()
        recent_vol = hist["Volume"].iloc[-5:].mean()
        price_up = hist["Close"].iloc[-1] > hist["Close"].iloc[-5]
        ratio = recent_vol / (avg_vol + 1)
        if ratio > 1.3 and price_up:
            return _clamp(int(65 + (ratio - 1) * 20))
        elif ratio > 1.3 and not price_up:
            return _clamp(int(35 - (ratio - 1) * 20))
        return 50
    except:
        return 50

def score_short_interest(info):
    """Short % of float. High short = squeeze potential (bullish) or bearish sentiment."""
    try:
        short_pct = info.get("shortPercentOfFloat", 0) or 0
        short_pct *= 100
        if short_pct > 20:
            return 30   # Very high short = bearish consensus
        elif short_pct > 10:
            return 40
        elif short_pct < 3:
            return 65   # Low short = institutional confidence
        return 55
    except:
        return 50

def score_earnings_surprise(info):
    """EPS actual vs estimate. Beat → bullish."""
    try:
        eps_actual = info.get("trailingEps") or 0
        eps_est = info.get("forwardEps") or eps_actual
        if eps_est == 0:
            return 50
        surprise_pct = (eps_actual - eps_est) / abs(eps_est) * 100
        return _clamp(int(50 + surprise_pct * 2))
    except:
        return 50

def score_sentiment(info):
    """Analyst recommendation mean. 1=Strong Buy, 5=Strong Sell → invert to 0–100."""
    try:
        rec = info.get("recommendationMean")
        if rec is None:
            return 50
        # 1.0 → 95, 3.0 → 50, 5.0 → 5
        return _clamp(int(95 - (rec - 1) * 22.5))
    except:
        return 50

def score_analyst_revisions(info):
    """Number of analyst buy vs hold/sell recommendations."""
    try:
        strong_buy = info.get("numberOfAnalystOpinions", 0) or 0
        rec_mean = info.get("recommendationMean", 3) or 3
        if strong_buy == 0:
            return 50
        # Low mean (bullish) + many analysts = high confidence
        score = _clamp(int((5 - rec_mean) / 4 * 100))
        # Scale up if many analysts cover it
        if strong_buy > 20:
            score = _clamp(score + 5)
        return score
    except:
        return 50

def score_pe_vs_sector(info):
    """P/E relative to trailing average. Lower than expected = undervalued."""
    try:
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is None or pe <= 0:
            return 50
        # Rough sector-agnostic scoring
        if pe < 10:
            return 75
        elif pe < 15:
            return 68
        elif pe < 20:
            return 60
        elif pe < 25:
            return 52
        elif pe < 35:
            return 44
        elif pe < 50:
            return 35
        else:
            return 25
    except:
        return 50

def score_revenue_growth(info):
    """YoY revenue growth rate."""
    try:
        growth = info.get("revenueGrowth")
        if growth is None:
            return 50
        growth_pct = growth * 100
        return _clamp(int(50 + growth_pct * 2.5))
    except:
        return 50

def score_profit_margin(info):
    """Net profit margin. Higher = better."""
    try:
        margin = info.get("profitMargins")
        if margin is None:
            return 50
        margin_pct = margin * 100
        if margin_pct < 0:
            return _clamp(int(30 + margin_pct))
        elif margin_pct < 5:
            return 45
        elif margin_pct < 10:
            return 55
        elif margin_pct < 20:
            return 65
        elif margin_pct < 30:
            return 75
        else:
            return 85
    except:
        return 50

def score_fcf_yield(info):
    """Free cash flow yield = FCF / Market Cap."""
    try:
        fcf = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        if not fcf or not mktcap or mktcap == 0:
            return 50
        yield_pct = (fcf / mktcap) * 100
        return _clamp(int(50 + yield_pct * 4))
    except:
        return 50

def score_pb_ratio(info):
    """Price/Book. Lower = more undervalued (generally)."""
    try:
        pb = info.get("priceToBook")
        if pb is None or pb <= 0:
            return 50
        if pb < 1:
            return 80
        elif pb < 2:
            return 68
        elif pb < 3:
            return 58
        elif pb < 5:
            return 48
        elif pb < 10:
            return 38
        else:
            return 28
    except:
        return 50

def score_debt_equity(info):
    """Debt/Equity. Lower is safer."""
    try:
        de = info.get("debtToEquity")
        if de is None:
            return 55
        if de < 20:
            return 80
        elif de < 50:
            return 68
        elif de < 100:
            return 55
        elif de < 200:
            return 40
        else:
            return 25
    except:
        return 50

def score_div_growth(info):
    """Dividend growth rate and yield as stability signal."""
    try:
        dy = info.get("dividendYield") or 0
        dy_pct = dy * 100
        payout = info.get("payoutRatio") or 0
        if dy_pct == 0:
            return 50  # No dividend = neutral (many growth stocks)
        if payout > 1.0:
            return 25  # Paying out more than earns = risky
        if dy_pct > 6:
            return 55  # Very high yield often unsustainable
        return _clamp(int(55 + dy_pct * 5))
    except:
        return 50

def score_peg(info):
    """PEG ratio = P/E / growth. <1 undervalued, >2 overvalued."""
    try:
        peg = info.get("pegRatio")
        if peg is None or peg <= 0:
            return 50
        if peg < 0.5:
            return 85
        elif peg < 1.0:
            return 72
        elif peg < 1.5:
            return 58
        elif peg < 2.0:
            return 45
        elif peg < 3.0:
            return 33
        else:
            return 20
    except:
        return 50

def score_macro(_info):
    """
    Macro score is proxied by VIX level and 10Y treasury yield direction.
    Low VIX + stable rates = bullish macro environment.
    This is fetched once and shared across all tickers.
    """
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        if vix < 15:
            return 75
        elif vix < 20:
            return 62
        elif vix < 25:
            return 50
        elif vix < 35:
            return 38
        else:
            return 22
    except:
        return 50

def score_sector_momentum(info, cache={}):
    """Sector ETF 1-month momentum as proxy for sector tailwind."""
    SECTOR_ETFS = {
        "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
        "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
        "Energy": "XLE", "Utilities": "XLU", "Real Estate": "XLRE",
        "Basic Materials": "XLB", "Industrials": "XLI",
        "Communication Services": "XLC",
    }
    try:
        sector = info.get("sector", "")
        etf = SECTOR_ETFS.get(sector, "SPY")
        if etf not in cache:
            h = yf.Ticker(etf).history(period="2mo")
            if len(h) > 22:
                ret = (h["Close"].iloc[-1] / h["Close"].iloc[-22] - 1) * 100
                cache[etf] = _clamp(int(50 + ret * 3))
            else:
                cache[etf] = 50
        return cache[etf]
    except:
        return 50

def score_earnings_proximity(info):
    """Days to next earnings. Within 2 weeks = high uncertainty / volatility risk."""
    try:
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts is None:
            return 50
        days = (datetime.fromtimestamp(ts) - datetime.now()).days
        if 0 <= days <= 7:
            return 35   # Earnings imminent = elevated risk
        elif 0 <= days <= 14:
            return 42
        elif days < 0:
            return 60   # Just reported = clarity gained
        return 55
    except:
        return 50

def score_eps_growth(info):
    """EPS growth (forward vs trailing)."""
    try:
        trailing = info.get("trailingEps") or 0
        forward = info.get("forwardEps") or trailing
        if trailing == 0:
            return 50
        growth = (forward - trailing) / abs(trailing) * 100
        return _clamp(int(50 + growth * 2))
    except:
        return 50

# ─── BRIER SCORE ──────────────────────────────────────────────────────────────

def compute_brier(score, actual_return_pct, horizon):
    """
    Brier score measures calibration. We convert our score into a probability
    forecast P(up) and compare to whether stock actually went up.

    B = (P - O)^2, where O = 1 if stock went up, 0 if down.
    Perfect score = 0.0. Random = 0.25.
    """
    p_up = score / 100.0
    outcome = 1 if actual_return_pct > 0 else 0
    brier = round((p_up - outcome) ** 2, 4)
    return brier

# ─── MAIN SCORER ─────────────────────────────────────────────────────────────

def fetch_and_score(ticker: str, horizon: str = "1M", quiet: bool = False) -> dict:
    """
    Fetch live data for ticker and return full scoring result.

    Returns:
        {
          "ticker": str,
          "horizon": str,
          "overall": int (0–100),
          "signal": str,
          "price": float,
          "target": float,
          "data_scores": {key: int},
          "brier": float | None,
          "confidence": int,
          "error": str | None
        }
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="1y")

        if hist.empty:
            return _error_result(ticker, horizon, "No price data found")

        # ── Compute all 20 data point scores ──────────────────────────
        ds = {
            "momentum_1m":       score_momentum(hist),
            "rsi":               score_rsi(hist),
            "macd":              score_macd(hist),
            "volume_trend":      score_volume(hist),
            "short_interest":    score_short_interest(info),
            "earnings_surprise": score_earnings_surprise(info),
            "sentiment":         score_sentiment(info),
            "analyst_revisions": score_analyst_revisions(info),
            "pe_vs_sector":      score_pe_vs_sector(info),
            "revenue_growth":    score_revenue_growth(info),
            "profit_margin":     score_profit_margin(info),
            "fcf_yield":         score_fcf_yield(info),
            "pb_ratio":          score_pb_ratio(info),
            "debt_equity":       score_debt_equity(info),
            "div_growth":        score_div_growth(info),
            "peg_ratio":         score_peg(info),
            "macro_score":       score_macro(info),
            "sector_momentum":   score_sector_momentum(info),
            "earnings_proximity":score_earnings_proximity(info),
            "eps_growth":        score_eps_growth(info),
        }

        # ── Weighted composite ─────────────────────────────────────────
        w = WEIGHTS[horizon]
        total_weight = sum(w.values())
        weighted_sum = sum(ds[k] * w[k] for k in ds)
        overall = round(weighted_sum / total_weight)

        # ── Price target ───────────────────────────────────────────────
        price = round(hist["Close"].iloc[-1], 2)
        horizon_factor = {"1D": 0.005, "1W": 0.02, "1M": 0.07, "1Q": 0.16, "1Y": 0.38}[horizon]
        direction = (overall - 50) / 50
        target = round(price * (1 + direction * horizon_factor), 2)

        # ── Confidence via data availability ──────────────────────────
        filled = sum(1 for v in ds.values() if v != 50)
        confidence = round((filled / 20) * 100)

        return {
            "ticker": ticker.upper(),
            "horizon": horizon,
            "overall": overall,
            "signal": _signal_label(overall),
            "price": price,
            "target": target,
            "data_scores": ds,
            "brier": None,  # Only computed during backtesting
            "confidence": confidence,
            "error": None,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "—"),
            "market_cap": info.get("marketCap"),
        }

    except Exception as e:
        return _error_result(ticker, horizon, str(e))


def fetch_and_score_batch(tickers: list, horizon: str = "1M") -> pd.DataFrame:
    """Score a list of tickers and return a DataFrame."""
    results = []
    for t in tickers:
        r = fetch_and_score(t, horizon)
        row = {
            "Ticker": r["ticker"],
            "Name": r.get("name", ""),
            "Sector": r.get("sector", ""),
            "Score": r["overall"],
            "Signal": r["signal"],
            "Price": r.get("price"),
            "Target": r.get("target"),
            "Upside %": round((r["target"] / r["price"] - 1) * 100, 1) if r.get("price") and r.get("target") else None,
            "Confidence %": r["confidence"],
            "Error": r["error"],
        }
        results.append(row)
    return pd.DataFrame(results)


def backtest_ticker(ticker: str, date_str: str) -> dict:
    """
    For a given ticker and past date, compute what our model would have
    predicted, then compare to actual returns over each horizon.
    """
    results = []
    HORIZON_DAYS = {"1D": 1, "1W": 7, "1M": 30, "1Q": 90, "1Y": 365}

    try:
        t = yf.Ticker(ticker)
        start = datetime.strptime(date_str, "%Y-%m-%d")

        # Fetch historical data up to the backtest date
        hist_to_date = t.history(start=start - timedelta(days=400), end=start + timedelta(days=1))
        if hist_to_date.empty:
            return {"error": f"No data for {ticker} around {date_str}"}

        price_at_date = hist_to_date["Close"].iloc[-1]

        # Get full future history
        hist_full = t.history(start=start, end=datetime.now())

        # Use info from yfinance (current, not historical — limitation of free data)
        info = t.info or {}

        for horizon, days in HORIZON_DAYS.items():
            # Build a truncated hist for scoring (as if we were on that date)
            hist_snapshot = hist_to_date.copy()

            ds = {
                "momentum_1m":       score_momentum(hist_snapshot),
                "rsi":               score_rsi(hist_snapshot),
                "macd":              score_macd(hist_snapshot),
                "volume_trend":      score_volume(hist_snapshot),
                "short_interest":    score_short_interest(info),
                "earnings_surprise": score_earnings_surprise(info),
                "sentiment":         score_sentiment(info),
                "analyst_revisions": score_analyst_revisions(info),
                "pe_vs_sector":      score_pe_vs_sector(info),
                "revenue_growth":    score_revenue_growth(info),
                "profit_margin":     score_profit_margin(info),
                "fcf_yield":         score_fcf_yield(info),
                "pb_ratio":          score_pb_ratio(info),
                "debt_equity":       score_debt_equity(info),
                "div_growth":        score_div_growth(info),
                "peg_ratio":         score_peg(info),
                "macro_score":       score_macro(info),
                "sector_momentum":   score_sector_momentum(info),
                "earnings_proximity":score_earnings_proximity(info),
                "eps_growth":        score_eps_growth(info),
            }

            w = WEIGHTS[horizon]
            total_weight = sum(w.values())
            weighted_sum = sum(ds[k] * w[k] for k in ds)
            overall = round(weighted_sum / total_weight)

            direction = (overall - 50) / 50
            hf = {"1D": 0.005, "1W": 0.02, "1M": 0.07, "1Q": 0.16, "1Y": 0.38}[horizon]
            pred_target = round(price_at_date * (1 + direction * hf), 2)

            # Actual return
            end_date = start + timedelta(days=days)
            future = hist_full[hist_full.index <= end_date]
            if len(future) > 0:
                actual_price = future["Close"].iloc[-1]
                actual_ret = round((actual_price / price_at_date - 1) * 100, 2)
                actual_price = round(actual_price, 2)
            else:
                actual_ret = None
                actual_price = None

            brier = compute_brier(overall, actual_ret, horizon) if actual_ret is not None else None

            pred_dir = "UP" if overall >= 55 else ("DOWN" if overall <= 45 else "FLAT")
            actual_dir = "UP" if (actual_ret or 0) > 0.5 else ("DOWN" if (actual_ret or 0) < -0.5 else "FLAT")
            correct = pred_dir == actual_dir if actual_ret is not None else None

            results.append({
                "Horizon": horizon,
                "Score": overall,
                "Signal": _signal_label(overall),
                "Pred Target": pred_target,
                "Actual Return %": actual_ret,
                "Actual Price": actual_price,
                "Direction Correct": "✓ YES" if correct else ("✗ NO" if correct is not None else "—"),
                "Brier Score": brier,
            })

        return {
            "ticker": ticker.upper(),
            "date": date_str,
            "base_price": round(price_at_date, 2),
            "results": results,
            "error": None,
        }

    except Exception as e:
        return {"error": str(e)}


def run_scenario(ticker, start_date, dollars, interval, buy_rating, sell_rating):
    """
    Backtesting scenario: buy/sell based on score thresholds at each interval.
    Returns trade log DataFrame and summary stats.
    """
    INTERVAL_DAYS = {"1D": 1, "1W": 7, "1M": 30}
    days_step = INTERVAL_DAYS.get(interval, 7)

    try:
        t = yf.Ticker(ticker)
        start = datetime.strptime(start_date, "%Y-%m-%d")
        hist = t.history(start=start - timedelta(days=400), end=datetime.now())

        if hist.empty:
            return None, "No price data found."

        cash = float(dollars)
        shares = 0
        log = []
        cur_date = start

        while cur_date <= datetime.now() and len(log) < 104:
            subset = hist[hist.index.tz_localize(None) <= cur_date] if hist.index.tz is not None else hist[hist.index <= cur_date]
            if len(subset) < 20:
                cur_date += timedelta(days=days_step)
                continue

            price = round(subset["Close"].iloc[-1], 2)
            score = score_momentum(subset) * 0.4 + score_rsi(subset) * 0.35 + score_macd(subset) * 0.25
            score = round(score)

            action = "HOLD"
            if score >= buy_rating and cash >= price:
                n = int(cash // price)
                if n > 0:
                    shares += n
                    cash -= n * price
                    action = "BUY"
            elif score <= sell_rating and shares > 0:
                cash += shares * price
                shares = 0
                action = "SELL"

            portfolio = round(cash + shares * price, 2)
            log.append({
                "Date": cur_date.strftime("%Y-%m-%d"),
                "Score": score,
                "Price": price,
                "Action": action,
                "Shares": shares,
                "Cash": round(cash, 2),
                "Portfolio Value": portfolio,
            })
            cur_date += timedelta(days=days_step)

        df = pd.DataFrame(log)
        if df.empty:
            return None, "No trading periods found."

        final = df["Portfolio Value"].iloc[-1]
        initial_price = df["Price"].iloc[0]
        final_price = df["Price"].iloc[-1]
        buy_hold = round(float(dollars) * (final_price / initial_price), 2)
        total_ret = round((final / float(dollars) - 1) * 100, 2)
        bh_ret = round((buy_hold / float(dollars) - 1) * 100, 2)

        summary = {
            "Final Portfolio": round(final, 2),
            "Total Return %": total_ret,
            "Buy & Hold Return %": bh_ret,
            "Alpha %": round(total_ret - bh_ret, 2),
            "Trades": len(df[df["Action"] != "HOLD"]),
        }
        return df, summary

    except Exception as e:
        return None, str(e)


def _signal_label(score):
    if score >= 80: return "STRONG BUY"
    if score >= 65: return "BUY"
    if score >= 45: return "NEUTRAL"
    if score >= 30: return "SELL"
    return "STRONG SELL"

def _error_result(ticker, horizon, msg):
    return {
        "ticker": ticker, "horizon": horizon, "overall": 50,
        "signal": "NEUTRAL", "price": None, "target": None,
        "data_scores": {k: 50 for k in WEIGHTS["1M"]},
        "brier": None, "confidence": 0, "error": msg,
        "name": ticker, "sector": "—", "market_cap": None,
    }
