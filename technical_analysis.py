import numpy as np
import pandas as pd

# ============================================================
#                HELPER — RSI & MACD
# ============================================================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast).mean()
    exp2 = df['Close'].ewm(span=slow).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal

    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_hist
    return df


# ============================================================
#        HITUNG SEMUA INDIKATOR TEKNIKAL
# ============================================================
def build_technical_indicators(df):

    # Moving Averages
    for win in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{win}'] = df['Close'].rolling(win).mean()

    # RSI
    df['RSI_14'] = calculate_rsi(df['Close'], 14)

    # MACD
    df = calculate_macd(df)

    # Volume MA
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()

    # Bollinger Bands
    mid = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Middle'] = mid
    df['BB_Upper'] = mid + std * 2
    df['BB_Lower'] = mid - std * 2

    # Support & resistance dynamic
    df['Resistance_20'] = df['High'].rolling(20).max()
    df['Support_20'] = df['Low'].rolling(20).min()

    return df.dropna()


# ============================================================
#     DETEKSI POLA ASCENDING TRIANGLE (OPSIONAL)
# ============================================================
def detect_ascending_triangle(df, lookback=90):
    recent = df.tail(lookback)

    resistance = recent["High"].max()
    lows = recent["Low"].values
    x = np.arange(len(lows))

    if len(lows) < 10:
        return {"detected": False, "target": None}

    # slope tren naik pada support (indikator ascending)
    slope = np.polyfit(x, lows, 1)[0]

    if slope <= 0:
        return {"detected": False, "target": None}

    height = resistance - recent["Low"].min()
    target = resistance + height

    return {
        "detected": True,
        "resistance": resistance,
        "support_slope": slope,
        "target": target
    }


# ============================================================
#      ANALISIS TEKNIKAL + SKOR + REKOMENDASI
# ============================================================
def analyze_technical(symbol, df):
    """
    Return structured technical analysis:
    {
        technical_score,
        recommendation,
        current_price,
        resistance,
        support,
        rsi,
        volume_ratio,
        ma_signal,
        triangle_target
    }
    """

    df_ta = build_technical_indicators(df.copy())
    last = df_ta.iloc[-1]

    price = last["Close"]

    # ============
    # Moving Avg
    # ============
    ma_vals = {w: last[f"MA_{w}"] for w in [5,10,20,50]}
    bullish_count = sum(price > ma_vals[w] for w in ma_vals)

    if bullish_count >= 3:
        ma_signal = "STRONG BUY"
    elif bullish_count >= 2:
        ma_signal = "NEUTRAL"
    else:
        ma_signal = "BEARISH"

    # ============
    # RSI
    # ============
    rsi = last["RSI_14"]

    # ============
    # VOLUME
    # ============
    vol_ratio = (
        last["Volume"] / last["Volume_MA_20"]
        if last["Volume_MA_20"] > 0
        else 1
    )

    # ============
    # SUPPORT / RESISTANCE
    # ============
    resistance = df["High"].tail(90).max()
    support = df["Low"].tail(90).min()

    # ============
    # Triangle Pattern
    # ============
    triangle = detect_ascending_triangle(df)
    triangle_target = triangle["target"] if triangle["detected"] else None

    # ============
    # Scoring System 0–100
    # ============
    score = 50

    # MA (30%)
    score += bullish_count * 5

    # RSI (20%)
    if 30 <= rsi <= 70:
        score += 10
    elif rsi > 70:
        score -= 5
    else:
        score += 5

    # Volume (20%)
    if vol_ratio > 1.5:
        score += 10
    elif vol_ratio < 0.8:
        score -= 5

    # MACD (30%)
    if last["MACD"] > last["MACD_Signal"]:
        score += 10
    else:
        score -= 5

    score = max(0, min(100, score))

    # ============
    # Recommendation
    # ============
    if score >= 70:
        rec = "STRONG BUY"
    elif score >= 60:
        rec = "BUY"
    elif score >= 50:
        rec = "NEUTRAL"
    elif score >= 40:
        rec = "CAUTION"
    else:
        rec = "STRONG SELL"

    return {
        "symbol": symbol,
        "technical_score": score,
        "recommendation": rec,
        "current_price": price,
        "resistance": float(resistance),
        "support": float(support),
        "rsi": float(rsi),
        "volume_ratio": float(vol_ratio),
        "ma_signal": ma_signal,
        "triangle_target": float(triangle_target) if triangle_target else None
    }
