import numpy as np
import pandas as pd

# ======================================
# 1. FITUR DASAR UNTUK PREDIKSI SIMPLE
# ======================================
def create_basic_features(data):
    df = data.copy()

    # Lag Prices
    df['Price_Lag_1'] = df['Close'].shift(1)
    df['Price_Lag_2'] = df['Close'].shift(2)
    df['Price_Lag_3'] = df['Close'].shift(3)

    # Moving Averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()

    # Volatility
    df['Volatility'] = df['Close'].rolling(10).std()

    # RSI 14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2

    return df.dropna()


# ==================================================
# 2. FITUR KOMPREHENSIF UNTUK MODEL ADVANCED
# ==================================================
def create_comprehensive_features(data, fundamental_data=None, fundamental_score=50):
    df = data.copy()

    # Rolling mean sebagai baseline
    df['Price_Rolling_Mean_20'] = df['Close'].rolling(20).mean()
    df['Price_Normalized'] = df['Close'] / df['Price_Rolling_Mean_20']

    # Lag & returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_{lag}'] = df['Close'].pct_change(lag)

    # Moving Averages
    for win in [5, 10, 20, 50, 100]:
        df[f'MA_{win}'] = df['Close'].rolling(win).mean()
        df[f'MA_Ratio_{win}'] = df['Close'] / df[f'MA_{win}']

    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # Volatility
    df['Vol_5'] = df['Close'].pct_change().rolling(5).std()
    df['Vol_20'] = df['Close'].pct_change().rolling(20).std()
    df['Vol_50'] = df['Close'].pct_change().rolling(50).std()

    # Support / Resistance
    df['Resistance_20'] = df['High'].rolling(20).max()
    df['Support_20'] = df['Low'].rolling(20).min()
    df['Price_vs_Resistance'] = df['Close'] / df['Resistance_20']
    df['Price_vs_Support'] = df['Close'] / df['Support_20']

    # Volume
    df['Vol_MA_5'] = df['Volume'].rolling(5).mean()
    df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Vol_MA_20']

    # RSI (7,14,21)
    for win in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(win).mean()
        loss = (-delta.clip(upper=0)).rolling(win).mean()
        rs = gain / loss
        df[f'RSI_{win}'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    mid = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Mid'] = mid
    df['BB_Upper'] = mid + std * 2
    df['BB_Lower'] = mid - std * 2
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Trend
    df['Trend_5'] = df['Close'].rolling(5).apply(lambda x: 1 if x[-1] > x[0] else -1)
    df['Trend_20'] = df['Close'].rolling(20).apply(lambda x: 1 if x[-1] > x[0] else -1)

    # Tambahkan Fundamental (konstan)
    if fundamental_data:
        df['Fundamental_Score'] = fundamental_score
        df['PE'] = fundamental_data.get('trailingPE', 0)
        df['PB'] = fundamental_data.get('priceToBook', 0)
        df['ProfitMargin'] = fundamental_data.get('profitMargins', 0)
        df['ROE'] = fundamental_data.get('returnOnEquity', 0)

    return df.dropna()


# ==================================
# 3. SKOR FUNDAMENTAL
# ==================================
def calculate_fundamental_score(f):
    if not f:
        return 50

    score = 50

    # Valuation
    if 0 < f.get('trailingPE', 0) < 25: score += 10
    if f.get('trailingPE', 0) > 40: score -= 10
    if 0 < f.get('priceToBook', 0) < 3: score += 5
    if f.get('priceToBook', 0) > 5: score -= 5

    # Profitability
    if f.get('profitMargins', 0) > 0.1: score += 10
    if f.get('profitMargins', 0) < 0: score -= 10
    if f.get('returnOnEquity', 0) > 0.15: score += 10
    if f.get('returnOnEquity', 0) < 0: score -= 5

    # Financial Health
    if f.get('debtToEquity', 0) < 1: score += 5
    if f.get('debtToEquity', 0) > 2: score -= 5
    if f.get('currentRatio', 0) > 1.5: score += 5
    if f.get('currentRatio', 0) < 1: score -= 5

    # Growth
    if f.get('earningsGrowth', 0) > 0.1: score += 5
    if f.get('earningsGrowth', 0) < -0.1: score -= 5
    if f.get('revenueGrowth', 0) > 0.1: score += 5
    if f.get('revenueGrowth', 0) < -0.1: score -= 5

    return max(0, min(100, score))


# ==================================
# 4. DATA 3 HARI TERAKHIR
# ==================================
def get_last_3_days_data(df):
    tail = df.tail(3)
    out = []
    for idx, row in tail.iterrows():
        out.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    return out
