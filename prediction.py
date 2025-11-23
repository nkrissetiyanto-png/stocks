import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error

from data_loader import get_cached_stock_data, get_cached_fundamental_data
from features import (
    create_basic_features,
    create_comprehensive_features,
    calculate_fundamental_score,
    get_last_3_days_data
)

# ======================================================
#                BASIC PREDICTION MODEL
# ======================================================
def basic_predict_stock_price(symbol, days_to_predict=1, force_update=False):
    """
    Prediksi dasar untuk saham stabil.
    """
    # Ambil data
    data = get_cached_stock_data(symbol, '2y', force_update)
    if data is None or len(data) < 100:
        return None, None, None

    # Buat fitur
    df = create_basic_features(data)
    hist_3 = get_last_3_days_data(data)

    feature_cols = [
        'Open','High','Low','Close','Volume',
        'Price_Lag_1','Price_Lag_2','Price_Lag_3',
        'MA_5','MA_10','MA_20','Volatility','RSI','MACD'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y_open = df['Open'].shift(-days_to_predict)
    y_close = df['Close'].shift(-days_to_predict)

    valid = ~y_open.isna()
    X, y_open, y_close = X[valid], y_open[valid], y_close[valid]

    if len(X) < 50:
        return None, None, None

    # Split train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_open_train, y_open_test = y_open[:split], y_open[split:]
    y_close_train, y_close_test = y_close[:split], y_close[split:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model_open = RandomForestRegressor(n_estimators=60, random_state=42)
    model_close = RandomForestRegressor(n_estimators=60, random_state=42)

    model_open.fit(X_train_scaled, y_open_train)
    model_close.fit(X_train_scaled, y_close_train)

    # Predict last
    last_scaled = scaler.transform(X.iloc[[-1]])
    pred_open = model_open.predict(last_scaled)[0]
    pred_close = model_close.predict(last_scaled)[0]

    # Range via MAE
    mae_open = mean_absolute_error(y_open_test, model_open.predict(X_test_scaled))
    mae_close = mean_absolute_error(y_close_test, model_close.predict(X_test_scaled))

    result = {
        "current_price": data["Close"].iloc[-1],
        "predicted_open": pred_open,
        "open_range": (pred_open - mae_open, pred_open + mae_open),
        "predicted_close": pred_close,
        "close_range": (pred_close - mae_close, pred_close + mae_close),
        "volatility": data['Close'].pct_change().std() * np.sqrt(252),
        "model_type": "basic"
    }

    return result, hist_3, data



# ======================================================
#              ADVANCED PREDICTION MODEL
# ======================================================
def advanced_predict_stock_price(symbol, days_to_predict=1, force_update=False):
    """
    Prediksi advanced (teknikal + fundamental).
    """
    # Load data teknikal
    data = get_cached_stock_data(symbol, '3y', force_update)
    if data is None or len(data) < 150:
        return None, None, None, None

    # Load fundamental
    fundamental = get_cached_fundamental_data(symbol, force_update)
    fund_score = calculate_fundamental_score(fundamental)

    hist_3 = get_last_3_days_data(data)

    # Build features
    df = create_comprehensive_features(data, fundamental, fund_score)

    # Siapkan dataset
    feature_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume']]
    X = df[feature_cols]
    y_open = df['Open'].shift(-days_to_predict)
    y_close = df['Close'].shift(-days_to_predict)

    valid = ~y_open.isna()
    X, y_open, y_close = X[valid], y_open[valid], y_close[valid]

    if len(X) < 100:
        return None, None, None, None

    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Model advanced
    model_open = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=8,
        min_samples_leaf=4, max_features='sqrt', random_state=42
    )
    model_close = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=8,
        min_samples_leaf=4, max_features='sqrt', random_state=42
    )

    model_open.fit(X_scaled, y_open)
    model_close.fit(X_scaled, y_close)

    # Predict
    last_scaled = scaler.transform(X.iloc[[-1]])
    pred_open = model_open.predict(last_scaled)[0]
    pred_close = model_close.predict(last_scaled)[0]

    # Ensemble STD → Confidence interval
    open_std = np.std([est.predict(last_scaled)[0] for est in model_open.estimators_])
    close_std = np.std([est.predict(last_scaled)[0] for est in model_close.estimators_])

    confidence = 1.96 * (1 - (fund_score / 200))  # semakin bagus fundamental → CI mengecil

    current_price = data['Close'].iloc[-1]

    open_range = (
        max(pred_open - confidence * open_std, current_price * 0.7),
        min(pred_open + confidence * open_std, current_price * 1.3),
    )
    close_range = (
        max(pred_close - confidence * close_std, current_price * 0.7),
        min(pred_close + confidence * close_std, current_price * 1.3),
    )

    # Fundamental adjustment
    if fund_score > 70:
        adj = 1 + (fund_score - 70) / 1000
        pred_open *= adj
        pred_close *= adj
    elif fund_score < 40:
        adj = 1 - (40 - fund_score) / 1000
        pred_open *= adj
        pred_close *= adj

    result = {
        "current_price": current_price,
        "predicted_open": pred_open,
        "open_range": open_range,
        "predicted_close": pred_close,
        "close_range": close_range,
        "volatility": data['Close'].pct_change().std() * np.sqrt(252),
        "fundamental_score": fund_score,
        "model_type": "advanced"
    }

    return result, hist_3, fundamental, data
