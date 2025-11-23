import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "stock_data"
LOG_FILE = "stock_prediction_log.json"

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
#   SERIALIZATION HELPER (SAFE UNTUK JSON)
# ============================================================
def to_serializable(obj):
    """Convert numpy, pandas, timestamps into JSON-safe types."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


# ============================================================
#   WRITE LOG PREDIKSI
# ============================================================
def write_prediction_log(symbol, prediction, hist3, fundamental=None):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "prediction": to_serializable(prediction),
        "last_3_days": to_serializable(hist3),
        "fundamental": to_serializable(fundamental)
    }

    if os.path.exists(LOG_FILE):
        try:
            logs = json.load(open(LOG_FILE))
        except:
            logs = []
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    return True


# ============================================================
#   READ LOG (DIGUNAKAN DI STREAMLIT)
# ============================================================
def read_prediction_log(limit=50):
    if not os.path.exists(LOG_FILE):
        return []

    try:
        logs = json.load(open(LOG_FILE))
    except:
        return []

    return logs[-limit:]


# ============================================================
#   CACHE LIST & CLEAR
# ============================================================
def list_cached_symbols():
    files = os.listdir(DATA_DIR)
    symbols = []
    for f in files:
        if f.endswith(".csv"):
            sym = f.replace(".csv", "").replace("_", ".")
            symbols.append(sym)
    return sorted(symbols)


def clear_cache(symbol=None):
    removed = []

    if symbol:
        targets = [
            f"{symbol.replace('.', '_')}.csv",
            f"{symbol.replace('.', '_')}_fundamental.json",
            f"{symbol.replace('.', '_')}_technical.json"
        ]
        for t in targets:
            path = os.path.join(DATA_DIR, t)
            if os.path.exists(path):
                os.remove(path)
                removed.append(path)
    else:
        # delete all cache files
        for f in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, f)
            if os.path.isfile(path):
                os.remove(path)
                removed.append(path)

    return removed
