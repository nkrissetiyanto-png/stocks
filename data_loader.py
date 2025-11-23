import os
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

DATA_DIR = "stock_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# LOAD DATA SAHAM (DENGAN CACHE)
# ===============================
def get_cached_stock_data(symbol, period='2y', force_update=False):
    cache_file = os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}.csv")

    # Cek cache masih fresh < 24 jam
    if not force_update and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(hours=24):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df
            except:
                pass

    # Ambil dari yfinance
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if not df.empty:
            df.to_csv(cache_file)
            return df
        return None
    except:
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return None


# ==================================
# LOAD DATA FUNDAMENTAL (DENGAN CACHE)
# ==================================
def get_cached_fundamental_data(symbol, force_update=False):
    cache_file = os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}_fundamental.json")

    # Cek cache < 7 hari
    if not force_update and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(days=7):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except:
                pass

    # Ambil dari yfinance
    try:
        info = yf.Ticker(symbol).info

        fundamental = {
            'trailingPE': info.get('trailingPE', 0),
            'forwardPE': info.get('forwardPE', 0),
            'priceToBook': info.get('priceToBook', 0),
            'priceToSales': info.get('priceToSalesTrailing12Months', 0),
            'profitMargins': info.get('profitMargins', 0),
            'returnOnEquity': info.get('returnOnEquity', 0),
            'debtToEquity': info.get('debtToEquity', 0),
            'currentRatio': info.get('currentRatio', 0),
            'earningsGrowth': info.get('earningsGrowth', 0),
            'revenueGrowth': info.get('revenueGrowth', 0),
            'dividendYield': info.get('dividendYield', 0),
            'marketCap': info.get('marketCap', 0),
            'beta': info.get('beta', 0)
        }

        with open(cache_file, "w") as f:
            json.dump(fundamental, f, indent=2)

        return fundamental
    except:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
        return {}
