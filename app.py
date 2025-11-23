import streamlit as st
import pandas as pd

from data_loader import get_cached_stock_data
from prediction import basic_predict_stock_price, advanced_predict_stock_price
from technical_analysis import analyze_technical, build_technical_indicators
from visualization import build_full_chart
from utils import (
    write_prediction_log, 
    read_prediction_log, 
    list_cached_symbols,
    clear_cache
)


# ============================================================
#   STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Nanang Stock AI",
    layout="wide"
)

st.title("📊 Nanang Stock AI — Prediction & Technical Analysis (Streamlit)")

st.markdown("""
### 🔥 Prediksi Harga Saham + Analisis Teknikal Lengkap  
• Model Basic → Cepat  
• Model Advanced → Akurat (Teknikal + Fundamental)  
• Grafik TradingView Style  
""")

# ============================================================
#   INPUT SECTION
# ============================================================

symbol = st.text_input("Masukkan simbol saham (contoh: AAPL, BBRI.JK, TSLA)", value="BBRI.JK").upper()
model_choice = st.selectbox("Pilih Model Prediksi:", ["Basic Prediction", "Advanced Prediction"])

col_run, col_clear = st.columns([2,1])
run_predict = col_run.button("🚀 Jalankan Prediksi")
clear_log_btn = col_clear.button("🧹 Clear Cache (symbol ini)")


# ============================================================
#   CLEAR CACHE
# ============================================================
if clear_log_btn:
    removed = clear_cache(symbol)
    st.success(f"Cache dibersihkan: {removed}")


# ============================================================
#   RUN PREDICTION
# ============================================================
if run_predict and symbol:

    # ---------- BASIC -----------
    if model_choice == "Basic Prediction":
        result, hist3, df = basic_predict_stock_price(symbol)

        if result is None:
            st.error("Data tidak cukup untuk prediksi model basic.")
            st.stop()

        st.subheader("📈 Hasil Prediksi — Basic Model")
        st.write(result)

        write_prediction_log(symbol, result, hist3)

    # ---------- ADVANCED -------
    else:
        result, hist3, fundamental, df = advanced_predict_stock_price(symbol)

        if result is None:
            st.error("Data tidak cukup untuk prediksi model advanced.")
            st.stop()

        st.subheader("🤖 Hasil Prediksi — Advanced Model")
        st.write(result)

        write_prediction_log(symbol, result, hist3, fundamental)


    # ============================================================
    #   ANALISIS TEKNIKAL
    # ============================================================
    st.subheader("📊 Analisis Teknikal Lengkap")

    df_ta = build_technical_indicators(df.copy())
    ta = analyze_technical(symbol, df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Technical Score", f"{ta['technical_score']}/100")
    col2.metric("Recommendation", ta["recommendation"])
    col3.metric("RSI (14)", f"{ta['rsi']:.2f}")

    st.write(ta)


    # ============================================================
    #   GRAFIK TRADINGVIEW STYLE
    # ============================================================
    st.subheader("📉 Grafik Harga & Indikator")

    charts = build_full_chart(df_ta, symbol)

    st.plotly_chart(charts["candlestick"], use_container_width=True)
    st.plotly_chart(charts["rsi"], use_container_width=True)
    st.plotly_chart(charts["macd"], use_container_width=True)
    st.plotly_chart(charts["volume"], use_container_width=True)
    st.plotly_chart(charts["bollinger"], use_container_width=True)


# ============================================================
#   LOG HISTORY
# ============================================================
st.subheader("🕑 Histori Prediksi Terakhir")

logs = read_prediction_log()
if logs:
    st.json(logs[-10:])
else:
    st.info("Belum ada log prediksi.")


# ============================================================
#   CACHE LIST
# ============================================================
st.subheader("📦 Cache Symbols")

cached = list_cached_symbols()
st.write(cached)
