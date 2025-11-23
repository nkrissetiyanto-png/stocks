import plotly.graph_objects as go
import pandas as pd

# ============================================================
#   CANDLESTICK + MOVING AVERAGES
# ============================================================
def plot_candlestick(df, symbol, ma_windows=[5, 20, 50]):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ))

    # Moving Averages
    for w in ma_windows:
        if f"MA_{w}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f"MA_{w}"],
                mode="lines",
                name=f"MA {w}",
                line=dict(width=1.8)
            ))

    fig.update_layout(
        title=f"{symbol} — Candlestick + Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ============================================================
#   RSI CHART
# ============================================================
def plot_rsi(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI_14'],
        mode="lines",
        name="RSI 14",
        line=dict(width=2)
    ))

    # Overbought / Oversold Bands
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)

    fig.update_layout(
        title="RSI (14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ============================================================
#   MACD CHART
# ============================================================
def plot_macd(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_Signal'],
        mode='lines',
        name='Signal',
        line=dict(width=2, dash="dot")
    ))

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Histogram'],
        name='Histogram',
        opacity=0.5
    ))

    fig.update_layout(
        title="MACD",
        xaxis_title="Date",
        yaxis_title="Value",
        height=300,
        barmode='overlay',
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ============================================================
#   VOLUME BAR CHART
# ============================================================
def plot_volume(df):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color="blue",
        opacity=0.6
    ))

    # Volume MA 20
    if "Volume_MA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Volume_MA_20"],
            mode="lines",
            name="MA Volume 20",
            line=dict(width=2, color="orange")
        ))

    fig.update_layout(
        title="Trading Volume",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ============================================================
#   BOLLINGER BANDS
# ============================================================
def plot_bollinger(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode="lines",
        name="Close Price",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Upper'],
        mode="lines",
        name="BB Upper",
        line=dict(width=1, color="red")
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Middle'],
        mode="lines",
        name="BB Middle",
        line=dict(width=1, color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Lower'],
        mode="lines",
        name="BB Lower",
        line=dict(width=1, color="green")
    ))

    fig.update_layout(
        title="Bollinger Bands",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ============================================================
#   MASTER PLOT (1 CALL)
# ============================================================
def build_full_chart(df, symbol):
    """Generate all charts needed in a Streamlit page."""
    components = {
        "candlestick": plot_candlestick(df, symbol),
        "rsi": plot_rsi(df),
        "macd": plot_macd(df),
        "volume": plot_volume(df),
        "bollinger": plot_bollinger(df)
    }
    return components
