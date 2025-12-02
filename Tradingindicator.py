import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 1) Fetch ~1 year of data from Yahoo Finance 

TICKER = "MSFT"      # change to "AAPL", "TSLA", etc.
RANGE = "1y"         # 1 year of data
INTERVAL = "1d"      # daily candles

def fetch_yahoo_chart(ticker: str, range_: str = "1y", interval: str = "1d") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={range_}&interval={interval}&includeAdjustedClose=true"
    )
    print("Downloading from:", url)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP status {resp.status} from Yahoo Finance")
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise SystemExit(
                "HTTP 429 (Too Many Requests) from Yahoo.\n"
                "Wait a few minutes, then run again."
            )
        elif e.code == 401:
            raise SystemExit(
                "HTTP 401 (Unauthorized) from Yahoo chart API.\n"
                "Try again later or with a different TICKER.\n"
                "If this keeps happening, your network is blocking Yahoo."
            )
        else:
            raise

    chart = data.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart API error: {chart['error']}")

    result = chart.get("result")
    if not result:
        raise RuntimeError("No result data in Yahoo chart response")

    result = result[0]
    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})
    quote = indicators.get("quote", [{}])[0]

    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    if not timestamps or not closes:
        raise RuntimeError("Empty time series from Yahoo")

    # Build DataFrame
    dates = [
        datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        for ts in timestamps
    ]

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )

    # Remove rows with missing Close prices
    df = df.dropna(subset=["Close"])
    return df

df = fetch_yahoo_chart(TICKER, RANGE, INTERVAL)


# 2) Prepare data & indicators (9 EMA, 12 EMA, Bollinger Bands)


df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

df["Open"] = df["Open"].astype(float)
df["High"] = df["High"].astype(float)
df["Low"] = df["Low"].astype(float)
df["Close"] = df["Close"].astype(float)
df["Volume"] = df["Volume"].astype(float)

# 9 EMA and 12 EMA
df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()

# Bollinger Bands: 20-period, 2 std
window = 20
mult = 2

rolling_mean = df["Close"].rolling(window=window).mean()
rolling_std = df["Close"].rolling(window=window).std()

df["BB_Middle"] = rolling_mean
df["BB_Upper"] = rolling_mean + mult * rolling_std
df["BB_Lower"] = rolling_mean - mult * rolling_std

# Daily returns
df["Return"] = df["Close"].pct_change()


# 3) Candlestick + EMA9 + EMA12 + Bollinger Bands + Volume


volume_colors = [
    "green" if close >= open_ else "red"
    for open_, close in zip(df["Open"], df["Close"])
]

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.7, 0.3],
)

# --- Candlestick ---
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    ),
    row=1, col=1,
)

# --- EMAs ---
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA9"],
        mode="lines",
        line=dict(color="orange", width=1.2),
        name="EMA 9",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA12"],
        mode="lines",
        line=dict(color="purple", width=1.2),
        name="EMA 12",
    ),
    row=1, col=1,
)

# --- Bollinger Bands ---
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BB_Upper"],
        mode="lines",
        line=dict(color="blue", width=1, dash="dash"),
        name="BB Upper",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BB_Middle"],
        mode="lines",
        line=dict(color="blue", width=1),
        name="BB Middle (20 MA)",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BB_Lower"],
        mode="lines",
        line=dict(color="blue", width=1, dash="dash"),
        name="BB Lower",
    ),
    row=1, col=1,
)

# Optional: shaded Bollinger band area
fig.add_trace(
    go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df["BB_Upper"]) + list(df["BB_Lower"][::-1]),
        fill="toself",
        fillcolor="rgba(0, 0, 255, 0.06)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ),
    row=1, col=1,
)

# --- Volume ---
fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=volume_colors,
        name="Volume",
    ),
    row=2, col=1,
)

fig.update_layout(
    title=f"{TICKER} - Candlestick with 9 EMA, 12 EMA, and Bollinger Bands",
    xaxis_rangeslider_visible=False,
    xaxis=dict(title="Date"),
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Volume"),
    template="plotly_white",
)

fig.show()


# 4) Distribution of daily returns


plt.figure(figsize=(8, 4))
sns.histplot(df["Return"].dropna(), bins=50, kde=True)
plt.title(f"{TICKER} - Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()