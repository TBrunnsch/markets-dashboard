# dashboard_finance_full_daily_bonds.py
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from pandas_datareader.data import DataReader
import requests
import altair as alt
import calendar

st.set_page_config(page_title="Markets & Rates Dashboard", layout="wide")
st.title("📊 Markets & Rates Dashboard")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")
years = st.sidebar.slider("Historical period (years)", min_value=1, max_value=5, value=2)
refresh = st.sidebar.button("🔄 Refresh Data")

end = pd.Timestamp.today()
start = end - pd.DateOffset(years=years)

# -------------------------
# Tickers
# -------------------------
TICKERS = {
    "SMI": "^SSMI",
    "Dow Jones": "^DJI",
    "DAX": "^GDAXI",
    "EUR/CHF": "EURCHF=X",
    "USD/CHF": "USDCHF=X",
    "Brent": "BZ=F",
    "Gold": "GC=F",
}

# -------------------------
# Fetch functions
# -------------------------
@st.cache_data(ttl=300)
def fetch_yfinance(tickers, start, end, interval="1d"):
    data = {}
    for name, tk in tickers.items():
        try:
            hist = yf.download(tk, start=start, end=end + pd.Timedelta(days=1),
                               interval=interval, progress=False)
            data[name] = hist if not hist.empty else None
        except Exception:
            data[name] = None
    return data

@st.cache_data(ttl=3600)
def fetch_fed_rate(start, end):
    df = DataReader("FEDFUNDS", "fred", start, end)
    df = df.resample('M').last().fillna(method='ffill')
    df = df.reset_index().rename(columns={"DATE": "date", "FEDFUNDS": "rate"})
    return df

@st.cache_data(ttl=3600)
def fetch_ecb_rate(start, end):
    df = DataReader("ECBDFR", "fred", start, end)
    df = df.resample('M').last().fillna(method='ffill')
    df = df.reset_index().rename(columns={"DATE": "date", "ECBDFR": "rate"})
    return df

@st.cache_data(ttl=3600)
def fetch_snb_policy_rate(start):
    url = "https://data.snb.ch/api/cube/snboffzisa/data/json/en"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        timeseries = data.get("timeseries", [])
        for ts in timeseries:
            header = ts.get("header", [])
            if header and "SNB policy rate" in header[0].get("dimItem", ""):
                values = ts.get("values", [])
                df = pd.DataFrame(values)
                if df.empty:
                    return None
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['date', 'value'])
                df = df[df['date'] >= start]  # Filter basierend auf dem Slider
                return df.sort_values('date')
        return None
    except requests.RequestException as e:
        st.warning(f"Fehler beim Abrufen der SNB Policy Rate: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_swiss_10y_bond_daily(start):
    url = "https://data.snb.ch/api/cube/rendeiduebd/data/json/en"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        timeseries = data.get("timeseries", [])
        for ts in timeseries:
            header = ts.get("header", [])
            if header:
                dim_item = {h['dim']: h['dimItem'] for h in header}
                if dim_item.get("Bond categories", "").startswith("Spot interest rates") and "10 year" in dim_item.get("Maturity","").lower():
                    values = ts.get("values", [])
                    df = pd.DataFrame(values)
                    if df.empty:
                        return None
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna(subset=['date','value'])
                    df = df[df['date'] >= start]  # Filter basierend auf dem Slider
                    return df.sort_values('date')
        return None
    except requests.RequestException as e:
        st.warning(f"Fehler beim Abrufen der 10Y Swiss Bond Rate: {e}")
        return None

# -------------------------
# Chart helper
# -------------------------
def create_chart(df, value_col, title, unit, daily=False):
    df = df.copy()
    if not daily:
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['month_name'] = df['month'].apply(lambda x: calendar.month_abbr[x])
        df['x_label'] = df['month_name'] + "\n" + df['year'].astype(str)
        x = alt.X('x_label:N', title='Month / Year', sort=None)
    else:
        x = alt.X('date:T', title='Date')
    chart = alt.Chart(df).mark_line().encode(
        x=x,
        y=alt.Y(f'{value_col}:Q', title=unit),
        tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip(f'{value_col}:Q', title=f'Value ({unit})')]
    ).properties(
        title=title,
        height=300
    )
    return chart

# -------------------------
# Fetch data
# -------------------------
with st.spinner("📥 Loading market data..."):
    yf_data = fetch_yfinance(TICKERS, start, end)

with st.spinner("📥 Loading interest rates..."):
    fed_df = fetch_fed_rate(start, end)
    ecb_df = fetch_ecb_rate(start, end)
    snb_df = fetch_snb_policy_rate(start)
    bond10y_df = fetch_swiss_10y_bond_daily(start)

# -------------------------
# Market Metrics
# -------------------------
st.subheader("Current Prices / Exchange Rates / Commodities")
cols = st.columns(len(TICKERS))
for i, (name, tk) in enumerate(TICKERS.items()):
    col = cols[i]
    hist = yf_data.get(name)
    if hist is None:
        col.metric(label=name, value="N/A")
    else:
        last_close = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last_close
        change = last_close - prev
        pct = (change / prev) * 100 if prev != 0 else 0
        col.metric(label=name, value=f"{last_close:,.2f}", delta=f"{pct:.2f}%")

st.markdown("---")

# -------------------------
# Market Charts (daily)
# -------------------------
st.subheader("Market / Commodity / Bonds Charts")
all_daily = list(TICKERS.keys()) + ["10Y Bundesobli"]
for i in range(0, len(all_daily), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(all_daily):
            name = all_daily[i + j]
            if name == "10Y Bundesobli":
                hist = bond10y_df
            else:
                hist = yf_data[name]
            with cols[j]:
                if hist is None or hist.empty:
                    st.warning(f"No data for {name}")
                else:
                    df = hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "value"}) if name != "10Y Bundesobli" else hist
                    df['date'] = pd.to_datetime(df['date'])
                    unit = (
                        "Indexpunkte" if name in ["SMI","Dow Jones","DAX"] else
                        "USD" if name in ["Brent","Gold"] else
                        "%" if name=="10Y Bundesobli" else
                        "CHF"
                    )
                    st.altair_chart(create_chart(df, "value", name, unit, daily=True), use_container_width=True)

st.markdown("---")

# -------------------------
# Interest Rates Charts (monthly)
# -------------------------
st.subheader("📈 Central Bank Interest Rates (Monthly)")
rates = [("USA Fed", fed_df, "%"), 
         ("EZB Leitzins", ecb_df, "%"), 
         ("SNB Leitzins", snb_df, "%")]

cols = st.columns(3)
for i, (label, df, unit) in enumerate(rates):
    with cols[i]:
        if df is None or df.empty:
            st.error(f"No data for {label}")
        else:
            last_val = df['rate'].iloc[-1] if 'rate' in df.columns else df['value'].iloc[-1]
            st.metric(label, f"{last_val:.2f} {unit}")
            val_col = 'rate' if 'rate' in df.columns else 'value'
            st.altair_chart(create_chart(df, val_col, label, unit, daily=False), use_container_width=True)

st.markdown("---")
st.caption("📌 Märkte & 10Y Oblis täglich, Leitzinsen monatlich: Fed (USA), EZB (Europa), SNB (Schweiz)")
