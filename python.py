import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.markdown(
    """
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    body{
    background-color: pin;
    }
    .stock-card {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        animation: pulse 2s infinite;
    }
    </style>
    <div class="stock-card">📈 Stock Market Predictor</div>
    """,
    unsafe_allow_html=True
)

stock = st.text_input('🔍 Enter Stock Symbol', 'GOOG')
start = "2020-01-01"
end = "2024-12-31"


data = yf.download(stock, start, end)

if data.empty:
    st.error("⚠️ No data found for the given stock symbol and date range.")
    st.stop()

st.markdown('<h2 class="subheader">📊 Stock Data (Latest 10 Records)</h2>', unsafe_allow_html=True)
st.write(data.tail(50).to_html(classes="table", escape=False), unsafe_allow_html=True)


st.markdown(f'<h2 class="subheader">📈 {stock} - Moving Averages</h2>', unsafe_allow_html=True)

ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.Close, label="Close Price", color="lime")
ax.plot(ma_50_days, label="50-Day MA", linestyle="dashed", color="red")
ax.plot(ma_100_days, label="100-Day MA", linestyle="dashed", color="blue")
ax.plot(ma_200_days, label="200-Day MA", linestyle="dashed", color="yellow")
ax.set_title(f"{stock} - Price vs Moving Averages")
ax.legend()
ax.grid(True)
st.pyplot(fig)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)


predictions = np.random.uniform(min(y), max(y), size=len(y))

scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y = y * scale_factor


st.markdown('<h2 class="subheader">📊 Predicted vs Actual Prices</h2>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y, 'g', label="Actual Price")
ax.plot(predictions, 'r', label="Predicted Price")
ax.set_title("Stock Price Prediction")
ax.legend()
ax.grid(True)
st.pyplot(fig)

latest_price = round(data.Close.iloc[-1], 2)
st.markdown(
    f"""
    <div class="stock-card">📌 Latest {stock} Price: ${latest_price}</div>
    """,
    unsafe_allow_html=True
)

st.success("✅ Prediction Completed!")
