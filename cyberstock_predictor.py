import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import streamlit as st
import pandas_ta as ta
from datetime import datetime
import os
from sklearn.externals import joblib

# Streamlit App Configuration
st.set_page_config(
    page_title="🚀 CyberStock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Retro Neon Theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #00FF00;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FF00FF;
    }
    .stButton>button {
        background-color: #00FF00;
        color: #000000;
        border-radius: 5px;
        border: 1px solid #FF00FF;
    }
    .stTextInput>div>div>input {
        background-color: #000000;
        color: #00FF00;
        border: 1px solid #FF00FF;
    }
    .stSelectbox>div>div>select {
        background-color: #000000;
        color: #00FF00;
        border: 1px solid #FF00FF;
    }
    .stSlider>div>div>div>div {
        background-color: #00FF00;
    }
    .stDataFrame {
        background-color: #000000;
        color: #00FF00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.title("🚀 CyberStock Predictor")
st.markdown("Predict future stock prices with **AI-powered precision** and a **retro neon vibe**!")

# Sidebar for User Input
st.sidebar.header("📊 User Input")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL").upper()
timeframe = st.sidebar.selectbox("Prediction Horizon:", ["7 Days", "30 Days", "90 Days"])
include_indicators = st.sidebar.checkbox("Include Technical Indicators", value=True)

# Fetch Historical Data
@st.cache_data
def fetch_data(symbol):
    try:
        data = yf.download(symbol, start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"))
        if data.empty:
            st.error(f"No data found for symbol: {symbol}. Please check the symbol and try again.")
            return None
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

if stock_symbol:
    data = fetch_data(stock_symbol)
    if data is not None:
        st.write(f"### 📈 Historical Data for {stock_symbol}")
        st.write(data.tail())

        # Add Technical Indicators
        if include_indicators:
            data['RSI'] = ta.rsi(data['Close'], length=14)
            data['EMA'] = ta.ema(data['Close'], length=20)
            data['MACD'] = ta.macd(data['Close'])['MACD_12_26_9']
            data.dropna(inplace=True)

        # Preprocess Data
        scaler_path = "scaler.save"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_close = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
            joblib.dump(scaler, scaler_path)

        # Prepare Training Data
        def create_dataset(dataset, time_step=60):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X, Y = create_dataset(scaled_close, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Build LSTM Model
        model_path = "lstm_model.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(100, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(50),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, Y, epochs=50, batch_size=32, verbose=1)
            model.save(model_path)

        # Predict Future Prices
        future_days = int(timeframe.split()[0])
        last_60_days = scaled_close[-time_step:]
        predictions = []

        for _ in range(future_days):
            x_input = last_60_days.reshape(1, -1)
            x_input = x_input.reshape((1, time_step, 1))
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred[0][0])
            last_60_days = np.append(last_60_days[1:], pred)

        # Inverse Transform Predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Create DataFrame for Predictions
        future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1, freq='B')[1:]
        predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})

        # Plot Historical and Predicted Data
        st.write("### 📊 Stock Price Predictions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Price', line=dict(color='#00FF00')))
        fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(color='#FF00FF')))
        fig.update_layout(
            title=f"{stock_symbol} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            font=dict(color="#00FF00"),
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show Predictions Table
        st.write("### 📅 Predicted Prices")
        st.dataframe(predictions_df.style.applymap(lambda x: "color: #00FF00"))

else:
    st.write("Please enter a valid stock symbol.")
