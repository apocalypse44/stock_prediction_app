import numpy as np
import streamlit as st
from datetime import date, datetime
import yfinance as yf
import nsepy
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as g_obs

st.set_page_config(page_title="Stock Prediction", page_icon="📈")
st.title("STOCK PREDICTION APP")
st.write("😀")

TODAY = date.today()
indices = ["NIFTY_50", "BANKNIFTY"]
stocks = st.text_input("STOCK SYMBOL")
st.write("Enter Stock For Prediction")

n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365

@st.cache
def load_data(ticker):
    if(ticker in indices):
        data = nsepy.get_history(symbol=ticker, start=date(2015,1,1), end=TODAY,index=True)
        data.reset_index(inplace=True)
    else:
        data = nsepy.get_history(symbol=ticker, start=date(2015,1,1), end=TODAY)
        data.reset_index(inplace=True)
    return data
data_load_state = st.text("Load Data.....")
data = load_data(stocks)
data_load_state.text("Loading data.....DONE")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = g_obs.Figure()
    fig.add_trace(g_obs.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(g_obs.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    fig2 = g_obs.Figure(data=[g_obs.Candlestick(x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'])])
    st.plotly_chart(fig2)
plot_raw_data()

#Prediction with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
pred = model.predict(future)

st.subheader('Forecast data')
st.write(pred.tail())

st.write("Prediction Graph")
fig1 = plot_plotly(model, pred)
st.plotly_chart(fig1)

st.write("Prediction Components")
fig2 = model.plot_components(pred)
st.write(fig2)