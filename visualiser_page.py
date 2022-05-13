import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import yfinance as yf
from millify import millify, prettify
from utils import *

@st.cache(allow_output_mutation=True)
def plot_charts(df):
    buttons = [
        {'count': 1, 'label': '1D', 'step': 'day', 'stepmode': 'backward'},
        {'count': 5, 'label': '5D', 'step': 'day', 'stepmode': 'backward'},
        {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
        {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
        {'count': 1, 'label': '1Y', 'step': 'year', 'stepmode': 'backward'},
        {'count': 5, 'label': '5Y', 'step': 'year', 'stepmode': 'backward'},
        {'label': 'Max', 'step': 'all'}
    ]

    start = df['Date'][0]
    end = df['Date'][len(df) - 1]
    time_diff = end - start
    if time_diff < pd.Timedelta(5 * 365, 'd'):
        buttons[5]['visible'] = False
    elif time_diff < pd.Timedelta(365, 'd'):
        buttons[4]['visible'] = False
    elif time_diff < pd.Timedelta(6 * 30, 'd'):
        buttons[3]['visible'] = False
    elif time_diff < pd.Timedelta(31, 'd'):
        buttons[2]['visible'] = False
    elif time_diff < pd.Timedelta(5, 'd'):
        buttons[1]['visible'] = False
    line = go.Figure(go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    marker={'color': '#EA4335'} if df['Close'].values[0] > \
                    df['Close'].values[-1] else {'color': '#39AA57'},
                    hovertemplate='Price: $%{y:.2f} at ' +
                    '%{x} <extra></extra>',
                    fill='tozeroy'
                )
    )
    line.update_xaxes(
        rangeslider_visible=False,
        rangeselector={'buttons': buttons},
        rangebreaks=[{'bounds': ['Sat', 'Mon']}],
        showspikes=True
    )

    line.update_yaxes(showspikes=True)
    candlestick = go.Figure(
        data=[go.Candlestick(x=df['Date'],
                            open=df['Open'],
                            close=df['Close'],
                            high=df['High'],
                            low=df['Low'])]
        
    )
    candlestick.update_layout(yaxis_title='Stock price', xaxis_title='Date')
    candlestick.update_xaxes(
        rangeslider_visible=False,
        rangeselector={'buttons': buttons},
        rangebreaks=[{'bounds': ['Sat', 'Mon']}]
    )
    return line, candlestick

def main():
    st.title('Stock Price Visualiser')
    st.write('The prices are updated every minute. Though not in real time' + \
        ' so please reload the page to see the updated prices.')
    st.write('---')
    ticker_list_asx, ticker_list_nasdaq = setup()
    options = ticker_list_asx + ticker_list_nasdaq
    sq2_idx = 0 if 'SQ2' not in options else options.index('SQ2')
    asx_code = st.sidebar.selectbox('Tickers', options, sq2_idx)
    currency = 'USD'
    if asx_code in ticker_list_asx:
        asx_code += '.AX'
        currency = 'AUD'
    now = datetime.datetime.now()
    info, df = get_data(asx_code, datetime.datetime(now.year, now.month, 
        now.day, now.hour, now.minute))
    col1, col2, = st.columns(2)
    col2.image(info['logo_url'])
    col2_1, col2_2 = st.columns(2)
    col2_1.metric('High', round(df['High'].values[-1], 2))
    col2_1.metric('Low', round(df['Low'].values[-1], 2))
    col2_2.metric('Volume', prettify(df['Volume'].values[-1]))
    col2_2.metric('Market Cap.', millify(info['marketCap'], 2))
    x = round((100 / df['Close'].values[-2]) * (df['Close'].values[-1] \
        - df['Close'].values[-2]), 2)
    col1.metric(info['longName'], f"${round(df['Close'].values[-1], 3)} \
    ({currency})", f"{x}% today")
    line, candlestick = plot_charts(df)
    checkbox = st.checkbox('Show Line Chart')
    if checkbox:
        st.plotly_chart(line, config=PLOT_CONFIG, use_container_width=True)
        st.write('---')
    st.plotly_chart(candlestick, config=PLOT_CONFIG, use_container_width=True)

if __name__ == "__main__":
    main()
