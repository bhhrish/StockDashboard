import streamlit as st
import yfinance as yf
import yahooquery as yq
import pandas as pd


PLOT_CONFIG = {
    'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 
                                'autoScale', 'resetScale'], 
    'displaylogo': False,
    'scrollZoom': True
}

ASX_TICKERS_URL = 'https://asx.api.markitdigital.com/asx-research/1.0/comp' + \
'anies/directory/file?access_token=83ff96335c2d45a094df02a206a39ff4.csv'

NASDAQ_TICKERS_URL = 'https://pkgstore.datahub.io/core/nasdaq-listings/nas' + \
'daq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv'

@st.cache_data
def setup():
    ticker_list = pd.read_csv(ASX_TICKERS_URL)
    ticker_list_asx = ticker_list['ASX code'].tolist()
    ticker_list = pd.read_csv(NASDAQ_TICKERS_URL)
    ticker_list_nasdaq = ticker_list['Symbol'].tolist()
    return [ticker_list_asx, ticker_list_nasdaq]

#@st.cache(allow_output_mutation=True, show_spinner=False)
@st.cache_data
def get_data(code, time):
    ticker = yf.Ticker(code)
    df = ticker.history(period='max')
    df.reset_index(inplace=True)
    day_data = ticker.history(period='1d', interval='5m').reset_index().rename({'Datetime': 'Date'}, axis=1)
    return [yq.Ticker(code), df, day_data]