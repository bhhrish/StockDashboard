import streamlit as st
import yfinance as yf
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

@st.cache
def setup():
    ticker_list = pd.read_csv(ASX_TICKERS_URL)
    ticker_list_asx = ticker_list['ASX code'].tolist()
    ticker_list = pd.read_csv(NASDAQ_TICKERS_URL)
    ticker_list_nasdaq = ticker_list['Symbol'].tolist()
    return ticker_list_asx, ticker_list_nasdaq

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_data(code, time=None):
    ticker = yf.Ticker(code)
    logo_url = ticker.info['logo_url']
    df = ticker.history(period='max')
    df.reset_index(inplace=True)
    return ticker.info, df