import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from millify import prettify

TICKERS = [
    [
        (yf.Ticker('^AXJO').history(period='1mo')).reset_index(), 
        (yf.Ticker('^GSPC').history(period='1mo')).reset_index()
    ], 
    [
        (yf.Ticker('^IXIC').history(period='1mo')).reset_index(), 
        (yf.Ticker('^DJI').history(period='1mo')).reset_index()
    ]
]
    
TITLES = (
    ('S&P/ASX 200', 'S&P 500'), 
    ('NASDAQ Composite', 'Dow Jones Industrial Average')
)

PLOT_CONFIG = {
    'modeBarButtonsToRemove': ['zoom', 'select', 'zoomIn', 'zoomOut', 
                                'autoScale', 'resetScale'], 
    'displaylogo': False, 
    'scrollZoom': True,
}

st.cache(allow_output_mutation=True)
def plot_indexes(): 
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        TITLES[i][j] for i in range(2) for j in range(2)])
    for i in range(2):
        for j in range(2):
            fig.add_trace(
                go.Scatter(
                        x=TICKERS[i][j]['Date'],
                        y=TICKERS[i][j]['Close'],
                        mode='lines',
                        marker={'color': '#EA4335'} if \
                        TICKERS[i][j]['Close'].values[0] >
                        TICKERS[i][j]['Close'].values[-1] \
                            else {'color': '#39AA57'},
                        hovertemplate='Price: $%{y:.2f} at ' +
                            '%{x} <extra></extra>'
                                
                ),
                row=i + 1, col=j + 1
            )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        rangebreaks=[{'bounds': ['Sat', 'Mon']}],
        showspikes=True
    )

    fig.update_yaxes(showspikes=True)
    return fig

def main():
    st.title('Stock Price Dashboard')
    for i in range(2):
        col1, col2 = st.columns(2)
        for j in range(2):
            col_to_use = col1 if j == 0 else col2
            x = round((100 / TICKERS[i][j]['Close'].values[-2]) * \
                (TICKERS[i][j]['Close'].values[-1] - TICKERS[i][j]['Close'].\
                    values[-2]), 2)
            col_to_use.metric(TITLES[i][j], 
            f"${prettify(round(TICKERS[i][j]['Close'].values[-1], 3))}", 
            f"{x}% today")

    if st.checkbox('Show 1 month line chart'):
        st.plotly_chart(plot_indexes(), config=PLOT_CONFIG, use_container_width=True)

if __name__ == "__main__":
    main()