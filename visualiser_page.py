import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import yfinance as yf
from millify import millify, prettify
from utils import *
import pytz
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

USE_CHROME = True

@st.cache_data
def plot_chart(df, name, day_data_len, chart_type="line"):
    buttons = [
        {"count": day_data_len * 5, "label": "1D", "step": "minute", "stepmode": "backward"},
        {"count": 5, "label": "5D", "step": "day", "stepmode": "backward"},
        {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
        {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
        {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
        {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
        {"label": "Max", "step": "all"},
    ]

    start = df["Date"].values[0]
    end = df["Date"].values[len(df) - 1]
    time_diff = end - start
    if time_diff < pd.Timedelta(5 * 365, "d"):
        buttons[5]["visible"] = False
    elif time_diff < pd.Timedelta(365, "d"):
        buttons[5]["visible"] = False
    elif time_diff < pd.Timedelta(6 * 30, "d"):
        buttons[3]["visible"] = False
    elif time_diff < pd.Timedelta(31, "d"):
        buttons[2]["visible"] = False
    elif time_diff < pd.Timedelta(5, "d"):
        buttons[1]["visible"] = False

    if chart_type == "line":
        chart = go.Figure(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                marker={"color": "black"},
                hovertemplate="Price: $%{y:.2f} at " + "%{x} <extra></extra>",
                name=name,
            )
        )
        chart.update_xaxes(showspikes=True)
        chart.update_yaxes(showspikes=True)
    else:
        chart = go.Figure(
            data=[
                go.Candlestick(
                    x=df["Date"], open=df["Open"], close=df["Close"], high=df["High"], low=df["Low"]
                )
            ]
        )

    chart.update_layout(
        yaxis_title="Stock price", xaxis_title="Date", margin={"t": 30}, hovermode="x unified"
    )
    chart.update_xaxes(
        rangeslider_visible=True,
        rangeselector={"buttons": buttons},
        rangebreaks=[{"bounds": ["Sat", "Mon"]}],
    )

    return chart


@st.cache_data
def plot_income_stmt(code, _ticker):
    income_stmt = _ticker.income_statement()
    revenue = income_stmt["TotalRevenue"].values[-2]
    cor = income_stmt["CostOfRevenue"].values[-2]
    gross_profit = income_stmt["GrossProfit"].values[-2]
    income = income_stmt["NetIncome"].values[-2]
    expenses = gross_profit - income
    fig = go.Figure(
        go.Bar(
            x=["Revenue", "Cost of Revenue", "Gross Profit", "Other Expenses", "Earnings"],
            y=[revenue, cor, gross_profit, expenses, income],
            base=[0, gross_profit, 0, income],
            text=[
                "$" + millify(revenue, 2),
                "$" + millify(cor, 2),
                "$" + millify(gross_profit, 2),
                "$" + millify(expenses, 2),
                "$" + millify(income, 2),
            ],
            textposition="outside",
            outsidetextfont={"color": "black", "size": 14},
            marker={"color": ["#2394DF", "#E64141", "#2DC97E", "#E64141", "#71E7D6"]},
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=0),
        xaxis=dict(fixedrange=True),
        yaxis=dict(showgrid=False, fixedrange=True),
    )
    return fig


def plot_treemap(*args, type="assets"):
    asset_labels = [
        "Assets",
        "Physical Assets",
        "Receivables",
        "Cash & Short term Investments",
        "Long term & Other Assets",
        "Inventory",
    ]
    fig = go.Figure(
        go.Treemap(
            labels=asset_labels
            if type == "assets"
            else ["Liabilities + Equity", "Equity", "Accounts Payable", "Debt"],
            parents=[""] + (["Assets"] * 5 if type == "assets" else ["Liabilities + Equity"] * 3),
            values=[sum(args), *args],
            marker_colors=["white"]
            + (["#2DC97E"] * 5 if type == "assets" else (["#2DC97E"] * 2) + ["#E64141"]),
            branchvalues="total",
            textinfo="text+label",
            hoverinfo="text+label",
            outsidetextfont={"color": "black", "size": 15},
            text=[""] + ["$" + millify(i, 2) for i in args],
        )
    )

    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    return fig


@st.cache_data
def fetch_past_financials(code, ipo_year):
    if USE_CHROME:
        DRIVER_PATH = "./drivers/chromedriver.exe"
        op = webdriver.ChromeOptions()
        op.add_argument("headless")
        driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=op)
    else:
        DRIVER_PATH = "./drivers/geckodriver.exe"
        op = webdriver.FirefoxOptions()
        op.add_argument("--headless")
        driver = webdriver.Firefox(executable_path=DRIVER_PATH, options=op)
    driver.maximize_window()
    driver.get(f"https://quickfs.net/company/{code}")
    wait = WebDriverWait(driver, 60)
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="ovr-table"]/tbody')))
    raw_table = driver.find_element(By.XPATH, '//*[@id="ovr-table"]/tbody').text.split("\n")
    driver.quit()

    year_index = raw_table[0].split(" ")
    data = {}
    year_len = (
        10
        if datetime.datetime.today().year - ipo_year + 1 > 10
        else len(year_index)
    )
    for i in raw_table[1:]:
        temp = i.rsplit(" ", maxsplit=year_len)
        data[temp[0]] = temp[-year_len:]

    data = pd.DataFrame(data, index=year_index[-year_len:])
    for cols in data.columns:
        data[cols] = data[cols].apply(
            lambda x: "".join(i for i in x if (i.isalnum() and i != "A") or i == "." or i == "-")
        )
        data[cols] = data[cols].apply(lambda x: np.nan if x == "-" else x)
        if cols == "Revenue" or cols == "Gross Profit" or cols == "Operating Profit":
            try:
                data[cols] = data[cols].apply(lambda x: int(x) * 1000000)
            except ValueError:
                pass

    return data


@st.cache_data
def plot_balance_sheet(code, _ticker):
    balance_sheet = _ticker.balance_sheet()
    physical_assets = balance_sheet["NetPPE"].tolist()[-1]
    receivables = balance_sheet["Receivables"].tolist()[-1]
    inventory = balance_sheet["Inventory"].tolist()[-1]
    cash_and_short_term = balance_sheet["CashCashEquivalentsAndShortTermInvestments"].tolist()[-1]
    long_term = balance_sheet["TotalNonCurrentAssets"].tolist()[-1]

    equity = balance_sheet["StockholdersEquity"].tolist()[-1]
    payable = balance_sheet["AccountsPayable"].tolist()[-1]
    debt = balance_sheet["LongTermDebt"].tolist()[-1] + (
        balance_sheet["CurrentDebt"].tolist()[-1]
        if "CurrentDebt" in balance_sheet.columns and balance_sheet["CurrentDebt"].tolist()[-1] != np.nan
        else 0
    )



    return [
        plot_treemap(physical_assets, receivables, cash_and_short_term, long_term, inventory),
        plot_treemap(equity, payable, debt, type="liabilities"),
    ]


def main():
    st.title("Stock Price Visualiser")
    st.write(
        "The prices are updated every minute. Though not in real time"
        + " so please reload the page to see the updated prices."
    )
    st.write("---")
    ticker_list_asx, ticker_list_nasdaq = setup()
    options = ticker_list_asx + ticker_list_nasdaq
    sq2_idx = 0 if "DMP" not in options else options.index("DMP")
    code = st.sidebar.selectbox("Tickers", options, sq2_idx)
    currency = "USD"
    if code in ticker_list_asx:
        code += ".AX"
        currency = "AUD"
        now = datetime.datetime.now(pytz.timezone("Australia/Sydney"))
        is_open = (now.hour > 9 and now.hour < 16) or (now.hour == 16 and now.minute <= 10)
    else:
        now = datetime.datetime.now(pytz.timezone("US/Eastern"))
        is_open = (now.hour == 9 and now.minute >= 30) or (now.hour > 9 and now.hour < 17)
    if bool(len(pd.bdate_range(now, now))) and is_open:
        _time = time.time()
    else:
        _time = None
    ticker, df, day_data = get_data(code, _time)
    info = ticker.summary_detail
    (
        col1,
        col2,
    ) = st.columns(2)
    col2.image(f"https://logo.clearbit.com/{ticker.summary_profile[code]['website'].split('www.')[-1]}")
    col2_1, col2_2 = st.columns(2)
    col2_1.metric("High", round(df["High"].values[-1], 2))
    col2_1.metric("Low", round(df["Low"].values[-1], 2))
    col2_2.metric("Volume", prettify(df["Volume"].values[-1]))
    col2_2.metric("Market Cap.", millify(info[code]["marketCap"], 2))
    x = round((100 / df["Close"].values[-2]) * (df["Close"].values[-1] - df["Close"].values[-2]), 2)
    diff = round(df["Close"].values[-1] - df["Close"].values[-2], 3)
    col1.metric(
        code.split(".AX")[0],
        f"${round(df['Close'].values[-1], 3)} \
    ({currency})",
        f"{'+' if diff > 0 else ''}{diff} ({x}%)",
    )

    with st.expander("View Business Summary"):
        st.write(ticker.summary_profile[code]["longBusinessSummary"])

    line, candlestick = plot_chart(
        pd.concat([df.drop(df.index[-1]), day_data]), code if ".AX" not in code else code[:-3], len(day_data)
    ), plot_chart(df, None, len(day_data), "candlestick")

    line_chart_tab, candlestick_chart_tab = st.tabs(["Line Chart", "Candle Stick Chart"])
    with line_chart_tab:
        code_idx = options.index(code if ".AX" not in code else code[:-3])
        companies = st.multiselect(
            "Compare to", options[:code_idx] + options[code_idx + 1 :], max_selections=5
        )
        colours = ["#FF8C00", "#0072C6", "#7030A0", "#008080", "#F2AF00"]
        for c, i in enumerate(companies):
            _, df_new, day_data_new = get_data(i if i in ticker_list_nasdaq else i + ".AX", time)
            df_new = pd.concat([df_new, day_data_new])
            line.add_trace(
                go.Scatter(
                    x=df_new["Date"],
                    y=df_new["Close"],
                    mode="lines",
                    marker={"color": colours[c]},
                    hovertemplate="Price: $%{y:.2f} at " + "%{x} <extra></extra>",
                    name=i,
                )
            )

        st.plotly_chart(line, config=PLOT_CONFIG, use_container_width=True)
    with candlestick_chart_tab:
        st.plotly_chart(candlestick, config=PLOT_CONFIG, use_container_width=True)

    st.subheader("Earnings & Revenue")
    try:
        st.plotly_chart(plot_income_stmt(code, ticker), config=PLOT_CONFIG, use_container_width=True)
    except KeyError:
        st.error("Unable to find data! :disappointed:")

    st.subheader("Balance Sheet")
    col1, col2 = st.columns(2)
    try:
        asset, liability = plot_balance_sheet(code, ticker)
        with col1:
            st.plotly_chart(asset, config=PLOT_CONFIG, use_container_width=True)
        with col2:
            st.plotly_chart(liability, config=PLOT_CONFIG, use_container_width=True)
    except KeyError:
        st.error("Unable to find data! :disappointed:")

    data = fetch_past_financials(
        code if ".AX" not in code else code[:-3] + ":" + "AU" if ".AX" in code else "US",
        pd.to_datetime(df["Date"].values[0]).year,
    )
    st.subheader("Past Performance")
    fig = go.Figure(
        go.Scatter(
            x=data.dropna().index,
            y=data["Revenue Growth"].dropna(),
            mode="lines+markers",
            line={"shape": "spline", "smoothing": 1.1},
            hovertemplate="Revenue Growth: %{y: .2f}%<extra></extra>",
            hoverlabel={"bgcolor": "black", "font_size": 13, "font": {"color": "white"}},
        )
    )

    fig.update_layout(
        xaxis=dict(tickvals=data.index, fixedrange=True),
        title={"text": "Revenue Growth", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        margin={"t": 30},
        yaxis_ticksuffix="%",
        xaxis_title="Year",
        yaxis_title="Growth",
        yaxis=dict(fixedrange=True),
    )

    st.plotly_chart(fig, config=PLOT_CONFIG, use_container_width=True)

    fig = go.Figure(
        go.Bar(
            x=data.dropna().index,
            y=data["Revenue"].dropna(),
            name="Revenue",
            hoverinfo="y+name",
            hoverlabel={"bgcolor": "black", "font_size": 13, "font": {"color": "white"}},
        )
    )

    fig.add_trace(
        go.Bar(
            x=data.dropna().index,
            y=data["Gross Profit"].dropna(),
            name="Gross Profit",
            hoverinfo="y+name",
            hoverlabel={"bgcolor": "black", "font_size": 13, "font": {"color": "white"}},
        )
    )

    fig.add_trace(
        go.Bar(
            x=data.dropna().index,
            y=data["Operating Profit"].dropna(),
            name="Operating Profit",
            hoverinfo="y+name",
            hoverlabel={"bgcolor": "black", "font_size": 13, "font": {"color": "white"}},
        )
    )

    fig.update_layout(
        xaxis=dict(tickvals=data.index, fixedrange=True),
        title={
            "text": "Revenue & Operating Income",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin={"t": 30},
        yaxis_tickprefix="$",
        xaxis_title="Year",
        yaxis_title="Amount",
        yaxis=dict(fixedrange=True),
    )

    st.plotly_chart(fig, config=PLOT_CONFIG, use_container_width=True)

    fig = go.Figure(
        go.Scatter(
            x=data.dropna().index,
            y=data["Gross Margin %"].dropna(),
            mode="lines+markers",
            name="Gross Margin",
            line={"shape": "spline", "smoothing": 1.1},
            hovertemplate="Gross Margin: %{y: .2f}%",
            hoverlabel={"bgcolor": "black", "font_size": 13, "font": {"color": "white"}},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.dropna().index,
            y=data["Operating Margin %"].dropna(),
            mode="lines+markers",
            name="Operating Margin",
            line={"shape": "spline", "smoothing": 1.1},
            hovertemplate="Operating Margin: %{y: .2f}%",
            hoverlabel={"bgcolor": "black", "font_size": 13, "font": {"color": "white"}},
        )
    )

    fig.update_layout(
        xaxis=dict(tickvals=data.index, fixedrange=True),
        title={"text": "Margins", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        margin={"t": 30},
        yaxis_ticksuffix="%",
        xaxis_title="Year",
        yaxis_title="Amount",
        yaxis=dict(fixedrange=True),
    )

    st.plotly_chart(fig, config=PLOT_CONFIG, use_container_width=True)

    fig = go.Figure(
        go.Bar(
            x=data.dropna().index,
            y=data["Earnings Per Share"].dropna(),
            hovertemplate="EPS: $%{y:.2f}<extra></extra>",
            hoverlabel={"bgcolor": "black", "font_size": 13, "font": {"color": "white"}},
            marker={
                "color": [
                    "#2DC97E" if float(data["Earnings Per Share"][i]) > 0 else "#E64141"
                    for i in range(len(data.index))
                ]
            },
        )
    )

    fig.update_layout(
        xaxis=dict(tickvals=data.index, fixedrange=True),
        title={"text": "Earnings Per Share", "y": 0.99, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        margin={"t": 30},
        yaxis_tickprefix="$",
        xaxis_title="Year",
        yaxis_title="Amount",
        yaxis=dict(fixedrange=True),
    )

    st.plotly_chart(fig, config=PLOT_CONFIG, use_container_width=True)


if __name__ == "__main__":
    main()
