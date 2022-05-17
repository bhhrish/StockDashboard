# Stock Price Dashboard

This is a simple dashboard to visualise and predict the stock prices of all of the ASX and NASDAQ-listed companies. The up-to-date data are downloaded from Yahoo! Finance API, upon request. The interactive charts were created using Plotly and the prediction was done using various machine learning models including: 
* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Support Vector Regression 
* k-Nearest Neighbour Regression
* Long Short Term Memory Network

Most of the models were implemented from scratch.

### Requirements:
* Streamlit for the front end. Dashboard is created using version 1.8.1
* Yfinance for downloading the data. Dashboard is created using version 0.1.7
* Pandas for data manipulation. Dashboard is created using version 1.3.5
* NumPy for performing a wide variety of mathematical operations on arrays. Dashboard is created using version 1.22.0
* Scikit-learn for building/training the SVR model. Dashboard is created using version 1.0.2
* Tensorflow for building/training the LSTM network. Dashboard is created using version 2.8.0

To run the dashboard, type:
```
streamlit run main.py
```
