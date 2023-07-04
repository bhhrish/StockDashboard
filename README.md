# Stock Price Dashboard

This is a simple dashboard to visualize and predict the stock prices of all the ASX and NASDAQ-listed companies. The up-to-date data is downloaded from the Yahoo! Finance API upon request. The interactive charts were created using Plotly, and the prediction was done using various machine learning models including:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Support Vector Regression
* k-Nearest Neighbor Regression
* Long Short-Term Memory Network

Most of the models were implemented from scratch.

## Installation

To run the dashboard, follow these steps:

1. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
2. Update the web driver to the latest version compatible with your system and place it in the `drivers` directory.
3. Set the `USE_CHROME` variable to `False` in the `visualiser_page.py` file if you are not using the Google Chrome browser.
4. Run the following command to start the dashboard:
  ```
  streamlit run main.py
  ```

## Usage
Once the dashboard is running, follow these steps to visualize and predict stock prices:

1. Select a company's stock from the available options.
2. Choose a prediction model from the provided list.
3. Explore the interactive charts to analyze historical data and visualize the predicted stock prices.

![App Demo](demo/streamlit-main-2023-07-04-13-07-20.gif)
