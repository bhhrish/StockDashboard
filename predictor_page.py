import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd
from models import *
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from plotly.subplots import make_subplots
from utils import *

def train_test_split(df, train_size):
    X = df['Open'].values
    y = df['Close'].values
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return X_train, X_test, y_train, y_test

def plot_actual_predicted(date, y_train, y_train_pred, y_test, y_pred):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Performance',
        'Testing Performance'))
    fig.add_scatter(
        x=date[1],
        y=y_train,
        mode='lines',
        marker={'color': 'lightblue'}, 
        name='Actual',
        showlegend=False,
        row=1, col=1
    )
    fig.add_scatter(
        x=date[1],
        y=y_train_pred,
        mode='lines',
        marker={'color': 'orange'},
        name='Predicted',
        showlegend=False,
        row=1, col=1
    )
    fig.add_scatter(
        x=date[0],
        y=y_test,
        mode='lines',
        name='Actual',
        marker={'color': 'lightblue'},
        row=1, col=2
    )
    fig.add_scatter(
        x=date[0],
        y=y_pred,
        mode='lines',
        name='Predicted',
        marker={'color': 'orange'},
        row=1, col=2
    )
    fig.update_layout(legend={
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.05,
        'xanchor': 'right',
        'x': 1}
    )
    return fig 

def show_data_size(train_size, test_size):
    st.sidebar.markdown("**Data Splits**")
    st.sidebar.write('Train set size')
    st.sidebar.info(train_size)
    st.sidebar.write('Test set size')
    st.sidebar.info(test_size)

def show_metrics(r_sq, mse):
    col1, col2 = st.columns(2)
    col1.write('Coefficient of determination $\left(R^2\\right)$:')
    col1.info(r_sq[0])
    col1.write('Error $(\\text{MSE})$:')
    col1.info(mse[0])
    col2.write('Coefficient of determination $\left(R^2\\right)$:')
    col2.info(r_sq[1])
    col2.write('Error $(\\text{MSE})$:')
    col2.info(mse[1])

def linear_regression(symbol, training_ratio):
    _, df = get_data(symbol)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    train_size = int(training_ratio * len(df))
    X_train, X_test, y_train, y_test = train_test_split(df, train_size)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    date = df['Date'].values[train_size:]
    show_data_size(X_train.shape[0], X_test.shape[0])
    st.subheader('Model Performance')
    st.plotly_chart(plot_actual_predicted((date, 
        df['Date'].values[:train_size]), y_train, 
        model.predict(X_train), y_test, y_pred), use_container_width=True)
    show_metrics((model.score(X_train, y_train), model.score(X_test, y_test)), 
        (mean_squared_error(y_train, model.predict(X_train)), 
        mean_squared_error(y_test, y_pred)))

#@st.cache
def run_decision_tree_regression(X_train, y_train, params):
    model = DecisionTreeRegressor(*params)
    model.fit(X_train, y_train)
    return model

def decision_tree_regressor(symbol, training_ratio):
    _, df = get_data(symbol)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    train_size = int(training_ratio * len(df))
    X_train, X_test, y_train, y_test = train_test_split(df, train_size)
    show_data_size(X_train.shape[0], X_test.shape[0])
    st.subheader('Hyperparameter Tuning')
    st.write('Select the hyperparameters for Decision Tree Regression')
    col1, col2 = st.columns(2)
    step_size = 100 if int(len(str(X_train.shape[0]))) > 3 else 10
    min_samples_split = col1.slider('Minimum samples in interal nodes', 2,
        X_train.shape[0], 2, step_size,
        help='The minimum number of samples required to split an internal ' +
        'node.')
    max_depth = col2.select_slider('Maximum depth of tree',
        list(range(1, X_train.shape[0] + 1, step_size)) + ['None'], 'None',
        help='The maximum depth of the tree. If None, then nodes are ' +
        'expanded until all leaves are pure or until all leaves contain less' +
        ' than the minimum samples in internal nodes.')
    max_depth = float('inf') if max_depth == 'None' else max_depth
    model = run_decision_tree_regression(X_train, 
    y_train, (min_samples_split, max_depth))
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    date = df['Date'].values[train_size:]
    st.subheader('Model Performance')
    st.plotly_chart(plot_actual_predicted((date, 
        df['Date'].values[:train_size]), y_train, y_train_pred, y_test, 
        y_pred), use_container_width=True)
    show_metrics((r2_score(y_train, y_train_pred), r2_score(y_test, y_pred)),
        (mean_squared_error(y_train, y_train_pred), 
        mean_squared_error(y_test, y_pred)))

@st.cache
def run_random_forest_regressor(X_train, y_train, params):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def random_forest_regressor(symbol, training_ratio):
    _, df = get_data(symbol)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    train_size = int(training_ratio * len(df))
    X_train, X_test, y_train, y_test = train_test_split(df, train_size)
    show_data_size(X_train.shape[0], X_test.shape[0])
    st.subheader('Hyperparameter Tuning')
    st.write('Select the hyperparameters for Random Forest Regression')
    col1, col2, col3 = st.columns(3)
    step_size = 100 if int(len(str(X_train.shape[0]))) > 3 else 10
    min_samples_split = col1.slider('Minimum samples in interal nodes', 2, 
        X_train.shape[0], 2, step_size, 
        help='The minimum number of samples required to split an internal ' +
        'node.')
    max_depth = col2.select_slider('Maximum depth of tree',
        list(range(1, X_train.shape[0] + 1, step_size)) + ['None'], 'None', 
        help='The maximum depth of the tree. If None, then nodes are ' +
        'expanded until all leaves are pure or until all leaves contain less' +
        ' than the minimum samples in internal nodes.')
    max_depth = float('inf') if max_depth == 'None' else max_depth
    n_estimators = col3.slider('Number of trees in the forest', 10, 1000, 
        100, 10, help='Number of trees in the forest.')
    st.warning('Random forest regressor may take some time to run depending' +
    ' on the size of the dataset. If time is a concern please decrease the ' +
    'number of trees in the forest. Though this may yield sub-optimal results.'
    )
    if st.button('Run Random Forest Regressor!'):
        model = run_random_forest_regressor(X_train, 
        y_train, (n_estimators, min_samples_split, max_depth))
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        date = df['Date'].values[train_size:]
        st.subheader('Model Performance')
        st.plotly_chart(plot_actual_predicted((date, 
            df['Date'].values[:train_size]),
            y_train, y_train_pred, y_test, y_pred),
            use_container_width=True)
        show_metrics((r2_score(y_train, y_train_pred), 
            r2_score(y_test, y_pred)),
            (mean_squared_error(y_train, y_train_pred), 
            mean_squared_error(y_test, y_pred)))

def svr(symbol, training_ratio):
    _, df = get_data(symbol)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    train_size = int(training_ratio * len(df))
    X_train, X_test, y_train, y_test = train_test_split(df, train_size)
    show_data_size(X_train.shape[0], X_test.shape[0])
    st.subheader('Hyperparameter Tuning')
    st.write('The radial basis function (RBF) is used as the kernel function.')
    st.write('Select the hyperparameters for Support Vector Regression')
    col1, col2 = st.columns(2)
    C = col1.slider('C', 0.1, 100.0, 100.0, help='C adds a penalty for each ' +
        'misclassified data point.')
    gamma = col2.slider('Gamma', 0.01, 10.0, 0.1, help='Gamma of RBF determi' +
        'nes the distance of influence of a single training point.')
    model = SVR(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train.reshape(-1, 1), y_train)
    y_pred = model.predict(X_test.reshape(-1, 1))
    y_train_pred = model.predict(X_train.reshape(-1, 1))
    date = df['Date'].values[train_size:]    
    st.subheader('Model Performance')
    st.plotly_chart(plot_actual_predicted((date, 
        df['Date'].values[:train_size]),
        y_train, y_train_pred, y_test, y_pred),
        use_container_width=True)
    show_metrics((r2_score(y_train, y_train_pred), r2_score(y_test, y_pred)),
        (mean_squared_error(y_train, y_train_pred), 
        mean_squared_error(y_test, y_pred)))

def knn_regressor(symbol, training_ratio):
    _, df = get_data(symbol)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    train_size = int(training_ratio * len(df))
    X_train, X_test, y_train, y_test = train_test_split(df, train_size)
    show_data_size(X_train.shape[0], X_test.shape[0])
    st.subheader('Hyperparameter Tuning')
    st.write('Select the hyperparameter for k-Nearest Neighbour Regression')
    mses = []
    for i in range(1, 51):
        model = KNeighborsRegressor(i)
        fitted = model.fit(X_train, y_train)
        y_pred = model.predict(X_test) 
        mses.append(mean_squared_error(y_test, y_pred))
    k = np.argmin(mses) + 1
    k = st.slider('k', 1, 50, int(k), help='k determines the number of neare' +
    'st neighbours to consider in the "voting" process.' + f'k = {int(k)}' +
    'minimises the mean squared error.')
    model = KNeighborsRegressor(k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train.reshape(-1, 1))
    date = df['Date'].values[train_size:]    
    st.subheader('Model Performance')
    st.plotly_chart(plot_actual_predicted((date, 
        df['Date'].values[:train_size]),
        y_train, y_train_pred, y_test, y_pred),
        use_container_width=True)
    show_metrics((r2_score(y_train, y_train_pred), r2_score(y_test, y_pred)),
        (mean_squared_error(y_train, y_train_pred), 
        mean_squared_error(y_test, y_pred)))

def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i: i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

@st.cache
def run_lstm(df, train_size):
    train_data = df['Close'].values[:train_size]
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = df['Close'].values[train_size:]
    test_data = test_data.reshape(-1, 1)
    test_data = scaler.transform(test_data)
    time_steps = 30 if test_data.shape[0] > 30 else int(test_data.shape[0] / 4)
    X_train, y_train = create_dataset(train_data, time_steps)
    X_train = np.reshape(X_train, (X_train.shape[0], time_steps, 1))
    model = keras.Sequential()
    model.add(layers.LSTM(150, return_sequences=False, input_shape=(
        X_train.shape[1], 1), activation='linear'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1, activation='linear'))
    model.add(layers.Activation('ReLU'))
    model.compile(optimizer='adam', loss='mean_squared_error', 
        metrics=['accuracy'])
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5,
        min_delta=1e-4
    )
    history = model.fit(X_train, y_train, epochs=50, batch_size=500, 
        validation_split=0.25, callbacks=[early_stopping], verbose=0)
    y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
    X_train = np.reshape(X_train, (X_train.shape[0], time_steps, 1))
    y_train_pred = scaler.inverse_transform(model.predict(X_train))
    X_test, y_test = create_dataset(test_data, time_steps)
    y_test = y_test.reshape(-1, 1)
    y_true = scaler.inverse_transform(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], time_steps, 1))
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    return (y_train_true, y_train_pred, y_true, y_pred, 
        history.history['loss'], history.history['val_loss'], time_steps)

def lstm(symbol, training_ratio):
    _, df = get_data(symbol)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    train_size = int(training_ratio * len(df))
    show_data_size(train_size, len(df) - train_size)
    if train_size < 100:
        st.warning('The current dataset is too small for LSTM to yield ' +
        'optimal results!')
    st.subheader('Hyperparameters')
    st.markdown('**Note:** hyperparameter tuning has been disabled as it is '
        + 'computationally expensive!')
    col1, col2, col3 = st.columns(3)
    col1.slider('Number of units in hidden LSTM layer', 1, 500, 150, 
        help='Number of neurons in the single LSTM layer', disabled=True)
    col2.slider('Number of units in Dense layer.', 1, 100, 25, 
        help='A dense layer is a layer where each neuron receives input from' +
        ' all neurons in the previous layer.', disabled=True)
    col3.slider('Dropout layer value', 0.1, 1.0, 0.2, help='The LSTM layer i' +
    's accompanied by a dropout layer, which is regularlisation technique ' + 
    ' to reduce overfitting.', disabled=True)
    col1, col2 = st.columns(2)
    col1.slider('Number of epochs', 1, 100, 50, help='The number of complete' +
    ' iterations of the dataset to be run.', disabled=True)
    col2.slider('Batch size', 1, train_size, 500, help='The number of sample' +
    's that will be passed through to the network at one time.', disabled=True)
    y_train, y_train_pred, y_true, y_pred, loss, val_loss, time_steps = \
        run_lstm(df, train_size)
    _, test_date = create_dataset(df['Date'].values[:train_size].
        reshape(-1, 1), time_steps)
    _, train_date = create_dataset(df['Date'].values[train_size:].
        reshape(-1, 1), time_steps)
    st.subheader('The Network')
    st.write('The network contains a single LSTM layer that is accompanied b' +
    'y a dropout layer, and two dense layers.')
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(loss) + 1)),
        y=loss,
        mode='lines',
        name='Train',
        hovertemplate=None

    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(loss) + 1)),
        y=val_loss,
        mode='lines',
        name='Test',
        hovertemplate=None
    ))
    fig.update_layout(hovermode='x unified')
    st.subheader('Model Performance')
    st.write('The number of epochs is set to 50, though early stopping ' +
    '(with patience = 5) has been implemented to reduce overfitting. We ' +
    f'can see from the plot below that the model only took {len(loss)} epochs!')
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(plot_actual_predicted((train_date, test_date), 
        y_train.flatten(), y_train_pred.flatten(), y_true.flatten(), 
        y_pred.flatten()), use_container_width=True)
    show_metrics((r2_score(y_train, y_train_pred), r2_score(y_true, y_pred)),
        (mean_squared_error(y_train, y_train_pred), 
        mean_squared_error(y_true, y_pred)))

def main():
    st.title('Predictor page title')
    st.write('Please hide the sidebar to see better ("unsquished") plots.')
    ticker_list_asx, ticker_list_nasdaq = setup()
    symbol = st.sidebar.selectbox('Codes', ticker_list_asx + 
        ticker_list_nasdaq)
    if symbol in ticker_list_asx:
        symbol += '.AX'
    model = st.sidebar.selectbox('Models', ['Linear Regression', 
                'Decision Tree Regression', 'Random Forest Regression', 
                'Support Vector Regression', 'k-Nearest Neighbour Regression', 
                'LSTM'])
    training_ratio = st.sidebar.slider('Training size ratio', 5, 95, 85, 1)
    models = {
       'Linear Regression': linear_regression,
       'Decision Tree Regression': decision_tree_regressor,
       'Random Forest Regression': random_forest_regressor,
       'Support Vector Regression': svr,
       'k-Nearest Neighbour Regression': knn_regressor,
       'LSTM': lstm
    }
    models[model](symbol, training_ratio / 100)    

if __name__ == "__main__":
    main()