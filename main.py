from trade import Stock, generate_market_open_dates
from pynetwork import Network, Relu, Dense, Sigmoid, mse, mse_prime
from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np
import mplfinance as mpf


def normalization(df):
    df_normalized = df.copy()
    min_max_values = {}
    
    for column in df_normalized.columns:
        min_value = df_normalized[column].min()
        max_value = df_normalized[column].max()
        
        min_max_values[column] = (min_value, max_value)
        
        # Avoid division by zero if all values are the same
        if min_value == max_value:
            df_normalized[column] = 0.0
        else:
            df_normalized[column] = (df_normalized[column] - min_value) / (max_value - min_value)
    
    return df_normalized, min_max_values

def denormalize_dataframe(df_normalized, min_max_values):
    df_denormalized = df_normalized.copy()
    
    for column in df_denormalized.columns:
        min_value, max_value = min_max_values[column]
        
        # Reverse the normalization
        df_denormalized[column] = df_denormalized[column] * (max_value - min_value) + min_value
    
    return df_denormalized


def create_train_data(stock: Stock,
                      start_date: datetime,
                      end_date: datetime,
                      prediction_period: int = 5):
    x_train = []
    y_train = []
    data, max_min_normal = normalization(stock.get_history(start_date, end_date))
    data = np.array(data)
    for i in range(len(data) - prediction_period - 2):
        x_train.append(data[i:i + prediction_period].flatten().T)
        y_train.append(data[i + prediction_period + 1])
    return data.tolist(), x_train, y_train, max_min_normal


def create_generate_data(data, prediction_period):
    x_train = []
    y_train = []
    data = np.array(data)
    for i in range(len(data) - prediction_period - 2):
        x_train.append(data[i:i + prediction_period].flatten().T)
        y_train.append(data[i + prediction_period + 1])
    return data.tolist(), x_train, y_train


def generate_prediction(network, n_predictions, data, x_train, y_train,
                        prediction_period):
    for i in range(n_predictions):
        prediction = network.predict(x_train[-1])
        data.append(prediction.reshape(5).tolist())
        data, x_train, y_train = create_generate_data(data, prediction_period)
    return data


facebook = Stock("META", "facebook")
end = dt.datetime.now() - dt.timedelta(days=1)
start = end - dt.timedelta(weeks=260)
prediction_period = 20
data, x_train, y_train, min_max_normal = create_train_data(facebook, start, end,
                                           prediction_period)
inputs = prediction_period * 5

network = Network([Dense(inputs, 100), Sigmoid(), Dense(100, 5), Sigmoid()], inputs, 5)
network.train(mse,
              mse_prime,
              x_train,
              y_train,
              epochs=10000,
              learning_rate=0.03,
                verbose=True)
print(pd.DataFrame(data))
data = generate_prediction(network, 360, data, x_train, y_train, prediction_period)

date_df = generate_market_open_dates(start, len(data)-1)

df = pd.DataFrame(data,
                  index=date_df,
                  columns=["Open", "High", "Low", "Close", "Volume"])
df = denormalize_dataframe(df, min_max_normal)
mpf.plot(df, type='line')
network.export_network(network, "network.pkl")
