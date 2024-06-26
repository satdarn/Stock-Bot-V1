from trade import Stock
from pynetwork import Network, Relu, Dense, normalization, mse, mse_prime
from datetime import datetime
import datetime as dt
import numpy as np

vectorized_normalization = np.vectorize(normalization)

def create_train_data(stock: Stock,
                      start_date: datetime,
                      end_date: datetime,
                      prediction_period: int = 5):
    x_train = []
    y_train = []
    data = np.array(stock.get_history(start_date, end_date))
    for i in range(len(data) - prediction_period-2):
        x_train.append(vectorized_normalization(data[i:i+prediction_period].flatten().T))
        y_train.append(vectorized_normalization(data[i+prediction_period+1]))
    return x_train, y_train
 
tesla = Stock("TSLA", "tesla")  
end = dt.datetime.now() - dt.timedelta(days=1)    
start = end - dt.timedelta(weeks=3) 
prediction_period = 4
x_train, y_train = create_train_data(tesla, start, end, prediction_period)
inputs = prediction_period * 4

network = Network([Dense(inputs, 50), Relu(), Dense(50, 30), Relu(), Dense(30,4), Relu()], inputs, 4)

network.train(mse, mse_prime, x_train, y_train, verbose = True)