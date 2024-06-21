import datetime as dt
import pandas as pd
import yfinance as yf  
import mplfinance as mpf



end = dt.datetime.now()
start = end - dt.timedelta(days=10)

class Stock:
    def __init__(self, ticker: str, company_name: str):
        self.ticker = yf.Ticker(ticker)
        self.company_name = company_name
        self.amount = 0
        self.history = self.get_history(start, end)

    def buy(self, amount: float):
        self.amount += amount

    def sell(self, amount: float):
        self.amount -= amount
    
    def get_history(self, start_date: dt.datetime, end_date: dt.datetime):
        history = self.ticker.history(start = start_date, end = end_date)
        history.drop(['Volume', 'Dividends', "Stock Splits"], axis=1, inplace=True)
        return history
    
tesla = Stock("TSLA", "tesla")