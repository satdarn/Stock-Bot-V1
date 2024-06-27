import datetime as dt
import pandas as pd
import yfinance as yf
import mplfinance as mpf

end = dt.datetime.now() - dt.timedelta(days=1)
start = end - dt.timedelta(
    days=8
)  #this is will give the last weekdays - today so 8 will give the last 5 weekdays


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
        history = self.ticker.history(start=start_date, end=end_date)
        history.drop(['Dividends', "Stock Splits"], axis=1, inplace=True)
        return history


def generate_market_open_dates(start_date: dt.datetime, days_later: int):
    current_date = start_date
    market_open_dates = [start_date]
    days_added = 0
    while days_added < days_later:
        current_date += dt.timedelta(days=1)
        if current_date.weekday() in range(5):  # 0 (Monday) to 4 (Friday)
            market_open_dates.append(current_date)
            days_added += 1
    return market_open_dates


if __name__ == "__main__":
    tesla = Stock("TSLA", "tesla")
    print(tesla.history)
