from newsapi import NewsApiClient
from trade import Stock
import numpy as np
import pandas as pd
from datetime import datetime
import json

api_key = "5cad3426d44d4742983615f05663aa60"
newsapi = NewsApiClient(api_key)
sources = 'bloomberg,business-insider,fortune,the-wall-street-journal'
domains = 'bloomberg.com,businessinsider.com,fortune.com,wsj.com'


def get_stock_news(stock: Stock, start_date: datetime, end_date: datetime):
    articles = newsapi.get_everything(q = stock.company_name, domains=domains, sources=sources, from_param=start_date, to=end_date, sort_by="relevancy")
    return articles["articles"]
def get_headlines(stock: Stock, start_date: datetime, end_date: datetime):
    headlines = {"date": [], "headlines": []}
    articles = get_stock_news(stock, start_date, end_date)
    for article in articles:
        date = article["publishedAt"]
        headline = article["title"]
        headlines["headlines"].append(headline)
        headlines["date"].append(date)

    headlines_df = pd.DataFrame(headlines)
    headlines_df["date"] = pd.to_datetime(headlines_df["date"])
    headlines_df = headlines_df.sort_values(by="date")
    return headlines_df
