from newsapi import NewsApiClient
from trade import Stock
from trade import tesla, start, end
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
    articles = get_stock_news(stock, start_date, end_date)
    return articles["articles"]