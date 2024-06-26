from news import get_stock_news
from trade import Stock
from datetime import datetime
from transformers import pipeline

pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def get_sentiment(stock: Stock, start:datetime, end:datetime):
    sentiment = []
    articles = get_stock_news(stock, start, end)
    sentiment_analysis = pipeline("sentiment-analysis")
    for article in articles:
        sentiment = sentiment_analysis(article["title"])
        sentiment.append(sentiment[0]["label"])
