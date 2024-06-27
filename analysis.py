from news import get_headlines
from trade import Stock, start, end
from datetime import datetime
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def get_sentiment(stock: Stock, start:datetime, end:datetime):
    sentiments = []
    stock_headlines_df = get_headlines(stock, start, end)
    for headline in stock_headlines_df["headlines"]:
        encoded_text  = tokenizer(headline, return_tensors="pt")
        out = model(**encoded_text)
        scores = out[0][0].detach().numpy()
        scores = [scores[0], scores[2]]
        scores = softmax(scores)
        sentiment = scores[1] - scores[0]
        sentiments.append(sentiment)
    stock_headlines_df["sentiment"] = sentiments
    return stock_headlines_df
        
tesla = Stock("TSLA", "tesla")
print(get_sentiment(tesla, start, end))


