# -*- coding: utf-8 -*-
"""
stockPred.py
imported from Google Colabratory
"""

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report

ticker = 'AAPL'
data = yf.download(ticker, '2016-01-01', '2023-12-10')

newsapi_key = 'cdad8cdb014e49a49c904729329ab195'

horizons = {5, 25, 50, 100, 200}
for h in horizons:
    data[f'{h}MA'] = data['Open'].rolling(window=h).mean()

data = data.dropna()

data['Future_Close'] = data['Close'].shift(-30).copy()
data = data.copy()
data['Target'] = (data['Future_Close'] > data['Close']).astype(int)
data = data.dropna(subset=['Future_Close'])

predictors = ['50MA', '100MA', '5MA', '200MA', '25MA', 'Volume', 'High']
data = data.drop(columns=['Adj Close', 'Future_Close'])

train_proportion = 0.8
train_size = int(len(data) * train_proportion)
forTest = data[predictors]

predstrain, ptest = forTest[:train_size], forTest[train_size:]
targtrain, preal = data['Target'][:train_size], data['Target'][train_size:]

model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(predstrain, targtrain)
predictions = model.predict(ptest)
accuracy = accuracy_score(preal, predictions)
classification = classification_report(preal, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification)
