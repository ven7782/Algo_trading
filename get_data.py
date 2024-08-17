import yfinance as yf
import pandas as pd

tickers = ['NVDA', 'QCOM']

def get_data(tickers):
    stock_data = {}
    for ticker in tickers:
        # df = yf.download(ticker, start="2024-08-01", end="2024-08-02", interval="1m")
        df = yf.download(ticker, start="2007-01-01", end="2020-05-08")
        stock_data[ticker] = df
    return stock_data

stock_data = get_data(tickers)

for ticker, df in stock_data.items():
    df.to_csv(f'data/{ticker}.csv')