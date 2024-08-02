import yfinance as yf
import pandas as pd
import talib

pd.options.mode.chained_assignment = None  # default='warn'
# pd.set_option('display.max_rows', None)

# tickers = ['NVDA', 'QCOM']
tickers = ['NVDA']

# Get the data from the CSV files
stock_data = {}
for ticker in tickers:
    df = pd.read_csv(f'data/{ticker}.csv', index_col='Date', parse_dates=True)
    stock_data[ticker] = df


# split the data into training, validation and test sets
training_data_time_range = ('2009-01-01', '2015-12-31')
validation_data_time_range = ('2016-01-01', '2016-12-31')
test_data_time_range = ('2017-01-01', '2020-05-08')

# split the data into training, validation and test sets
training_data = {}
validation_data = {}
test_data = {}

for ticker, df in stock_data.items():
    training_data[ticker] = df.loc[training_data_time_range[0]:training_data_time_range[1]]
    validation_data[ticker] = df.loc[validation_data_time_range[0]:validation_data_time_range[1]]
    test_data[ticker] = df.loc[test_data_time_range[0]:test_data_time_range[1]]

# print shape of training, validation and test data
ticker = 'NVDA'
print(f'Training data shape for {ticker}: {training_data[ticker].shape}')
print(f'Validation data shape for {ticker}: {validation_data[ticker].shape}')
print(f'Test data shape for {ticker}: {test_data[ticker].shape}')

# Display the first 5 rows of the data
print(stock_data['NVDA'].head())
print("###################")

def add_technical_indicators(df):
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['RSI'] = talib.RSI(df['Close'])
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])

    # drop NaN values
    df.dropna(inplace=True)

    # keep Open, High, Low, Close, Volume, MACD, Signal, RSI, CCI, ADX
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'MACD_Signal', 'RSI', 'CCI', 'ADX']]

    return df
    
# add technical indicators to the training data for each stock
for ticker, df in training_data.items():
    training_data[ticker] = add_technical_indicators(df)

# add technical indicators to the validation data for each stock
for ticker, df in validation_data.items():
    validation_data[ticker] = add_technical_indicators(df)

# add technical indicators to the test data for each stock
for ticker, df in test_data.items():
    test_data[ticker] = add_technical_indicators(df)

# print the first 5 rows of the data
print('Shape of training data for NVDA:', training_data['NVDA'].shape)
print('Shape of validation data for NVDA:', validation_data['NVDA'].shape)
print('Shape of test data for NVDA:', test_data['NVDA'].shape)
print("###################")
print(training_data['NVDA'].head())