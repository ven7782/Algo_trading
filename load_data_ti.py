import yfinance as yf
import pandas as pd
import numpy as np
import talib
import sys

start = "2010-01-04"
end = "2020-05-08"

pd.options.mode.chained_assignment = None  # default='warn'
# pd.set_option('display.max_rows', None)

# tickers = ['NVDA', 'QCOM']
tickers = ['NVDA']

def add_turbulence(data):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df = data.copy()
    turbulence_index = calculate_turbulence(df)
    # Set 'date' as the index
    turbulence_index.set_index('date', inplace=True)

    # Drop rows where 'turbulence' is 0
    df = turbulence_index[turbulence_index['turbulence'] != 0]
    return df

def calculate_turbulence(data):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    df = data.copy()
    df_price_pivot = df[['Close']]
    # use returns to calculate turbulence
    df_price_pivot = df_price_pivot.pct_change()

    unique_date = df.index.tolist()
    # start after a year
    start = 252
    turbulence_index = [0] * start
    # turbulence_index = [0]
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        # use one year rolling window to calcualte covariance
        hist_price = df_price_pivot[
            (df_price_pivot.index < unique_date[i])
            & (df_price_pivot.index >= unique_date[i - 252])
        ]
        # Drop tickers which has number missing values more than the "oldest" ticker
        filtered_hist_price = hist_price.iloc[
            hist_price.isna().sum().min() :
        ].dropna(axis=1)

        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
            filtered_hist_price, axis=0
        )
        # cov_temp = hist_price.cov()
        # current_temp=(current_price - np.mean(hist_price,axis=0))

        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
            current_temp.values.T
        )
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)
    try:
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
    except ValueError:
        raise Exception("Turbulence information could not be added.")
    return turbulence_index

def back_252(date):
    # Set the target date
    target_date = pd.Timestamp(date)

    # Generate a date range going backwards to include enough days to capture 252 trading days
    all_dates = pd.date_range(end=target_date, periods=400, freq='B')  # 'B' stands for business days

    # The first date in the list should be 252 trading days behind the target date
    return all_dates[-252]

# Get the data from the CSV files
stock_data = {}
for ticker in tickers:
    df = pd.read_csv(f'data/{ticker}.csv', index_col='Date', parse_dates=True)
    stock_data[ticker] = df


# split the data into training, validation and test sets
training_data_time_range = ('2008-11-13', '2015-12-31')
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

def add_technical_indicators(df, ticker):
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
    df['RSI'] = talib.RSI(df['Close'])
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
    df_temp = pd.read_csv(f'data/{ticker}.csv', index_col='Date', parse_dates=True)
    df_temp = df_temp.loc[back_252(df.index.min()):df.index.max()]
    # turbulence_index = calculate_turbulence(df_temp)
    # print(turbulence_index)
    df['TINDEX'] = add_turbulence(df_temp)

    # drop NaN values
    df.dropna(inplace=True)

    # keep Open, High, Low, Close, Volume, MACD, Signal, RSI, CCI, ADX
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'MACD_Signal', 'RSI', 'CCI', 'ADX', 'TINDEX']]

    return df
    
# add technical indicators to the training data for each stock
for ticker, df in training_data.items():
    training_data[ticker] = add_technical_indicators(df, ticker)

# add technical indicators to the validation data for each stock
# for ticker, df in validation_data.items():
#     validation_data[ticker] = add_technical_indicators(df, ticker)

# # add technical indicators to the test data for each stock
# for ticker, df in test_data.items():
#     test_data[ticker] = add_technical_indicators(df, ticker)

# print the first 5 rows of the data
print('Shape of training data for NVDA:', training_data['NVDA'].shape)
print('Shape of validation data for NVDA:', validation_data['NVDA'].shape)
print('Shape of test data for NVDA:', test_data['NVDA'].shape)
print("###################")
print(training_data['NVDA'].head())