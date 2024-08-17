import pandas as pd

# Set the target date
target_date = pd.Timestamp('2010-01-04')

# Generate a date range going backwards to include enough days to capture 252 trading days
all_dates = pd.date_range(end=target_date, periods=400, freq='B')  # 'B' stands for business days

# The first date in the list should be 252 trading days behind the target date
date_252_trading_days_behind = all_dates[-252]

print(f"252 trading days before {target_date.date()} was {date_252_trading_days_behind.date()}")