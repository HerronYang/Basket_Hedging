import yfinance as yf
import pandas as pd

# List your tickers
tickers = ['AAPL', 'SPY', 'FXE', 'EWJ', 'GLD', 'QQQ', 'SHV', 'DBA', 'USO', 'XBI', 'ILF', 'EPP', 'FEZ']
# Define date range
start_date = '2020-01-01'
end_date = '2021-03-30'

# Download data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Combine adjusted close prices and daily returns
combined_df = pd.concat([data, daily_returns], axis=1, keys=['Adjusted Close', 'Daily Return'])

# Save to CSV
combined_df.to_csv('ticker_data_with_returns.csv')

print("Data saved to 'ticker_data_with_returns.csv'")
