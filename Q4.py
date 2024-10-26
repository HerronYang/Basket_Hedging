import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from Q3 import daily_returns, weights_beta_hedge, weights_beta_neutral

# Load the CSV file, focusing on the daily returns section
file_path = 'ticker_data.csv'  # Replace with your actual file path
daily_returns_data = pd.read_csv(file_path, skiprows=2, usecols=[0] + list(range(14, 27)))

# Rename columns based on ticker symbols for daily returns
daily_returns_data.columns = ['Date', 'AAPL', 'DBA', 'EPP', 'EWJ', 'FEZ', 'FXE', 'GLD', 'ILF', 'QQQ', 'SHV', 'SPY', 'USO', 'XBI']
daily_returns_data['Date'] = pd.to_datetime(daily_returns_data['Date'])
daily_returns_data.set_index('Date', inplace=True)

# Define the date range for Q4
start_date = '2021-01-03'
end_date = '2021-03-30'

# Filter daily returns for the specified date range
daily_returns_filtered = daily_returns_data.loc[start_date:end_date]
num_assets = daily_returns_filtered.shape[1]

# Calculate realized returns for both portfolios
realized_hedge_returns = daily_returns_filtered.dot(weights_beta_hedge)
realized_neutral_returns = daily_returns_filtered.dot(weights_beta_neutral)
market_returns = daily_returns_filtered['SPY']  # SPY as the market benchmark

print("Realized Daily Returns for Beta Hedging Portfolio (Q3b):")
print(realized_hedge_returns)

print("\nRealized Daily Returns for Beta Neutral Portfolio (Q3c):")
print(realized_neutral_returns)

######################################################################################################################

# Function to calculate metrics
def calculate_metrics(portfolio_returns, market_returns):
    return {
        "Expected Return": np.mean(portfolio_returns),
        "Volatility": np.std(portfolio_returns),
        "95% VaR": np.percentile(portfolio_returns, 5),
        "Skewness": skew(portfolio_returns),
        "Kurtosis": kurtosis(portfolio_returns),
        "Beta": np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    }

# Calculate metrics for each portfolio
hedge_metrics = calculate_metrics(realized_hedge_returns, market_returns)
neutral_metrics = calculate_metrics(realized_neutral_returns, market_returns)

# Display the metrics for both portfolios
print("Performance Metrics for Beta Hedging Portfolio (Q3b):")
for key, value in hedge_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nPerformance Metrics for Beta Neutral Portfolio (Q3c):")
for key, value in neutral_metrics.items():
    print(f"{key}: {value:.4f}")

# Plot distribution of returns for both portfolios
plt.figure(figsize=(12, 6))

# Distribution plot for Beta Hedging Portfolio
plt.subplot(1, 2, 1)
plt.hist(realized_hedge_returns, bins=30, alpha=0.7, color='blue')
plt.title('Return Distribution - Beta Hedging Portfolio (Q3b)')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')

# Distribution plot for Beta Neutral Portfolio
plt.subplot(1, 2, 2)
plt.hist(realized_neutral_returns, bins=30, alpha=0.7, color='green')
plt.title('Return Distribution - Beta Neutral Portfolio (Q3c)')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()
