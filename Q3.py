import pandas as pd
import numpy as np 
from scipy.optimize import minimize
from Q2 import expected_returns, cov_matrix

# Load the data
data = pd.read_csv('ticker_data.csv', skiprows=2, usecols=range(14))
# Rename columns to ticker symbols
data.columns = ['Date', 'AAPL', 'DBA', 'EPP', 'EWJ', 'FEZ', 'FXE', 'GLD', 'ILF', 'QQQ', 'SHV', 'SPY', 'USO', 'XBI']
data['Date'] = pd.to_datetime(data['Date'])  # Convert Date column to datetime
data.set_index('Date', inplace=True)

# Filter data for one year up to January 1, 2021
subset_data = data.loc['2020-01-01':'2021-01-01']

# Calculate daily returns for each ticker if not done
daily_returns = subset_data.pct_change().dropna()

# Define the market returns as SPY returns
market_returns = daily_returns['SPY']

# Calculate beta for each asset with respect to SPY
betas = {}
for ticker in daily_returns.columns:
    cov_with_market = daily_returns[ticker].cov(market_returns)
    var_market = market_returns.var()
    betas[ticker] = cov_with_market / var_market

# Beta of Apple (AAPL) with respect to the S&P 500
beta_A = betas['AAPL']

######################################################################################################################

num_assets = daily_returns.shape[1]

# Define the objective function to minimize portfolio variance
def hedging_objective(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Fully invested
    {'type': 'eq', 'fun': lambda weights: np.dot(list(betas.values()), weights) - beta_A}  # Target beta = beta of AAPL
]

# Initial guess for weights
initial_weights = np.ones(num_assets) / num_assets

# Perform the optimization
opt_result_hedge = minimize(hedging_objective, initial_weights, args=(cov_matrix,), constraints=constraints)
weights_beta_hedge = opt_result_hedge.x

# Create a DataFrame for the optimal weights with ticker names and formatted weights
hedge_portfolio_df = pd.DataFrame({
    'Ticker': daily_returns.columns,
    'Optimal Weight': weights_beta_hedge
})

# Format the weights to 4 decimal places for better readability
hedge_portfolio_df['Optimal Weight'] = hedge_portfolio_df['Optimal Weight'].round(4)

######################################################################################################################

# Expand the investment universe to include Apple (AAPL) in the list of tickers
investment_universe = list(betas.keys())  # All tickers, including AAPL
num_assets = len(investment_universe)

# Define the objective function to minimize portfolio variance
def hedging_objective(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraints for Beta Neutral Portfolio with target beta of 0
constraints = [
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Fully invested
    {'type': 'eq', 'fun': lambda weights: np.dot(list(betas.values()), weights)}  # Target beta = 0
]

# Initial guess for weights
initial_weights = np.ones(num_assets) / num_assets

# Perform the optimization to find the Beta Neutral Portfolio with target beta = 0
opt_result_neutral = minimize(hedging_objective, initial_weights, args=(cov_matrix,), constraints=constraints)
weights_beta_neutral = opt_result_neutral.x
# Create a DataFrame for better readability
neutral_portfolio_df = pd.DataFrame({
    'Ticker': investment_universe,
    'Optimal Weight': weights_beta_neutral.round(4)
})

# Calculate the expected return of the Beta Neutral Portfolio
expected_return_neutral = np.dot(weights_beta_neutral, expected_returns)

if __name__ == '__main__':
    print("Betas for each security with respect to SPY:", betas)
    print("Beta of Apple (AAPL) with respect to SPY:", beta_A)

    print("Optimal Weights for Beta Hedging Portfolio (Q3b):")
    print(hedge_portfolio_df.to_string(index=False))

    expected_return_hedged_strategy = np.dot(weights_beta_hedge, expected_returns)
    print(f"\nExpected Return of the Hedged Strategy: {expected_return_hedged_strategy:.4f}")

    print("Optimal Weights for Beta Neutral Portfolio (Q3c):")
    print(neutral_portfolio_df.to_string(index=False))
    print(f"\nExpected Return of the Beta Neutral Portfolio: {expected_return_neutral:.4f}")
