import pandas as pd 
import numpy as np
from scipy.optimize import minimize

# Load the data
data = pd.read_csv('ticker_data.csv', skiprows=2, usecols=range(14))
# Rename columns to ticker symbols
data.columns = ['Date', 'AAPL', 'DBA', 'EPP', 'EWJ', 'FEZ', 'FXE', 'GLD', 'ILF', 'QQQ', 'SHV', 'SPY', 'USO', 'XBI']
data['Date'] = pd.to_datetime(data['Date'])  # Convert Date column to datetime
data.set_index('Date', inplace=True)

# Filter data for one year up to January 1, 2021
subset_data = data.loc['2020-01-01':'2021-01-01']

# Calculate daily returns from Adjusted Close data
daily_returns = subset_data.pct_change().dropna()

# Compute expected returns and covariance matrix
expected_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

######################################################################################################################

# Assume these betas are given
beta_a = 0.5
beta_b = 1.5
target_beta = 1.0  # You can replace this with any target beta

# Calculate alpha to achieve target beta using two-fund theorem
alpha = (target_beta - beta_b) / (beta_a - beta_b)

# Define the objective function to minimize portfolio variance
def objective(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraints
# 1. The portfolio should have a target beta as a linear combination of betas for beta_a and beta_b
# 2. The sum of weights must equal 1
constraints = [
    {'type': 'eq', 'fun': lambda weights: np.dot(expected_returns, weights) - (alpha * expected_returns + (1 - alpha) * expected_returns).sum()},
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
]

# Initial guess for weights
num_assets = len(expected_returns)
initial_weights = np.ones(num_assets) / num_assets

# Solve the optimization
opt_result = minimize(objective, initial_weights, args=(cov_matrix,), constraints=constraints)

# Retrieve the optimal weights for the target beta portfolio
weights_target_beta = opt_result.x
# Convert the weights to a readable format with ticker names
optimal_weights_df = pd.DataFrame({
    'Ticker': expected_returns.index,
    'Optimal Weight': weights_target_beta
})

######################################################################################################################

if __name__ == '__main__':
    print("Expected Returns:\n", expected_returns)
    print("\nCovariance Matrix:\n", cov_matrix)

    optimal_weights_df['Optimal Weight'] = optimal_weights_df['Optimal Weight'].round(4)

    print("Optimal Weights for Target Beta Portfolio:")
    print(optimal_weights_df.to_string(index=False))
