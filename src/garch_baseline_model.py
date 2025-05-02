# Hybrid GARCH-ML Model for Bitcoin Volatility Forecasting
# Phase 2: GARCH Modeling (Baseline)

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("Starting Phase 2: GARCH Modeling (Baseline)")

# Load processed data from Phase 1
try:
    # Try to load saved processed data
    df = pd.read_csv('processed_bitcoin_data.csv', index_col=0, parse_dates=True)
    print("Loaded processed data from CSV file.")
except FileNotFoundError:
    print("Processed data file not found. Loading and processing raw data...")
    # Load raw data and process it (simplified version of Phase 1)
    df = pd.read_csv('data/btc_1d_data_2018_to_2025.csv')
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    
    # Select and rename OHLCV columns to lowercase
    ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ohlcv.columns = ohlcv.columns.str.lower()
    
    # Calculate daily log returns
    ohlcv['log_return'] = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
    
    # Implement Garman-Klass volatility estimator
    ohlcv['gk_volatility'] = np.sqrt(
        0.5 * np.log(ohlcv['high'] / ohlcv['low'])**2 - 
        (2 * np.log(2) - 1) * np.log(ohlcv['close'] / ohlcv['open'])**2
    )
    
    # Remove NaN values
    df = ohlcv.dropna()
    print("Raw data processed successfully.")

# Display basic info
print("\nDataset overview:")
print(df.head())

# 2.1 Implement GARCH Model
print("\n2.1 Implement GARCH Model")

# Scale log returns for numerical stability (multiply by 100)
scaled_returns = df['log_return'] * 100

# Split data into training and testing sets
train_size = int(len(scaled_returns) * 0.8)
train_returns = scaled_returns[:train_size]
test_returns = scaled_returns[train_size:]

print(f"Training set size: {len(train_returns)}")
print(f"Test set size: {len(test_returns)}")

# Fit GARCH(1,1) model with normal distribution
print("\nFitting GARCH(1,1) model...")
garch_model = arch_model(train_returns, vol='Garch', p=1, q=1, dist='normal')
garch_result = garch_model.fit(disp='off')

# Print model summary
print("\nGARCH Model Summary:")
print(garch_result.summary())

# Extract model parameters
omega = garch_result.params['omega']
alpha = garch_result.params['alpha[1]']
beta = garch_result.params['beta[1]']

print(f"\nGARCH(1,1) Parameters:")
print(f"omega: {omega:.8f}")
print(f"alpha: {alpha:.8f}")
print(f"beta: {beta:.8f}")
print(f"Persistence (alpha + beta): {alpha + beta:.8f}")

# Generate in-sample conditional volatility forecasts
in_sample_vol = garch_result.conditional_volatility

# Get one-step-ahead forecasts for the test period
forecasts = []
test_vol = []
realized_vol = df['gk_volatility'][train_size:] * 100  # Scale realized volatility too

# Iteratively update the model with new observations
test_index = df.index[train_size:]
for i in range(len(test_returns)):
    # Get forecast for next day
    forecast = garch_result.forecast(horizon=1)
    next_vol = np.sqrt(forecast.variance.iloc[-1, 0])
    forecasts.append(next_vol)
    
    # Update the model with the actual return
    if i < len(test_returns) - 1:  # Skip the last iteration to avoid IndexError
        garch_result = garch_result.update(test_returns.iloc[i:i+1])

# Convert forecasts to numpy array
forecasts = np.array(forecasts)

# Rescale forecasts back to original scale
forecasts = forecasts / 100
realized_vol = realized_vol / 100

# 2.2 Evaluate GARCH Performance
print("\n2.2 Evaluate GARCH Performance")

# Create DataFrame for evaluation
eval_df = pd.DataFrame({
    'date': test_index,
    'realized_volatility': realized_vol.values,
    'garch_forecast': forecasts
})

# Calculate performance metrics
mse = mean_squared_error(eval_df['realized_volatility'], eval_df['garch_forecast'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(eval_df['realized_volatility'], eval_df['garch_forecast'])

print("\nGARCH Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.8f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.8f}")
print(f"Mean Absolute Error (MAE): {mae:.8f}")

# Visualize GARCH forecasts against realized volatility
plt.figure(figsize=(14, 7))
plt.plot(eval_df['date'], eval_df['realized_volatility'], label='Realized Volatility (GK)', color='blue')
plt.plot(eval_df['date'], eval_df['garch_forecast'], label='GARCH(1,1) Forecast', color='red', alpha=0.7)
plt.title('GARCH(1,1) Volatility Forecasts vs. Realized Volatility', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('garch_forecasts_vs_realized.png')
plt.close()
print("Created: garch_forecasts_vs_realized.png")

# Scatter plot of forecasts vs. realized volatility
plt.figure(figsize=(10, 10))
plt.scatter(eval_df['realized_volatility'], eval_df['garch_forecast'], alpha=0.5)
plt.plot([0, max(eval_df['realized_volatility'].max(), eval_df['garch_forecast'].max())], 
         [0, max(eval_df['realized_volatility'].max(), eval_df['garch_forecast'].max())], 
         'r--')
plt.title('GARCH Forecasts vs. Realized Volatility', fontsize=16)
plt.xlabel('Realized Volatility (GK)', fontsize=12)
plt.ylabel('GARCH Forecast', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('garch_forecast_scatter.png')
plt.close()
print("Created: garch_forecast_scatter.png")

# Analyze forecast errors
eval_df['error'] = eval_df['garch_forecast'] - eval_df['realized_volatility']

plt.figure(figsize=(14, 7))
plt.plot(eval_df['date'], eval_df['error'], color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('GARCH Forecast Errors', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Forecast Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('garch_forecast_errors.png')
plt.close()
print("Created: garch_forecast_errors.png")

# Plot error distribution
plt.figure(figsize=(12, 6))
sns.histplot(eval_df['error'], kde=True, bins=50)
plt.title('Distribution of GARCH Forecast Errors', fontsize=16)
plt.xlabel('Forecast Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('garch_error_distribution.png')
plt.close()
print("Created: garch_error_distribution.png")

# Save the evaluation data for next phases
eval_df.to_csv('garch_evaluation.csv')
print("\nGARCH evaluation results saved to 'garch_evaluation.csv'")

# Save in-sample volatility estimates for feature engineering
in_sample_vol_df = pd.DataFrame(
    {'date': df.index[:train_size], 'garch_vol': in_sample_vol / 100}
)
in_sample_vol_df.to_csv('in_sample_garch_vol.csv')

# Combine in-sample and out-of-sample GARCH volatility for full dataset
full_garch_vol = pd.concat([
    pd.DataFrame({'date': df.index[:train_size], 'garch_vol': in_sample_vol / 100}),
    pd.DataFrame({'date': eval_df['date'], 'garch_vol': eval_df['garch_forecast']})
])
full_garch_vol.set_index('date', inplace=True)

# Add GARCH volatility to the original dataframe
df['garch_vol'] = full_garch_vol['garch_vol']

# Save the enhanced dataset for Phase 3
df.to_csv('bitcoin_data_with_garch.csv')
print("Enhanced dataset saved to 'bitcoin_data_with_garch.csv'")

print("\nPhase 2: GARCH Modeling (Baseline) completed!")
print("The baseline GARCH model has been implemented and evaluated.")
print("Data is ready for Phase 3: Feature Engineering for Machine Learning.")

# Summary of findings
print("\nSummary of GARCH Modeling:")
print(f"1. The GARCH(1,1) model shows {'high' if alpha + beta > 0.9 else 'moderate'} persistence of volatility (alpha + beta = {alpha + beta:.4f})")
print(f"2. Baseline performance metrics: RMSE = {rmse:.6f}, MAE = {mae:.6f}")
print(f"3. The model {'overestimates' if eval_df['error'].mean() > 0 else 'underestimates'} volatility on average by {abs(eval_df['error'].mean()):.6f}")
print(f"4. GARCH model captures {100 * (1 - mse / np.var(eval_df['realized_volatility'])):.2f}% of the variance in realized volatility")