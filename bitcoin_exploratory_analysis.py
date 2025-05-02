# Hybrid GARCH-ML Model for Bitcoin Volatility Forecasting
# Phase 1: Data Preparation & Exploratory Analysis

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("Starting Bitcoin Volatility Forecasting Project - Phase 1")

# 1.1 Data Loading and Cleaning
print("\n1.1 Data Loading and Cleaning")

# Load Bitcoin daily data from CSV
df = pd.read_csv('data/btc_1d_data_2018_to_2025.csv')

# Display the first few rows
print("\nFirst few rows of raw data:")
print(df.head())

# Display data information
print("\nDataset info:")
print(df.info())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Parse 'Open time' as datetime and set as index
df['Open time'] = pd.to_datetime(df['Open time'])
df = df.set_index('Open time')

# Select and rename OHLCV columns to lowercase
ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
ohlcv.columns = ohlcv.columns.str.lower()

# Check for missing values
print("\nMissing values count:")
print(ohlcv.isnull().sum())

# 1.2 Calculate Returns and Volatility Measures
print("\n1.2 Calculate Returns and Volatility Measures")

# Calculate daily log returns
ohlcv['log_return'] = np.log(ohlcv['close'] / ohlcv['close'].shift(1))

# Implement Garman-Klass volatility estimator
ohlcv['gk_volatility'] = np.sqrt(
    0.5 * np.log(ohlcv['high'] / ohlcv['low'])**2 - 
    (2 * np.log(2) - 1) * np.log(ohlcv['close'] / ohlcv['open'])**2
)

# Remove NaN values resulting from calculations
ohlcv = ohlcv.dropna()

# Display results
print("\nData with returns and volatility measures:")
print(ohlcv.head())

# Basic statistics of returns and volatility
print("\nLog Returns Statistics:")
print(ohlcv['log_return'].describe())

print("\nGarman-Klass Volatility Statistics:")
print(ohlcv['gk_volatility'].describe())

# 1.3 Exploratory Data Analysis
print("\n1.3 Exploratory Data Analysis")

# Visualize Bitcoin price evolution over time
plt.figure(figsize=(14, 7))
plt.plot(ohlcv.index, ohlcv['close'], color='blue')
plt.title('Bitcoin Price Evolution (2018-2025)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('bitcoin_price_evolution.png')
plt.close()
print("Created: bitcoin_price_evolution.png")

# Plot log returns with volatility clustering patterns
plt.figure(figsize=(14, 7))
plt.plot(ohlcv.index, ohlcv['log_return'], color='green', alpha=0.7)
plt.title('Bitcoin Daily Log Returns (2018-2025)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Log Return', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('bitcoin_log_returns.png')
plt.close()
print("Created: bitcoin_log_returns.png")

# Analyze the realized volatility proxy
plt.figure(figsize=(14, 7))
plt.plot(ohlcv.index, ohlcv['gk_volatility'], color='red', alpha=0.7)
plt.title('Bitcoin Garman-Klass Volatility (2018-2025)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GK Volatility', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('bitcoin_gk_volatility.png')
plt.close()
print("Created: bitcoin_gk_volatility.png")

# Check for stationarity of returns
def check_stationarity(time_series, title):
    """
    Check stationarity using ADF test and return results
    """
    print(f"\nStationarity Check for {title}")
    # Perform Augmented Dickey-Fuller test
    result = adfuller(time_series.dropna())
    
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    # Interpret results
    if result[1] <= 0.05:
        print(f"Result: {title} is stationary (reject null hypothesis)")
    else:
        print(f"Result: {title} is non-stationary (fail to reject null hypothesis)")

# Check stationarity for returns and volatility
check_stationarity(ohlcv['log_return'], 'Log Returns')
check_stationarity(ohlcv['gk_volatility'], 'GK Volatility')

# Examine autocorrelation patterns in returns and squared returns (volatility proxy)
plt.figure(figsize=(14, 10))

# ACF for returns
plt.subplot(2, 2, 1)
plot_acf(ohlcv['log_return'], lags=40, alpha=0.05, title='ACF of Log Returns')

# PACF for returns
plt.subplot(2, 2, 2)
plot_pacf(ohlcv['log_return'], lags=40, alpha=0.05, title='PACF of Log Returns')

# ACF for squared returns (volatility clustering)
plt.subplot(2, 2, 3)
plot_acf(ohlcv['log_return']**2, lags=40, alpha=0.05, title='ACF of Squared Log Returns')

# PACF for squared returns
plt.subplot(2, 2, 4)
plot_pacf(ohlcv['log_return']**2, lags=40, alpha=0.05, title='PACF of Squared Log Returns')

plt.tight_layout()
plt.savefig('autocorrelation_analysis.png')
plt.close()
print("Created: autocorrelation_analysis.png")

# Yearly statistics
yearly_stats = ohlcv.groupby(ohlcv.index.year).agg({
    'log_return': ['mean', 'std'],
    'gk_volatility': ['mean', 'std']
})

print("\nYearly Statistics:")
print(yearly_stats)

# Visualize yearly volatility and returns
plt.figure(figsize=(14, 7))
plt.bar(yearly_stats.index, yearly_stats[('gk_volatility', 'mean')], alpha=0.7, color='orangered')
plt.title('Yearly Average Volatility (2018-2025)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average GK Volatility', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(yearly_stats.index)
plt.savefig('yearly_volatility.png')
plt.close()
print("Created: yearly_volatility.png")

# Visualize distribution of log returns
plt.figure(figsize=(14, 7))
sns.histplot(ohlcv['log_return'], kde=True, bins=100, color='green')
plt.title('Distribution of Log Returns', fontsize=16)
plt.xlabel('Log Return', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('log_returns_distribution.png')
plt.close()
print("Created: log_returns_distribution.png")

# Volatility clustering visualization (scatter plot of returns vs lagged returns)
plt.figure(figsize=(10, 10))
plt.scatter(ohlcv['log_return'].shift(1), ohlcv['log_return'], alpha=0.5, s=10)
plt.title('Volatility Clustering: Returns vs Lagged Returns', fontsize=16)
plt.xlabel('Previous Day Return', fontsize=12)
plt.ylabel('Current Day Return', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
plt.savefig('volatility_clustering.png')
plt.close()
print("Created: volatility_clustering.png")

# Create a correlation heatmap for different metrics
correlation_data = pd.DataFrame({
    'log_return': ohlcv['log_return'],
    'abs_return': np.abs(ohlcv['log_return']),
    'squared_return': ohlcv['log_return']**2,
    'gk_volatility': ohlcv['gk_volatility'],
    'volume': ohlcv['volume']
})

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Returns, Volatility and Volume', fontsize=16)
plt.savefig('correlation_heatmap.png')
plt.close()
print("Created: correlation_heatmap.png")

# Analyze recent data (last 100 days)
recent_data = ohlcv.iloc[-100:]

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(recent_data.index, recent_data['close'], color='blue')
plt.title('Recent Bitcoin Price (Last 100 Days)', fontsize=16)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(recent_data.index, recent_data['gk_volatility'], color='red')
plt.title('Recent Volatility (Last 100 Days)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('GK Volatility', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('recent_price_volatility.png')
plt.close()
print("Created: recent_price_volatility.png")

print("\nPhase 1: Data Preparation & Exploratory Analysis Completed!")
print("The data is now ready for Phase 2: GARCH Modeling (Baseline)")

# Save processed data for subsequent phases
ohlcv.to_csv('processed_bitcoin_data.csv')
print("\nProcessed data saved to 'processed_bitcoin_data.csv'")

# Print dataset summary for next steps
print("\nDataset Summary for Next Steps:")
print(f"Time range: {ohlcv.index.min()} to {ohlcv.index.max()}")
print(f"Total observations after preprocessing: {len(ohlcv)}")
print(f"Mean log return: {ohlcv['log_return'].mean():.6f}")
print(f"Mean GK volatility: {ohlcv['gk_volatility'].mean():.6f}")