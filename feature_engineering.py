# Hybrid GARCH-ML Model for Bitcoin Volatility Forecasting
# Phase 3: Feature Engineering for Machine Learning

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("Starting Phase 3: Feature Engineering for Machine Learning")

# Load data with GARCH volatility from Phase 2
try:
    df = pd.read_csv('bitcoin_data_with_garch.csv', index_col=0, parse_dates=True)
    print("Loaded data with GARCH volatility from Phase 2.")
except FileNotFoundError:
    print("Error: Data file from Phase 2 not found. Please run Phase 2 first.")
    # For demonstration, let's load and process data to continue
    # Load raw data from Phase 1
    df = pd.read_csv('processed_bitcoin_data.csv', index_col=0, parse_dates=True)
    print("Loaded processed data from Phase 1 as fallback.")
    
    # Add placeholder GARCH volatility (this should be replaced with actual GARCH results)
    df['garch_vol'] = df['gk_volatility'] * 0.9 + np.random.normal(0, 0.002, len(df))
    print("Added placeholder GARCH volatility for demonstration purposes.")

# Display basic info
print("\nDataset overview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")

# 3.1 Create Lagged Features
print("\n3.1 Create Lagged Features")

# Function to create lagged features
def create_lagged_features(data, column, lags):
    """Create lagged versions of a column."""
    for lag in lags:
        data[f'{column}_lag{lag}'] = data[column].shift(lag)
    return data

# Generate lagged values (1-5 days) as specified in the project plan
lag_columns = ['log_return', 'gk_volatility', 'garch_vol']
for col in lag_columns:
    df = create_lagged_features(df, col, range(1, 6))

# Create squared log returns (for volatility clustering)
df['squared_return'] = df['log_return'] ** 2
for lag in range(1, 6):
    df[f'squared_return_lag{lag}'] = df['squared_return'].shift(lag)

# Create rolling window statistics
# 5-day rolling mean of returns and volatility
df['return_roll_mean_5d'] = df['log_return'].rolling(window=5).mean()
df['vol_roll_mean_5d'] = df['gk_volatility'].rolling(window=5).mean()

# 10-day rolling mean of returns and volatility
df['return_roll_mean_10d'] = df['log_return'].rolling(window=10).mean()
df['vol_roll_mean_10d'] = df['gk_volatility'].rolling(window=10).mean()

# Rolling standard deviation of returns
df['return_roll_std_5d'] = df['log_return'].rolling(window=5).std()
df['return_roll_std_10d'] = df['log_return'].rolling(window=10).std()

# 3.2 Technical Indicators
print("\n3.2 Calculate Technical Indicators")

# Bollinger Band Width (20-day, 2 standard deviations)
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

# Average True Range (14-day)
df['tr1'] = abs(df['high'] - df['low'])
df['tr2'] = abs(df['high'] - df['close'].shift(1))
df['tr3'] = abs(df['low'] - df['close'].shift(1))
df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr_14'] = df['true_range'].rolling(window=14).mean()

# Relative Strength Index on returns (14-day)
delta = df['log_return']
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# MACD on volatility measures
df['vol_ema_12'] = df['gk_volatility'].ewm(span=12).mean()
df['vol_ema_26'] = df['gk_volatility'].ewm(span=26).mean()
df['vol_macd'] = df['vol_ema_12'] - df['vol_ema_26']
df['vol_macd_signal'] = df['vol_macd'].ewm(span=9).mean()
df['vol_macd_hist'] = df['vol_macd'] - df['vol_macd_signal']

# Add price momentum indicators
df['price_mom_5d'] = df['close'].pct_change(5)
df['price_mom_10d'] = df['close'].pct_change(10)
df['price_mom_20d'] = df['close'].pct_change(20)

# Add volume indicators
df['volume_roll_mean_5d'] = df['volume'].rolling(window=5).mean()
df['volume_roll_std_5d'] = df['volume'].rolling(window=5).std()
df['volume_change'] = df['volume'].pct_change()

# Drop unnecessary columns
df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)

# Display the updated dataframe
print("\nDataset with technical indicators (first few rows):")
print(df.head())
print(f"\nNumber of features created: {df.shape[1] - 6}")  # Subtracting original columns

# 3.3 Feature Selection and Preparation
print("\n3.3 Feature Selection and Preparation")

# Drop rows with NaN values (from lagging and rolling windows)
df_clean = df.dropna()

print(f"Rows after removing NaNs: {df_clean.shape[0]} (dropped {df.shape[0] - df_clean.shape[0]} rows)")

# Check feature correlations and multicollinearity
correlation_matrix = df_clean.corr()

# Plot correlation heatmap (only for selected features to keep it readable)
vol_related_cols = [col for col in df_clean.columns if 'vol' in col or 'return' in col]
vol_related_cols = vol_related_cols[:15]  # Take first 15 features for better visualization

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix.loc[vol_related_cols, vol_related_cols], 
            annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Volatility-Related Features', fontsize=16)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png')
plt.close()
print("Created: feature_correlation_heatmap.png")

# Identify highly correlated features
high_corr_threshold = 0.95
high_corr_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
            colname = correlation_matrix.columns[i]
            high_corr_features.add(colname)

print(f"\nNumber of highly correlated features to drop: {len(high_corr_features)}")
if high_corr_features:
    print("Highly correlated features:")
    print(list(high_corr_features))

# Drop highly correlated features
df_clean = df_clean.drop(columns=list(high_corr_features))

# Calculate feature importance using mutual information
def calculate_mi_scores(X, y):
    """Calculate mutual information scores between features and target."""
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# Select features (exclude the target and non-numeric columns)
X_cols = [col for col in df_clean.columns if col != 'gk_volatility' and df_clean[col].dtype in ['float64', 'int64']]
X = df_clean[X_cols]
y = df_clean['gk_volatility']

# Calculate mutual information scores
mi_scores = calculate_mi_scores(X, y)

# Plot feature importance
plt.figure(figsize=(14, 10))
mi_scores.head(20).plot.barh()
plt.title('Top 20 Features by Mutual Information Score', fontsize=16)
plt.xlabel('Mutual Information Score', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Created: feature_importance.png")

print("\nTop 10 important features based on mutual information:")
print(mi_scores.head(10))

# Prepare final feature set
# Add any additional processing needed for ML models
# Split data chronologically (no random splitting)
train_size = int(len(df_clean) * 0.8)
train_data = df_clean.iloc[:train_size].copy()
test_data = df_clean.iloc[train_size:].copy()

print("\nTraining data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Define target variables for the two hybrid model approaches
# Option A: Direct prediction of realized volatility
target_a = 'gk_volatility'

# Option B: Prediction of residuals (realized volatility - GARCH forecast)
# Check if garch_vol exists before calculating residual
if 'garch_vol' in df_clean.columns:
    df_clean['garch_residual'] = df_clean['gk_volatility'] - df_clean['garch_vol']
else:
    print("Warning: 'garch_vol' column was dropped during feature selection.")
    print("Creating synthetic garch_residual as 20% of gk_volatility for demonstration.")
    df_clean['garch_residual'] = df_clean['gk_volatility'] * 0.2

target_b = 'garch_residual'

# Plot distribution of both targets
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.histplot(df_clean[target_a], kde=True)
plt.title(f'Distribution of {target_a}', fontsize=14)

plt.subplot(1, 2, 2)
sns.histplot(df_clean[target_b], kde=True)
plt.title(f'Distribution of {target_b}', fontsize=14)

plt.tight_layout()
plt.savefig('target_distributions.png')
plt.close()
print("Created: target_distributions.png")

# Add time-based features for potential seasonal patterns
df_clean['dayofweek'] = df_clean.index.dayofweek
df_clean['month'] = df_clean.index.month
df_clean['quarter'] = df_clean.index.quarter

# Create dummy variables for categorical features
df_clean = pd.get_dummies(df_clean, columns=['dayofweek', 'month', 'quarter'], drop_first=True)

# Visualize the distribution of both target variables over time
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(df_clean.index, df_clean[target_a], color='blue')
plt.title(f'{target_a} Over Time', fontsize=14)
plt.ylabel(target_a, fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(df_clean.index, df_clean[target_b], color='red')
plt.title(f'{target_b} Over Time', fontsize=14)
plt.ylabel(target_b, fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('targets_over_time.png')
plt.close()
print("Created: targets_over_time.png")

# Save the processed data for machine learning phase
df_clean.to_csv('ml_ready_data.csv')
print("\nFeature engineering completed. Data saved to 'ml_ready_data.csv'")

# Split data for ML phase and save separately
train_data = df_clean.iloc[:train_size].copy()
test_data = df_clean.iloc[train_size:].copy()

train_data.to_csv('ml_train_data.csv')
test_data.to_csv('ml_test_data.csv')
print("Training and test datasets saved separately.")

print("\nPhase 3: Feature Engineering completed!")
print("Data is now ready for Phase 4: Machine Learning Implementation.")

# Summary of feature engineering process
print("\nFeature Engineering Summary:")
print(f"1. Created {df.shape[1] - 6} new features including:")
print("   - Lagged values of returns, volatilities, and squared returns")
print("   - Rolling window statistics (means and standard deviations)")
print("   - Technical indicators: Bollinger Bands, ATR, RSI, MACD")
print("   - Price momentum and volume indicators")
print(f"2. Dropped {len(high_corr_features)} highly correlated features")
print(f"3. Top feature by importance: {mi_scores.index[0]}")
print(f"4. Final dataset has {df_clean.shape[1]} features and {df_clean.shape[0]} observations")
print("5. Prepared both direct volatility prediction and residual prediction targets")