# Hybrid GARCH-ML Model for Bitcoin Volatility Forecasting
# Phase 4: Machine Learning Implementation

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import plot_importance
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("Starting Phase 4: Machine Learning Implementation")

# 4.1 Data Splitting
print("\n4.1 Loading Split Data")

# Load train and test data from Phase 3
try:
    train_data = pd.read_csv('ml_train_data.csv', index_col=0, parse_dates=True)
    test_data = pd.read_csv('ml_test_data.csv', index_col=0, parse_dates=True)
    print("Loaded pre-split training and test data from Phase 3.")
except FileNotFoundError:
    print("Error: Split data files not found. Loading the full dataset...")
    try:
        # Load the full dataset and split it
        df_clean = pd.read_csv('ml_ready_data.csv', index_col=0, parse_dates=True)
        print("Loaded full dataset from Phase 3.")
        
        # Split data chronologically
        train_size = int(len(df_clean) * 0.8)
        train_data = df_clean.iloc[:train_size].copy()
        test_data = df_clean.iloc[train_size:].copy()
        print("Split data into training and test sets.")
    except FileNotFoundError:
        print("Error: ML-ready data file not found. Please run Phase 3 first.")
        # For demonstration purposes, load the raw data
        df = pd.read_csv('processed_bitcoin_data.csv', index_col=0, parse_dates=True)
        print("Loaded processed data from Phase 1 as fallback.")
        
        # Add placeholder GARCH volatility and residuals
        df['garch_vol'] = df['gk_volatility'] * 0.9 + np.random.normal(0, 0.002, len(df))
        df['garch_residual'] = df['gk_volatility'] - df['garch_vol']
        
        # Use a small set of features for demonstration
        for col in ['log_return', 'gk_volatility', 'garch_vol']:
            for lag in range(1, 3):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        df = df.dropna()
        
        # Split data chronologically
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size].copy()
        test_data = df.iloc[train_size:].copy()
        print("Created simplified data for demonstration.")

print(f"\nTraining data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Define target variables for both approaches
# Option A: Direct prediction of realized volatility
target_a = 'gk_volatility'

# Option B: Prediction of residuals
target_b = 'garch_residual'

# Make sure the target variables exist in the data
if target_b not in train_data.columns:
    print(f"Creating {target_b} as it was not found in the data")
    train_data[target_b] = train_data['gk_volatility'] - train_data['garch_vol']
    test_data[target_b] = test_data['gk_volatility'] - test_data['garch_vol']

# Define features excluding targets and non-feature columns
exclude_cols = ['gk_volatility', 'garch_residual', 'open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in train_data.columns if col not in exclude_cols]

print(f"\nNumber of features used: {len(feature_cols)}")

# Prepare feature and target data
X_train = train_data[feature_cols]
y_train_a = train_data[target_a]
y_train_b = train_data[target_b]

X_test = test_data[feature_cols]
y_test_a = test_data[target_a]
y_test_b = test_data[target_b]

# Store test_data dates for later visualization
test_dates = test_data.index

# 4.2 Model Training - Random Forest
print("\n4.2 Model Training - Tree-based Methods")
print("\nTraining Random Forest for Direct Volatility Prediction (Option A)...")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV for Random Forest on direct volatility prediction
rf_grid_a = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0
)

# Fit the model
rf_grid_a.fit(X_train, y_train_a)

# Get best parameters and model
print(f"Best Random Forest parameters for Option A: {rf_grid_a.best_params_}")
rf_model_a = rf_grid_a.best_estimator_

# Train Random Forest for residual prediction (Option B)
print("\nTraining Random Forest for Residual Prediction (Option B)...")
rf_grid_b = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0
)

# Fit the model
rf_grid_b.fit(X_train, y_train_b)

# Get best parameters and model
print(f"Best Random Forest parameters for Option B: {rf_grid_b.best_params_}")
rf_model_b = rf_grid_b.best_estimator_

# Evaluate Random Forest models
rf_pred_a = rf_model_a.predict(X_test)
rf_pred_b = rf_model_b.predict(X_test)

# For Option B, add GARCH forecast to get the final prediction
# For Option B, add GARCH forecast to get the final prediction
if 'garch_vol' in test_data.columns:
    rf_full_pred_b = test_data['garch_vol'].values + rf_pred_b
else:
    print("Warning: 'garch_vol' column not found. Using a synthetic placeholder.")
    # Create a simple placeholder (20% of the actual volatility)
    rf_full_pred_b = 0.8 * test_data['gk_volatility'].values + rf_pred_b

# Calculate performance metrics for Random Forest
rf_a_rmse = np.sqrt(mean_squared_error(y_test_a, rf_pred_a))
rf_a_mae = mean_absolute_error(y_test_a, rf_pred_a)
rf_a_r2 = r2_score(y_test_a, rf_pred_a)

rf_b_rmse = np.sqrt(mean_squared_error(y_test_a, rf_full_pred_b))
rf_b_mae = mean_absolute_error(y_test_a, rf_full_pred_b)
rf_b_r2 = r2_score(y_test_a, rf_full_pred_b)

print("\nRandom Forest Performance Metrics:")
print(f"Option A (Direct Prediction) - RMSE: {rf_a_rmse:.6f}, MAE: {rf_a_mae:.6f}, R²: {rf_a_r2:.6f}")
print(f"Option B (Residual Prediction) - RMSE: {rf_b_rmse:.6f}, MAE: {rf_b_mae:.6f}, R²: {rf_b_r2:.6f}")

# Feature importance analysis for Random Forest
rf_importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model_a.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(14, 10))
sns.barplot(x='importance', y='feature', data=rf_importances.head(20))
plt.title('Random Forest Feature Importance (Top 20)', fontsize=16)
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()
print("Created: rf_feature_importance.png")

# 4.3 Model Training - XGBoost
print("\n4.3 Model Training - XGBoost")
print("\nTraining XGBoost for Direct Volatility Prediction (Option A)...")

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# GridSearchCV for XGBoost on direct volatility prediction
xgb_grid_a = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid=xgb_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0
)

# Fit the model
xgb_grid_a.fit(X_train, y_train_a)

# Get best parameters and model
print(f"Best XGBoost parameters for Option A: {xgb_grid_a.best_params_}")
xgb_model_a = xgb_grid_a.best_estimator_

# Train XGBoost for residual prediction (Option B)
print("\nTraining XGBoost for Residual Prediction (Option B)...")
xgb_grid_b = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid=xgb_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0
)

# Fit the model
xgb_grid_b.fit(X_train, y_train_b)

# Get best parameters and model
print(f"Best XGBoost parameters for Option B: {xgb_grid_b.best_params_}")
xgb_model_b = xgb_grid_b.best_estimator_

# Evaluate XGBoost models
xgb_pred_a = xgb_model_a.predict(X_test)
xgb_pred_b = xgb_model_b.predict(X_test)

# For Option B, add GARCH forecast to get the final prediction
# For Option B, add GARCH forecast to get the final prediction
if 'garch_vol' in test_data.columns:
    xgb_full_pred_b = test_data['garch_vol'].values + xgb_pred_b
else:
    print("Warning: 'garch_vol' column not found. Using a synthetic placeholder.")
    # Create a simple placeholder (20% of the actual volatility)
    xgb_full_pred_b = 0.8 * test_data['gk_volatility'].values + xgb_pred_b

# Calculate performance metrics for XGBoost
xgb_a_rmse = np.sqrt(mean_squared_error(y_test_a, xgb_pred_a))
xgb_a_mae = mean_absolute_error(y_test_a, xgb_pred_a)
xgb_a_r2 = r2_score(y_test_a, xgb_pred_a)

xgb_b_rmse = np.sqrt(mean_squared_error(y_test_a, xgb_full_pred_b))
xgb_b_mae = mean_absolute_error(y_test_a, xgb_full_pred_b)
xgb_b_r2 = r2_score(y_test_a, xgb_full_pred_b)

print("\nXGBoost Performance Metrics:")
print(f"Option A (Direct Prediction) - RMSE: {xgb_a_rmse:.6f}, MAE: {xgb_a_mae:.6f}, R²: {xgb_a_r2:.6f}")
print(f"Option B (Residual Prediction) - RMSE: {xgb_b_rmse:.6f}, MAE: {xgb_b_mae:.6f}, R²: {xgb_b_r2:.6f}")

# Feature importance analysis for XGBoost
plt.figure(figsize=(14, 10))
plot_importance(xgb_model_a, max_num_features=20)
plt.title('XGBoost Feature Importance (Top 20)', fontsize=16)
plt.tight_layout()
plt.savefig('xgb_feature_importance.png')
plt.close()
print("Created: xgb_feature_importance.png")

# Compare GARCH baseline with ML models
# Get GARCH forecasts for comparing with ML models
if 'garch_vol' in test_data.columns:
    garch_forecasts = test_data['garch_vol'].values
else:
    print("Warning: 'garch_vol' column not found for GARCH comparison. Using a synthetic placeholder.")
    # Create a simple placeholder (80% of the actual volatility)
    garch_forecasts = test_data['gk_volatility'].values * 0.8
garch_rmse = np.sqrt(mean_squared_error(y_test_a, garch_forecasts))
garch_mae = mean_absolute_error(y_test_a, garch_forecasts)
garch_r2 = r2_score(y_test_a, garch_forecasts)

print("\nGARCH Baseline Performance:")
print(f"RMSE: {garch_rmse:.6f}, MAE: {garch_mae:.6f}, R²: {garch_r2:.6f}")

# Calculate performance improvement over GARCH
print("\nPerformance Improvement over GARCH Baseline:")
print(f"RF Option A: {100 * (garch_rmse - rf_a_rmse) / garch_rmse:.2f}% RMSE reduction")
print(f"RF Option B: {100 * (garch_rmse - rf_b_rmse) / garch_rmse:.2f}% RMSE reduction")
print(f"XGB Option A: {100 * (garch_rmse - xgb_a_rmse) / garch_rmse:.2f}% RMSE reduction")
print(f"XGB Option B: {100 * (garch_rmse - xgb_b_rmse) / garch_rmse:.2f}% RMSE reduction")

# Visualize model predictions against actual volatility
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test_a, label='Actual Volatility', color='black', linewidth=2)
plt.plot(test_dates, garch_forecasts, label='GARCH', color='blue', alpha=0.7)
plt.plot(test_dates, rf_full_pred_b, label='RF (Residual)', color='green', alpha=0.7)
plt.plot(test_dates, xgb_full_pred_b, label='XGB (Residual)', color='red', alpha=0.7)
plt.title('Volatility Forecasts Comparison', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('model_comparison.png')
plt.close()
print("Created: model_comparison.png")

# Scatter plots of predicted vs actual volatility
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.scatter(y_test_a, garch_forecasts, alpha=0.5)
plt.plot([y_test_a.min(), y_test_a.max()], [y_test_a.min(), y_test_a.max()], 'r--')
plt.title('GARCH Forecast vs Actual', fontsize=12)
plt.xlabel('Actual Volatility', fontsize=10)
plt.ylabel('Predicted Volatility', fontsize=10)

plt.subplot(1, 4, 2)
plt.scatter(y_test_a, rf_pred_a, alpha=0.5)
plt.plot([y_test_a.min(), y_test_a.max()], [y_test_a.min(), y_test_a.max()], 'r--')
plt.title('RF (Direct) vs Actual', fontsize=12)
plt.xlabel('Actual Volatility', fontsize=10)
plt.ylabel('Predicted Volatility', fontsize=10)

plt.subplot(1, 4, 3)
plt.scatter(y_test_a, rf_full_pred_b, alpha=0.5)
plt.plot([y_test_a.min(), y_test_a.max()], [y_test_a.min(), y_test_a.max()], 'r--')
plt.title('RF (Residual) vs Actual', fontsize=12)
plt.xlabel('Actual Volatility', fontsize=10)
plt.ylabel('Predicted Volatility', fontsize=10)

plt.subplot(1, 4, 4)
plt.scatter(y_test_a, xgb_full_pred_b, alpha=0.5)
plt.plot([y_test_a.min(), y_test_a.max()], [y_test_a.min(), y_test_a.max()], 'r--')
plt.title('XGB (Residual) vs Actual', fontsize=12)
plt.xlabel('Actual Volatility', fontsize=10)
plt.ylabel('Predicted Volatility', fontsize=10)

plt.tight_layout()
plt.savefig('scatter_comparisons.png')
plt.close()
print("Created: scatter_comparisons.png")

# Save predictions for hybrid model construction
predictions_df = pd.DataFrame({
    'date': test_dates,
    'actual_volatility': y_test_a,
    'garch_forecast': garch_forecasts,
    'rf_direct_pred': rf_pred_a,
    'rf_residual_full': rf_full_pred_b,
    'xgb_direct_pred': xgb_pred_a,
    'xgb_residual_full': xgb_full_pred_b
})
predictions_df.set_index('date', inplace=True)
predictions_df.to_csv('ml_predictions.csv')
print("\nML predictions saved to 'ml_predictions.csv'")

# Save models for later use
import pickle

# Save Random Forest models
with open('rf_model_direct.pkl', 'wb') as f:
    pickle.dump(rf_model_a, f)
with open('rf_model_residual.pkl', 'wb') as f:
    pickle.dump(rf_model_b, f)

# Save XGBoost models
with open('xgb_model_direct.pkl', 'wb') as f:
    pickle.dump(xgb_model_a, f)
with open('xgb_model_residual.pkl', 'wb') as f:
    pickle.dump(xgb_model_b, f)

print("\nModels saved as pickle files.")

print("\nPhase 4: Machine Learning Implementation completed!")
print("The ML models have been trained and evaluated.")
print("Data is ready for Phase 5: Hybrid Model Construction.")

# Summary of findings
print("\nSummary of Machine Learning Phase:")
print(f"1. Best performing model: {'XGBoost' if min(xgb_a_rmse, xgb_b_rmse) < min(rf_a_rmse, rf_b_rmse) else 'Random Forest'} using {'direct prediction' if min(xgb_a_rmse, rf_a_rmse) < min(xgb_b_rmse, rf_b_rmse) else 'residual prediction'}")
print(f"2. Improvement over GARCH baseline: {100 * (garch_rmse - min(rf_a_rmse, rf_b_rmse, xgb_a_rmse, xgb_b_rmse)) / garch_rmse:.2f}% RMSE reduction")
print(f"3. Most important feature: {rf_importances.iloc[0]['feature']}")
print(f"4. Option B (residual prediction) {'outperforms' if min(rf_b_rmse, xgb_b_rmse) < min(rf_a_rmse, xgb_a_rmse) else 'underperforms'} Option A (direct prediction)")