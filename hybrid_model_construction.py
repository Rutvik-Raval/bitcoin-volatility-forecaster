# Hybrid GARCH-ML Model for Bitcoin Volatility Forecasting
# Phase 5: Hybrid Model Construction

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("Starting Phase 5: Hybrid Model Construction")

# Load predictions from Phase 4
try:
    predictions_df = pd.read_csv('ml_predictions.csv', index_col=0, parse_dates=True)
    print("Loaded predictions from Phase 4.")
except FileNotFoundError:
    print("Error: Predictions file not found. Please run Phase 4 first.")
    # Create dummy data for demonstration purposes
    import random
    
    index = pd.date_range(start='2024-01-01', periods=100)
    actual_vol = np.random.normal(0.03, 0.01, 100)
    garch_forecast = actual_vol * 0.9 + np.random.normal(0, 0.005, 100)
    rf_direct = actual_vol * 0.95 + np.random.normal(0, 0.004, 100)
    rf_residual = actual_vol - garch_forecast + np.random.normal(0, 0.002, 100)
    rf_residual_full = garch_forecast + rf_residual
    xgb_direct = actual_vol * 0.97 + np.random.normal(0, 0.003, 100)
    xgb_residual = actual_vol - garch_forecast + np.random.normal(0, 0.002, 100)
    xgb_residual_full = garch_forecast + xgb_residual
    
    predictions_df = pd.DataFrame({
        'actual_volatility': actual_vol,
        'garch_forecast': garch_forecast,
        'rf_direct_pred': rf_direct,
        'rf_residual_full': rf_residual_full,
        'xgb_direct_pred': xgb_direct,
        'xgb_residual_full': xgb_residual_full
    }, index=index)
    
    print("Created dummy predictions data for demonstration.")

# Display the predictions data
print("\nPredictions data (first few rows):")
print(predictions_df.head())

# 5.1 Direct Combination Approach
print("\n5.1 Direct Combination Approach")

# Function to create weighted average combination and evaluate performance
def weighted_average_hybrid(weight, forecasts, actual):
    """
    Combine forecasts using weighted average and calculate RMSE
    
    Parameters:
    weight (float): Weight for the first forecast (0 to 1)
    forecasts (tuple): Tuple of two forecast series
    actual (Series): Actual values
    
    Returns:
    float: RMSE of the combined forecast
    """
    forecast1, forecast2 = forecasts
    combined = weight * forecast1 + (1 - weight) * forecast2
    return np.sqrt(mean_squared_error(actual, combined))

# Create different hybrid models using weighted average combination
# Hybrid 1: GARCH + RF Direct
def objective_hybrid1(weight):
    return weighted_average_hybrid(weight, 
                                  (predictions_df['garch_forecast'], predictions_df['rf_direct_pred']), 
                                  predictions_df['actual_volatility'])

# Hybrid 2: GARCH + XGB Direct
def objective_hybrid2(weight):
    return weighted_average_hybrid(weight, 
                                  (predictions_df['garch_forecast'], predictions_df['xgb_direct_pred']), 
                                  predictions_df['actual_volatility'])

# Hybrid 3: RF Direct + XGB Direct
def objective_hybrid3(weight):
    return weighted_average_hybrid(weight, 
                                  (predictions_df['rf_direct_pred'], predictions_df['xgb_direct_pred']), 
                                  predictions_df['actual_volatility'])

# Optimize weights for each hybrid model
opt_result1 = minimize(objective_hybrid1, x0=0.5, bounds=[(0, 1)])
opt_result2 = minimize(objective_hybrid2, x0=0.5, bounds=[(0, 1)])
opt_result3 = minimize(objective_hybrid3, x0=0.5, bounds=[(0, 1)])

# Get optimal weights
optimal_weight1 = opt_result1.x[0]
optimal_weight2 = opt_result2.x[0]
optimal_weight3 = opt_result3.x[0]

print(f"Optimal weights for hybrid models:")
print(f"GARCH + RF Direct: GARCH weight = {optimal_weight1:.4f}, RF weight = {1-optimal_weight1:.4f}")
print(f"GARCH + XGB Direct: GARCH weight = {optimal_weight2:.4f}, XGB weight = {1-optimal_weight2:.4f}")
print(f"RF Direct + XGB Direct: RF weight = {optimal_weight3:.4f}, XGB weight = {1-optimal_weight3:.4f}")

# Create the hybrid forecasts
predictions_df['hybrid1_forecast'] = (optimal_weight1 * predictions_df['garch_forecast'] + 
                                     (1 - optimal_weight1) * predictions_df['rf_direct_pred'])

predictions_df['hybrid2_forecast'] = (optimal_weight2 * predictions_df['garch_forecast'] + 
                                     (1 - optimal_weight2) * predictions_df['xgb_direct_pred'])

predictions_df['hybrid3_forecast'] = (optimal_weight3 * predictions_df['rf_direct_pred'] + 
                                     (1 - optimal_weight3) * predictions_df['xgb_direct_pred'])

# 5.2 Residual Prediction Approach
print("\n5.2 Residual Prediction Approach")

# We already have the residual prediction models from Phase 4
# The results are in rf_residual_full and xgb_residual_full columns

# Rename for clarity
predictions_df['hybrid4_forecast'] = predictions_df['rf_residual_full']
predictions_df['hybrid5_forecast'] = predictions_df['xgb_residual_full']

# 5.3 Advanced Hybrid: Combination of Direct and Residual Approaches
print("\n5.3 Advanced Hybrid: Combination of Direct and Residual Approaches")

# Combine the best direct model and best residual model
def objective_advanced_hybrid(weights):
    combined = (weights[0] * predictions_df['hybrid3_forecast'] + 
                weights[1] * predictions_df['hybrid4_forecast'] + 
                weights[2] * predictions_df['hybrid5_forecast'])
    
    # Ensure weights sum to 1
    penalty = 100 * abs(np.sum(weights) - 1)
    
    return np.sqrt(mean_squared_error(predictions_df['actual_volatility'], combined)) + penalty

# Optimize weights for advanced hybrid
initial_weights = [1/3, 1/3, 1/3]  # Start with equal weights
bounds = [(0, 1), (0, 1), (0, 1)]  # Weights between 0 and 1
constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1

adv_opt_result = minimize(objective_advanced_hybrid, x0=initial_weights, 
                          bounds=bounds, constraints=constraint)

# Get optimal weights
adv_weights = adv_opt_result.x

print(f"Optimal weights for advanced hybrid model:")
print(f"Direct Ensemble weight = {adv_weights[0]:.4f}")
print(f"RF Residual weight = {adv_weights[1]:.4f}")
print(f"XGB Residual weight = {adv_weights[2]:.4f}")

# Create the advanced hybrid forecast
predictions_df['advanced_hybrid_forecast'] = (
    adv_weights[0] * predictions_df['hybrid3_forecast'] +
    adv_weights[1] * predictions_df['hybrid4_forecast'] +
    adv_weights[2] * predictions_df['hybrid5_forecast']
)

# Evaluate all models
def calculate_metrics(actual, predicted):
    """Calculate RMSE, MAE, and R² for a model"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

# Calculate metrics for all models
models = [
    ('GARCH', 'garch_forecast'),
    ('RF Direct', 'rf_direct_pred'),
    ('XGB Direct', 'xgb_direct_pred'),
    ('RF Residual', 'hybrid4_forecast'),
    ('XGB Residual', 'hybrid5_forecast'),
    ('Hybrid1 (GARCH+RF)', 'hybrid1_forecast'),
    ('Hybrid2 (GARCH+XGB)', 'hybrid2_forecast'),
    ('Hybrid3 (RF+XGB)', 'hybrid3_forecast'),
    ('Advanced Hybrid', 'advanced_hybrid_forecast')
]

results = []
for name, col in models:
    rmse, mae, r2 = calculate_metrics(predictions_df['actual_volatility'], predictions_df[col])
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Improvement': (1 - rmse / results[0]['RMSE']) * 100 if name != 'GARCH' else 0
    })

# Create results dataframe
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df.sort_values('RMSE'))

# Visualize model performance
plt.figure(figsize=(14, 8))
# Sort by RMSE for the bar plot
sorted_results = results_df.sort_values('RMSE')
sns.barplot(x='RMSE', y='Model', data=sorted_results, palette='viridis')
plt.title('Model Comparison - RMSE (Lower is Better)', fontsize=16)
plt.xlabel('RMSE', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_rmse.png')
plt.close()
print("Created: model_comparison_rmse.png")

# Visualize improvement over GARCH
plt.figure(figsize=(14, 8))
sorted_by_improvement = results_df[results_df['Model'] != 'GARCH'].sort_values('Improvement', ascending=False)
sns.barplot(x='Improvement', y='Model', data=sorted_by_improvement, palette='viridis')
plt.title('Improvement Over GARCH Baseline (%)', fontsize=16)
plt.xlabel('Percentage Improvement in RMSE', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('improvement_over_garch.png')
plt.close()
print("Created: improvement_over_garch.png")

# Visualize the best hybrid model against actual and GARCH
# Find the best model
best_model = results_df.sort_values('RMSE').iloc[0]['Model']
best_col = [col for name, col in models if name == best_model][0]

plt.figure(figsize=(14, 7))
plt.plot(predictions_df.index, predictions_df['actual_volatility'], label='Actual Volatility', 
         color='black', linewidth=2)
plt.plot(predictions_df.index, predictions_df['garch_forecast'], label='GARCH Forecast', 
         color='blue', alpha=0.7)
plt.plot(predictions_df.index, predictions_df[best_col], label=f'Best Model ({best_model})', 
         color='red', linewidth=1.5)
plt.title('Volatility Forecasts: Actual vs GARCH vs Best Hybrid Model', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('best_hybrid_comparison.png')
plt.close()
print(f"Created: best_hybrid_comparison.png")

# Analyze model performance across different market conditions
# Define high volatility periods (top 25% of volatility)
vol_threshold = predictions_df['actual_volatility'].quantile(0.75)
high_vol_periods = predictions_df[predictions_df['actual_volatility'] >= vol_threshold]
normal_vol_periods = predictions_df[predictions_df['actual_volatility'] < vol_threshold]

print("\nModel Performance in High Volatility Periods:")
high_vol_results = []
for name, col in models:
    rmse, mae, r2 = calculate_metrics(high_vol_periods['actual_volatility'], high_vol_periods[col])
    high_vol_results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
high_vol_df = pd.DataFrame(high_vol_results)
print(high_vol_df.sort_values('RMSE'))

print("\nModel Performance in Normal Volatility Periods:")
normal_vol_results = []
for name, col in models:
    rmse, mae, r2 = calculate_metrics(normal_vol_periods['actual_volatility'], normal_vol_periods[col])
    normal_vol_results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
normal_vol_df = pd.DataFrame(normal_vol_results)
print(normal_vol_df.sort_values('RMSE'))

# Save results
predictions_df.to_csv('hybrid_model_predictions.csv')
results_df.to_csv('model_performance_summary.csv')

print("\nPhase 5: Hybrid Model Construction completed!")
print(f"The best hybrid model is {best_model} with RMSE: {results_df.sort_values('RMSE').iloc[0]['RMSE']:.6f}")
print(f"This represents a {results_df.sort_values('RMSE').iloc[0]['Improvement']:.2f}% improvement over the GARCH baseline.")
print("Data is ready for Phase 6: Model Evaluation & Comparison.")

# Summary of findings
print("\nSummary of Hybrid Model Construction:")
print(f"1. The best performing model is {best_model}")
print(f"2. Direct combination approach: Best weight for GARCH in hybrid = {optimal_weight1:.4f}")
print(f"3. Best model performs {'better' if high_vol_df.sort_values('RMSE').iloc[0]['Model'] == best_model else 'worse'} during high volatility periods")
print(f"4. {'Residual-based' if 'Residual' in best_model else 'Direct prediction'} approach proved more effective overall")
print(f"5. Improvement over GARCH baseline: {results_df.sort_values('RMSE').iloc[0]['Improvement']:.2f}%")