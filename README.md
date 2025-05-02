# Hybrid GARCH-ML Model for Bitcoin Volatility Forecasting

## Project Overview

This project implements a hybrid approach for forecasting Bitcoin price volatility by combining traditional GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models with modern Machine Learning techniques. The hybrid model leverages the strengths of both approaches to achieve superior forecasting accuracy compared to standard GARCH models.

## Table of Contents

1. [Data Description](#data-description)
2. [Project Structure](#project-structure)
3. [Implementation Phases](#implementation-phases)
4. [Key Findings](#key-findings)
5. [Technical Requirements](#technical-requirements)
6. [How to Run](#how-to-run)
7. [Results](#results)

## Data Description

- **Source**: Bitcoin daily OHLCV data (Open, High, Low, Close, Volume) from 2018-01-01 to 2025-04-28
- **File**: `btc_1d_data_2018_to_2025.csv`
- **Frequency**: Daily (1d)

## Project Structure

The implementation follows a structured approach divided into 7 phases:

```
├── bitcoin-exploratory-analysis.py    # Phase 1: Data Preparation & EDA
├── garch-baseline-model.py            # Phase 2: GARCH Modeling (Baseline)
├── feature-engineering.py             # Phase 3: Feature Engineering
├── ml-implementation.py               # Phase 4: Machine Learning Implementation
├── hybrid-model-construction.py       # Phase 5: Hybrid Model Construction
├── model-evaluation-visualization.py  # Phase 6 & 7: Evaluation and Visualization
├── README.md                          # Project documentation
└── data/
    ├── btc_1d_data_2018_to_2025.csv   # Raw Bitcoin price data
    ├── processed_bitcoin_data.csv     # Processed data from Phase 1
    ├── bitcoin_data_with_garch.csv    # Dataset with GARCH volatility from Phase 2
    ├── ml_ready_data.csv              # Feature-engineered data from Phase 3
    ├── ml_predictions.csv             # ML model predictions from Phase 4
    ├── hybrid_model_predictions.csv   # Hybrid model predictions from Phase 5
    └── model_performance_summary.csv  # Final performance metrics
```

## Implementation Phases

### Phase 1: Data Preparation & Exploratory Analysis
- Data loading and cleaning
- Calculation of returns and volatility measures (Garman-Klass estimator)
- Statistical analysis of returns and volatility
- Visualization of Bitcoin price evolution and volatility patterns

### Phase 2: GARCH Modeling (Baseline)
- Implementation of GARCH(1,1) model using the `arch` library
- Parameter estimation and interpretation
- One-step ahead conditional variance forecasts
- Baseline performance evaluation

### Phase 3: Feature Engineering for Machine Learning
- Creation of lagged features (returns, volatility, GARCH forecast)
- Rolling window statistics
- Technical indicators (Bollinger Bands, ATR, RSI, MACD)
- Feature selection and correlation analysis

### Phase 4: Machine Learning Implementation
- Chronological data splitting (training/test)
- Training of Random Forest and XGBoost models
- Hyperparameter tuning via cross-validation
- Feature importance analysis
- Two prediction approaches:
  - Direct prediction of realized volatility
  - Prediction of residuals (realized volatility - GARCH forecast)

### Phase 5: Hybrid Model Construction
- Weighted average combination approach
- Residual prediction approach
- Optimization of combination weights
- Performance comparison of different hybrid strategies

### Phase 6 & 7: Model Evaluation, Comparison and Visualization
- Comprehensive performance evaluation across different metrics
- Analysis of model performance in different market conditions
- Error distribution analysis
- Creation of visualizations for result interpretation
- Final conclusions and recommendations

## Key Findings

1. The hybrid model achieved significant improvement over the traditional GARCH model, with approximately 15-25% reduction in forecasting errors.
2. The best performing model was the residual-based approach using XGBoost to predict the difference between realized volatility and GARCH forecasts.
3. Machine learning components were particularly valuable during high volatility periods, where GARCH models tend to underestimate volatility.
4. The most important features for volatility prediction included recent volatility measures, trading volume indicators, and price momentum.
5. The hybrid approach demonstrated better directional accuracy in predicting volatility changes compared to GARCH alone.

## Technical Requirements

### Core Libraries
- **Data manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Econometrics**: arch
- **Machine Learning**: scikit-learn, xgboost

### Environment Setup
- Python 3.x
- Jupyter Notebook/Lab (optional, for interactive exploration)

## How to Run

1. Install the required dependencies:
```
pip install pandas numpy matplotlib seaborn arch scikit-learn xgboost
```

2. Run the phases in sequence:
```
python bitcoin-exploratory-analysis.py
python garch-baseline-model.py
python feature-engineering.py
python ml-implementation.py
python hybrid-model-construction.py
python model-evaluation-visualization.py
```

Alternatively, you can run the Jupyter notebooks in sequence if you prefer an interactive approach.

## Results

The project demonstrates that hybrid GARCH-ML models provide superior Bitcoin volatility forecasts compared to traditional GARCH models. The key improvements include:

1. Reduced forecasting errors (RMSE, MAE)
2. Better directional accuracy
3. More consistent performance across different market conditions
4. Ability to incorporate additional information beyond historical volatility

These improvements make the hybrid approach valuable for applications such as risk management, options pricing, and trading strategy development in the Bitcoin market.

## License

MIT License