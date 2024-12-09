# Regression Models Comparative Analysis
Understanding Machine Learning Regression Models - This sample python code predicts stock prices for the next five days using historical data through various regression models.

## Objective
* Understanding the behaviors of regression models  
* Practical insights for implementing models  
* Comparing different algorithmic approaches  
* Learning how models interpret financial data

## Regression Models Comparison Matrix

| Model | Best Use Case | Advantages | Disadvantages |
| ----- | ------------- | ---------- | ------------- |
| Linear Regression | Predicting linear trends | Easy to interpret, fast computation | Limited to linear relationships, sensitive to outliers |
| Ridge Regression | Handling multi-collinearity | Prevents overfitting, stable predictions | Doesn't perform feature selection | 
| Lasso Regression | Feature selection | Reduces model complexity, handles high-dimensional data | Can underperform with highly correlated features |
| Random Forest | Complex, non-linear data | Handles multiple feature interactions, robust to outliers | Less interpretable, computationally expensive |
| Gradient Boosting | Intricate pattern recognition | High predictive accuracy, handles complex interactions | Requires careful hyperparameter tuning, prone to overfitting |
| K-Nearest Neighbors | Localized pattern recognition | Simple, no assumptions about data distribution | Computationally expensive for large datasets, sensitive to outliers |
| Neural Network| Complex non-linear patterns. deep feature learning | Captures complex non-linear patterns, highly flexible and accurate | Risk of overfitting, requires careful tuning |

## Model Training Constants - Configuration Parameters

* `'TEST_SIZE'` = 0.4: 40% of data for testing
* `'RANDOM_STATE'` = 42: Ensures reproducibility
* `'NUM_ESTIMATORS'` = 50: Number of trees/estimators
* `'ALPHA '`= 1.0: Regularization strength
* `'MAX_ITER'` = 10000: Maximum training iterations

## The Coding Steps:
* Data Download: Stock data is obtained from Yahoo Finance via the yfinance library.
* Data Preparation: The data is processed by creating target columns, removing missing values, and scaling with StandardScaler.
* Model Training and Evaluation: Regression models are trained and evaluated using the `'train_test_split'` function from `'sklearn.model_selection'`.
* Prediction: The models predict stock prices for the next five days.
* Performance Evaluation: Model performance is measured using mean squared error (MSE), mean absolute error (MAE), and accuracy.

## The Coding Approach
* Historical Data Analysis: Pulls stock data for one year, including `'Open'`, `'High'`, `'Low'`, `'Close'`, and `'Volume'`.
* Multiple Regression Models: Implements and evaluates models such as `'Linear Regression'`, `'Ridge'`, `'Lasso'`, `'Random Forest'`, `'Gradient Boosting'`, `'K-Nearest Neighbors'`, and `'Neural Networks'`.
* Performance Metrics: Reports `'Mean Squared Error'` (MSE), `'Mean Absolute Error'` (MAE), and `'accuracy'` for each model.

## Python Libraries

| Library | Purpose |
|---------|---------|
| sys | Command-line argument parsing |
| yfinance | Stock data retrieval |
| pandas | Data manipulation and cleaning |
| numpy | Numerical computations |
| scikit-learn | Machine learning models and metrics |

    import sys
    import yfinance
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.neural_network import MLPRegressor

## Training Strategy - Function Analysis: `'train_evaluate_models()'`

| Model Initialization | Parameters |
|----------------------|------------|
| RandomForestRegressor | `'n_estimators=50'`: Creates 50 decision trees. `'random_state=42'`: Ensures consistent results across runs |
| RidgeRegression | `'alpha=1.0'`: Regularization strength to prevent overfitting |
| LassoRegression | `'alpha=1.0'`: Feature selection and regularization. `'max_iter=10000'`: Allows more iterations for convergence |
| GradientBoostingRegressor | Similar to RandomForest, but uses sequential model improvement |
| KNeighborsRegressor | `'n_neighbors=5'`: Considers 5 closest data points for prediction |
| NeuralNetworkRegressor | `'hidden_layer_sizes=(50, 100)'`: Two hidden layers. `'early_stopping=True'`: Prevents model overfitting |

        regression_models = {
            "RandomForestRegressor": RandomForestRegressor(n_estimators=NUM_ESTIMATORS, random_state=RANDOM_STATE),
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(alpha=ALPHA),
            "LassoRegression": Lasso(alpha=ALPHA, max_iter=MAX_ITER),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=NUM_ESTIMATORS, random_state=RANDOM_STATE),
            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
            "NeuralNetworkRegressor": MLPRegressor(hidden_layer_sizes=(50, 100), max_iter=MAX_ITER, random_state=RANDOM_STATE, tol=0.001, early_stopping=True)
        }

    
| Training Strategy |  |
|-------------------|------|
| Iterative Model Training | Trains each model across multiple prediction horizons. Comprehensive performance assessment. |
| Performance Metrics Calculation | Computes prediction errors. `'Mean Squared Error (MSE)'` , `'Mean Absolute Error (MAE)'` and aggregates errors across different prediction days. |
| Accuracy Computation | Normalizes error against maximum target value. Percentage-based performance indicator. |
| Purpose and Insights | Demonstrate model-specific behaviors. Compare predictive capabilities. Understand regression model nuances. |

      for name, model in regression_models.items():
              mse, mae = 0, 0
              predictions_list = []
              for day_target_train, day_target_test in zip(y_train, y_test):
                  model.fit(X_train, day_target_train)
                  predictions = predict_next_5_days_regression(model, X_test)
                  mse += mean_squared_error(day_target_test, predictions)
                  mae += mean_absolute_error(day_target_test, predictions)
                  predictions_list.append(predictions[0])
              avg_mse, avg_mae = mse / len(y_train), mae / len(y_train)
              accuracy = 1 - (avg_mae / max([max(y) for y in y_test]))
              print(f"{name}: MSE = {avg_mse:.2f}, MAE = {avg_mae:.2f}, Accuracy = {100 * accuracy:.2f}%, Prediction for next 5 days - {predictions_list}")


## Performance Metrics 
| Metric | Description |
|-------|--------|
| Mean Squared Error (MSE) | Measures average squared difference between predictions and actual values. Penalizes larger errors more significantly. Helps identify models with consistent predictions |
| Mean Absolute Error (MAE)| Measures average absolute difference between predictions and actual values. Provides straightforward error interpretation. Less sensitive to extreme outliers |
| Custom Accuracy Percentage | Normalizes error against data scale. Provides intuitive performance understanding. Enables cross-model comparability Evaluates|


## How It Works
* Input Stock Ticker: Run the [stock.py](./stock.py) with a stock ticker, e.g., 
  
      python stock.py AAPL

* Data Download: Fetches one year of stock data from Yahoo Finance.
* Preprocessing: Scales features. Splits data into training and testing sets.
* Model Training: Trains models on historical data.
* Evaluate Metrics: Evaluate their performance.
* Prediction: Outputs predictions for the next 5 days and key metrics.
  
<img src="./PredictionResult.png">


## What We Learn
* No single "best" model
* Model performance varies with data characteristics
* Importance of systematic model comparison
* Understanding model-specific strengths and limitations


## Future Exploration
* Advanced feature engineering
* Hyperparameter optimization
* Expanded model comparison techniques


⚠️ Methodology Disclaimer: Focused on model comparison, not definitive predictions.
