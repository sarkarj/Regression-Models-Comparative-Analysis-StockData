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

# Constants
TEST_SIZE = 0.4
RANDOM_STATE = 42
NUM_ESTIMATORS = 50
ALPHA = 1.0
MAX_ITER = 10000


def download_stock_data(ticker: str, period: str = '1y') -> pd.DataFrame:
    """
    Downloads stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        period (str, optional): Time period for data download. Defaults to '1y'.

    Returns:
        pd.DataFrame: Stock data.
    """
    try:
        stock_data = yfinance.download(ticker, period=period)
        data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Create target columns
        for i in range(1, 6):
            data[f'Target_{i}'] = data['Close'].shift(-i)

        # Remove rows with missing values
        data.dropna(inplace=True)

        # Round numeric columns to 2 decimal places
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data[numeric_cols] = data[numeric_cols].round(2)

        return data

    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)


def prepare_data(data: pd.DataFrame) -> tuple:
    """
    Prepares data for training.

    Args:
        data (pd.DataFrame): Stock data.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Define feature columns
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = data[feature_cols].values

    # Define target columns
    y_targets = [data[f'Target_{i}'].values for i in range(1, 6)]

    # Split data into training and testing sets
    X_train, X_test = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)
    y_train = [y[:len(X_train)] for y in y_targets]
    y_test = [y[len(X_train):] for y in y_targets]

    # Scale data using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def predict_next_5_days_regression(model, X_test_scaled: np.ndarray) -> list:
    """
    Predicts next 5 days using regression model.

    Args:
        model: Regression model.
        X_test_scaled (np.ndarray): Scaled test data.

    Returns:
        list: Predictions for next 5 days.
    """
    predictions = model.predict(X_test_scaled)
    predictions_rounded = np.round(predictions, 2)
    return predictions_rounded.tolist()


def train_evaluate_models(X_train: np.ndarray, X_test: np.ndarray, y_train: list, y_test: list) -> None:
    """
    Trains and evaluates models.

    Args:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Test data.
        y_train (list): Training target values.
        y_test (list): Test target values.
    """
    regression_models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=NUM_ESTIMATORS, random_state=RANDOM_STATE),
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=ALPHA),
        "LassoRegression": Lasso(alpha=ALPHA, max_iter=MAX_ITER),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=NUM_ESTIMATORS, random_state=RANDOM_STATE),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
        "NeuralNetworkRegressor": MLPRegressor(hidden_layer_sizes=(50, 100), max_iter=MAX_ITER, random_state=RANDOM_STATE, tol=0.001, early_stopping=True)
    }

    print("\nRegression Models:")
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


def main() -> None:
    if len(sys.argv) < 2:
        print("Please provide a stock ticker. Usage: python stock_models.py AAPL")
        sys.exit(1)

    ticker = sys.argv[1]

    stock_data = download_stock_data(ticker)

    X_train, X_test, y_train, y_test = prepare_data(stock_data)

    train_evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()