import pandas as pd
from pathlib import Path


CLEAN_DATA_PATH = Path("data/clean_prices.csv")
FEATURES_DATA_PATH = Path("data/features.csv")


def load_clean_prices(file_path):
    # loads cleaned price data from CSV
    prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return prices


def calculate_daily_returns(prices):
    # calculates daily percentage returns for each asset
    returns = prices.pct_change()
    return returns

def create_features(prices, returns, target_stock):

    """
    Creates model-ready financial features from cleaned price data.

    Features created:
    1. Daily returns
    2. 5-day momentum
    3. 20-day momentum
    4. 5-day rolling volatility
    5. 20-day rolling volatility
    6. 5-day moving average ratio
    7. 20-day moving average ratio
    """
    features = pd.DataFrame(index=prices.index)

    # Daily return
    features[f"{target_stock}_daily_return"] = returns[target_stock]

    # Momentum features
    features[f"{target_stock}_5d_momentum"] = prices[target_stock].pct_change(periods=5)
    features[f"{target_stock}_20d_momentum"] = prices[target_stock].pct_change(periods=20)

    # Rolling volatility features
    features[f"{target_stock}_5d_rolling_volatility"] = (returns[target_stock].rolling(window=5).std())

    features[f"{target_stock}_20d_rolling_volatility"] = (returns[target_stock].rolling(window=20).std())

    # Moving average ratio features
    features[f"{target_stock}_5d_ma_ratio"] = (prices[target_stock] / prices[target_stock].rolling(window=5).mean())

    features[f"{target_stock}_20d_ma_ratio"] = (prices[target_stock] / prices[target_stock].rolling(window=20).mean())
    return features


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    clean_prices = load_clean_prices(CLEAN_DATA_PATH)

    features = create_features(clean_prices)

    features.to_csv(FEATURES_DATA_PATH)

    print(f"Saved engineered features to {FEATURES_DATA_PATH}")
    print(features.head())
    print()
    print("Feature columns:")
    print(features.columns.tolist())
    print()
    print("Missing values after feature engineering:")
    print(features.isna().sum())


if __name__ == "__main__":
    main()