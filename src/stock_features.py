import pandas as pd

def calculate_daily_returns(prices):
    # calculates daily percentage returns for each asset
    returns = prices.pct_change()
    return returns

#    Creates stock features
#    1. Daily returns
#    2. 5-day momentum
#    3. 20-day momentum
#    4. 5-day rolling volatility
#    5. 20-day rolling volatility
#    6. 5-day moving average ratio
#    7. 20-day moving average ratio

def create_features(prices, returns, target_stock):
    # error raising
    if target_stock not in prices.columns:
        raise ValueError(f"{target_stock} not found in prices data.")

    if target_stock not in returns.columns:
        raise ValueError(f"{target_stock} not found in returns data.")
    
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