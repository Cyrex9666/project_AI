import pandas as pd

def create_direction_target(data):

    # Creates target variable from OHLCV data.

    # For each date:
    # target = 1 if tomorrow's closing price is higher than today's closing price
    # target = 0 if tomorrow's closing price is lower than or equal to today's closing price


    close = data["Close"]

    # Safety check: if Close is accidentally a DataFrame, convert it to a Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    tomorrow_return = close.pct_change().shift(-1)

    target = (tomorrow_return > 0).astype(int)

    # Final row has no tomorrow, so mark it as NA
    target[tomorrow_return.isna()] = pd.NA

    return target