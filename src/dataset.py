import pandas as pd

from data_extraction import download_close_price_data
from data_cleaning import clean_prices
from stock_features import calculate_daily_returns, create_features
from targets import create_direction_target

# Final processing of the data pipeline, creating the X inputs and Y targets
def create_model_dataset(prices, target_stock):

    returns = calculate_daily_returns(prices)

    X = create_features(
        prices=prices,
        returns=returns,
        target_stock=target_stock
    )

    y = create_direction_target(
        returns=returns,
        target_stock=target_stock
    )

    dataset = X.copy()
    dataset["target"] = y

    # remove rows where features/target could not be calculated, example 5 day ma @ day 1...
    dataset = dataset.dropna()

    return dataset

def build_dataset(tickers, target_stock, start_date, end_date):
    raw_prices = download_close_price_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )

    clean_price_data = clean_prices(raw_prices)

    dataset = create_model_dataset(
        prices=clean_price_data,
        target_stock=target_stock
    )

    return dataset

