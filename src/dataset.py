from data_extraction import download_full_data
from data_cleaning import clean_market_data
from stock_features import create_stock_features1, create_stock_features2, create_stock_features3
from targets import create_direction_target

    # Final processing of the data pipeline.

    # Creates:
    # X = model input features
    # y = target variable

    # feature_set=1 uses the original 7 close-price features.
    # feature_set=2 uses the improved OHLCV feature set.
def create_model_dataset(data, feature_set):
    if feature_set == 1:
        X = create_stock_features1(data)
    elif feature_set == 2:
        X = create_stock_features2(data)
    elif feature_set == 3:
        X = create_stock_features3(data)

    else:
        raise ValueError("feature_set must be either 1, 2 or 3.")

    y = create_direction_target(data)

    dataset = X.copy()
    dataset["target"] = y

    # Remove rows where features/target could not be calculated.
    # Example:
    # - 20-day moving average cannot be calculated for the first 19 rows
    # - tomorrow's return cannot be calculated for the final row
    dataset = dataset.dropna()

    return dataset


def build_dataset(ticker, start_date, end_date, feature_set):

    # Downloads, cleans, and prepares the final model dataset
    # for one selected stock


    raw_data = download_full_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )

    clean_data = clean_market_data(raw_data)

    dataset = create_model_dataset(
        data=clean_data,
        feature_set=feature_set
    )

    return dataset