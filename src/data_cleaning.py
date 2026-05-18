import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw_prices.csv")
CLEAN_DATA_PATH = Path("data/clean_prices.csv")


def load_raw_prices(file_path):
    # fetches prices 
    prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return prices


def clean_prices(prices):
    # for each col, check each row:
    # 1. since index is date, sort prices from oldest to newest
    # 2. remove duplicate dates
    # 3. copy previous known date if no price existed
    # 4. copy next known date if no price existed
    # 5. remove columns that are completely empty
    prices = prices.sort_index() 

    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.ffill()
    prices = prices.bfill()
    prices = prices.dropna(axis=1, how="all")

    return prices


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    raw_prices = load_raw_prices(RAW_DATA_PATH)

    clean_data = clean_prices(raw_prices)

    clean_data.to_csv(CLEAN_DATA_PATH)

    print(f"Saved cleaned price data to {CLEAN_DATA_PATH}")
    print(clean_data.head())
    print()
    print("Missing values after cleaning:")
    print(clean_data.isna().sum())


if __name__ == "__main__":
    main()