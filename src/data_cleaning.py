import pandas as pd

def load_raw_prices(file_path):
    # fetches prices 
    prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return prices


#   for each col, check each row:
#   1. since index is date, sort prices from oldest to newest
#   2. remove duplicate dates
#   3. remove columns that are completely empty
#   4. copy previous known date if no price existed

def clean_prices(prices):
    prices = prices.sort_index() 

    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill()

    return prices