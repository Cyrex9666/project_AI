import yfinance as yf
import pandas as pd

# downloads Open, High, Low, Close, Volume data for selected stock(s)
def download_full_data(ticker, start_date, end_date):
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data