import yfinance as yf
import pandas as pd
from pathlib import Path

# Commonwealth bank, Westpac, National Australia Bank, ASX200 index
TICKERS = ["CBA.AX", "WBC.AX", "NAB.AX", "ANZ.AX", "^AXJO"]
START_DATE = "2018-01-01"
END_DATE = None

def download_close_price_data(tickers, start_date, end_date):
    # downloads the closing price of the selected stock
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    # returns closing column per stock form start_date to end_date
    prices = data["Close"]
    return prices

def main():
    # this creates a file path for the incoming raw_prices.csv
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    prices = download_close_price_data(TICKERS, START_DATE, END_DATE)

    output_path = output_dir / "raw_prices.csv"
    prices.to_csv(output_path)

    
    print(f"Saved price data to {output_path}")
    print(prices.head())


if __name__ == "__main__":
    main()