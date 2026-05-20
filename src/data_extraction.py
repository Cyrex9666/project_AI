import yfinance as yf

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