from dataset import build_dataset


tickers = ["CBA.AX", "WBC.AX", "NAB.AX", "ANZ.AX"]
target_stock = "CBA.AX"

dataset = build_dataset(
    tickers=tickers,
    target_stock=target_stock,
    start_date="2015-01-01",
    end_date="2025-01-01"
)
print(dataset.head())
print(dataset.tail())
print(dataset.shape)