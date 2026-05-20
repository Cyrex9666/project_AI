import pandas as pd

# if tomorrows return is positive then 1, else 0
def convert_return_to_target(tomorrow_return):
    
    # incase tomorrow cannot be retrieved since indexing ahead, then return an NA
    # so that dropna can be used in dataset
    if pd.isna(tomorrow_return):
        return pd.NA
    
    if tomorrow_return > 0:
        return 1
    else:
        return 0




#   creates the target variable for the model.
#
#   for each date
#   - target = 1 if the stock goes up the next trading day
#   - target = 0 if the stock goes down or remains the next trading day

def create_direction_target(returns, target_stock):

    if target_stock not in returns.columns:
        raise ValueError(f"{target_stock} not found in returns data.")

    # daily return
    stock_returns = returns[target_stock]

    # target of today is tomorrows price, therefore let tomorrows's return be todays target
    tomorrow_returns = stock_returns.shift(-1)
    target = tomorrow_returns.apply(convert_return_to_target)

    return target