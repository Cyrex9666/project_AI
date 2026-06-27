import pandas as pd
import numpy as np


# percent change in price
def calculate_daily_return(close):
    return close.pct_change()

# percent change in closing price over a window
def calculate_momentum(close, window):
    return close.pct_change(periods=window)

# standard deviation of prices within window
def calculate_rolling_volatility(close, window):
    returns = close.pct_change()
    return returns.rolling(window=window).std()

# compares closing price to the moving average for a given window
def calculate_ma_ratio(close, window):
    moving_average = close.rolling(window=window).mean()
    return close / moving_average

# relative strength index (0-100)
def calculate_rsi(close, window=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(window=window).mean()
    average_loss = loss.rolling(window=window).mean()

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi

# movin average convergence/divergence
def calculate_macd_histogram(close, short_window=12, long_window=26, signal_window=9):
    # ema_short is the 12 day exponential moving average
    # ema_long is the 26 day exponential moving average
    ema_short = close.ewm(span=short_window, adjust=False).mean()
    ema_long = close.ewm(span=long_window, adjust=False).mean()

    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()

    macd_histogram = macd - macd_signal

    return macd_histogram

# bolinger band percent tells the model where the price sits relatiev to those bands
# (0 - 1, where 0.5 is in the middle)
def calculate_bollinger_percent_b(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()

    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    percent_b = (close - lower_band) / (upper_band - lower_band)

    return percent_b

# distance between bands
def calculate_bollinger_band_width(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()

    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    band_width = (upper_band - lower_band) / rolling_mean

    return band_width

# compares current volume to average volume 
def calculate_volume_ratio(volume, window=20):
    average_volume = volume.rolling(window=window).mean()
    return volume / average_volume

# calculates intraday range
def calculate_daily_range(high, low, close):
    return (high - low) / close


def calculate_atr_ratio(high, low, close, window=14):
    previous_close = close.shift(1)

    high_low = high - low
    high_previous_close = (high - previous_close).abs()
    low_previous_close = (low - previous_close).abs()

    true_range = pd.concat(
        [high_low, high_previous_close, low_previous_close],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(window=window).mean()

    atr_ratio = atr / close

    return atr_ratio

# featureset 3 from the study
# typical price: average of high, low, and close
def calculate_hlc3_ratio(high, low, close, window=20):
    hlc3 = (high + low + close) / 3
    hlc3_average = hlc3.rolling(window=window).mean()

    return hlc3 / hlc3_average


# exponential moving average ratio
def calculate_ema_ratio(close, window=20):
    ema = close.ewm(span=window, adjust=False).mean()

    return close / ema


# triple exponential moving average ratio
def calculate_tema_ratio(close, window=20):
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()

    tema = (3 * ema1) - (3 * ema2) + ema3

    return close / tema


# weighted moving average helper
def calculate_wma(series, window):
    weights = np.arange(1, window + 1)

    wma = series.rolling(window=window).apply(
        lambda values: np.dot(values, weights) / weights.sum(),
        raw=True
    )

    return wma


# fibonacci weighted moving average ratio
def calculate_fwma_ratio(close, window=13):
    fibonacci_weights = []

    a = 1
    b = 1

    for i in range(window):
        fibonacci_weights.append(a)
        a, b = b, a + b

    weights = np.array(fibonacci_weights)

    fwma = close.rolling(window=window).apply(
        lambda values: np.dot(values, weights) / weights.sum(),
        raw=True
    )

    return close / fwma


# hull moving average ratio
def calculate_hma_ratio(close, window=20):
    half_window = int(window / 2)
    sqrt_window = int(np.sqrt(window))

    wma_half = calculate_wma(close, half_window)
    wma_full = calculate_wma(close, window)

    raw_hma = (2 * wma_half) - wma_full
    hma = calculate_wma(raw_hma, sqrt_window)

    return close / hma


# simple holt-winter style moving average ratio
# this is a smoothed trend estimate using level + trend
def calculate_hwma_ratio(close, alpha=0.2, beta=0.1):
    hwma = pd.Series(index=close.index, dtype=float)

    level = close.iloc[0]
    trend = 0

    for i in range(len(close)):
        price = close.iloc[i]

        if i == 0:
            hwma.iloc[i] = price
            continue

        previous_level = level

        level = (alpha * price) + ((1 - alpha) * (level + trend))
        trend = (beta * (level - previous_level)) + ((1 - beta) * trend)

        hwma.iloc[i] = level + trend

    return close / hwma


# kaufman adaptive moving average ratio
def calculate_kama_ratio(close, er_window=10, fast_window=2, slow_window=30):
    price_change = (close - close.shift(er_window)).abs()
    volatility = close.diff().abs().rolling(window=er_window).sum()

    efficiency_ratio = price_change / volatility

    fast_sc = 2 / (fast_window + 1)
    slow_sc = 2 / (slow_window + 1)

    smoothing_constant = (
        efficiency_ratio * (fast_sc - slow_sc) + slow_sc
    ) ** 2

    kama = pd.Series(index=close.index, dtype=float)

    kama.iloc[0] = close.iloc[0]

    for i in range(1, len(close)):
        if pd.isna(smoothing_constant.iloc[i]):
            kama.iloc[i] = pd.NA
        else:
            if pd.isna(kama.iloc[i - 1]):
                previous_kama = close.iloc[i - 1]
            else:
                previous_kama = kama.iloc[i - 1]

            kama.iloc[i] = previous_kama + smoothing_constant.iloc[i] * (
                close.iloc[i] - previous_kama
            )

    return close / kama


# symmetric weighted moving average ratio
def calculate_swma_ratio(close, window=5):
    if window % 2 == 0:
        raise ValueError("SWMA window should be odd, for example 5 or 7.")

    midpoint = window // 2

    weights = []

    for i in range(window):
        weight = midpoint + 1 - abs(i - midpoint)
        weights.append(weight)

    weights = np.array(weights)

    swma = close.rolling(window=window).apply(
        lambda values: np.dot(values, weights) / weights.sum(),
        raw=True
    )

    return close / swma


# on-balance volume change
# normalised by average volume so the numbers do not become massive
def calculate_obv_change(close, volume, window=20):
    price_direction = np.sign(close.diff()).fillna(0)

    obv = (price_direction * volume).cumsum()

    average_volume = volume.rolling(window=window).mean()

    obv_change = obv.diff() / average_volume

    return obv_change


# price volume trend change
# also normalised by average volume
def calculate_pvt_change(close, volume, window=20):
    daily_return = close.pct_change()

    pvt_daily_change = daily_return * volume

    average_volume = volume.rolling(window=window).mean()

    pvt_change = pvt_daily_change / average_volume

    return pvt_change


# ichimoku-style trend signal
# important: we do not use forward-shifted cloud values because that can leak future information
def calculate_ichimoku_signal(high, low, close):
    tenkan_sen = (
        high.rolling(window=9).max() + low.rolling(window=9).min()
    ) / 2

    kijun_sen = (
        high.rolling(window=26).max() + low.rolling(window=26).min()
    ) / 2

    ichimoku_signal = (tenkan_sen - kijun_sen) / close

    return ichimoku_signal

def create_stock_features1(data):
    
    # Improved feature set.

    # Uses OHLCV data.

    # Features:
    # 1. Daily return
    # 2. 5-day momentum
    # 3. 5-day rolling volatility
    # 4. 5-day moving average ratio
    # 5. RSI 14
    # 6. MACD histogram
    # 7. Bollinger percent B
    # 8. Bollinger band width
    # 9. Volume ratio 20d
    # 10. Daily range
    # 11. ATR 14 ratio

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    features = pd.DataFrame(index=data.index)

    features["daily_return"] = calculate_daily_return(close)

    features["5d_momentum"] = calculate_momentum(close, window=5)

    features["5d_rolling_volatility"] = calculate_rolling_volatility(close, window=5)

    features["5d_ma_ratio"] = calculate_ma_ratio(close, window=5)

    features["rsi_14"] = calculate_rsi(close, window=14)

    features["macd_histogram"] = calculate_macd_histogram(close)

    features["bollinger_percent_b"] = calculate_bollinger_percent_b(close)

    features["bollinger_band_width"] = calculate_bollinger_band_width(close)

    features["volume_ratio_20d"] = calculate_volume_ratio(volume, window=20)

    features["daily_range"] = calculate_daily_range(high, low, close)

    features["atr_14_ratio"] = calculate_atr_ratio(high, low, close, window=14)

    return features


def create_stock_features2(data):

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    features = pd.DataFrame(index=data.index)

    # momentum / trend direction 
    features["daily_return"] = calculate_daily_return(close)
    features["5d_momentum"] = calculate_momentum(close, window=5)
    features["macd_histogram"] = calculate_macd_histogram(close)

    # reversion / extension
    features["rsi_14"] = calculate_rsi(close, window=14)
    features["bollinger_percent_b"] = calculate_bollinger_percent_b(close)

    # volatility or risk regime
    features["bollinger_band_width"] = calculate_bollinger_band_width(close)
    features["daily_range"] = calculate_daily_range(high, low, close)
    features["5d_rolling_volatility"] = calculate_rolling_volatility(close, window=5)

    # volume / market anticipation
    features["volume_ratio_20d"] = calculate_volume_ratio(volume, window=20)

    return features

def create_stock_features3(data):

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    features = pd.DataFrame(index=data.index)

    # momentum / trend direction 
    features["daily_return"] = calculate_daily_return(close)
    features["5d_momentum"] = calculate_momentum(close, window=5)
    features["macd_histogram"] = calculate_macd_histogram(close)
    #============================================================
    features["hlc3_ratio"] = calculate_hlc3_ratio(high=high,low=low,close=close,window=20)
    features["fwma_ratio"] = calculate_fwma_ratio(close=close,window=13)
    features["swma_ratio"] = calculate_swma_ratio(close=close,window=5)
    # features["tema_ratio"] = calculate_tema_ratio(close=close,window=20)
    # features["ema_20_ratio"] = calculate_ema_ratio(close=close,window=20)
    # features["hma_ratio"] = calculate_hma_ratio(close=close,window=20)
    # features["hwma_ratio"] = calculate_hwma_ratio(close=close,alpha=0.2,beta=0.1)
    # features["ichimoku_signal"] = calculate_ichimoku_signal(high=high,low=low,close=close)

    # reversion / extension
    features["rsi_14"] = calculate_rsi(close, window=14)
    features["bollinger_percent_b"] = calculate_bollinger_percent_b(close)
    #============================================================
    # features["kama_ratio"] = calculate_kama_ratio(close=close,er_window=10,fast_window=2,slow_window=30)

    # volatility or risk regime
    features["bollinger_band_width"] = calculate_bollinger_band_width(close)
    features["daily_range"] = calculate_daily_range(high, low, close)
    features["5d_rolling_volatility"] = calculate_rolling_volatility(close, window=5)
    #============================================================

    # volume / market anticipation
    features["volume_ratio_20d"] = calculate_volume_ratio(volume, window=20)
    #============================================================
    # features["obv_change"] = calculate_obv_change(close=close,volume=volume,window=20)
    # features["pvt_change"] = calculate_pvt_change(close=close,volume=volume,window=20)

    return features