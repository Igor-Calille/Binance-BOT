import numpy as np

class Indicadores:
    def __init__ (self):
        pass

    def compute_RSI(self, stocks_open, window=14):
        diff = stocks_open.diff(1).dropna()
        gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
        loss = (-diff.where(diff <0,0)).rolling(window=window).mean()
        RS = gain / loss

        return 100 - (100 / (1 + RS))
    
    def compute_Bollinger_Bands(stocks_open, window=20, nstd=2):
        rolling_mean = stocks_open.rolling(window=window).mean()
        rolling_std = stocks_open.rolling(window=window).std()
        bollinger_high = rolling_mean + (nstd * rolling_std)
        bollinger_low = rolling_mean - (nstd * rolling_std)
        return bollinger_high, bollinger_low
    
    def Media_movel(stocks_open, window):
        return stocks_open.rolling(window=window).mean()
    
    def media_movel_exponecial(stocks_open, window):
        return stocks_open.ewm(span=window, adjust=False).mean()
    
    def media_movel_ponderada(stocks_open, window):
        weights = np.arange(1, window + 1)
        return stocks_open.rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    
    def media_movel_kaufman(stocks_open, window):
        change = abs(stocks_open.diff(window))
        volatility = stocks_open.diff().abs().rolling(window=window).sum()
        er = change / volatility
        sc = (er * (2/(2+1) - 2/(window+1)) + 2/(window+1))**2
        kama = np.zeros_like(stocks_open)
        for i in range(window, len(stocks_open)):
            kama[i] = kama[i-1] + sc[i] * (stocks_open[i] - kama[i-1])
        return kama
    
    def media_movel_hull(self, stocks_open, window):
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        wma_half = self.media_movel_ponderada(stocks_open, half_window)
        wma_full = self.media_movel_ponderada(stocks_open, window)
        hull_ma = self.media_movel_ponderada(2 * wma_half - wma_full, sqrt_window)
        return hull_ma

    def media_movel_triangular(self, stocks_open, window):
        return self.Media_movel(self.Media_movel(stocks_open, window), window)
    
    def compute_MACD(self, stocks_open, short_window=12, long_window=26, signal_window=9):
        short_ema = stocks_open.ewm(span=short_window, adjust=False).mean()
        long_ema = stocks_open.ewm(span=long_window, adjust=False).mean()

        macd_line = short_ema - long_ema

        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        macd_histogram = macd_line - signal_line

        return macd_histogram
    
    def get_stochastic_rsi(self, data_value, window=14, stochastic_window=14):
        rsi = self.compute_RSI(data_value, window)

        min_rsi = rsi.rolling(window=stochastic_window).min()
        max_rsi = rsi.rolling(window=stochastic_window).max()
        stochastica_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)

        return stochastica_rsi
    


