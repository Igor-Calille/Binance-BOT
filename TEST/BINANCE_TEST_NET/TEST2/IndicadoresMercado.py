import numpy as np

class Indicadores:
    def __init__(self):
        pass

    def compute_RSI(self, series, window=14):
        """
        RSI clássico:
        - Calcula diferença entre candles consecutivos
        - separa ganhos e perdas
        - faz média móvel
        """
        diff = series.diff(1).dropna()
        gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
        loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
        RS = gain / loss
        return 100 - (100 / (1 + RS))

    def compute_Bollinger_Bands(self, series, window=20, nstd=2):
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        bollinger_high = rolling_mean + (nstd * rolling_std)
        bollinger_low = rolling_mean - (nstd * rolling_std)
        return bollinger_high, bollinger_low

    def Media_movel(self, series, window):
        return series.rolling(window=window).mean()

    def media_movel_exponecial(self, series, window):
        return series.ewm(span=window, adjust=False).mean()

    def media_movel_ponderada(self, series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    def media_movel_kaufman(self, series, window):
        """
        Exemplo de Adaptive Moving Average (KAMA).
        Implementação simples; pode requerer otimizações.
        """
        change = abs(series.diff(window))
        volatility = series.diff().abs().rolling(window=window).sum()
        er = change / volatility
        sc = (er * (2/(2+1) - 2/(window+1)) + 2/(window+1))**2
        kama = np.zeros(len(series))
        kama[0:window] = series.iloc[0:window]  # inicia nos primeiros 'window' valores

        for i in range(window, len(series)):
            kama[i] = kama[i-1] + sc[i] * (series.iloc[i] - kama[i-1])
        return kama

    def media_movel_hull(self, series, window):
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        wma_half = self.media_movel_ponderada(series, half_window)
        wma_full = self.media_movel_ponderada(series, window)
        hull_ma = self.media_movel_ponderada(2 * wma_half - wma_full, sqrt_window)
        return hull_ma

    def media_movel_triangular(self, series, window):
        return self.Media_movel(self.Media_movel(series, window), window)

    def compute_MACD(self, series, short_window=12, long_window=26, signal_window=9):
        short_ema = series.ewm(span=short_window, adjust=False).mean()
        long_ema = series.ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_histogram

    def get_stochastic_rsi(self, series, window=14, stochastic_window=14):
        """
        Stoch RSI = (RSI - min(RSI)) / (max(RSI) - min(RSI)) em 'stochastic_window'.
        """
        rsi = self.compute_RSI(series, window)
        min_rsi = rsi.rolling(window=stochastic_window).min()
        max_rsi = rsi.rolling(window=stochastic_window).max()
        return (rsi - min_rsi) / (max_rsi - min_rsi)
