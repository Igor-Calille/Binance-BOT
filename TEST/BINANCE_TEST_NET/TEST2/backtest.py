import numpy as np
import pandas as pd

class Backtest:
    def __init__(self):
        pass

    def check_signal_accuracy(self, data):
        """
        data: DataFrame com colunas:
          - 'Close'
          - 'signal_ml' (1 = compra, -1 = venda, 0 = sem sinal)
        
        Avalia a acurácia de previsão do candle seguinte:
        - Se signal_ml = +1 e o Close[t+1] > Close[t], conta como acerto.
        - Se signal_ml = -1 e o Close[t+1] < Close[t], conta como acerto.
        
        Retorna (accuracy, total_signals, correct_signals).
        """
        data = data.copy()

        # price_change do próximo candle
        data['price_change'] = data['Close'].shift(-1) - data['Close']

        # Sinais corretos se:
        # (signal_ml == 1) & (price_change > 0)  ou
        # (signal_ml == -1) & (price_change < 0)
        conditions = [
            (data['signal_ml'] == 1) & (data['price_change'] > 0),
            (data['signal_ml'] == -1) & (data['price_change'] < 0),
        ]
        data['correct_signal'] = np.select(conditions, [1, 1], default=0)

        correct_signals = data['correct_signal'].sum()
        total_signals   = (data['signal_ml'] != 0).sum()  # ignora candles sem sinal
        accuracy = correct_signals / total_signals if total_signals > 0 else 0

        return accuracy, total_signals, correct_signals


