import numpy as np
import backtrader as bt
import pandas as pd

class Backtest:
    def __init__(self):
        pass

    def check_signal_accuracy(self, data):
        """
        data: DataFrame com colunas:
          - 'Close'
          - 'signal_ml' (1 = compra, -1 = venda, 0 = sem sinal)
        Gera price_change, compara com signal_ml e calcula acertos.
        
        Retorna: (accuracy, total_signals, correct_signals)
        """
        # Calcula a mudança de preço do PRÓXIMO candle
        data['price_change'] = data['Close'].shift(-1) - data['Close']

        # Sinais corretos:
        # - (signal_ml == 1) & (price_change > 0)  => previu alta e subiu
        # - (signal_ml == -1) & (price_change < 0) => previu queda e caiu
        conditions = [
            (data['signal_ml'] == 1) & (data['price_change'] > 0),
            (data['signal_ml'] == -1) & (data['price_change'] < 0),
        ]
        data['correct_signal'] = np.select(conditions, [1, 1], default=0)

        correct_signals = data['correct_signal'].sum()
        total_signals = (data['signal_ml'] != 0).sum()
        accuracy = correct_signals / total_signals if total_signals > 0 else 0
        return accuracy, total_signals, correct_signals
    
