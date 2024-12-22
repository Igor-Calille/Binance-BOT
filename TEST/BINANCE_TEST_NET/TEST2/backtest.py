import numpy as np
import backtrader as bt

class Backtest:
    def __init__(self):
        pass

    def check_signal_accuracy(self, data):
        # Calcular a mudança de preço do dia seguinte
        data['price_change'] = data['Close'].shift(-1) - data['Close']

        # Definir condições para um sinal correto de compra ou venda
        conditions = [
            (data['signal_ml'] > 0) & (data['price_change'] > 0),  # Compra seguida de aumento de preço
            (data['signal_ml'] < 0) & (data['price_change'] < 0),  # Venda seguida de queda de preço
        ]
        choices = [1, 1]  # Ambos são sinais corretos, independente de serem compra ou venda
        data['correct_signal'] = np.select(conditions, choices, default=0)

        # Calcular a acurácia do sinal
        correct_signals = data['correct_signal'].sum()
        total_signals = np.count_nonzero(data['signal_ml'])  # Conta todos os sinais emitidos, ignorando zeros

        accuracy = correct_signals / total_signals if total_signals > 0 else 0  # Evita divisão por zero

        return accuracy, total_signals, correct_signals
    
class MLStrategy(bt.Strategy):
    params = (
        ('risk_per_trade', 0.98),  # Risco máximo por operação (2%)
        ('slippage', 0.0005),  # Exemplo de 0,05% de slippage
        ('transaction_fee', 0.001),  # Exemplo de taxa de 0,1%
    )

    def next(self):
        self.close_price = self.data.close[0] * (1 + self.params.slippage)

        if self.data.signal_ml[0] == 1 and self.broker.get_cash() > 0:
            amount_to_risk = self.broker.get_cash() * self.params.risk_per_trade
            size = amount_to_risk / (self.close_price * (1 + self.params.transaction_fee))
            self.buy(size=size)
        elif self.data.signal_ml[0] == -1 and self.position:
            self.sell(size=self.position.size)
