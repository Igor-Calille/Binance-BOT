import backtrader as bt
from backtest import MLStrategy
import pandas as pd

# Adicionar função para rodar backtesting
def run_backtesting(data_file='df.csv', initial_cash=10000.0):
    # Carregar os dados históricos com sinais preditivos
    data = pd.read_csv(data_file, parse_dates=['Close Time'], index_col='Close Time')

    # Adicionar os sinais e preços como dados para backtrader
    class PandasData(bt.feeds.PandasData):
        lines = ('signal_ml',)
        params = (
            ('datetime', None),
            ('open', 'Open'),
            ('high', 'High'),
            ('low', 'Low'),
            ('close', 'Close'),
            ('volume', 'Volume'),
            ('signal_ml', 'signal_ml'),
        )

    # Configuração do ambiente de backtesting
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MLStrategy)

    # Adicionar os dados ao backtesting
    data_feed = PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Configuração inicial do portfólio
    cerebro.broker.setcash(initial_cash)  # Capital inicial
    cerebro.broker.setcommission(commission=0.001)  # Taxa de 0,1%

    # Rodar o backtesting
    print("Capital inicial: {:.2f}".format(cerebro.broker.getvalue()))
    results = cerebro.run()
    print("Capital final: {:.2f}".format(cerebro.broker.getvalue()))

    # Exibir o gráfico de resultados
    cerebro.plot()

# Rodar o backtesting ao final
if __name__ == "__main__":
    run_backtesting()
