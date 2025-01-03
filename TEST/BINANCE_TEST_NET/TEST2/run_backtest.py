import backtrader as bt
import pandas as pd
from backtest import MLStrategy

# Função para rodar o backtesting
def run_backtesting(data_file='df.csv', initial_cash=10000.0, start_date=None):
    # Carregar os dados históricos
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

    # Estratégia personalizada para iniciar trades após a data definida
    class FilteredMLStrategy(bt.Strategy):
        params = (('start_date', None),)

        def __init__(self):
            self.start_date = pd.to_datetime(self.params.start_date) if self.params.start_date else None

        def next(self):
            # Ignorar trades antes da data de início
            if self.start_date and self.datas[0].datetime.date(0) < self.start_date.date():
                return

            # Obter saldo atual e posição
            cash = self.broker.getcash()  # Saldo em caixa
            position = self.getposition(self.datas[0]).size  # Quantidade do ativo
            close_price = self.data.close[0]
            signal = self.data.signal_ml[0]  # Obtém o sinal atual

            # Debugging: Exibir informações básicas
            print(f"Data: {self.datas[0].datetime.date(0)}, Signal: {signal}, Cash: {cash:.2f}, Position: {position}, Close Price: {close_price:.2f}")

            # Garantir que o preço seja válido
            if not close_price or close_price <= 0:
                print("Preço inválido. Ignorando.")
                return

            # Sinal de compra
            if signal == 1:
                size = (cash * 0.98) / close_price  # Usa 98% do caixa para compra
                size = max(size, 0.001)  # Garantir tamanho mínimo
                self.buy(size=size)
                print(f"Comprando {size:.4f} unidades a {close_price:.2f}")

            # Sinal de venda
            elif signal == -1:
                self.sell(size=position)  # Vende tudo
                print(f"Vendendo {position:.4f} unidades a {close_price:.2f}")


    # Configuração do ambiente de backtesting
    cerebro = bt.Cerebro()
    cerebro.addstrategy(FilteredMLStrategy, start_date=start_date)

    # Adicionar os dados ao ambiente
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

# Executar o backtesting
if __name__ == "__main__":
    # Especificar a data de início para os trades
    run_backtesting(data_file='df.csv', start_date='2024-12-20')


