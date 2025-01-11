import backtrader as bt
import pandas as pd
import numpy as np

from IndicadoresMercado import Indicadores
from MLModels import using_RandomForestRegressor

# ------------------------------------
# CONFIGURAÇÕES DA "NOVA" ESTRATÉGIA
# ------------------------------------
STOP_LOSS_PCT = 0.03      # 3% de Stop Loss
TAKE_PROFIT_PCT = 0.05    # 5% de Take Profit
RISK_PERCENT_CAPITAL = 0.70
MIN_NOTIONAL = 10.0       # Mínimo de 10 USDT para permitir a compra

# Capital inicial do backtest
INITIAL_CASH = 10000.0

# Data inicial do backtest
BACKTEST_START_DATE = "2024-12-28"

class ExtendedPandasData(bt.feeds.PandasData):
    """
    Feed que inclui coluna 'signal_ml' no DataFrame.
    Se não houver 'signal_ml', definimos como 0.
    """
    lines = ('signal_ml',)
    params = (
        ('signal_ml', 'signal_ml'),
        ('datetime', None),
        ('time', -1),
        ('open',  'Open'),
        ('high',  'High'),
        ('low',   'Low'),
        ('close', 'Close'),
        ('volume','Volume'),
        ('openinterest', -1),
    )

    def __init__(self):
        super().__init__()
        if 'signal_ml' not in self.p.dataname.columns:
            self.p.dataname['signal_ml'] = 0

# ------------------------------------
# ESTRATÉGIA USANDO ORDENS NATIVAS
# ------------------------------------
class FullReplicateStrategy(bt.Strategy):
    """
    Estratégia que:
      - Compõe posição (self.position) usando self.buy() 
        quando sinal=+1 e não estamos comprados.
      - DCA se já estamos comprados e sinal=+1 novamente.
      - Fecha (self.close()) se sinal=-1.
      - Aplica STOP LOSS e TAKE PROFIT checando 
        se o preço atual ultrapassou as bandas definidas.
      - Checa minNotional (size * preço >= 10).
    Assim, o Backtrader enxerga as ORDENS e mostra 
    setas de Buy/Sell no gráfico.
    """

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d %H:%M')
        print(f"[{dt}] {txt}")

    def __init__(self):
        # Guardamos a referência ao campo 'signal_ml' do DataFrame
        self.signal = self.datas[0].signal_ml

    def next(self):
        # Preço atual do candle
        current_price = self.datas[0].close[0]

        # Se temos posição aberta, checar SL e TP
        if self.position.size > 0:  # posição comprada
            entry_price = self.position.price  # preço médio da posição
            stop_loss_price   = entry_price * (1 - STOP_LOSS_PCT)
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)

            if current_price <= stop_loss_price:
                self.log(f"[SL HIT] Price={current_price:.2f}, SL={stop_loss_price:.2f}")
                # Fecha posição (gera seta de SELL no gráfico)
                self.close()
                return

            if current_price >= take_profit_price:
                self.log(f"[TP HIT] Price={current_price:.2f}, TP={take_profit_price:.2f}")
                self.close()
                return

        # Ler o sinal do modelo
        signal_ml = int(self.signal[0])  # +1 ou -1
        cash_available = self.broker.getcash()
        value_total    = self.broker.getvalue()  # cash + valor da posição

        # 1) Se não temos posição e sinal=+1 => Compra
        if self.position.size == 0 and signal_ml == 1:
            # Tenta comprar ~70% do capital
            amount_usdt = value_total * RISK_PERCENT_CAPITAL
            if amount_usdt > cash_available:  # não gastar mais do que o cash
                amount_usdt = cash_available

            # Calcula size
            size = amount_usdt / current_price
            # Checa minNotional
            if (size * current_price) >= MIN_NOTIONAL:
                self.log(f"ABRINDO COMPRA => size={size:.6f} (Close={current_price:.2f})")
                self.buy(size=size)  # gera a seta de compra
            else:
                self.log("Compra abortada: minNotional não atinge 10 USDT")

        # 2) Se já temos posição comprada e sinal=+1 => DCA (comprar mais)
        elif self.position.size > 0 and signal_ml == 1:
            # Exemplo de DCA
            amount_usdt = cash_available * RISK_PERCENT_CAPITAL
            if (amount_usdt > 0) and (amount_usdt >= MIN_NOTIONAL):
                dca_size = amount_usdt / current_price
                if (dca_size * current_price) >= MIN_NOTIONAL:
                    self.log(f"DCA extra => size={dca_size:.6f} (Close={current_price:.2f})")
                    self.buy(size=dca_size)
                else:
                    self.log("DCA abortada: minNotional não atinge 10 USDT")

        # 3) Se temos posição comprada e sinal=-1 => Fechar tudo
        elif self.position.size > 0 and signal_ml == -1:
            self.log("Fechando posição => vendendo tudo (sinal=-1).")
            self.close()  # gera seta vermelha de saída

        # 4) Se não temos posição e sinal=-1 => não faz nada 
        #    (ou poderia shortar, mas não é o caso).

    def notify_order(self, order):
        """
        Callback do Backtrader para acompanhar as ordens.
        Podemos logar fills, status, etc.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # A ordem ainda está em processamento.
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"ORD BUY EXEC: {order.executed.size:.4f} @ {order.executed.price:.4f}")
            else:  # sell
                self.log(f"ORD SELL EXEC: {order.executed.size:.4f} @ {order.executed.price:.4f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Ordem cancelada / rejeitada / sem margem.")
        # Zera a referência da ordem
        order = None

    def notify_trade(self, trade):
        """
        Callback para acompanhar a evolução do trade.
        Podemos logar PnL parcial, etc.
        """
        if not trade.isclosed:
            return
        self.log(f"TRADE FECHADO => PnL bruto: {trade.pnl:.2f}, PnL líquido: {trade.pnlcomm:.2f}")

# ---------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE BACKTEST
# ---------------------------------------------------------------------
def run_backtest():
    df = pd.read_csv('ml_data.csv', index_col=0)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Filtra a data inicial
    df = df.loc[df.index >= BACKTEST_START_DATE]

    # Se não existir 'signal_ml', recalcular via ML
    if 'signal_ml' not in df.columns:
        print("[INFO] 'signal_ml' não encontrado. Recalculando via ML...")

        ind = Indicadores()
        needed = [
            'RSI_14','RSI_7','MACD_12_26_9','MACD_5_35_5','StochRSI_14','StochRSI_7'
        ]
        for col in needed:
            if col not in df.columns:
                if col == 'RSI_14':
                    df['RSI_14'] = ind.compute_RSI(df['Close'], 14)
                elif col == 'RSI_7':
                    df['RSI_7']  = ind.compute_RSI(df['Open'], 7)
                elif col == 'MACD_12_26_9':
                    df['MACD_12_26_9'] = ind.compute_MACD(df['Close'], 12, 26, 9)
                elif col == 'MACD_5_35_5':
                    df['MACD_5_35_5']  = ind.compute_MACD(df['Open'], 5, 35, 5)
                elif col == 'StochRSI_14':
                    df['StochRSI_14']  = ind.get_stochastic_rsi(df['Close'], 14, 14)
                elif col == 'StochRSI_7':
                    df['StochRSI_7']   = ind.get_stochastic_rsi(df['Open'], 7)
        df.dropna(inplace=True)

        features = [
            'Open', 'Close', 'High', 'Low', 'Volume',
            'Quote Asset Volume','Number of Trades',
            'Taker Buy Base Asset Volume','Taker Buy Quote Asset Volume',
            'RSI_14','RSI_7','MACD_12_26_9','MACD_5_35_5',
            'StochRSI_14','StochRSI_7'
        ]
        df.dropna(subset=features + ['Target'], inplace=True)

        X = df[features]
        y = df['Target']
        model = using_RandomForestRegressor.fixed_params_RandomForestRegressor(X, y)

        df['signal_ml'] = 0
        for i in range(len(df) - 1):
            row_features = df.iloc[i][features].values.reshape(1, -1)
            predicted_close = model.predict(row_features)[0]
            current_close   = df['Close'].iloc[i]
            df.at[df.index[i], 'signal_ml'] = 1 if (predicted_close > current_close) else -1

    # Configura e roda o Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(INITIAL_CASH)  
    cerebro.broker.setcommission(commission=0.001)  # Ex.: 0.1%

    datafeed = ExtendedPandasData(dataname=df)
    cerebro.adddata(datafeed)

    cerebro.addstrategy(FullReplicateStrategy)

    print(f"===== Iniciando Backtest a partir de {BACKTEST_START_DATE} =====")
    results = cerebro.run()
    print("===== Backtest finalizado =====")
    final_value = cerebro.broker.getvalue()
    print(f"[BT-Broker] Valor final: {final_value:.2f}")

    # Plot com as setas de Buy/Sell
    cerebro.plot(style='candlestick')

if __name__ == '__main__':
    run_backtest()
