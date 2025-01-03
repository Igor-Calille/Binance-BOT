# backtest_estrategia.py
"""
Backtest offline que replica a lógica de 'app.py' (DCA, SL/TP, 
vendas parciais, etc.), porém:
 - Ao final de cada candle, o valor do broker do BT
   é forçado a refletir (quote_balance + base_balance*close).
 - Inicia a simulação a partir de data específica.
 - Limita a compra/venda ao saldo real disponível.
"""

import backtrader as bt
import pandas as pd
import numpy as np

from IndicadoresMercado import Indicadores
from MLModels import using_RandomForestRegressor

# -------------------------------
# REPLICA CONFIGS DO app.py
# -------------------------------
SYMBOL = "BNBUSDT"
BASE_ASSET = "BNB"
QUOTE_ASSET = "USDT"

STOP_LOSS_PCT = 0.15   
TAKE_PROFIT_PCT = 0.22 
RISK_PERCENT_CAPITAL = 0.70   
MIN_NOTIONAL = 10.0         # Ex.: 10 USDT min p/ operar

# Ajuste se quiser outro capital inicial:
INITIAL_QUOTE_BALANCE = 10000.0
INITIAL_BASE_BALANCE  = 0.0

# DATA DE INÍCIO do backtest (exemplo):
BACKTEST_START_DATE = "2024-07-06"

# -------------------------------
# FEED PERSONALIZADO
# -------------------------------
class ExtendedPandasData(bt.feeds.PandasData):
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

# -------------------------------
# STRATEGY: REPRODUZ LÓGICA DO app.py
# -------------------------------
class FullReplicateStrategy(bt.Strategy):
    """
    Reproduz a lógica do app.py, mas no final de cada candle
    atualizamos self.broker.set_cash(...) para que o "Valor final"
    reportado reflita de fato (quote_balance + base_balance*preço).
    """

    def __init__(self):
        # Estado de posição:
        self.open_position   = False
        self.entry_price     = 0.0
        self.entry_quantity  = 0.0

        # "Saldos"
        self.quote_balance   = INITIAL_QUOTE_BALANCE
        self.base_balance    = INITIAL_BASE_BALANCE

        self.signal = self.datas[0].signal_ml

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d %H:%M')
        print(f"[{dt}] {txt}")

    def next(self):
        current_price = self.datas[0].close[0]

        # 1) Stop Loss / Take Profit
        if self.open_position:
            stop_loss_price   = self.entry_price * (1 - STOP_LOSS_PCT)
            take_profit_price = self.entry_price * (1 + TAKE_PROFIT_PCT)
            if current_price <= stop_loss_price:
                self.log(f"[SL HIT] Price={current_price:.4f}, SL={stop_loss_price:.4f}")
                self.close_position_all(current_price)
            elif current_price >= take_profit_price:
                self.log(f"[TP HIT] Price={current_price:.4f}, TP={take_profit_price:.4f}")
                self.close_position_all(current_price)

        # 2) Ler o sinal
        signal_ml = int(self.signal[0])

        # 3) Se não temos posição aberta
        if not self.open_position:
            if signal_ml == 1:
                # "Compra ~70% do saldo USDT"
                if self.quote_balance > MIN_NOTIONAL:
                    amount_usdt = self.quote_balance * RISK_PERCENT_CAPITAL
                    amount_usdt = min(amount_usdt, self.quote_balance)  # travar no saldo
                    qty_buy     = amount_usdt / current_price
                    qty_buy     = self.adjust_quantity(qty_buy)

                    cost = qty_buy * current_price
                    if cost > self.quote_balance:
                        cost = self.quote_balance
                        qty_buy = self.adjust_quantity(cost / current_price)

                    if qty_buy > 0:
                        self.log(f"ABRINDO COMPRA => {qty_buy:.6f} BNB (DCA)")
                        self.quote_balance -= cost
                        self.base_balance  += qty_buy
                        self.open_position   = True
                        self.entry_price     = current_price
                        self.entry_quantity  = qty_buy

            else:  # signal=-1 => vender BNB parado (sem posição)
                if self.base_balance > 0.001:
                    qty_sell = self.base_balance * RISK_PERCENT_CAPITAL
                    qty_sell = self.adjust_quantity(qty_sell)
                    if qty_sell > self.base_balance:
                        qty_sell = self.base_balance
                    if qty_sell > 0:
                        self.log(f"Vendendo BNB => {qty_sell:.6f} (sinal=-1, sem pos).")
                        revenue = qty_sell * current_price
                        self.base_balance  -= qty_sell
                        self.quote_balance += revenue
                        if self.base_balance < 0:
                            self.base_balance = 0

        # 4) Já temos posição comprada
        else:
            if signal_ml == 1:
                # DCA
                if self.quote_balance > MIN_NOTIONAL:
                    amount_usdt = self.quote_balance * RISK_PERCENT_CAPITAL
                    amount_usdt = min(amount_usdt, self.quote_balance)
                    qty_buy     = amount_usdt / current_price
                    qty_buy     = self.adjust_quantity(qty_buy)

                    cost = qty_buy * current_price
                    if cost > self.quote_balance:
                        cost = self.quote_balance
                        qty_buy = self.adjust_quantity(cost / current_price)

                    if qty_buy > 0:
                        self.log(f"DCA extra: comprando +{qty_buy:.6f} BNB.")
                        self.quote_balance -= cost
                        self.base_balance  += qty_buy
                        # Recalcula preço médio
                        old_val   = self.entry_price * self.entry_quantity
                        new_val   = current_price * qty_buy
                        total_val = old_val + new_val
                        self.entry_quantity += qty_buy
                        self.entry_price     = total_val / self.entry_quantity
            else:
                # signal=-1 => vende RISK_PERCENT_CAPITAL do base_balance
                if self.base_balance > 0.001:
                    qty_to_sell = self.base_balance * RISK_PERCENT_CAPITAL
                    qty_to_sell = self.adjust_quantity(qty_to_sell)
                    if qty_to_sell > self.base_balance:
                        qty_to_sell = self.base_balance
                    if qty_to_sell > 0:
                        self.log(f"Fechando posição => vendendo {qty_to_sell:.6f} BNB (sinal=-1).")
                        revenue = qty_to_sell * current_price
                        self.base_balance  -= qty_to_sell
                        self.quote_balance += revenue
                        if self.base_balance < 0:
                            self.base_balance = 0
                # Zera posição
                self.open_position  = False
                self.entry_price    = 0.0
                self.entry_quantity = 0.0

        # 5) Ajusta o "valor do broker" p/ refletir saldo manual
        total_value = self.quote_balance + (self.base_balance * current_price)
        self.broker.set_cash(total_value)

        # Log de saldo a cada candle
        self.log(
            f"Saldo => {self.quote_balance:.2f} USDT, "
            f"{self.base_balance:.6f} BNB, "
            f"entry_price={self.entry_price:.4f}, "
            f"signal={signal_ml}, "
            f"ValorTotal={total_value:.2f}"
        )

    def close_position_all(self, current_price):
        """Fecha posição vendendo TODO o base_balance (para SL ou TP)."""
        qty_to_sell = self.adjust_quantity(self.base_balance)
        if qty_to_sell > self.base_balance:
            qty_to_sell = self.base_balance
        if qty_to_sell > 0:
            self.log(f"Fechando posição [SL/TP] => vendendo {qty_to_sell:.6f} BNB.")
            revenue = qty_to_sell * current_price
            self.base_balance  -= qty_to_sell
            self.quote_balance += revenue
            if self.base_balance < 0:
                self.base_balance = 0
        self.open_position   = False
        self.entry_price     = 0.0
        self.entry_quantity  = 0.0

    @staticmethod
    def adjust_quantity(qty):
        return round(qty, 6)

# -------------------------------
# FUNÇÃO PRINCIPAL
# -------------------------------
def run_backtest():
    # 1) Ler CSV gerado pelo app.py
    df = pd.read_csv('ml_data.csv', index_col=0)

    # 2) Converte index e filtra data
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.loc[df.index >= BACKTEST_START_DATE]

    # 3) Se não existir 'signal_ml', recalcular
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

    # 4) Configura Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(10000.0)  # meramente simbólico
    cerebro.broker.setcommission(commission=0.001)

    datafeed = ExtendedPandasData(dataname=df)
    cerebro.adddata(datafeed)

    cerebro.addstrategy(FullReplicateStrategy)

    print(f"===== Iniciando Backtest a partir de {BACKTEST_START_DATE} =====")
    cerebro.run()
    print("===== Backtest finalizado =====")

    # Valor final do broker => agora deve refletir nosso "valor_total"
    print(f"[BT-Broker] Valor final: {cerebro.broker.getvalue():.2f}")

if __name__ == '__main__':
    run_backtest()
