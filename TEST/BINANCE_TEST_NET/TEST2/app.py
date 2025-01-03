import os
import time
import math
import logging
import subprocess
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client

from IndicadoresMercado import Indicadores
from MLModels import using_RandomForestRegressor
from backtest import Backtest  # <- seu arquivo backtest.py

# -----------------------------------------------------------------------
#                       CONFIGURAÇÕES DO LOG
# -----------------------------------------------------------------------
logging.basicConfig(
    filename='trade.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
#                       CONFIGURAÇÕES DO BOT
# -----------------------------------------------------------------------
load_dotenv()

SYMBOL = "BNBUSDT"
BASE_ASSET = "BNB"
QUOTE_ASSET = "USDT"

# Exemplo de SL e TP fixos para Altcoins
STOP_LOSS_PCT = 0.15   
TAKE_PROFIT_PCT = 0.22 

RISK_PERCENT_CAPITAL = 0.70   # 70% do saldo p/ comprar ou vender (DCA)
RETRAIN_INTERVAL = 5
MAX_CANDLES = 5555

# Intervalos
KLINE_INTERVAL = Client.KLINE_INTERVAL_1HOUR   # candle de 1 hora
SL_TP_CHECK_INTERVAL = 25 * 60  # 25 minutos => 1500s

# Credenciais (testnet ou principal)
testnet_api_key = os.getenv('TESTNET_API_KEY')
testnet_api_secret = os.getenv('TESTNET_API_SECRET')
client = Client(testnet_api_key, testnet_api_secret)
client.API_URL = 'https://testnet.binance.vision/api'

api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')
client2 = Client(api_key, api_secret)

# Inicia serviço de tempo (Windows)
start_service = subprocess.run(["net", "start", "w32time"], capture_output=True, text=True)
logger.info("Iniciando serviço de tempo do Windows...")
logger.info(start_service.stdout)
if start_service.stderr:
    logger.error(start_service.stderr)

# -----------------------------------------------------------------------
#             VARIÁVEIS GLOBAIS DE CONTROLE DE POSIÇÃO
# -----------------------------------------------------------------------
open_position = False
entry_price   = 0.0
entry_quantity= 0.0

# -----------------------------------------------------------------------
#                       FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------
def get_asset_balance(asset: str) -> float:
    """Retorna o saldo 'free' (disponível) de um ativo."""
    try:
        balance = client.get_asset_balance(asset=asset)
        if balance:
            return float(balance['free'])
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Erro ao obter saldo de {asset}: {e}")
        return 0.0

def get_symbol_price(symbol: str) -> float:
    """Retorna o último preço do par (ticker)."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        logger.error(f"Erro ao obter preço do par {symbol}: {e}")
        return 0.0

def get_lot_size_limits(symbol: str):
    """Retorna (min_qty, max_qty, step_size) do par."""
    try:
        symbol_info = client.get_symbol_info(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    min_qty = float(f['minQty'])
                    max_qty = float(f['maxQty'])
                    step_size = float(f['stepSize'])
                    return min_qty, max_qty, step_size
        return None, None, None
    except Exception as e:
        logger.error(f"Erro ao obter LOT_SIZE para {symbol}: {e}")
        return None, None, None

def adjust_quantity(symbol: str, quantity: float) -> float:
    """Ajusta 'quantity' para respeitar min_qty, max_qty e step_size."""
    try:
        min_qty, max_qty, step_size = get_lot_size_limits(symbol)
        if min_qty is None or max_qty is None or step_size is None:
            logger.error(f"Erro: Não foi possível obter os limites p/ {symbol}")
            return 0

        adjusted_quantity = math.floor(quantity / step_size) * step_size
        casas_decimais = len(str(step_size).split('.')[1])
        adjusted_quantity = round(adjusted_quantity, casas_decimais)

        if adjusted_quantity < min_qty:
            logger.warning(f"Qtd ajustada abaixo do mínimo: {adjusted_quantity} < {min_qty}")
            return 0
        if adjusted_quantity > max_qty:
            logger.warning(f"Qtd ajustada acima do máximo: {adjusted_quantity} > {max_qty}")
            return max_qty
        
        return adjusted_quantity
    except Exception as e:
        logger.error(f"Erro ao ajustar qtd: {e}")
        return 0

def buy_market(symbol: str, quantity: float):
    """Executa ordem de compra a mercado."""
    try:
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        logger.info(f"Ordem de COMPRA: {order}")
        return order
    except Exception as e:
        logger.error(f"Erro ao comprar {symbol}: {e}")
        return None

def sell_market(symbol: str, quantity: float):
    """Executa ordem de venda a mercado (só se tivermos BNB)."""
    try:
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        logger.info(f"Ordem de VENDA: {order}")
        return order
    except Exception as e:
        logger.error(f"Erro ao vender {symbol}: {e}")
        return None

def wait_for_next_close():
    """
    Aguarda até o próximo fechamento do candle (KLINE_INTERVAL).
    Exemplo: se KLINE_INTERVAL=1h, aguarda fechar a hora.
    """
    try:
        server_time = client.get_server_time()
        current_time = int(time.time() * 1000)
        time_diff = server_time['serverTime'] - current_time
        adjusted_time = int(time.time() * 1000) + time_diff

        last_kline = client.get_klines(symbol=SYMBOL, interval=KLINE_INTERVAL, limit=1)[0]
        next_close_time = last_kline[6]
        wait_time = (next_close_time - adjusted_time) / 1000

        if wait_time > 0:
            logger.info(f"Aguardando candle fechar em {wait_time:.2f}s...")
            time.sleep(wait_time)
    except Exception as e:
        logger.error(f"Erro ao aguardar candle: {e}")
        time.sleep(60)

# -----------------------------------------------------------------------
#  CHECAGEM DE STOP LOSS E TAKE PROFIT (SEM SHORT)
# -----------------------------------------------------------------------
def check_sl_tp():
    """
    Roda periodicamente (25 min). Se existe posição comprada (open_position=True),
    checa se preço atual atingiu SL ou TP. Se sim, vende tudo.
    """
    global open_position, entry_price, entry_quantity

    if not open_position:
        return  # Não há posição aberta, nada a fazer

    symbol_price = get_symbol_price(SYMBOL)
    if symbol_price <= 0:
        return

    # STOP LOSS se price <= entry * (1 - STOP_LOSS_PCT)
    # TAKE PROFIT se price >= entry * (1 + TAKE_PROFIT_PCT)
    stop_loss_price    = entry_price * (1 - STOP_LOSS_PCT)
    take_profit_price  = entry_price * (1 + TAKE_PROFIT_PCT)

    # Checa SL
    if symbol_price <= stop_loss_price:
        logger.info(f"[SL HIT] Preço={symbol_price:.4f}, SL={stop_loss_price:.4f}")
        qty_to_sell = entry_quantity
        qty_to_sell = adjust_quantity(SYMBOL, qty_to_sell)
        if qty_to_sell > 0:
            logger.info(f"Fechando posição (COMPRA) => vendendo {qty_to_sell} {BASE_ASSET}")
            sell_market(SYMBOL, qty_to_sell)
        open_position = False
        entry_price   = 0.0
        entry_quantity= 0.0

    # Checa TP
    elif symbol_price >= take_profit_price:
        logger.info(f"[TP HIT] Preço={symbol_price:.4f}, TP={take_profit_price:.4f}")
        qty_to_sell = entry_quantity
        qty_to_sell = adjust_quantity(SYMBOL, qty_to_sell)
        if qty_to_sell > 0:
            logger.info(f"Fechando posição (COMPRA) => vendendo {qty_to_sell} {BASE_ASSET}")
            sell_market(SYMBOL, qty_to_sell)
        open_position = False
        entry_price   = 0.0
        entry_quantity= 0.0

# -----------------------------------------------------------------------
#     FUNÇÕES DE ML / BACKTEST (iguais às versões anteriores)
# -----------------------------------------------------------------------
def check_training_accuracy(X, y, df, model):
    df_train = pd.DataFrame(index=X.index)
    df_train['Close'] = df.loc[X.index, 'Close']
    df_train['Next_Close'] = y

    predictions = model.predict(X)
    df_train['Predicted_Close'] = predictions

    df_train['signal_ml'] = np.where(
        df_train['Predicted_Close'] > df_train['Close'], 1, -1
    )

    df_train['price_change'] = df_train['Next_Close'] - df_train['Close']
    conditions = [
        (df_train['signal_ml'] == 1) & (df_train['price_change'] > 0),
        (df_train['signal_ml'] == -1) & (df_train['price_change'] < 0),
    ]
    df_train['correct_signal'] = np.select(conditions, [1, 1], default=0)

    correct_signals = df_train['correct_signal'].sum()
    total_signals   = len(df_train)
    accuracy        = correct_signals / total_signals if total_signals > 0 else 0
    return accuracy, total_signals, correct_signals

def generate_signals_for_all_history(df, model):
    df = df.copy()
    features = [
        'Open', 'Close', 'High', 'Low', 'Volume',
        'Quote Asset Volume','Number of Trades',
        'Taker Buy Base Asset Volume','Taker Buy Quote Asset Volume',
        'RSI_14', 'RSI_7',
        'MACD_12_26_9', 'MACD_5_35_5',
        'StochRSI_14', 'StochRSI_7'
    ]
    df.dropna(subset=features, inplace=True)
    df['signal_ml'] = 0

    for i in range(len(df) - 1):
        row_features = df.iloc[i][features]
        X = pd.DataFrame([row_features], columns=features)
        predicted_close = model.predict(X)[0]
        current_close   = df['Close'].iloc[i]
        df.iloc[i, df.columns.get_loc('signal_ml')] = 1 if (predicted_close > current_close) else -1

    return df

# Carrega histórico
logger.info("Carregando dados históricos...")
klines = client2.get_historical_klines(SYMBOL, KLINE_INTERVAL, "1 jan, 2024")
df = pd.DataFrame(
    klines,
    columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time','Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume','Taker Buy Quote Asset Volume',
        'Ignore'
    ]
)

df['Open Time']  = pd.to_datetime(df['Open Time'],  unit='ms')
df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
df.set_index('Close Time', inplace=True)

df = df.astype({
    'Open': 'float',
    'High': 'float',
    'Low': 'float',
    'Close': 'float',
    'Volume': 'float',
    'Quote Asset Volume': 'float',
    'Taker Buy Base Asset Volume': 'float',
    'Taker Buy Quote Asset Volume': 'float'
})

# Calcula indicadores
indicadores = Indicadores()
df['RSI_14']       = indicadores.compute_RSI(df['Close'], 14)
df['RSI_7']        = indicadores.compute_RSI(df['Open'], 7)
df['MACD_12_26_9'] = indicadores.compute_MACD(df['Close'], 12, 26, 9)
df['MACD_5_35_5']  = indicadores.compute_MACD(df['Open'], 5, 35, 5)
df['StochRSI_14']  = indicadores.get_stochastic_rsi(df['Close'], 14, 14)
df['StochRSI_7']   = indicadores.get_stochastic_rsi(df['Open'], 7)

df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

logger.info(f"DF shape após indicadores e dropna: {df.shape}")

# Treinamento inicial
def train_model(df: pd.DataFrame):
    features = [
        'Open', 'Close', 'High', 'Low', 'Volume',
        'Quote Asset Volume','Number of Trades',
        'Taker Buy Base Asset Volume','Taker Buy Quote Asset Volume',
        'RSI_14', 'RSI_7',
        'MACD_12_26_9', 'MACD_5_35_5',
        'StochRSI_14', 'StochRSI_7'
    ]
    df.dropna(subset=features + ['Target'], inplace=True)

    X = df[features]
    y = df['Target']

    logger.info(f"Tamanho de X: {X.shape}. Treinando RandomForestRegressor...")

    # Salva CSV p/ auditoria (opcional)
    X_join = X.copy()
    X_join['Target'] = y
    X_join.to_csv('ml_data.csv', index=True)

    model = using_RandomForestRegressor.fixed_params_RandomForestRegressor(X, y)
    return model, X, y

def get_signal(df: pd.DataFrame, model):
    features = [
        'Open', 'Close', 'High', 'Low', 'Volume',
        'Quote Asset Volume','Number of Trades',
        'Taker Buy Base Asset Volume','Taker Buy Quote Asset Volume',
        'RSI_14', 'RSI_7',
        'MACD_12_26_9', 'MACD_5_35_5',
        'StochRSI_14', 'StochRSI_7'
    ]
    last_row = df.iloc[-1][features]
    X_last  = pd.DataFrame([last_row], columns=features)
    predicted_close = model.predict(X_last)[0]
    current_close   = df['Close'].iloc[-1]

    # +1 => prevê alta, -1 => prevê queda
    return 1 if (predicted_close > current_close) else -1

from backtest import Backtest
model, X_train, y_train = train_model(df)
acc_train, total_train, correct_train = check_training_accuracy(X_train, y_train, df, model)
logger.info(
    f"[ACURÁCIA TREINO] TotalRows={total_train}, Acertos={correct_train}, Acur={acc_train*100:.2f}%"
)

# -----------------------------------------------------------------------
#   THREAD PRINCIPAL DO MODELO (RODA EM INTERVALO DE CANDLE)
# -----------------------------------------------------------------------
def dca_model_loop():
    global open_position, entry_price, entry_quantity
    global model, X_train, y_train

    candle_count = 0

    while True:
        # Espera o candle fechar
        wait_for_next_close()

        # Atualiza DF com último candle
        last_kline = client.get_klines(symbol=SYMBOL, interval=KLINE_INTERVAL, limit=1)[0]
        new_row = pd.DataFrame([last_kline], columns=[
            'Open Time','Open','High','Low','Close','Volume',
            'Close Time','Quote Asset Volume','Number of Trades',
            'Taker Buy Base Asset Volume','Taker Buy Quote Asset Volume',
            'Ignore'
        ])
        new_row['Open Time']  = pd.to_datetime(new_row['Open Time'],  unit='ms')
        new_row['Close Time'] = pd.to_datetime(new_row['Close Time'], unit='ms')
        for c in ['Open','High','Low','Close','Volume',
                  'Quote Asset Volume','Number of Trades',
                  'Taker Buy Base Asset Volume','Taker Buy Quote Asset Volume']:
            new_row[c] = new_row[c].astype(float)
        new_row.set_index('Close Time', inplace=True)

        df.loc[new_row.index[0], new_row.columns] = new_row.iloc[0]
        if len(df) > MAX_CANDLES:
            df.drop(df.index[0], inplace=True)

        # Recalcula indicadores
        indicadores = Indicadores()
        df['RSI_14']       = indicadores.compute_RSI(df['Close'], 14)
        df['RSI_7']        = indicadores.compute_RSI(df['Open'], 7)
        df['MACD_12_26_9'] = indicadores.compute_MACD(df['Close'], 12, 26, 9)
        df['MACD_5_35_5']  = indicadores.compute_MACD(df['Open'], 5, 35, 5)
        df['StochRSI_14']  = indicadores.get_stochastic_rsi(df['Close'], 14, 14)
        df['StochRSI_7']   = indicadores.get_stochastic_rsi(df['Open'], 7)
        df['Target']       = df['Close'].shift(-1)

        candle_count += 1

        # Re-treina a cada RETRAIN_INTERVAL
        if candle_count % RETRAIN_INTERVAL == 0:
            logger.info(f"Re-treinando modelo (candles={candle_count})...")
            model, X_train, y_train = train_model(df)
            acc2, tot2, cor2 = check_training_accuracy(X_train, y_train, df, model)
            logger.info(
                f"[ReTreino] Tamanho={tot2}, Acertos={cor2}, Acur={acc2*100:.2f}%"
            )
            # Backtest
            df_bt = generate_signals_for_all_history(df.copy(), model)
            acc_b, tot_b, cor_b = Backtest().check_signal_accuracy(df_bt)
            logger.info(
                f"[Backtest FULL] TotTrades={tot_b}, Corretos={cor_b}, Acur={acc_b*100:.2f}%"
            )

        # Sincroniza relógio
        sync_time = subprocess.run(["w32tm", "/resync"], capture_output=True, text=True)
        logger.info("Sincronizando relógio do sistema...")
        if sync_time.stdout:
            logger.info(sync_time.stdout)
        if sync_time.stderr:
            logger.error(sync_time.stderr)

        # Lê sinal do modelo
        signal_ml = get_signal(df, model)
        df['signal_ml'] = 0
        df.loc[df.index[-1], 'signal_ml'] = signal_ml

        symbol_price  = get_symbol_price(SYMBOL)
        quote_balance = get_asset_balance(QUOTE_ASSET)
        base_balance  = get_asset_balance(BASE_ASSET)

        logger.info(f"[Candle={candle_count}] Sinal={signal_ml}, Saldo= {quote_balance:.2f} USDT, {base_balance:.4f} BNB")

        # Se não temos posição aberta:
        if not open_position:
            if signal_ml == 1:  
                # Compra ~70% do saldo USDT
                if quote_balance > 10:
                    qty_buy = (quote_balance * RISK_PERCENT_CAPITAL) / symbol_price
                    qty_buy = adjust_quantity(SYMBOL, qty_buy)
                    if qty_buy > 0:
                        logger.info(f"ABRINDO COMPRA => {qty_buy} BNB (DCA)")
                        buy_market(SYMBOL, qty_buy)
                        open_position   = True
                        entry_price     = symbol_price
                        entry_quantity  = qty_buy

            else:  
                # signal = -1 => vender BNB se tiver (sem abrir short)
                if base_balance > 0.001:
                    qty_sell = base_balance * RISK_PERCENT_CAPITAL
                    qty_sell = adjust_quantity(SYMBOL, qty_sell)
                    if qty_sell > 0:
                        logger.info(f"Vendendo BNB => {qty_sell} BNB (pois sinal=-1).")
                        sell_market(SYMBOL, qty_sell)
                        # Como não há "posição" formal, apenas esvaziamos BNB
                        open_position  = False
                        entry_price    = 0.0
                        entry_quantity = 0.0

        else:
            # Já temos posição comprada
            if signal_ml == 1:
                # DCA extra (compra mais)
                if quote_balance > 10:
                    qty_buy = (quote_balance * RISK_PERCENT_CAPITAL) / symbol_price
                    qty_buy = adjust_quantity(SYMBOL, qty_buy)
                    if qty_buy > 0:
                        logger.info(f"DCA extra: comprando +{qty_buy} BNB.")
                        buy_market(SYMBOL, qty_buy)
                        # Se quiser calcular preço médio, faça algo como:
                        old_value   = entry_quantity * entry_price
                        new_value   = qty_buy * symbol_price
                        total_value = old_value + new_value
                        entry_quantity += qty_buy
                        entry_price     = total_value / entry_quantity

            else:
                # Se o sinal ficou -1 e temos BNB => vendemos (fechamos posição)
                if base_balance > 0.001:
                    qty_to_sell = base_balance * RISK_PERCENT_CAPITAL
                    qty_to_sell = adjust_quantity(SYMBOL, qty_to_sell)
                    if qty_to_sell > 0:
                        logger.info(f"Fechando posição => vendendo {qty_to_sell} BNB (sinal=-1).")
                        sell_market(SYMBOL, qty_to_sell)
                open_position  = False
                entry_price    = 0.0
                entry_quantity = 0.0

        # Log final
        quote_balance = get_asset_balance(QUOTE_ASSET)
        base_balance  = get_asset_balance(BASE_ASSET)
        logger.info(f"Saldo pós: {quote_balance:.2f} USDT, {base_balance:.4f} BNB, entry_price={entry_price:.4f}")

        # Backtest rápido
        df_copy = df.copy()
        acc_x, tot_x, cor_x = Backtest().check_signal_accuracy(df_copy)
        logger.info(
            f"[Backtest FULL] TotTrades={tot_x}, Cor={cor_x}, Acur={acc_x*100:.2f}%"
        )
        logger.info("====================================")

# -----------------------------------------------------------------------
#   THREAD DE SL/TP (RODA A CADA 25 MIN)
# -----------------------------------------------------------------------
def sl_tp_loop():
    while True:
        logger.info("[SL/TP CHECK] Verificando Stop Loss / Take Profit...")
        check_sl_tp()
        time.sleep(SL_TP_CHECK_INTERVAL)

# -----------------------------------------------------------------------
#                               MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Criamos 2 threads:
    # 1) dca_model_loop(): segue o candle e faz DCA
    # 2) sl_tp_loop(): checa SL e TP a cada 25 min
    thread_dca = threading.Thread(target=dca_model_loop, daemon=True)
    thread_sl  = threading.Thread(target=sl_tp_loop,   daemon=True)

    thread_dca.start()
    thread_sl.start()

    # Mantém main vivo
    while True:
        time.sleep(60)  # ou qualquer outro valor
