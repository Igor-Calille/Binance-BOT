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
from backtest import Backtest  # <- classe que avalia acerto direcional
# Se tiver um "BacktestTrades" para PnL, poderia importar também.

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



SYMBOL = "SOLUSDT"      # Par a ser operado
BASE_ASSET = "SOL"
QUOTE_ASSET = "USDT"

# Stop Loss e Take Profit para timeframe de 1h
STOP_LOSS_PCT = 0.06     # ex: 5%
TAKE_PROFIT_PCT = 0.08   # ex: 7%

# Risco e DCA
RISK_PERCENT_CAPITAL = 0.95   # 70% do saldo p/ comprar/vender
RETRAIN_INTERVAL = 5          # Re-treina a cada 5 candles
MAX_CANDLES = 5555            # Máx. candles guardados em memória

# Intervalo de 1 hora
KLINE_INTERVAL = Client.KLINE_INTERVAL_1HOUR
SL_TP_CHECK_INTERVAL = 25 * 60  # 25 minutos => 1500s

# Credenciais – escolha se é Testnet ou conta real
testnet_api_key = os.getenv('TESTNET_API_KEY')
testnet_api_secret = os.getenv('TESTNET_API_SECRET')
client = Client(testnet_api_key, testnet_api_secret)
client.API_URL = 'https://testnet.binance.vision/api'  # Testnet

api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')
client2 = Client(api_key, api_secret)  # Usado para histórico

# Inicia serviço de tempo no Windows
start_service = subprocess.run(["net", "start", "w32time"], capture_output=True, text=True)
logger.info("Iniciando serviço de tempo do Windows...")
logger.info(start_service.stdout)
if start_service.stderr:
    logger.error(start_service.stderr)

# -----------------------------------------------------------------------
#             VARIÁVEIS GLOBAIS DE CONTROLE DE POSIÇÃO
# -----------------------------------------------------------------------
open_position   = False
entry_price     = 0.0
entry_quantity  = 0.0

# -----------------------------------------------------------------------
#                       FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------
def get_asset_balance(asset: str) -> float:
    """Retorna o saldo 'free' (disponível) de um ativo na conta da Binance."""
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
    """Retorna o último preço (ticker) do par."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        logger.error(f"Erro ao obter preço do par {symbol}: {e}")
        return 0.0

def get_symbol_info_filters(symbol: str):
    """Obtém filtros do par, p.ex. min_qty, max_qty, step_size, min_notional."""
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            return None
        filters_dict = {}
        for f in info['filters']:
            f_type = f['filterType']
            if f_type == 'LOT_SIZE':
                filters_dict['min_qty'] = float(f['minQty'])
                filters_dict['max_qty'] = float(f['maxQty'])
                filters_dict['step_size'] = float(f['stepSize'])
            elif f_type == 'MIN_NOTIONAL':
                filters_dict['min_notional'] = float(f['minNotional'])
        return filters_dict
    except Exception as e:
        logger.error(f"Erro ao obter symbol_info para {symbol}: {e}")
        return None

def adjust_quantity(symbol: str, quantity: float, price: float = None) -> float:
    """
    Ajusta 'quantity' para respeitar os limites de min_qty, max_qty, step_size e min_notional.
    """
    try:
        fdict = get_symbol_info_filters(symbol)
        if not fdict:
            logger.error(f"Não foi possível obter filtros de {symbol}")
            return 0

        min_qty = fdict.get('min_qty', 0)
        max_qty = fdict.get('max_qty', float('inf'))
        step_size = fdict.get('step_size', 1)
        min_notional = fdict.get('min_notional', 10)

        # Ajusta pela step_size
        adjusted_quantity = math.floor(quantity / step_size) * step_size
        casas_decimais = len(str(step_size).split('.')[1]) if '.' in str(step_size) else 0
        adjusted_quantity = round(adjusted_quantity, casas_decimais)

        # Checa min_qty e max_qty
        if adjusted_quantity < min_qty:
            logger.warning(f"Quantidade ajustada abaixo do mínimo: {adjusted_quantity} < {min_qty}")
            return 0
        if adjusted_quantity > max_qty:
            logger.warning(f"Quantidade ajustada acima do máximo: {adjusted_quantity} > {max_qty}")
            adjusted_quantity = max_qty

        # Checa min_notional
        if price is None or price <= 0:
            price = get_symbol_price(symbol)
            if price <= 0:
                logger.error(f"Preço inválido para {symbol}: {price}")
                return 0

        notional = adjusted_quantity * price
        if notional < min_notional:
            logger.warning(f"Valor total ({notional:.2f}) < minNotional ({min_notional}). Abortar ajuste.")
            return 0

        return adjusted_quantity
    except Exception as e:
        logger.error(f"Erro ao ajustar quantidade para {symbol}: {e}")
        return 0


def buy_market(symbol: str, quantity: float):
    """Envia ordem de compra a mercado."""
    try:
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        logger.info(f"Ordem de COMPRA: {order}")
        return order
    except Exception as e:
        logger.error(f"Erro ao comprar {symbol}: {e}")
        return None

def sell_market(symbol: str, quantity: float):
    """Envia ordem de venda a mercado."""
    try:
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        logger.info(f"Ordem de VENDA: {order}")
        return order
    except Exception as e:
        logger.error(f"Erro ao vender {symbol}: {e}")
        return None

def wait_for_next_close():
    """
    Aguarda até o próximo fechamento do candle de 1h, 
    usando a hora de fechamento do kline atual.
    """
    try:
        server_time = client.get_server_time()
        current_time = int(time.time() * 1000)
        time_diff = server_time['serverTime'] - current_time
        adjusted_time = int(time.time() * 1000) + time_diff

        last_kline = client.get_klines(symbol=SYMBOL, interval=KLINE_INTERVAL, limit=1)[0]
        next_close_time = last_kline[6]  # "close time" do kline atual
        wait_time = (next_close_time - adjusted_time) / 1000

        if wait_time > 0:
            logger.info(f"Aguardando candle fechar em {wait_time:.2f}s...")
            time.sleep(wait_time)
    except Exception as e:
        logger.error(f"Erro ao aguardar candle: {e}")
        time.sleep(60)

# -----------------------------------------------------------------------
#  CHECAGEM DE STOP LOSS E TAKE PROFIT (RODA EM THREAD)
# -----------------------------------------------------------------------
def check_sl_tp():
    """
    Executado a cada SL_TP_CHECK_INTERVAL.
    Se há posição aberta, verifica se o preço bateu SL ou TP. Se sim, vende tudo.
    """
    global open_position, entry_price, entry_quantity

    if not open_position:
        return

    symbol_price = get_symbol_price(SYMBOL)
    if symbol_price <= 0:
        return

    stop_loss_price   = entry_price * (1 - STOP_LOSS_PCT)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)

    # STOP LOSS
    if symbol_price <= stop_loss_price:
        logger.info(f"[SL HIT] Preço={symbol_price:.4f}, SL={stop_loss_price:.4f}")
        qty_to_sell = adjust_quantity(SYMBOL, entry_quantity, price=symbol_price)
        if qty_to_sell > 0:
            logger.info(f"Fechando posição => vendendo {qty_to_sell} {BASE_ASSET}")
            sell_market(SYMBOL, qty_to_sell)
        open_position  = False
        entry_price    = 0.0
        entry_quantity = 0.0

    # TAKE PROFIT
    elif symbol_price >= take_profit_price:
        logger.info(f"[TP HIT] Preço={symbol_price:.4f}, TP={take_profit_price:.4f}")
        qty_to_sell = adjust_quantity(SYMBOL, entry_quantity, price=symbol_price)
        if qty_to_sell > 0:
            logger.info(f"Fechando posição => vendendo {qty_to_sell} {BASE_ASSET}")
            sell_market(SYMBOL, qty_to_sell)
        open_position  = False
        entry_price    = 0.0
        entry_quantity = 0.0

# -----------------------------------------------------------------------
#     FUNÇÕES DE ML E BACKTEST
# -----------------------------------------------------------------------
def check_training_accuracy(X, y, df, model):
    """
    Verifica a acurácia direcional do modelo (se Close futuro > Close atual).
    """
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
    """
    Gera 'signal_ml' = +1/-1 em todo o histórico, 
    comparando 'predicted_close' e 'current_close'.
    """
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

# Carrega histórico inicial
logger.info("Carregando dados históricos (desde 1 jan 2024)...")
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

def train_model(df: pd.DataFrame):
    """Treina RandomForestRegressor c/ features e target."""
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

    # (Opcional) salva CSV
    X_join = X.copy()
    X_join['Target'] = y
    X_join.to_csv('ml_data.csv', index=True)

    model = using_RandomForestRegressor.fixed_params_RandomForestRegressor(X, y)
    return model, X, y

def get_signal(df: pd.DataFrame, model):
    """Retorna +1 se previsão for alta, -1 se for queda, usando último candle."""
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

    return 1 if (predicted_close > current_close) else -1

# Treinamento inicial do modelo
model, X_train, y_train = train_model(df)
acc_train, total_train, correct_train = check_training_accuracy(X_train, y_train, df, model)
logger.info(
    f"[ACURÁCIA TREINO] TotalRows={total_train}, Acertos={correct_train}, "
    f"Acur={acc_train*100:.2f}%"
)

# -----------------------------------------------------------------------
#   THREAD PRINCIPAL (CANDLE A CANDLE)
# -----------------------------------------------------------------------
def dca_model_loop():
    global open_position, entry_price, entry_quantity
    global model, X_train, y_train

    candle_count = 0

    while True:
        # Espera o candle de 1h fechar
        wait_for_next_close()

        # Pega o último candle e atualiza df
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

        # Adiciona ao df
        df.loc[new_row.index[0], new_row.columns] = new_row.iloc[0]
        if len(df) > MAX_CANDLES:
            df.drop(df.index[0], inplace=True)

        # Recalcula indicadores nas últimas linhas
        indicadores = Indicadores()
        df['RSI_14']       = indicadores.compute_RSI(df['Close'], 14)
        df['RSI_7']        = indicadores.compute_RSI(df['Open'], 7)
        df['MACD_12_26_9'] = indicadores.compute_MACD(df['Close'], 12, 26, 9)
        df['MACD_5_35_5']  = indicadores.compute_MACD(df['Open'], 5, 35, 5)
        df['StochRSI_14']  = indicadores.get_stochastic_rsi(df['Close'], 14, 14)
        df['StochRSI_7']   = indicadores.get_stochastic_rsi(df['Open'], 7)
        df['Target']       = df['Close'].shift(-1)

        candle_count += 1

        # A cada RETRAIN_INTERVAL, re-treina
        if candle_count % RETRAIN_INTERVAL == 0:
            logger.info(f"Re-treinando modelo (candles={candle_count})...")
            model, X_train, y_train = train_model(df)
            acc2, tot2, cor2 = check_training_accuracy(X_train, y_train, df, model)
            logger.info(
                f"[ReTreino] Tamanho={tot2}, Acertos={cor2}, Acur={acc2*100:.2f}%"
            )
            # Backtest "completo" (direcional) no dataset
            df_bt = generate_signals_for_all_history(df.copy(), model)
            acc_b, tot_b, cor_b = Backtest().check_signal_accuracy(df_bt)
            logger.info(
                f"[Backtest FULL] TotTrades={tot_b}, Corretos={cor_b}, Acur={acc_b*100:.2f}%"
            )

        # Sincroniza relógio no Windows
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

        # Consulta saldo e preço
        symbol_price  = get_symbol_price(SYMBOL)
        quote_balance = get_asset_balance(QUOTE_ASSET)
        base_balance  = get_asset_balance(BASE_ASSET)

        logger.info(
            f"[Candle={candle_count}] Sinal={signal_ml}, "
            f"Saldo= {quote_balance:.2f} USDT, {base_balance:.4f} BNB"
        )

        # Lógica de entradas/saídas
        if not open_position:
            if signal_ml == 1:
                # Compra ~70% do saldo em USDT
                if quote_balance > 10:
                    qty_buy = (quote_balance * RISK_PERCENT_CAPITAL) / symbol_price
                    qty_buy = adjust_quantity(SYMBOL, qty_buy, price=symbol_price)
                    if qty_buy > 0:
                        logger.info(f"ABRINDO COMPRA => {qty_buy} BNB")
                        buy_market(SYMBOL, qty_buy)
                        open_position   = True
                        entry_price     = symbol_price
                        entry_quantity  = qty_buy
            else:
                # Se sinal=-1, mas não temos posição => vender BNB leftover
                if base_balance > 0.0001:
                    qty_sell = adjust_quantity(SYMBOL, base_balance, price=symbol_price)
                    if qty_sell > 0:
                        logger.info(f"Zerando BNB => {qty_sell} BNB (sinal=-1).")
                        sell_market(SYMBOL, qty_sell)
                # Zera variáveis
                open_position  = False
                entry_price    = 0.0
                entry_quantity = 0.0
        else:
            # Já estamos comprados
            if signal_ml == 1:
                # DCA
                if quote_balance > 10:
                    qty_buy = (quote_balance * RISK_PERCENT_CAPITAL) / symbol_price
                    qty_buy = adjust_quantity(SYMBOL, qty_buy, price=symbol_price)
                    if qty_buy > 0:
                        logger.info(f"DCA extra: comprando +{qty_buy} BNB.")
                        buy_market(SYMBOL, qty_buy)
                        # Recalcula preço médio
                        old_val  = entry_quantity * entry_price
                        new_val  = qty_buy * symbol_price
                        total    = old_val + new_val
                        entry_quantity += qty_buy
                        entry_price     = total / entry_quantity
            else:
                # Sinal=-1 => Fecha posição (vende tudo)
                logger.info("Fechando posição => vendendo tudo.")
                qty_to_sell = adjust_quantity(SYMBOL, base_balance, price=symbol_price)
                if qty_to_sell > 0:
                    sell_market(SYMBOL, qty_to_sell)
                open_position  = False
                entry_price    = 0.0
                entry_quantity = 0.0

        # Checa saldo novamente
        quote_balance = get_asset_balance(QUOTE_ASSET)
        base_balance  = get_asset_balance(BASE_ASSET)
        logger.info(
            f"Saldo pós: {quote_balance:.2f} USDT, {base_balance:.4f} BNB, "
            f"entry_price={entry_price:.4f}"
        )

        # Backtest rápido do df atual (acurácia direcional)
        df_copy = df.copy()
        acc_x, tot_x, cor_x = Backtest().check_signal_accuracy(df_copy)
        logger.info(
            f"[Backtest QUICK] TotTrades={tot_x}, Cor={cor_x}, Acur={acc_x*100:.2f}%"
        )
        logger.info("====================================")

# -----------------------------------------------------------------------
#   THREAD DE SL/TP (roda a cada 25 min p/ checar SL e TP)
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

    # Comando para iniciar o serviço de sincronização de tempo do win
    start_service = subprocess.run(["net", "start", "w32time"], capture_output=True, text=True)
    print("Iniciando serviço de tempo:")
    print(start_service.stdout)
    print(start_service.stderr)

    # Thread 1: acompanha o candle e executa a lógica DCA
    thread_dca = threading.Thread(target=dca_model_loop, daemon=True)
    # Thread 2: verifica SL e TP periodicamente
    thread_sl  = threading.Thread(target=sl_tp_loop,   daemon=True)

    thread_dca.start()
    thread_sl.start()

    # Mantém main vivo
    while True:
        time.sleep(60)
