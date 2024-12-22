from binance.client import Client
from dotenv import load_dotenv
import os
import time
import pandas as pd
from datetime import datetime
from IndicadoresMercado import Indicadores
from MLModels import using_RandomForestRegressor
import numpy as np
import subprocess
from backtest import Backtest

load_dotenv()

# Utilize as chaves de API do Testnet
testnet_api_key = os.getenv('TESTNET_API_KEY')
testnet_api_secret = os.getenv('TESTNET_API_SECRET')

# Conecte-se ao cliente Binance Testnet
client = Client(testnet_api_key, testnet_api_secret)
client.API_URL = 'https://testnet.binance.vision/api'  # Endpoint para o Binance Spot Testnet

api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

client2 = Client(api_key, api_secret)

# Comando para iniciar o serviço de sincronização de tempo do win
start_service = subprocess.run(["net", "start", "w32time"], capture_output=True, text=True)
print("Iniciando serviço de tempo:")
print(start_service.stdout)
print(start_service.stderr)

# Função para obter o saldo de USDT
def get_usdt_balance():
    balance = client.get_asset_balance(asset='USDT')
    return float(balance['free']) if balance else 0.0

# Função para obter o saldo de BTC
def get_btc_balance():
    balance = client.get_asset_balance(asset='BTC')
    return float(balance['free']) if balance else 0.0

# Função para obter o preço de BTCUSDT
def get_btcusdt_price():
    ticker = client.get_symbol_ticker(symbol='BTCUSDT')
    return float(ticker['price'])

# Função para comprar BTC com USDT
def buy_btc(quantity):
    try:
        order = client.order_market_buy(symbol='BTCUSDT', quantity=quantity)
        print("Ordem de compra executada:", order)
        return order
    except Exception as e:
        print("Erro ao comprar BTC:", e)
        return None

# Função para vender BTC
def sell_btc(quantity):
    try:
        order = client.order_market_sell(symbol='BTCUSDT', quantity=quantity)
        print("Ordem de venda executada:", order)
        return order
    except Exception as e:
        print("Erro ao vender BTC:", e)
        return None

# Função para obter os limites permitidos para o par de negociação
def get_lot_size_limits(symbol):
    symbol_info = client.get_symbol_info(symbol)
    if symbol_info:
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                min_qty = float(filter['minQty'])
                max_qty = float(filter['maxQty'])
                step_size = float(filter['stepSize'])
                return min_qty, max_qty, step_size
    return None, None, None

# Função para ajustar a quantidade com base nos limites de LOT_SIZE
def adjust_quantity(quantity, min_qty, max_qty, step_size):
    if quantity < min_qty:
        return 0  # Quantidade muito pequena, ignora a operação
    if quantity > max_qty:
        quantity = max_qty  # Limita ao máximo permitido
    # Ajusta para o múltiplo mais próximo do step_size
    quantity = quantity - (quantity % step_size)
    return round(quantity, 6)  # Arredonda para 6 casas decimais

# Carregar histórico inicial de dados e calcular indicadores
klines = client2.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 jan, 2024")
df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                   'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                                   'Taker Buy Quote Asset Volume', 'Ignore'])

df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
df.set_index('Close Time', inplace=True)

# Converte colunas numéricas
df = df.astype({
    'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float',
    'Volume': 'float', 'Quote Asset Volume': 'float', 
    'Taker Buy Base Asset Volume': 'float', 'Taker Buy Quote Asset Volume': 'float'
})

# Inicializa o arquivo CSV para logs
log_file = "trade_log.csv"
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write("execution_time,action,signal_ml,btc_quantity,usdt_balance,btc_balance,btcusdt_price,close_now,predicted_close\n")

successful_trades = 0  # Contador de trades bem-sucedidas
last_close_time = None  # Armazena o último fechamento de candle para evitar duplicações

# Função para esperar até o próximo fechamento de candle
def wait_for_next_close():
    server_time = client.get_server_time()
    current_time = int(time.time() * 1000)
    time_diff = server_time['serverTime'] - current_time
    adjusted_time = int(time.time() * 1000) + time_diff  # Sincroniza com o horário da Binance
    
    last_kline = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=1)[0]
    next_close_time = last_kline[6]  # Próximo Close Time em milissegundos
    wait_time = (next_close_time - adjusted_time) / 1000  # Tempo restante em segundos
    
    if wait_time > 0:
        print(f"Aguardando até o próximo fechamento de candle em {wait_time:.2f} segundos...")
        time.sleep(1)

# Função principal de trading contínuo
def trading_bot():
    global successful_trades, last_close_time
    indicadores = Indicadores()  # Instância dos indicadores

    min_qty, max_qty, step_size = get_lot_size_limits("BTCUSDT")

    while True:
        # Espera até o próximo fechamento de candle
        wait_for_next_close()

        # Obtém o último candle
        last_kline = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=1)[0]
        close_time = last_kline[6]  # Tempo de fechamento do último candle

        # Evita duplicação: processa apenas se o fechamento for diferente do último
        if close_time != last_close_time:
            last_close_time = close_time  # Atualiza o último fechamento para o atual

            # Adiciona novo kline ao DataFrame e recalcula indicadores
            new_kline = [last_kline]
            new_row = pd.DataFrame(new_kline, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                                       'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                                                       'Taker Buy Quote Asset Volume', 'Ignore'])
            new_row['Open Time'] = pd.to_datetime(new_row['Open Time'], unit='ms')
            new_row['Close Time'] = pd.to_datetime(new_row['Close Time'], unit='ms')
            new_row.set_index('Close Time', inplace=True)

            # Apenas converte colunas numéricas para float
            new_row[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
                     'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']] = \
                new_row[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
                         'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']].astype(float)

            # Atualiza o DataFrame principal com os novos dados
            df.loc[new_row.index[0], new_row.columns] = new_row.iloc[0]

            # Recalcula os indicadores
            df['RSI_14'] = indicadores.compute_RSI(df['Close'], 14)
            df['RSI_7'] = indicadores.compute_RSI(df['Open'], 7)
            df['MACD_12_26_9'] = indicadores.compute_MACD(df['Close'], 12, 26, 9)
            df['MACD_5_35_5'] = indicadores.compute_MACD(df['Open'], 5, 35, 5)
            df['StochRSI_14'] = indicadores.get_stochastic_rsi(df['Close'], 14, 14)
            df['StochRSI_7'] = indicadores.get_stochastic_rsi(df['Open'], 7)

            # Definindo as features para o modelo
            features = ['Open', 'Close', 'High', 'Low', 'Volume', 'Quote Asset Volume', 'Number of Trades',
                        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'RSI_14', 'RSI_7',
                        'MACD_12_26_9', 'MACD_5_35_5', 'StochRSI_14', 'StochRSI_7']
            
            df['Target'] = df['Close'].shift(-1)  # Target para o modelo
            
            # Prepara X e y para o modelo, evitando NaNs no y
            df.dropna(inplace=True)
            X = df[features]
            y = df['Target']

            # Atualiza os dados para o modelo e faz a previsão
            model = using_RandomForestRegressor.fixed_params_RandomForestRegressor(X, y)
            df['Predicted_Close'] = model.predict(X)

            df['signal_ml'] = np.where(df['Predicted_Close'] > df['Close'], 1, -1)
            signal_ml = df['signal_ml'].iloc[-1]
            close_now = df['Close'].iloc[-1]

            # Executa a ordem de compra ou venda baseada no sinal
            usdt_balance = get_usdt_balance()
            btc_balance = get_btc_balance()
            btcusdt_price = get_btcusdt_price()
            action = None
            btc_quantity = 0

            
            
            if signal_ml == 1 and usdt_balance > 10:
                print("Sinal de compra detectado!")
                
                btc_quantity = (usdt_balance * 0.98) / btcusdt_price  
                btc_quantity = adjust_quantity(btc_quantity, min_qty, max_qty, step_size)
                print(f"Quantidade de compra ajustada: {btc_quantity}")

                if btc_quantity > 0:  # Verifica se a quantidade ajustada ainda é válida
                    #order = buy_btc(btc_quantity)
                    action = "buy"
                else:
                    print("Quantidade de compra ajustada é muito baixa, operação ignorada.")
                
            
            elif signal_ml == -1 and btc_balance > min_qty:
                print("Sinal de venda detectado!")
                
                btc_quantity = adjust_quantity(btc_balance, min_qty, max_qty, step_size)
                print(f"Quantidade de venda ajustada: {btc_quantity}")

                if btc_quantity > 0:  # Verifica se a quantidade ajustada ainda é válida
                    #order = sell_btc(btc_quantity)
                    action = "sell"
                else:
                    print("Quantidade de venda ajustada é muito baixa, operação ignorada.")


            backtest = Backtest()
            df_copy = df.copy()
            accuracy, total_signals, correct_signals = backtest.check_signal_accuracy(df_copy)

            print(f"Preço atual de BTCUSDT: {btcusdt_price}")
            print(f"Saldo de USDT: {usdt_balance}")
            print(f"Saldo de BTC: {btc_balance}")
            print(f"Sinal do modelo: {signal_ml}")
            print(f"Acurácia do sinal: {accuracy} -> ({correct_signals}/{total_signals})")

            sync_time = subprocess.run(["w32tm", "/resync"], capture_output=True, text=True)
            print("Sincronizando relógio do sistema:")
            print(sync_time.stdout)
            print(sync_time.stderr)
                
            # Registro do momento exato da execução com informações detalhadas
            execution_time = datetime.now()
            with open(log_file, 'a') as f:
                f.write(f"{execution_time},{action},{signal_ml},{btc_quantity},{usdt_balance},{btc_balance},{close_now},{btcusdt_price},"
                        f"{df['Predicted_Close'].iloc[-1]}\n")

            df.to_csv('df.csv')

            with open('df.csv', 'a') as f:
                f.write("\n")

trading_bot()
