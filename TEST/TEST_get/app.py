from binance.client import Client
from dotenv import load_dotenv
import os
import time

# Carregar as variáveis de ambiente
load_dotenv()
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# Inicializar o cliente Binance
client = Client(api_key, api_secret)

# Função para sincronizar o tempo local com o tempo do servidor da Binance
def sync_time():
    server_time = client.get_server_time()
    local_time = int(time.time() * 1000)
    client.TIME_OFFSET = server_time['serverTime'] - local_time

# Sincronizar o horário
sync_time()

# Função para obter o saldo de USDT
def get_usdt_balance():
    try:
        balance = client.get_asset_balance(asset='USDT')  # Ajuste de recvWindow
        return float(balance['free']) if balance else 0.0
    except Exception as e:
        print("Erro ao obter saldo de USDT:", e)
        sync_time()  # Re-sincronize o horário em caso de erro e tente novamente
        return 0.0
    
def get_price():
    ticker = client.get_symbol_ticker(symbol='BNBUSDT')
    return float(ticker['price'])

def sell_btc(quantity):
    try:
        order = client.order_market_sell(symbol='BTCUSDT', quantity=quantity)
        print("Ordem de venda executada:", order)
        return order
    except Exception as e:
        print("Erro ao vender BTC:", e)
        return None
response = sell_btc(0.19141)



usdt_balance = get_usdt_balance()
print("Saldo de USDT:", usdt_balance)

btcusdt_price = get_price()
print("Preço de BNBUSDT:", btcusdt_price)
