import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

# Utilize as chaves de API do Testnet
testnet_api_key = os.getenv('TESTNET_API_KEY')
testnet_api_secret = os.getenv('TESTNET_API_SECRET')

# Conecte-se ao cliente Binance Testnet
client = Client(testnet_api_key, testnet_api_secret)
client.API_URL = 'https://testnet.binance.vision/api'


# Função para obter o saldo de USDT
def get_usdt_balance():
    balance = client.get_asset_balance(asset='USDT')
    return float(balance['free']) if balance else 0.0

# Função para obter o saldo de BTC
def get_btc_balance():
    balance = client.get_asset_balance(asset='BTC')
    return float(balance['free']) if balance else 0.0

usdt = get_usdt_balance()
btc = get_btc_balance()
print(f"Saldo de USDT: {usdt}")
print(f"Saldo de BTC: {btc}")