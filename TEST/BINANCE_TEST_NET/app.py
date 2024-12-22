from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('TESTNET_API_KEY')
api_secret = os.getenv('TESTNET_API_SECRET')

client = Client(api_key, api_secret)
client.API_URL = 'https://testnet.binance.vision/api'


def get_usdt_balance():
    balance = client.get_asset_balance(asset='USDT')
    return float(balance['free']) if balance else 0.0

# Função para obter o preço de BTCUSDT na Testnet
def get_btcusdt_price():
    ticker = client.get_symbol_ticker(symbol='BTCUSDT')
    return float(ticker['price'])

# Função para obter as informações de LOT_SIZE
def get_lot_size(symbol):
    info = client.get_symbol_info(symbol)
    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            return float(f['minQty']), float(f['stepSize'])
    return None, None

def buy_btc_with_usdt(quant):
    try:
        order = client.order_market_buy(
            symbol='BTCUSDT',
            quantity=quant
        )
        print("Ordem de compra realizada com sucesso:", order)
    except Exception as e:
        print("Erro ao comprar BTC:", e)

# Exemplo de uso
usdt_balance = get_usdt_balance()
print("Saldo de USDT na Testnet:", usdt_balance)

btcusdt_price = get_btcusdt_price()
print("Preço de BTCUSDT na Testnet:", btcusdt_price)

# Verifica as restrições de LOT_SIZE para BTCUSDT
min_qty, step_size = get_lot_size('BTCUSDT')
print(f"Quantidade mínima: {min_qty}, Incremento: {step_size}")


btc_quantity_to_buy = (usdt_balance * 0.1) / btcusdt_price

btc_quantity_to_buy = round(btc_quantity_to_buy // step_size * step_size, 6)

#if btc_quantity_to_buy >= min_qty:
    #buy_btc_with_usdt(round(btc_quantity_to_buy, 6))
#else:
    #print("Saldo insuficiente para comprar BTC")