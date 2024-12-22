from binance.client import Client
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import backtrader as bt

from IndicadoresMercado import Indicadores
from MLModels import using_RandomForestRegressor 
from backtest import Backtest, MLStrategy

load_dotenv()

api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

client = Client(api_key, api_secret)

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 jan, 2024")

df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                   'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                                   'Taker Buy Quote Asset Volume', 'Ignore'])

df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
df.set_index('Close Time', inplace=True)

df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
df['Volume'] = df['Volume'].astype(float)
df['Quote Asset Volume'] = df['Quote Asset Volume'].astype(float)
df['Taker Buy Base Asset Volume'] = df['Taker Buy Base Asset Volume'].astype(float)
df['Taker Buy Quote Asset Volume'] = df['Taker Buy Quote Asset Volume'].astype(float)

features = ['Open', 'Close', 'High', 'Low', 'Volume', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
df['target'] = df['Close'].shift(-1)

indicadores = Indicadores()
# Indicadores
df['RSI_14'] = indicadores.compute_RSI(df['Close'], 14)
df['RSI_7'] = indicadores.compute_RSI(df['Open'], 7)
features.append('RSI_14')
features.append('RSI_7')

df['MACD_12_26_9'] = indicadores.compute_MACD(df['Close'], 12, 26, 9)  
df['MACD_5_35_5'] = indicadores.compute_MACD(df['Open'], 5, 35, 5)
features.append('MACD_12_26_9')
features.append('MACD_5_35_5')

df['StochRSI_14'] = indicadores.get_stochastic_rsi(df['Close'], 14, 14)
df['StochRSI_7'] = indicadores.get_stochastic_rsi(df['Open'], 7, 7)
features.append('StochRSI_14')
features.append('StochRSI_7')

df.dropna(inplace=True)

X = df[features]
y = df['target']



model = using_RandomForestRegressor.fixed_params_RandomForestRegressor(X, y)

df['Predicted_Close'] = model.predict(X)

df['signal_ml'] = np.where(df['Predicted_Close'] > df['Close'], 1, -1) 

df_copy = df.copy()
accuracy, total_signals, correct_signals = Backtest().check_signal_accuracy(df_copy)

print(f"Accuracy: {accuracy}")
print(f"Total signals: {total_signals}")
print(f"Correct signals: {correct_signals}")

class PandasData(bt.feeds.PandasData):
    lines = ('signal_ml',)
    params = (('signal_ml', -3),)

data_feed = PandasData(dataname=df, open='Open', high='High', low='Low', close='Close', volume='Volume')

cerebro = bt.Cerebro()
cerebro.adddata(data_feed)
cerebro.addstrategy(MLStrategy)
cerebro.broker.setcash(10000.0)
cerebro.run()
print(f"Final value: {cerebro.broker.getvalue()}")
cerebro.plot()


df.to_csv('BTCUSDT.csv')