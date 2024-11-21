import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import talib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# Завантаження даних
ticker = "AAPL"
data = yf.download(ticker, start="2022-11-01", end="2023-11-01")

# Перевірка пропущених значень
missing_values = data.isnull().sum()

# Графік ціни закриття
plt.figure(figsize=(14, 6))
plt.plot(data['Close'], label="Ціна закриття")
plt.title(f"Ціна закриття акцій {ticker}")
plt.legend()
plt.show()

# Описова статистика
stats = data['Close'].describe()

# Декомпозиція часового ряду
close_prices = data['Close'].dropna()
result = seasonal_decompose(close_prices, model='additive', period=30)
trend, seasonal, residual = result.trend, result.seasonal, result.resid

plt.figure(figsize=(14, 8))
plt.subplot(4, 1, 1); plt.plot(close_prices, label='Оригінальний ряд'); plt.legend()
plt.subplot(4, 1, 2); plt.plot(trend, label='Тренд', color='orange'); plt.legend()
plt.subplot(4, 1, 3); plt.plot(seasonal, label='Сезонність', color='green'); plt.legend()
plt.subplot(4, 1, 4); plt.plot(residual, label='Випадкова компонента', color='red'); plt.legend()
plt.tight_layout(); plt.show()

# Розрахунок технічних індикаторів
data['SMA_7'] = data['Close'].rolling(window=7).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['Volatility'] = data['Close'].rolling(window=30).std()

plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1); plt.plot(data['Close'], label='Ціна закриття', color='blue')
plt.plot(data['SMA_7'], label='SMA (7 днів)', color='orange')
plt.plot(data['SMA_30'], label='SMA (30 днів)', color='green'); plt.legend()
plt.subplot(3, 1, 2); plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--'); plt.axhline(30, color='blue', linestyle='--'); plt.legend()
plt.subplot(3, 1, 3); plt.plot(data['Volatility'], label='Волатильність (30 днів)', color='gray'); plt.legend()
plt.tight_layout(); plt.show()

support_level, resistance_level = data['Close'].min(), data['Close'].max()
crosses = (data['SMA_7'] > data['SMA_30']).astype(int).diff()
cross_dates = data.index[crosses == 1]

# Прогнозування
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

model_ewm = ExponentialSmoothing(train, trend='add').fit()
forecast_ewm = model_ewm.forecast(len(test))

arima_model = sm.tsa.ARIMA(train, order=(5, 1, 0)).fit()
forecast_arima = arima_model.forecast(steps=len(test))[0]

mse_ewm, mae_ewm = mean_squared_error(test, forecast_ewm), mean_absolute_error(test, forecast_ewm)
mse_arima, mae_arima = mean_squared_error(test, forecast_arima), mean_absolute_error(test, forecast_arima)

plt.figure(figsize=(14, 6))
plt.plot(train, label='Навчальна вибірка')
plt.plot(test, label='Тестова вибірка', color='orange')
plt.plot(test.index, forecast_ewm, label='Прогноз (EWM)', color='green')
plt.plot(test.index, forecast_arima, label='Прогноз (ARIMA)', color='red')
plt.legend(); plt.show()
