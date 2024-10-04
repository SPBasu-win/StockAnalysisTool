import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load stock data
data = pd.read_csv('stock_data2.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Plot stock price over time
plt.figure(figsize=(10,6))
plt.plot(data.index, data['Close'])
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Calculate daily returns
data['Return'] = data['Close'].pct_change()

# Plot daily returns
plt.figure(figsize=(10,6))
plt.plot(data.index, data['Return'])
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()

# Prepare data for modeling
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define features and target variable
X_train = train_data[['Open', 'High', 'Low']]
y_train = train_data['Close']
X_test = test_data[['Open', 'High', 'Low']]
y_test = test_data['Close']

# Create a classical linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Compute the mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error (Classical): {mse}')

# Use the classical model to predict future stock prices
future_prices = []
for i in range(30):
    future_input = np.array([[data['Open'].iloc[-1], data['High'].iloc[-1], data['Low'].iloc[-1]]])
    future_price = model.predict(future_input)[0]
    future_prices.append(future_price)

# Analyze the trend of the predicted prices
trend = np.polyfit(range(len(future_prices)), future_prices, 1)

# Determine whether the stock price is expected to increase or decrease
if trend[0] > 0:
    print('The stock price is expected to increase (Classical).')
else:
    print('The stock price is expected to decrease (Classical).')