from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ

IBMQ.save_account('fb3edb5caba5da52fd8299558cebee528d1bcf1a64d42c44073480cd82f85a49f5a2e9ce5004a870b3f8a425b3c4e48bc2e270ed9d7b2b366b9b2345f5085db5')

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

# Create a quantum circuit for linear regression
qc = QuantumCircuit(3, 1)
qc.x(0)
qc.h(1)
qc.cx(1, 2)
qc.barrier()
qc.h(0)
qc.barrier()
qc.measure([0], [0])

# Transpile the circuit for the Aer simulator
qc = transpile(qc, AerSimulator())

# Run the circuit on the Aer simulator
job = AerSimulator().run(qc)
result = job.result()

# Get the counts from the result
counts = result.get_counts()

# Define a function to compute the expectation value of the circuit
def compute_expectation_value(params):
    qc = QuantumCircuit(3, 1)
    qc.x(0)
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()
    qc.h(0)
    qc.barrier()
    qc.measure([0], [0])
    qc = transpile(qc, AerSimulator())
    job = AerSimulator().run(qc)
    result = job.result()
    counts = result.get_counts()
    expectation_value = 0
    for outcome in counts:
        expectation_value += int(outcome, 2) * counts[outcome]
    return expectation_value / sum(counts.values())

# Define a function to compute the gradient of the expectation value
def compute_gradient(params):
    eps = 1e-6
    gradient = []
    for i in range(len(params)):
        params_plus_eps = params.copy()
        params_plus_eps[i] += eps
        params_minus_eps = params.copy()
        params_minus_eps[i] -= eps
        gradient.append((compute_expectation_value(params_plus_eps) - compute_expectation_value(params_minus_eps)) / (2 * eps))
    return gradient

# Define a function to optimize the parameters using gradient descent
def optimize_params(params, learning_rate, num_iterations):
    for i in range(num_iterations):
        grad = compute_gradient(params)
        params = [params[j] - learning_rate * grad[j] for j in range(len(params))]
    return params

# Optimize the parameters using gradient descent
params = [0.5, 0.5, 0.5]
params = optimize_params(params, 0.1, 100)

# Compute the expectation value of the circuit with the optimized parameters
optimized_expectation_value = compute_expectation_value(params)

# Use the expectation value as the predicted value
y_pred = [optimized_expectation_value] * len(y_test)

# Compute the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print(f'Mean Squared Error: {mse}')

# Use the quantum circuit to predict future stock prices
future_prices = []
for i in range(30):
    params = [0.5, 0.5, 0.5]
    params = optimize_params(params, 0.1, 100)
    optimized_expectation_value = compute_expectation_value(params)
    future_prices.append(optimized_expectation_value)

# Analyze the trend of the predicted prices
trend = np.polyfit(range(len(future_prices)), future_prices, 1)

# Determine whether the stock price is expected to increase or decrease
if trend[0] > 0:
    print('The stock price is expected to increase.')
else:
    print('The stock price is expected to decrease.')