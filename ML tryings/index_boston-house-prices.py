# Делаем нейрон на торче и керасе для предсказания стоимости жилья

import torch
import tensorflow
import pandas as pd
import sklearn
import kagglehub
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

path = kagglehub.dataset_download("vikrishnan/boston-house-prices")
data_path = '/root/.cache/kagglehub/datasets/vikrishnan/boston-house-prices/versions/1/housing.csv'
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = pd.read_csv(data_path, delim_whitespace=True, header=None, names=column_names)

print(data.head())
print(data.columns)

# разбиваем данные на свойства жилья и его стоимость по отдельности
x = data.drop(columns='MEDV') # MEDV - цена на жильё
y = data['MEDV']

# делим данные на тренировочную и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'Training data shape: {x_train.shape}')
print(f'Test data shape: {x_test.shape}')

# нейрон на керасе

# создаём модель
keras_neuron = Sequential()

# добавляем один нейрон (Dense layer) с одним входом (input_dim=X_train.shape[1]) и активацией 'linear'
keras_neuron.add(Dense(1, input_dim=x_train.shape[1], activation='linear'))

# компилируем модель
keras_neuron.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

# нейрон на торче

# моздаем модель
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

torch_neuron = SimpleModel(x_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(torch_neuron.parameters(), lr=0.01)

# Преобразуем данные в тензоры PyTorch
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# обучаем модели
keras_neuron.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # Обнуляем градиенты
    outputs = torch_neuron(x_train_tensor)  # Прогоняем данные через модель
    loss = criterion(outputs, y_train_tensor)  # Считаем ошибку
    loss.backward()  # Вычисляем градиенты
    optimizer.step()  # Обновляем веса


# Keras: Предсказания на тестовых данных
y_pred_keras = keras_neuron.predict(x_test)

# PyTorch: Предсказания на тестовых данных
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_pred_pytorch = torch_neuron(x_test_tensor).detach().numpy()

# Выводим результаты для сравнения
print(f"Keras predictions: {y_pred_keras[:5].flatten()}")
print(f"PyTorch predictions: {y_pred_pytorch[:5].flatten()}")

# Предсказания Keras на тестовых данных
y_pred_keras = keras_neuron.predict(x_test)

# Вычисляем MSE и R^2 для Keras
mse_keras = mean_squared_error(y_test, y_pred_keras)
r2_keras = r2_score(y_test, y_pred_keras)

print(f"Keras - Mean Squared Error: {mse_keras}")
print(f"Keras - R² Score: {r2_keras}")

# Предсказания PyTorch на тестовых данных
y_pred_pytorch = torch_neuron(x_test_tensor).detach().numpy()

# Вычисляем MSE и R^2 для PyTorch
mse_pytorch = mean_squared_error(y_test, y_pred_pytorch)
r2_pytorch = r2_score(y_test, y_pred_pytorch)

print(f"PyTorch - Mean Squared Error: {mse_pytorch}")
print(f"PyTorch - R² Score: {r2_pytorch}")

# Фактические значения для тестовой выборки
y_test_np = y_test.values

# График "Фактические vs Предсказанные значения"
plt.figure(figsize=(14, 6))

# График для Keras
plt.subplot(1, 2, 1)
plt.scatter(y_test_np, y_pred_keras.flatten(), color='blue')
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], 'k--', lw=2)
plt.title('Keras: Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# График для PyTorch
plt.subplot(1, 2, 2)
plt.scatter(y_test_np, y_pred_pytorch.flatten(), color='red')
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], 'k--', lw=2)
plt.title('PyTorch: Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.tight_layout()
plt.show()

# График остатков (Residual Plot)
plt.figure(figsize=(14, 6))

# Остатки для Keras
residuals_keras = y_test_np - y_pred_keras.flatten()
plt.subplot(1, 2, 1)
plt.scatter(y_pred_keras.flatten(), residuals_keras, color='blue')
plt.hlines(y=0, xmin=min(y_pred_keras.flatten()), xmax=max(y_pred_keras.flatten()), colors='k', lw=2)
plt.title('Keras: Residuals')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')

# Остатки для PyTorch
residuals_pytorch = y_test_np - y_pred_pytorch.flatten()
plt.subplot(1, 2, 2)
plt.scatter(y_pred_pytorch.flatten(), residuals_pytorch, color='red')
plt.hlines(y=0, xmin=min(y_pred_pytorch.flatten()), xmax=max(y_pred_pytorch.flatten()), colors='k', lw=2)
plt.title('PyTorch: Residuals')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()