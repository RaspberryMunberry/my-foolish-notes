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
keras_neuron.compile(optimizer='adam', loss='mean_squared_error')

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
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# Keras: Предсказания на тестовых данных
y_pred_keras = keras_neuron.predict(x_test)

# PyTorch: Предсказания на тестовых данных
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_pred_pytorch = torch_neuron(x_test_tensor).detach().numpy()

# Выводим результаты для сравнения
print(f"Keras predictions: {y_pred_keras[:5].flatten()}")
print(f"PyTorch predictions: {y_pred_pytorch[:5].flatten()}")