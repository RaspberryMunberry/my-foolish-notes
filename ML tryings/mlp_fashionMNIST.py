import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time

BATCH_SIZE = 96

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Отзеркаливание с вероятностью 50%
    transforms.RandomResizedCrop(size=28, scale=(0.9, 1.1)),  # Масштабирование случайным образом
    transforms.ToTensor(),  # Преобразование в тензор
])

train_dataset = tv.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = tv.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 196),
    nn.BatchNorm1d(196),
    nn.ReLU(),
    nn.Linear(196, 120),
    nn.BatchNorm1d(120),
    nn.ReLU(),
    nn.Linear(120, 60),
    nn.BatchNorm1d(60),
    nn.ReLU(),
    nn.Linear(60, 10)
)

criterion = nn.CrossEntropyLoss()  # Функция потерь для классификации
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # насколько я понимаю, L2 стабилизация задаётся weight_decay=1e-4
epochs = 10
best_val_loss = float('inf')
patience = 3 # порог эпох без улучшения результата
patience_counter = 0 

# функция для трени и теста

def START_THE_EVIL_THINGS():
    global best_val_loss, patience_counter # Почему то без этого вылезает ошибка о том что best_val_loss - локальная переменная, хотя внутри функции нормально используются и другие переменные типа epochs и patience, объявленные ранее вместе с best_val_loss
    global_start = time.time()
    for ep in range(epochs):
        start = time.time()

        train_loss, train_acc = 0., 0.
        train_passed, train_iters = 0, 0

        test_loss, test_acc = 0., 0.
        test_passed, test_iters = 0, 0

        #треним
        model.train()
        for images, labels in train:
            # Очистка градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Обратный проход
            loss.backward()
            optimizer.step()

            # Обновляем счетчики
            train_loss += loss.item()
            train_acc += (outputs.argmax(dim=1) == labels).sum().item()
            train_passed += len(images)
            train_iters += 1

        # Средняя ошибка и точность
        avg_train_loss = train_loss / train_iters
        avg_train_acc = train_acc / train_passed

        #тестим
        model.eval()
        with torch.no_grad():
            for images, labels in test:
                # Прямой проход
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Обновляем счетчики
                test_loss += loss.item()
                test_acc += (outputs.argmax(dim=1) == labels).sum().item()
                test_passed += len(images)
                test_iters += 1

        # Средняя ошибка и точность
        avg_test_loss = test_loss / test_iters
        avg_test_acc = test_acc / test_passed

        print(f'Эпоха: {ep+1}/{epochs}, время: {round(time.time()-start,2)}\n\ttr_loss: {round(avg_train_loss,5)} ts_loss: {round(avg_test_loss,5)}\n\ttr_acc: {round(avg_train_acc*100,2)}% ts_acc: {round(avg_test_acc*100,2)}%')

        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Ранняя остановка")
                break


    print(f'Всего времени: {round(time.time()-global_start,2)}')

    START_THE_EVIL_THINGS()