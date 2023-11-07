from mnist import MNIST
import random
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
mndata = MNIST('./train_data')
images, labels = mndata.load_training()

# Преобразование изображений в массивы numpy и их нормализация
images = np.array(images)
labels = np.array(labels)

# Нормализация значений пикселей (диапазон [0, 255] -> [0, 1])
images = images / 255.0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Создание модели k-NN
knn = KNeighborsClassifier(n_neighbors=10)  # Установите значение k по вашему выбору

# Обучение модели на обучающих данных
knn.fit(X_train, y_train)

# Предсказание меток для тестовых данных
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Точность модели:", accuracy)

correct_indices = np.where(y_test == y_pred)[0]
incorrect_indices = np.where(y_test != y_pred)[0]

# Создайте подплоты для 5x5 изображений
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Выведите 25 случайных правильных изображений
for i in range(25):
    ax = axes[i // 5, i % 5]
    index = random.choice(correct_indices)
    image_data = X_test[index]  # Получите изображение из тестовых данных
    image_data = image_data.reshape(28, 28)  # Измените размерность

    # Предсказание метки для изображения с помощью модели k-NN
    predicted_label = knn.predict([image_data.flatten()])[0]
    actual_label = y_test[index]  # Получите реальную метку

    # Отобразите изображение и реальную метку
    ax.imshow(image_data, cmap='gray')
    ax.set_title(f"Actual Label: {actual_label}\nPredicted Label: {predicted_label}")
    ax.axis('off')  # Отключите оси координат


# Создайте подплоты для 5x5 изображений для неправильных предсказаний
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Выведите 25 случайных неправильных изображений
for i in range(25):
    ax = axes[i // 5, i % 5]
    index = random.choice(incorrect_indices)
    image_data = X_test[index]  # Получите изображение из тестовых данных
    image_data = image_data.reshape(28, 28)  # Измените размерность

    # Предсказание метки для изображения с помощью модели k-NN
    predicted_label = knn.predict([image_data.flatten()])[0]
    actual_label = y_test[index]  # Получите реальную метку

    # Отобразите изображение
    ax.imshow(image_data, cmap='gray')
    ax.set_title(f"Actual Label: {actual_label}\nPredicted Label: {predicted_label}")
    ax.axis('off')  # Отключите оси координат

# Показать изображения в окне
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Получите матрицу ошибок
confusion = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок с помощью seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title("Матрица ошибок (Confusion Matrix)")
plt.show()
