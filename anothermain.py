from mnist import MNIST
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Загрузка данных
mndata = MNIST('./train_data')
images, labels = mndata.load_training()

# Преобразование изображений в массивы numpy и их нормализация
images = np.array(images)
labels = np.array(labels)
images = images / 255.0

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Создание модели SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Обучение модели на обучающих данных
svm.fit(X_train, y_train)

# Предсказание меток для тестовых данных
y_pred_svm = svm.predict(X_test)

# Оценка производительности модели SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Точность модели SVM:", accuracy_svm)

# Получите матрицу ошибок
confusion = confusion_matrix(y_test, y_pred_svm)

# Визуализация матрицы ошибок с помощью seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title("Матрица ошибок (Confusion Matrix)")
plt.show()

# Получите индексы правильных и неправильных предсказаний
correct_indices = np.where(y_test == y_pred_svm)[0]
incorrect_indices = np.where(y_test != y_pred_svm)[0]

# Создание подплотов для 5x5 изображений
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Выведите 25 случайных правильных изображений
for i in range(25):
    ax = axes[i // 5, i % 5]
    index = random.choice(correct_indices)
    image_data = X_test[index].reshape(28, 28)
    predicted_label = svm.predict([image_data.flatten()])[0]

    ax.imshow(image_data, cmap='gray')
    ax.set_title(f"Predicted Label: {predicted_label}")
    ax.axis('off')

# Создание подплотов для 5x5 изображений для неправильных предсказаний
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Выведите 25 случайных неправильных изображений
for i in range(25):
    ax = axes[i // 5, i % 5]
    index = random.choice(incorrect_indices)
    image_data = X_test[index].reshape(28, 28)
    predicted_label = svm.predict([image_data.flatten()])[0]

    ax.imshow(image_data, cmap='gray')
    ax.set_title(f"Predicted Label: {predicted_label}")
    ax.axis('off')

# Показать изображения в окне
plt.show()
