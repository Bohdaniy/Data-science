# Необхідні бібліотеки
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Створення набору даних за допомогою DatasetGenerator
print("Створення набору даних за допомогою DatasetGenerator...")
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f'Feature{i+1}' for i in range(X.shape[1])])
df['Target'] = y
df.to_csv('Database.csv', index=False)
print("Набір даних згенеровано і збережено у файл 'generated_classification_dataset.csv'.")

# 2. Завантаження набору даних для класифікації з сайту Kaggle
print("Завантаження набору даних з Kaggle...")
try:
    kaggle_df = pd.read_csv('C:/Users/Admin/.kaggle/mushrooms.csv')  
    print("Набір даних Kaggle завантажено:")
    print(kaggle_df.head())
except FileNotFoundError:
    print("Помилка: файл 'mushrooms.csv' не знайдено.")

# 3. (Опціонально) Генерація набору даних для класифікації за допомогою стандартних бібліотек
print("Генерація випадкового набору даних для класифікації...")
X_random = np.random.rand(500, 3)  # 500 прикладів, 3 ознаки
y_random = np.random.randint(0, 2, 500)  # Два класи
random_df = pd.DataFrame(X_random, columns=['Feature1', 'Feature2', 'Feature3'])
random_df['Target'] = y_random
random_df.to_csv('Random_Database.csv', index=False)
print("Випадковий набір даних згенеровано і збережено у файл 'Random_Database.csv'.")

# 4. Моделювання для отриманих наборів даних з використанням регресійного аналізу
print("Проведення моделювання з використанням регресійного аналізу для згенерованого набору даних...")
# Завантаження згенерованого набору даних
df = pd.read_csv('Database.csv')
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точність моделі на згенерованому наборі даних: {accuracy:.2f}')
print("Проведення моделювання для випадково згенерованого набору даних...")
random_df = pd.read_csv('Random_Database.csv')
X_random = random_df.drop(columns=['Target'])
y_random = random_df['Target']
X_train, X_test, y_train, y_test = train_test_split(X_random, y_random, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точність моделі на випадково згенерованому наборі даних: {accuracy:.2f}')