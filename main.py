import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def run_advanced_project():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Загрузка
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Price'] = data.target

    # 2. Обучение
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Предсказание
    predictions = model.predict(X_test)

    # --- НОВОЕ: ВИЗУАЛИЗАЦИЯ ---
    print("Создание графика результатов...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    plt.xlabel('Реальная цена ($100k)')
    plt.ylabel('Предсказанная цена ($100k)')
    plt.title('Сравнение реальных цен с предсказаниями модели')
    # Сохраняем график в файл (это круто для GitHub)
    plt.savefig('result_plot.png')
    print("График сохранен как result_plot.png")
    plt.show()
    # 4. Сохранение модели
    joblib.dump(model, 'models/house_price_model.pkl')


if __name__ == "__main__":
    run_advanced_project()