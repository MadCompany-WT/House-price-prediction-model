import pandas as pd
import numpy as np
import os
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def run_project():
    print("--- Запуск проекта: Прогнозирование цен на жилье ---")

    # 1. СОЗДАНИЕ ПАПОК (чтобы структура была правильной)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 2. ЗАГРУЗКА ДАННЫХ
    print("Загрузка данных о жилье...")
    housing = fetch_california_housing()

    # Создаем таблицу (DataFrame)
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target  # Целевая цена (в сотнях тысяч $)

    # Сохраняем данные в файл в папку data
    csv_path = 'data/housing_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Данные сохранены в: {csv_path}")

    # 3. ПОДГОТОВКА ДАННЫХ К ОБУЧЕНИЮ
    X = df.drop('Price', axis=1)  # Все колонки кроме цены
    y = df['Price']  # Только цена

    # Разделяем на тренировочные (80%) и тестовые (20%) данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ
    # Используем Random Forest — это один из самых мощных алгоритмов для таких задач
    print("Обучение модели... Пожалуйста, подождите.")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. ОЦЕНКА ТОЧНОСТИ
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 30)
    print(f"РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print(f"Средняя ошибка (MAE): {mae:.3f} (в сотнях тысяч $)")
    print(f"Точность (R2 Score): {r2:.2f} (макс. 1.0)")
    print("=" * 30)

    # 6. СОХРАНЕНИЕ ОБУЧЕННОЙ МОДЕЛИ
    model_file = 'models/house_price_model.pkl'
    joblib.dump(model, model_file)
    print(f"\nМодель успешно сохранена в: {model_file}")
    print("Проект выполнен без ошибок!")


if __name__ == "__main__":
    run_project()
