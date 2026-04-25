import pandas as pd
import os
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_model():
    os.makedirs('models', exist_ok=True)

    # Мәліметтерді жүктеу
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target

    # Модельді үйрету
    print("Модель үйретілуде...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Сақтау
    joblib.dump(model, 'models/house_price_model.pkl')
    print("Модель дайын!")


if __name__ == "__main__":
    train_model()