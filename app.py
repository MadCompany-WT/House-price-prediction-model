import streamlit as st
import pandas as pd
import joblib

# Настройка страницы
st.set_page_config(page_title="Прогноз цен на жилье", page_icon="🏠")

st.title("🏠 Калькулятор стоимости жилья (California)")
st.write("Передвигайте ползунки слева, чтобы предсказать цену недвижимости.")


# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('models/house_price_model.pkl')


model = load_model()

# Настройка боковой панели (ползунки)
st.sidebar.header("Параметры дома:")


def user_input_features():
    med_inc = st.sidebar.slider('Средний доход в районе ($)', 0.5, 15.0, 3.5)
    house_age = st.sidebar.slider('Возраст дома (лет)', 1, 52, 28)
    ave_rooms = st.sidebar.slider('Среднее кол-во комнат', 1, 10, 5)
    ave_bedrms = st.sidebar.slider('Среднее кол-во спален', 1, 5, 1)
    pop = st.sidebar.slider('Население района', 3, 35000, 1400)
    occup = st.sidebar.slider('Жителей в одном доме', 1, 10, 3)
    lat = st.sidebar.slider('Широта (Latitude)', 32.5, 42.0, 35.6)
    long = st.sidebar.slider('Долгота (Longitude)', -124.3, -114.3, -119.5)

    data = {
        'MedInc': med_inc, 'HouseAge': house_age, 'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms, 'Population': pop, 'AveOccup': occup,
        'Latitude': lat, 'Longitude': long
    }
    return pd.DataFrame(data, index=[0])


df_input = user_input_features()

# Кнопка расчета
if st.button('Рассчитать цену'):
    prediction = model.predict(df_input)
    # Цены в датасете в сотнях тысяч долларов
    final_price = prediction[0] * 100000

    st.success(f"### Примерная стоимость: ${final_price:,.0f}")
    st.info("Это предсказание на основе обученной модели Random Forest.")