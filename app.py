import streamlit as st
import pandas as pd
import joblib

# Интерфейс баптаулары
st.set_page_config(page_title="Үй бағасын болжау", page_icon="🏠")

st.title("🏠 Тұрғын үй құнын есептеу (Қазақстан)")
st.write("Мәліметтерді толтырып, үйдің болжамды бағасын теңгемен біліңіз.")

# Модельді жүктеу
model = joblib.load('models/house_price_model.pkl')

# Слайдерлер (Қолданушыға түсінікті параметрлер)
st.sidebar.header("Үй параметрлері:")

# 1. Айлық табыс (Теңгемен)
monthly_income = st.sidebar.number_input('Отбасының айлық табысы (теңге)', min_value=100000, max_value=10000000,
                                         value=650000, step=50000)

# 2. Квадрат метр
area = st.sidebar.slider('Үйдің ауданы (квадрат метр)', 20, 300, 80)

# 3. Бөлмелер саны
rooms = st.sidebar.slider('Жалпы бөлме саны', 1, 10, 4)
bedrooms = st.sidebar.slider('Жатын бөлме саны', 1, 5, 2)

# 4. Адам саны
household_size = st.sidebar.slider('Үйде тұратын адам саны', 1, 15, 5)

# 5. Үйдің жасы
house_age = st.sidebar.slider('Үйдің жасы (жыл)', 1, 60, 15)

# --- МАТЕМАТИКАЛЫҚ АЙНАЛДЫРУ ---
# 1$ = 450тг деп алсақ:
# Модель табысты "10 000$" бірлігінде күтеді
yearly_income_usd = (monthly_income * 12) / 500
med_inc_scaled = yearly_income_usd / 10000

# Квадрат метрді модельдің "AveRooms" бағанына бейімдеу
# (орташа есеппен 1 бөлме 25-30 кв.м)
ave_rooms = area / 8

# Координаттарды орташа мәнде жасырын қалдырамыз (Калифорния үшін стандартты)
lat = 35.6
long = -119.5
pop = 1500  # Орташа халық саны

# Модельге жіберілетін мәліметтер
input_data = pd.DataFrame({
    'MedInc': [med_inc_scaled],
    'HouseAge': [house_age],
    'AveRooms': [ave_rooms],
    'AveBedrms': [bedrooms / 1.5],  # коэффициент
    'Population': [pop],
    'AveOccup': [household_size],
    'Latitude': [lat],
    'Longitude': [long]
})

# Есептеу батырмасы
if st.button('Бағаны есептеу'):
    # Модель болжамы (нәтиже 100 000$ бірлігінде шығады)
    raw_prediction = model.predict(input_data)

    # Теңгеге айналдыру: нәтиже * 100,000 * 450
    final_price_kzt = raw_prediction[0] * 100000 * 500

    # Нақтырақ болуы үшін ауданға байланысты түзету (шартты)
    # Егер 4 бөлмелі үй болса, баға тым арзан болмауы керек
    if final_price_kzt < 15000000:
        final_price_kzt *= 1.5

        multiplier = 0.8

        final_price_kzt = final_price_kzt * multiplier

    st.success(f"### Үйдің болжамды бағасы: {final_price_kzt:,.0f} теңге")

    st.info(f"""
    **Талдау:**
    - Айлық табыс: {monthly_income:,.0f} ₸
    - Үй ауданы: {area} м²
    - Бөлме саны: {rooms}
    """)