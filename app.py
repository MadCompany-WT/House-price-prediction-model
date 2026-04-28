import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. Бет баптаулары мен стильдері
st.set_page_config(page_title="Qyzylorda Smart Property AI", page_icon="🏠", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .district-card {
        background: white;
        padding: 22px; 
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-top: 6px solid #00d4ff;
        text-align: center;
        transition: 0.3s;
        margin-bottom: 15px;
    }
    .district-card:hover { transform: translateY(-8px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .district-card h4 {
        color: #000000 !important;
        font-weight: bold !important;
        margin-top: 10px;
        font-size: 19px;
    }
    .price-tag { color: #2e7d32; font-weight: bold; font-size: 22px; }
    .m2-price { color: #666666; font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)


# Модельді жүктеу
@st.cache_resource
def load_model():
    return joblib.load('models/house_price_model.pkl')


model = load_model()

# 2. Sidebar: Интеллектуалды басқару панелі
with st.sidebar:
    st.header("⚙️ Параметрлер")
    st.divider()

    # ТҮРІН ТАҢДАУ (ГАЛОЧКА)
    is_house = st.checkbox("🏡 Жер үй (Жер учаскесімен)", value=False)

    area = st.slider("📐 Үй ауданы (м²)", 20, 400, 85)
    rooms = st.slider("🚪 Бөлме саны", 1, 8, 3)

    if is_house:
        # Жер үйге арналған гектар (сотка)
        land_size = st.slider("🌱 Жер көлемі (Гектармен)", 0.02, 1.0, 0.06, step=0.01)
        st.caption(f"Бұл шамамен {land_size * 100:.1f} сотка")
        floor_val = 1  # Жер үй әрқашан 1-қабат деп алынады
    else:
        # Пәтерге арналған қабат (1-5)
        floor_val = st.select_slider("🏢 Қай қабатта? (Макс 5)", options=[1, 2, 3, 4, 5], value=3)
        land_size = 0

    income = st.number_input("💰 Айлық табыс (₸)", value=650000, step=50000)
    st.divider()
    st.write(f"Таңдалды: **{'Жер үй' if is_house else 'Пәтер'}**")


districts = {
    "Аудан": ["Орталық", "Сырдария", "Арай", "Шұғыла", "Титов", "КБИ", "Сельмаш", "Ақмешіт"],
    "Коэф": [1.35, 1.28, 1.15, 1.12, 0.85, 0.92, 0.88, 1.10],
    "Icon": ["🏛️", "🏙️", "🏡", "🌇", "🏭", "🚉", "🏗️", "🆕"]
}
df_dist = pd.DataFrame(districts)


def calc_smart_price(row):
    USD_KZT = 380
    MULTIPLIER = 0.9
    QYZ_INDEX = 0.4

    # Модельге дайындық
    med_inc = (income * 12) / USD_KZT / 10000
    input_data = pd.DataFrame({
        'MedInc': [med_inc], 'HouseAge': [15], 'AveRooms': [area / 25],
        'AveBedrms': [1.2], 'Population': [1500], 'AveOccup': [4],
        'Latitude': [34.0], 'Longitude': [-118.0]
    })

    raw_pred = model.predict(input_data)[0]
    base_price = raw_pred * 100000 * USD_KZT * MULTIPLIER * row['Коэф'] * QYZ_INDEX

    # ЖЕР ҮЙ мен ПӘТЕР логикасы
    if is_house:
        # Жер үй: Гектар үшін баға қосылады (1 сотка ~ 1.5 млн тг орташа)
        land_value = land_size * 100 * 1500000
        final_price = base_price + land_value
    else:
        # Пәтер: Қабат коэффициенті (2,3,4-қабаттар +10%)
        if floor_val in [2, 3, 4]:
            final_price = base_price * 1.1
        else:
            final_price = base_price * 0.9

    return int(final_price)


df_dist['Баға'] = df_dist.apply(calc_smart_price, axis=1)

# 4. Негізгі Бөлім
st.title("🏢 AI Qyzylorda Property Expert")
st.write(f"Қазіргі таңдау: **{area} м², {rooms} бөлмелі {'Жер үй' if is_house else 'Пәтер'}**")

# Карточкаларды шығару
for i in range(0, len(df_dist), 4):
    cols = st.columns(4)
    for j in range(4):
        if i + j < len(df_dist):
            row = df_dist.iloc[i + j]
            with cols[j]:
                st.markdown(f"""
                <div class="district-card">
                    <div style="font-size: 45px; margin-bottom: 10px;">{row['Icon']}</div>
                    <h4>{row['Аудан']}</h4>
                    <p class="price-tag">{row['Баға']:,} ₸</p>
                    <p class="m2-price">{int(row['Баға'] / area):,} ₸/м²</p>
                </div>
                """, unsafe_allow_html=True)

# 5. Аналитика
st.divider()
col_l, col_r = st.columns([1.5, 1])

with col_l:
    st.subheader("📊 Бағаларды салыстыру")
    fig = px.bar(df_dist, x='Аудан', y='Баға', color='Баға', color_continuous_scale='Blues')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("💡 Таңдау сипаттамасы")
    if is_house:
        st.success(
            f"Сіз **{land_size * 100:.0f} сотка** жері бар жеке үйді таңдадыңыз. Жер үйлердің бағасы учаске көлеміне тікелей байланысты.")
    else:
        status = "Алтын қабат" if floor_val in [2, 3, 4] else "Шеткі қабат"
        st.info(f"Сіз **{floor_val}-қабаттағы** пәтерді таңдадыңыз. Қызылордада бұл **{status}** болып есептеледі.")

    avg = df_dist['Баға'].mean()
    st.write(f"**Қала бойынша орташа баға:** {int(avg):,} ₸")

# st.balloons() - Алып тасталды