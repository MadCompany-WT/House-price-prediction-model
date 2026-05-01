import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# 1. Бет баптаулары мен ТҮЗЕТІЛГЕН СТИЛЬДЕР (Dark Mode)
st.set_page_config(page_title="Qyzylorda Property & Currency AI", page_icon="💰", layout="wide")

st.markdown("""
    <style>
    /* Негізгі фон мен мәтін түсі */
    .main { background-color: #0e1117; color: #ffffff; }

    /* Sidebar (сол жақ панель) дизайны */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] label { color: #ffffff !important; font-weight: 600; font-size: 16px; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #00d4ff !important; }

    /* Негізгі беттегі тақырыптар */
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Inter', sans-serif; }
    p, span, label { color: #e6edf3 !important; }

    /* Метрикалар (карточкалар) */
    div[data-testid="stMetric"] {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
    }
    div[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 32px !important; }

    /* Input өрістерін реттеу */
    input { color: white !important; }
    .stNumberInput div div input { background-color: #0d1117 !important; border: 1px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load('models/house_price_model.pkl')


model = load_model()

# 2. Sidebar: ПАРАМЕТРЛЕР (Динамикалық логика)
with st.sidebar:
    st.header("🏠 Үй параметрлері")
    is_house = st.checkbox("🏡 Жер үй", value=False)

    area = st.number_input("Үй ауданы (м²)", min_value=20, max_value=500, value=85)
    rooms = st.slider("Бөлме саны", 1, 10, 3)

    if is_house:
        # ЖЕР ҮЙГЕ ТӘН ПАРАМЕТРЛЕР
        land_sotka = st.slider("🌱 Жер көлемі (сотка)", 1, 20, 6)
        house_floors = st.selectbox("🏘️ Үй неше қабатты?", [1, 2, 3], index=0)
        age = st.slider("⏳ Салынған жылы (жасы)", 1, 60, 10)
        floor_impact = 1.0  # Жер үйде қабат әсері жоқ
    else:
        # ПӘТЕРГЕ ТӘН ПАРАМЕТРЛЕР
        floor_level = st.select_slider("🏢 Пәтер нешінші қабатта?", options=[1, 2, 3, 4, 5], value=3)
        age = st.slider("⏳ Үйдің жасы (жыл)", 1, 60, 20)
        # Қабат логикасы: 2,3,4 - алтын қабаттар
        floor_map = {1: 0.9, 2: 1.1, 3: 1.15, 4: 1.1, 5: 0.85}
        floor_impact = floor_map[floor_level]
        land_sotka = 0
        house_floors = 1

    repair_map = {"Черновой": 0.8, "Орташа": 1.0, "Еуро": 1.3}
    repair = st.selectbox("🛠 Жөндеу", list(repair_map.keys()), index=1)

    st.divider()
    st.header("📈 Базалық экономика")
    current_usd = st.number_input("Ағымдағы доллар курсы (₸)", value=480)
    income = st.number_input("Айлық табыс (₸)", value=650000)


# 3. Есептеу функциясы (Доллар мен Пайыздық өсімді ескере отырып)
def get_price(usd_rate, dist_mult=1.0):
    MULTIPLIER = 0.8
    QYZ_INDEX = 0.4

    med_inc = (income * 12) / usd_rate / 10000
    input_df = pd.DataFrame({
        'MedInc': [med_inc], 'HouseAge': [age], 'AveRooms': [area / 25],
        'AveBedrms': [1.2], 'Population': [1500], 'AveOccup': [4],
        'Latitude': [34.0], 'Longitude': [-118.0]
    })

    raw_pred = model.predict(input_df)[0]
    # Негізгі баға есебі
    price = raw_pred * 100000 * usd_rate * MULTIPLIER * dist_mult * QYZ_INDEX * repair_map[repair] * floor_impact

    # Егер жер үй болса, жердің және қабаттың құны
    if is_house:
        price += (land_sotka * 1500000)  # 1 сотка ~ 1.5 млн
        if house_floors > 1: price *= (1 + (house_floors * 0.1))

    return int(price)


districts = {"Орталық": 1.35, "Сырдария": 1.28, "Сол Жағалау": 1.32, "Арай": 1.15, "Титов": 0.85}

# 4. НЕГІЗГІ БЕТ: Валюталық және мүліктік талдау
st.title("💰 Qyzylorda Property & Currency AI")
st.write(
    f"Қазіргі таңдау: **{area} м², {rooms} бөлмелі {'жер үй (' + str(land_sotka) + ' сот.)' if is_house else 'пәтер (' + str(floor_level) + '-қабат)'}**")

st.divider()
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🚀 Валюталық шок симуляторы")
    # Доллардың болашақтағы ықтимал курсы
    future_usd = st.slider("Егер доллар осы бағаға көтерілсе (₸):", 400, 800, 550)

    p_now = get_price(current_usd, districts["Орталық"])
    p_future = get_price(future_usd, districts["Орталық"])

    # ПАЙЫЗДЫҚ ӨСІМ
    perc_increase = ((p_future - p_now) / p_now) * 100

    st.metric("Болжамды жаңа баға", f"{p_future:,} ₸", delta=f"+{int(perc_increase)}% өсім")
    st.info(f"Доллар **{future_usd - current_usd} ₸**-ге өссе, үй бағасы **{int(perc_increase)}%**-ға қымбаттайды.")

with col2:
    st.subheader("📊 Курсқа тәуелділік графигі")
    usd_range = np.arange(400, 801, 20)
    prices_range = [get_price(u, districts["Орталық"]) for u in usd_range]

    fig_line = px.line(x=usd_range, y=prices_range,
                       labels={'x': 'Доллар курсы (₸)', 'y': 'Үй бағасы (₸)'},
                       template="plotly_dark")
    fig_line.add_vline(x=current_usd, line_dash="dash", line_color="red", annotation_text="Ағымдағы")
    st.plotly_chart(fig_line, use_container_width=True)

# 5. Аудандарға әсері
st.divider()
st.subheader(f"🏘️ Аудандардағы баға өзгерісі ({future_usd} ₸ курсымен)")

cols = st.columns(len(districts))
for i, (name, mult) in enumerate(districts.items()):
    with cols[i]:
        new_p = get_price(future_usd, mult)
        old_p = get_price(current_usd, mult)
        st.metric(name, f"{int(new_p / 1e6)} млн ₸", f"+{int((new_p - old_p) / 1e6)} млн")

# 6. Қорытынды
st.divider()
st.subheader("🧠 AI Аналитика")
if perc_increase > 15:
    st.error(
        f"⚠️ **Ескерту:** Валюта бағамының өсуі нарыққа үлкен қысым береді. {area} м² {'үй' if is_house else 'пәтер'} үшін баға {int(perc_increase)}%-ға қымбаттауы мүмкін.")
else:
    st.success("✅ **Тұрақтылық:** Доллардың бұл деңгейдегі өсімі үй бағасына сыни әсер етпейді.")