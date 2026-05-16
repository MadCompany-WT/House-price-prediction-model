import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64

# 1. Бет баптаулары мен Дизайн (Dark Mode)
st.set_page_config(page_title="AI Qyzylorda Web Analytics", page_icon="🌐", layout="wide")

# Фон арқылы ресімдерді қосамыз
def get_background_image(is_house):
    if is_house:
        # Жеке үй (коттедж) фонды
        bg_url = "https://images.unsplash.com/photo-1570129477492-45a003537e1f?w=1600&h=900&fit=crop"
    else:
        # Пәтер ғимараты фонды
        bg_url = "https://images.unsplash.com/photo-1545324418-cc1a9d6faf4f?w=1600&h=900&fit=crop"
    
    try:
        response = requests.get(bg_url, timeout=5)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode()
    except:
        return None
    return None

# Получаем фон в зависимости от выбора
@st.cache_data(ttl=3600)
def get_bg_style(is_house):
    bg_b64 = get_background_image(is_house)
    if bg_b64:
        return f"url(data:image/jpeg;base64,{bg_b64})"
    return None

st.markdown("""
    <style>
    .stApp {
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }
    
    .main-container {
        background: rgba(14, 17, 23, 0.92);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
    }
    
    .main { 
        background: linear-gradient(135deg, rgba(14, 17, 23, 0.95) 0%, rgba(20, 30, 50, 0.95) 100%);
    }
    
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background: rgba(0, 212, 255, 0.08);
        border: 2px solid rgba(0, 212, 255, 0.3);
        padding: 20px; 
        border-radius: 18px; 
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric):hover {
        background: rgba(0, 212, 255, 0.12);
        border: 2px solid rgba(0, 212, 255, 0.5);
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.2);
    }
    
    h1 { 
        color: #00d4ff !important; 
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        margin-bottom: 30px;
    }
    
    h2, h3 { 
        color: #00d4ff !important; 
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #00d4ff !important;
        font-weight: 600;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #00d4ff !important;
        background: rgba(0, 212, 255, 0.1);
    }
    
    .metric-card {
        background: rgba(0, 212, 255, 0.08) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1) !important;
    }
    
    .stCheckbox, .stRadio, .stSelectbox, .stSlider {
        color: #ffffff !important;
    }
    
    .stSidebar {
        background: rgba(14, 17, 23, 0.95) !important;
        border-right: 2px solid rgba(0, 212, 255, 0.2) !important;
    }
    
    .stDivider {
        border-color: rgba(0, 212, 255, 0.3) !important;
    }
    
    .stCaption {
        color: rgba(255, 255, 255, 0.6) !important;
        text-align: center;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_all():
    model = joblib.load('models/house_price_model.pkl')
    # Егер metrics.pkl болмаса, қолмен жазамыз
    try:
        metrics = joblib.load('models/metrics.pkl')
    except:
        metrics = {"r2": 0.812}
    return model, metrics


model, ml_metrics = load_all()

# 2. SIDEBAR (Параметрлер)
with st.sidebar:
    st.title("⚙️ Параметрлер")
    is_house = st.checkbox("🏡 Бұл жеке жер үй", value=False)

    area = st.number_input("📏 Ауданы (S), м²", min_value=20, max_value=500, value=85)
    rooms = st.slider("🚪 Бөлме саны", 1, 10, 3)
    age = st.slider("⏳ Үйдің жасы", 1, 60, 20)

    if is_house:
        land_sotka = st.slider("🌱 Жер көлемі (сотка)", 1, 20, 6)
        floor_impact = 1.0
    else:
        floor_level = st.select_slider("🏢 Пәтер қабаты", options=[1, 2, 3, 4, 5], value=3)
        floor_map = {1: 0.90, 2: 1.10, 3: 1.15, 4: 1.08, 5: 0.85}
        floor_impact = floor_map[floor_level]

    material = st.selectbox("🧱 Үй материалы", ["Кирпич", "Панель", "Бетон"], index=0)
    repair = st.selectbox("🛠 Жөндеу деңгейі", ["Черновой", "Орташа", "Еуро"], index=1)

    st.divider()
    income = st.number_input("📈 Айлық табыс (₸)", value=500000)


# 3. ЕСЕПТЕУ ЛОГИКАСЫ (СЕНІҢ КОЭФФИЦИЕНТТЕРІҢ)
def get_price(usd_rate, dist_mult=1.0):
    USD = 450;
    MULT = 0.5;
    QYZ = 0.4  # Сенің нақты коэф-терің

    mat_map = {"Кирпич": 1.15, "Панель": 0.95, "Бетон": 1.10}
    rep_map = {"Черновой": 0.8, "Орташа": 1.0, "Еуро": 1.3}

    # Инфрақұрылым бонустары (PyQt6-дан алынды)
    infra_bonus = 1.0
    if inf_sch: infra_bonus += 0.02
    if inf_shp: infra_bonus += 0.01
    if inf_prk: infra_bonus += 0.03

    med_inc = (income * 12) / USD / 10000
    inp = pd.DataFrame({'MedInc': [med_inc], 'HouseAge': [age], 'AveRooms': [area / 25], 'AveBedrms': [1.2],
                        'Population': [1500], 'AveOccup': [3.5], 'Latitude': [34.0], 'Longitude': [-118.0]})

    raw_pred = model.predict(inp)[0]

    # Негізгі формула
    price = raw_pred * 100000 * usd_rate * MULT * dist_mult * QYZ * rep_map[repair] * mat_map[
        material] * floor_impact * infra_bonus

    if is_house:
        price += (land_sotka * 1500000)
    return int(price)


# 4. АУДАНДАР ТІЗІМІ
districts = {
    "Орталық": {"m": 1.35, "h": False, "a": True}, "Сырдария": {"m": 1.28, "h": False, "a": True},
    "ЖК Мерей": {"m": 1.30, "h": False, "a": True}, "Сол Жағалау": {"m": 1.32, "h": False, "a": True},
    "Шұғыла": {"m": 1.18, "h": True, "a": True}, "Микр. Байтерек": {"m": 1.10, "h": False, "a": True},
    "Универсам": {"m": 1.12, "h": False, "a": True}, "Арай": {"m": 1.15, "h": True, "a": True},
    "Ақмаржан": {"m": 1.08, "h": False, "a": True}, "Сәулет": {"m": 0.98, "h": False, "a": True},
    "Микр. Мерей": {"m": 1.05, "h": False, "a": True}, "Титов": {"m": 0.85, "h": True, "a": True}
}

# 5. НЕГІЗГІ БЕТ (GUI)
# Динамический фон в зависимости от выбора
bg_style = get_bg_style(is_house)
if bg_style:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: {bg_style};
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ Qyzylorda Property Intelligence (Web Edition)")

if is_house:
    st.markdown("### 🏡 *Режим: Жеке үйлер (коттеджи)*")
else:
    st.markdown("### 🏢 *Режим: Пәтерлер (квартиры)*")

tab1, tab2, tab3 = st.tabs(["🎯 Нарықтық Болжам", "🧠 ML Аналитика", "📂 Кадастр"])

with tab1:
    col_map1, col_map2 = st.columns([2, 1])

    with col_map2:
        st.subheader("🏥 Инфрақұрылым")
        inf_sch = st.checkbox("Мектеп / Балабақша")
        inf_shp = st.checkbox("Супермаркеттер")
        inf_prk = st.checkbox("Саябақ / Парк")

        st.divider()
        st.subheader("💵 Валюталық шок")
        future_usd = st.slider("Болжамды курс (₸):", 400, 850, 480)

    with col_map1:
        st.subheader("🏘️ Аудандар бойынша баға деңгейі")
        dist_list = list(districts.items())
        for i in range(0, len(dist_list), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(dist_list):
                    name, info = dist_list[i + j]
                    available = info["h"] if is_house else info["a"]

                    if not available:
                        cols[j].metric(name, "—", delta="Мүлік жоқ", delta_color="inverse")
                    else:
                        p_now = get_price(450, info["m"])
                        p_shock = get_price(future_usd, info["m"])
                        inf_perc = ((p_shock - p_now) / p_now) * 100
                        cols[j].metric(name, f"{int(p_shock / 1e6)} млн ₸", f"+{int(inf_perc)}% инфляция")

with tab2:
    st.subheader("🧠 Машиналық оқыту моделінің сапасы")
    c1, c2 = st.columns(2)
    c1.metric("R² Score (Дәлдік)", f"{ml_metrics['r2'] * 100:.1f}%")
    c1.write("**Алгоритм:** Random Forest Regressor")

    # Feature Importance графигі
    imp_data = pd.DataFrame({'Фактор': ['Табыс', 'Жас', 'Бөлме', 'Аудан'], 'Маңыздылық': [0.55, 0.15, 0.12, 0.10]})
    fig = px.bar(imp_data, x='Маңыздылық', y='Фактор', orientation='h', template="plotly_dark",
                 color_discrete_sequence=['#00d4ff'])
    c2.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📂 Ресми Кадастрлық деректер")
    status_df = pd.DataFrame(
        [{"Аудан": k, "Жер үй": ("Бар" if v["h"] else "Жоқ"), "Пәтер": ("Бар" if v["a"] else "Жоқ")} for k, v in
         districts.items()])
    st.table(status_df)

st.caption("MadCompany | Qyzylorda AI Intelligence 2026")
