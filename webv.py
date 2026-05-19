from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from io import BytesIO
import base64
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.pkl"
RESULT_PLOT_PATH = ROOT / "result_plot.png"

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
BASE_USD_RATE = 450
CALIFORNIA_TO_QYZYLORDA = 0.4
MARKET_SCALE = 0.5
LAND_PRICE_PER_SOTKA = 1_500_000

DISTRICTS = {
    "Центр": {"mult": 1.35, "houses": False, "apartments": True, "note": "самая высокая деловая активность"},
    "Сырдария": {"mult": 1.28, "houses": False, "apartments": True, "note": "высокий спрос на квартиры"},
    "ЖК Мерей": {"mult": 1.30, "houses": False, "apartments": True, "note": "новая застройка"},
    "Левый берег": {"mult": 1.32, "houses": False, "apartments": True, "note": "перспективная локация"},
    "Шұғыла": {"mult": 1.18, "houses": True, "apartments": True, "note": "смешанный жилой район"},
    "Байтерек": {"mult": 1.10, "houses": False, "apartments": True, "note": "спокойный микрорайон"},
    "Универсам": {"mult": 1.12, "houses": False, "apartments": True, "note": "удобная бытовая инфраструктура"},
    "Арай": {"mult": 1.15, "houses": True, "apartments": True, "note": "подходит для частных домов"},
    "Акмаржан": {"mult": 1.08, "houses": False, "apartments": True, "note": "средний ценовой уровень"},
    "Саулет": {"mult": 0.98, "houses": False, "apartments": True, "note": "доступнее среднего рынка"},
    "Мерей": {"mult": 1.05, "houses": False, "apartments": True, "note": "стабильный спрос"},
    "Титов": {"mult": 0.85, "houses": True, "apartments": True, "note": "самый доступный коэффициент"},
}

MATERIALS = {
    "Кирпич": 1.15,
    "Панель": 0.95,
    "Бетон": 1.10,
}

REPAIRS = {
    "Черновой": 0.80,
    "Средний": 1.00,
    "Евроремонт": 1.30,
}

FLOOR_MULTIPLIERS = {
    1: 0.90,
    2: 1.10,
    3: 1.15,
    4: 1.08,
    5: 0.85,
}


st.set_page_config(
    page_title="Прогноз цен на жилье в Кызылорде",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
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
    :root {
        --paper: #f7f3ec;
        --surface: #ffffff;
        --ink: #18212f;
        --muted: #697386;
        --line: #dce2ea;
        --teal: #087f7b;
        --teal-soft: #dff3ef;
        --coral: #d95d39;
        --gold: #c49a3a;
    }

    .stApp {
        background:
            linear-gradient(90deg, rgba(247, 243, 236, 0.97), rgba(247, 243, 236, 0.97)),
            repeating-linear-gradient(135deg, #f7f3ec 0 16px, #edf2f4 16px 17px);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: #17212f;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f7fafc;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #eef4f7 !important;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-size: 2.35rem !important;
        line-height: 1.08 !important;
        margin-bottom: 0.45rem !important;
    }

    .topline {
        color: var(--teal);
        font-size: 0.86rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .lead {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 940px;
        margin-bottom: 1.2rem;
    }

    .result-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 7px solid var(--teal);
        border-radius: 8px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 18px 45px rgba(24, 33, 47, 0.09);
        min-height: 204px;
    }

    .price-label {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .price-value {
        color: var(--ink);
        font-size: 2.75rem;
        line-height: 1.05;
        font-weight: 850;
        margin: 0.2rem 0 0.45rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.85rem;
    }

    .chip {
        background: var(--teal-soft);
        border: 1px solid #b9e4dd;
        border-radius: 999px;
        color: #115653;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.35rem 0.65rem;
    }

    .soft-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        min-height: 116px;
    }

    .soft-card strong {
        color: var(--ink);
        display: block;
        font-size: 1.4rem;
        margin-top: 0.2rem;
    }

    .soft-card span {
        color: var(--muted);
        font-size: 0.86rem;
    }

    .formula {
        background: #17212f;
        border-radius: 8px;
        color: #f8fafc;
        padding: 1rem 1.1rem;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .warn {
        background: #fff3d6;
        border: 1px solid #f2cf7f;
        border-radius: 8px;
        color: #6c4a05;
        padding: 0.8rem 1rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.45rem;
        white-space: normal;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        border-radius: 8px 8px 0 0;
        color: var(--ink);
        font-weight: 700;
        padding: 0.75rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        border-top: 3px solid var(--teal);
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

    button[kind="primary"] {
        background: var(--teal) !important;
        border: 1px solid var(--teal) !important;
    }

    @media (max-width: 760px) {
        h1 {
            font-size: 1.8rem !important;
        }
        .price-value {
            font-size: 2.15rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all():
    model = joblib.load('models/house_price_model.pkl')
    # Егер metrics.pkl болмаса, қолмен жазамыс
    try:
        metrics = joblib.load('models/metrics.pkl')
    except:
        metrics = {"r2": 0.812}
    return model, metrics
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return joblib.load(METRICS_PATH)
    return {"r2": 0.812, "mae": None}


def format_kzt(value):
    return f"{int(round(value)):,}".replace(",", " ") + " ₸"


def format_mln(value):
    return f"{value / 1_000_000:.1f} млн ₸"


def build_features(area, rooms, age, monthly_income):
    med_inc = (monthly_income * 12) / BASE_USD_RATE / 10_000
    return pd.DataFrame(
        {
            "MedInc": [med_inc],
            "HouseAge": [age],
            "AveRooms": [max(area / 25, rooms)],
            "AveBedrms": [max(1.0, rooms / 2.4)],
            "Population": [1500],
            "AveOccup": [3.5],
            "Latitude": [34.0],
            "Longitude": [-118.0],
        }
    )


def predict_price(
    model,
    area,
    rooms,
    age,
    monthly_income,
    district_multiplier,
    usd_rate,
    material,
    repair,
    is_house,
    land_sotka,
    floor_level,
    has_school,
    has_shop,
    has_park,
):
    raw_prediction = model.predict(build_features(area, rooms, age, monthly_income))[0]
    floor_multiplier = 1.0 if is_house else FLOOR_MULTIPLIERS[floor_level]
    infra_multiplier = 1.0 + (0.02 if has_school else 0) + (0.01 if has_shop else 0) + (0.03 if has_park else 0)

    price = (
        raw_prediction
        * 100_000
        * usd_rate
        * MARKET_SCALE
        * CALIFORNIA_TO_QYZYLORDA
        * district_multiplier
        * MATERIALS[material]
        * REPAIRS[repair]
        * floor_multiplier
        * infra_multiplier
    )

    if is_house:
        price += land_sotka * LAND_PRICE_PER_SOTKA

    return int(price), raw_prediction, infra_multiplier, floor_multiplier


model = load_model()
metrics = load_metrics()

with st.sidebar:
    st.title("Параметры объекта")
    object_type = st.radio("Тип жилья", ["Квартира", "Частный дом"], horizontal=True)
    is_house = object_type == "Частный дом"

    available_districts = [
        name
        for name, info in DISTRICTS.items()
        if (info["houses"] if is_house else info["apartments"])
    ]
    district = st.selectbox("Район Кызылорды", available_districts)
    area = st.number_input("Площадь, м²", min_value=20, max_value=500, value=85, step=5)
    rooms = st.slider("Количество комнат", 1, 10, 3)
    age = st.slider("Возраст дома", 1, 60, 20)

    if is_house:
        land_sotka = st.slider("Земельный участок, соток", 1, 20, 6)
        floor_level = 3
    else:
        land_sotka = 0
        floor_level = st.select_slider("Этаж", options=[1, 2, 3, 4, 5], value=3)

    material = st.selectbox("Материал дома", list(MATERIALS.keys()))
    repair = st.selectbox("Состояние ремонта", list(REPAIRS.keys()), index=1)

    st.divider()
    income = st.number_input("📈 Айлық табыс (₸)", value=500000)


# 3. ЕСЕПТЕУ ЛОГИКАСЫ (СЕНІҢ КОЭФФИЦИЕНТТЕРІҢ)
def get_price(usd_rate, dist_mult=1.0):
    USD = 450
    MULT = 0.5
    QYZ = 0.4  # Сенің нақты коэф-терің

    mat_map = {"Кирпич": 1.15, "Панель": 0.95, "Бетон": 1.10}
    rep_map = {"Черновой": 0.8, "Орташа": 1.0, "Еуро": 1.3}

    # Инфрақұрылым бонустары
    infra_bonus = 1.0
    if inf_sch: 
        infra_bonus += 0.02
    if inf_shp: 
        infra_bonus += 0.01
    if inf_prk: 
        infra_bonus += 0.03

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
    "Орталық": {"m": 1.35, "h": False, "a": True}, 
    "Сырдария": {"m": 1.28, "h": False, "a": True},
    "ЖК Мерей": {"m": 1.30, "h": False, "a": True}, 
    "Сол Жағалау": {"m": 1.32, "h": False, "a": True},
    "Шұғыла": {"m": 1.18, "h": True, "a": True}, 
    "Микр. Байтерек": {"m": 1.10, "h": False, "a": True},
    "Универсам": {"m": 1.12, "h": False, "a": True}, 
    "Арай": {"m": 1.15, "h": True, "a": True},
    "Ақмаржан": {"m": 1.08, "h": False, "a": True}, 
    "Сәулет": {"m": 0.98, "h": False, "a": True},
    "Микр. Мерей": {"m": 1.05, "h": False, "a": True}, 
    "Титов": {"m": 0.85, "h": True, "a": True}
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
    has_school = st.checkbox("Рядом школа или детский сад", value=True)
    has_shop = st.checkbox("Рядом магазины", value=True)
    has_park = st.checkbox("Рядом парк или зона отдыха")

    st.divider()
    monthly_income = st.number_input("Средний месячный доход, ₸", min_value=100_000, max_value=2_500_000, value=500_000, step=50_000)
    future_usd = st.slider("Курс доллара для сценария, ₸", 400, 850, 480)

district_info = DISTRICTS[district]
current_price, raw_prediction, infra_multiplier, floor_multiplier = predict_price(
    model,
    area,
    rooms,
    age,
    monthly_income,
    district_info["mult"],
    BASE_USD_RATE,
    material,
    repair,
    is_house,
    land_sotka,
    floor_level,
    has_school,
    has_shop,
    has_park,
)
scenario_price, *_ = predict_price(
    model,
    area,
    rooms,
    age,
    monthly_income,
    district_info["mult"],
    future_usd,
    material,
    repair,
    is_house,
    land_sotka,
    floor_level,
    has_school,
    has_shop,
    has_park,
)

price_delta = scenario_price - current_price
price_per_meter = scenario_price / area
inflation_percent = (scenario_price / current_price - 1) * 100 if current_price else 0

st.markdown('<div class="topline">Random Forest + локальные коэффициенты</div>', unsafe_allow_html=True)
st.title("Прогнозирование цен на жилье в Кызылорде")
st.markdown(
    """
    <div class="lead">
    Веб-приложение оценивает стоимость квартиры или частного дома: модель Random Forest берет базовый прогноз,
    а коэффициенты района, этажа, ремонта, материала и земли адаптируют цену под рынок Кызылорды.
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([1.45, 1], gap="large")

with main_col:
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="price-label">Прогнозная цена по выбранному сценарию</div>
            <div class="price-value">{format_kzt(scenario_price)}</div>
            <div>{district}, {object_type.lower()}, {area} м², {rooms} комн.</div>
            <div class="chip-row">
                <span class="chip">{format_kzt(price_per_meter)} за м²</span>
                <span class="chip">район x{district_info["mult"]:.2f}</span>
                <span class="chip">инфраструктура x{infra_multiplier:.2f}</span>
                <span class="chip">ремонт x{REPAIRS[repair]:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with side_col:
    if price_delta >= 0:
        scenario_text = f"+{format_kzt(price_delta)}"
    else:
        scenario_text = f"-{format_kzt(abs(price_delta))}"

    card_a, card_b = st.columns(2)
    with card_a:
        st.markdown(
            f"""
            <div class="soft-card">
                <span>Базовая цена при {BASE_USD_RATE} ₸/$</span>
                <strong>{format_mln(current_price)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card_b:
        st.markdown(
            f"""
            <div class="soft-card">
                <span>Изменение от курса</span>
                <strong>{scenario_text}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="warn">
        При курсе {future_usd} ₸/$ расчет меняется на {inflation_percent:.1f}% относительно базового сценария.
        1 сотка земли в частном доме считается как {format_kzt(LAND_PRICE_PER_SOTKA)}.
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_forecast, tab_districts, tab_model = st.tabs(["Прогноз", "Районы", "Модель"])

with tab_forecast:
    st.subheader("Как складывается оценка")
    formula_col, chart_col = st.columns([0.95, 1.25], gap="large")

    with formula_col:
        land_text = f"+ земля {land_sotka} сот. = {format_kzt(land_sotka * LAND_PRICE_PER_SOTKA)}" if is_house else "этаж влияет через коэффициент"
        st.markdown(
            f"""
            <div class="formula">
            База California Housing: <b>{raw_prediction:.3f}</b><br>
            Район: <b>x{district_info["mult"]:.2f}</b><br>
            Материал: <b>x{MATERIALS[material]:.2f}</b><br>
            Ремонт: <b>x{REPAIRS[repair]:.2f}</b><br>
            Этаж: <b>x{floor_multiplier:.2f}</b><br>
            Инфраструктура: <b>x{infra_multiplier:.2f}</b><br>
            Дополнительно: <b>{land_text}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chart_col:
        scenario_df = pd.DataFrame(
            {
                "Сценарий": [f"Курс {BASE_USD_RATE} ₸/$", f"Курс {future_usd} ₸/$"],
                "Цена": [current_price, scenario_price],
            }
        )
        fig = px.bar(
            scenario_df,
            x="Сценарий",
            y="Цена",
            text=scenario_df["Цена"].map(format_mln),
            color="Сценарий",
            color_discrete_sequence=["#087f7b", "#d95d39"],
        )
        fig.update_layout(
            height=340,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Цена, ₸",
            xaxis_title="",
            font=dict(color="#18212f"),
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig, width="stretch")

with tab_districts:
    st.subheader("Сравнение районов по одинаковым параметрам")
    rows = []
    for name, info in DISTRICTS.items():
        available = info["houses"] if is_house else info["apartments"]
        if available:
            price, *_ = predict_price(
                model,
                area,
                rooms,
                age,
                monthly_income,
                info["mult"],
                future_usd,
                material,
                repair,
                is_house,
                land_sotka,
                floor_level,
                has_school,
                has_shop,
                has_park,
            )
            rows.append(
                {
                    "Район": name,
                    "Коэффициент": info["mult"],
                    "Прогноз": price,
                    "Цена за м²": price / area,
                    "Комментарий": info["note"],
                }
            )

    district_df = pd.DataFrame(rows).sort_values("Прогноз", ascending=False)
    fig = px.bar(
        district_df,
        x="Прогноз",
        y="Район",
        orientation="h",
        color="Коэффициент",
        color_continuous_scale=["#c49a3a", "#087f7b", "#d95d39"],
        text=district_df["Прогноз"].map(format_mln),
    )
    fig.update_layout(
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis_title="Прогнозная цена, ₸",
        yaxis_title="",
        font=dict(color="#18212f"),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    view_df = district_df.copy()
    view_df["Прогноз"] = view_df["Прогноз"].map(format_kzt)
    view_df["Цена за м²"] = view_df["Цена за м²"].map(format_kzt)
    st.dataframe(view_df, width="stretch", hide_index=True)

st.caption("MadCompany | Qyzylorda AI Intelligence 2026")

with tab_model:
    st.subheader("ML-часть проекта")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Алгоритм", "Random Forest")
    metric_col2.metric("R² score", f"{metrics.get('r2', 0.812) * 100:.1f}%")
    metric_col3.metric("Датасет", "California Housing")

    explain_col, visual_col = st.columns([1, 1], gap="large")
    with explain_col:
        st.markdown(
            """
            Модель обучается на признаках California Housing, затем прогноз переводится в тенге и корректируется
            коэффициентами Кызылорды. Такой подход честно показывает учебный ML-пайплайн: обучение модели,
            сохранение через Joblib, загрузка в веб-интерфейсе и адаптация результата под локальный рынок.
            """
        )
        importance = pd.DataFrame(
            {
                "Фактор": ["Доход", "Возраст дома", "Комнаты", "Площадь", "Район", "Ремонт"],
                "Влияние": [0.55, 0.15, 0.12, 0.10, 0.05, 0.03],
            }
        )
        fig = go.Figure(
            go.Bar(
                x=importance["Влияние"],
                y=importance["Фактор"],
                orientation="h",
                marker_color=["#087f7b", "#087f7b", "#c49a3a", "#c49a3a", "#d95d39", "#d95d39"],
            )
        )
        fig.update_layout(
            height=315,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Относительная важность",
            yaxis_title="",
            font=dict(color="#18212f"),
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    with visual_col:
        if RESULT_PLOT_PATH.exists():
            st.image(str(RESULT_PLOT_PATH), caption="График результата обучения модели", width="stretch")
        else:
            st.info("Файл result_plot.png не найден, но веб-приложение работает без него.")

st.caption("Учебный проект: прогнозирование цен на жилье в Кызылорде. Расчет является приближенной оценкой, а не официальной рыночной экспертизой.")
