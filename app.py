import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Беттің дизайны мен баптаулары
st.set_page_config(page_title="Qyzylorda House Prediction", page_icon="🌙", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 15px; border-left: 5px solid #00d4ff; }
    div.stButton > button:first-child {
        background-color: #00d4ff; color: black; border: None; font-weight: bold; width: 100%; height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# Модельді жүктеу
model = joblib.load('models/house_price_model.pkl')

st.title("🌙 AI Qyzylorda Realtor: Сыр өңірі бойынша сараптама")
st.write("Қызылорда қаласының нарығына бейімделген интеллектуалды бағалау жүйесі.")

# 2. Мәліметтерді енгізу
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📍 Орналасуы мен Табыс")

    # Қызылорда аудандары және олардың коэффициенттері
    districts = {
        "Орталық (Қорқыт ата, Әйтеке би көшелері)": [34.5, -118.5, 1.1],
        "Арай, Шұғыла мөлтек аудандары": [34.2, -118.4, 1.05],
        "Сырдария мөлтек ауданы (Жаңа қала)": [34.0, -118.3, 1.15],
        "1-6 мөлтек аудандары": [33.8, -118.2, 0.95],
        "Сельмаш, Титов, КБИ аудандары": [33.5, -118.1, 0.85]
    }

    selected_district = st.selectbox("Ауданды/Мөлтек ауданды таңдаңыз:", list(districts.keys()))
    income = st.number_input("Отбасының айлық табысы (₸):", 100000, 10000000, 650000, step=50000)

    st.write("---")
    st.subheader("🏠 Үй сипаттамасы")
    area = st.slider("Үй ауданы (м²):", 20, 400, 80)
    rooms = st.slider("Бөлме саны:", 1, 10, 3)
    house_age = st.slider("Үйдің жасы (жыл):", 1, 60, 10)

with col2:
    st.subheader("📊 Сараптама нәтижесі")

    if st.button("НӘТИЖЕНІ ЕСЕПТЕУ"):
        # Коэффициенттер
        USD_KZT = 500
        MULTIPLIER = 0.8
        # Қызылорда бағасы Калифорниядан шамамен 2.5 есе арзан (0.4 коэффициенті)
        QYZYLORDA_INDEX = 0.4

        lat, long, dist_mult = districts[selected_district]

        # Модельге дайындық
        med_inc = (income * 12) / USD_KZT / 10000
        input_data = pd.DataFrame({
            'MedInc': [med_inc], 'HouseAge': [house_age], 'AveRooms': [area / 20],
            'AveBedrms': [1.1], 'Population': [1200], 'AveOccup': [3.5],
            'Latitude': [lat], 'Longitude': [long]
        })

        # Бағаны есептеу
        raw_price = model.predict(input_data)[0]
        final_price = raw_price * 100000 * USD_KZT * MULTIPLIER * dist_mult * QYZYLORDA_INDEX

        # Нәтижені шығару
        st.metric("Болжамды баға:", f"{final_price:,.0f} ₸")

        c1, c2 = st.columns(2)
        c1.write(f"**Аудан:** {selected_district}")
        c2.write(f"**1 м² құны:** {final_price / area:,.0f} ₸")

        # Спидометр (Қолжетімділік)
        affordability = (income * 60) / final_price

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(affordability * 100, 100),
            title={'text': "Қолжетімділік (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00d4ff"},
                'steps': [
                    {'range': [0, 30], 'color': "#ff4b4b"},
                    {'range': [30, 70], 'color': "#ffa500"},
                    {'range': [70, 100], 'color': "#00d4ff"}],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)

        # Кеңес
        if affordability < 0.4:
            st.error("⚠️ Қызылорда нарығы үшін бұл баға сіздің табысыңыздан тым жоғары.")
        elif affordability < 0.7:
            st.warning("🔶 Бағасы қолайлы, бірақ жинақталған қаражат қажет болады.")
        else:
            st.success("✅ Керемет! Бұл үй сіздің қаржылық мүмкіндігіңізге толық сай келеді.")

        st.balloons()
    else:
        st.info("Сол жақтағы мәліметтерді толтырып, 'Нәтижені есептеу' батырмасын басыңыз.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/ea/Qyzylorda_city.jpg", caption="Қызылорда қаласы")

st.sidebar.markdown("---")
st.sidebar.write("Автор: MadCompany-WT")
st.sidebar.caption("Жоба 1$ = 500₸ курсы бойынша есептелген.")