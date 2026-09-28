import streamlit as st
import requests

API = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="FinSight AI", layout="wide")
st.title("FinSight AI")
st.caption("Adaptive • Explainable • Multi-Asset Financial AI")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Recommendations", "RAG Chat", "Drift"])

with tab1:
    asset = st.selectbox("Asset", ["stock", "crypto", "reit"])
    symbol = st.text_input("Yahoo Finance symbol", "RELIANCE.NS" if asset=="stock" else "BTC-USD" if asset=="crypto" else "EMBASSY.NS")
    if st.button("Forecast"):
        r = requests.post(f"{API}/forecast", json={"asset": asset, "symbol": symbol}, timeout=120)
        st.json(r.json())

with tab2:
    income = st.number_input("Annual income", min_value=0.0, value=100000.0)
    liabilities = st.number_input("Liabilities", min_value=0.0, value=20000.0)
    risk = st.slider("Risk score", 0.0, 1.0, 0.65)
    horizon = st.number_input("Horizon (years)", min_value=1.0, value=8.0)
    age = st.number_input("Age", min_value=18.0, value=30.0)
    if st.button("Recommend"):
        r = requests.post(f"{API}/recommend", json={
            "income": income, "liabilities": liabilities, "risk_score": risk,
            "horizon_years": horizon, "age": age
        }, timeout=30)
        st.json(r.json())

with tab3:
    q = st.text_area("Ask a finance question", "What does diversification mean?")
    if st.button("Ask"):
        r = requests.post(f"{API}/chat", json={"query": q, "k": 5}, timeout=120)
        st.json(r.json())

with tab4:
    st.write("Use the API /api/drift endpoint with reference/current feature windows.")
