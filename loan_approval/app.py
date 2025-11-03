# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Loan Approval AI", page_icon="Money Bag", layout="centered")

st.markdown("""
<style>
    .main {background:#f8f9fa;}
    .stButton>button {background:#007bff; color:white; font-weight:bold;}
    .approved {padding:1rem; background:#d4edda; border:2px solid #28a745;
               border-radius:10px; text-align:center; font-size:1.5rem; color:#155724;}
    .rejected {padding:1rem; background:#f8d7da; border:2px solid #dc3545;
               border-radius:10px; text-align:center; font-size:1.5rem; color:#721c24;}
</style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    path = "loan_approval_best.pkl"
    if not os.path.exists(path):
        st.error(f"Model file **{path}** not found.")
        st.stop()
    return joblib.load(path)

model = load_model()
st.success("Model loaded")

# ------------------- UI -------------------
st.image("https://img.icons8.com/fluency/100/000000/money-bag.png", width=80)
st.title("Loan Approval Predictor")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        deps = st.selectbox("Dependents", [0,1,2,3,4,5])
        edu  = st.selectbox("Education", ["Graduate","Not Graduate"])
        emp  = st.selectbox("Self-Employed?", ["Yes","No"])
        inc  = st.number_input("Annual Income (₹)", 100000, 10000000, 5000000, 100000)
        loan = st.number_input("Loan Amount (₹)", 100000, 40000000, 15000000, 100000)

    with col2:
        term = st.slider("Loan Term (years)", 2, 20, 10, 2)
        cibil= st.slider("CIBIL Score", 300, 900, 750, 10)
        res  = st.number_input("Residential Assets (₹)", 0, 50000000, 8000000, 100000)
        com  = st.number_input("Commercial Assets (₹)", 0, 50000000, 3000000, 100000)
        lux  = st.number_input("Luxury Assets (₹)", 0, 50000000, 12000000, 100000)
        bank = st.number_input("Bank Assets (₹)", 0, 30000000, 4000000, 100000)

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ------------------- PREDICTION -------------------
if submitted:
    df = pd.DataFrame([{
        'no_of_dependents': deps,
        'education': edu,
        'self_employed': emp,
        'income_annum': inc,
        'loan_amount': loan,
        'loan_term': term,
        'cibil_score': cibil,
        'residential_assets_value': res,
        'commercial_assets_value': com,
        'luxury_assets_value': lux,
        'bank_asset_value': bank
    }])

    prob = model.predict_proba(df)[0][1]

    # ---- OPTIONAL CIBIL PENALTY (makes demo more realistic) ----
    if cibil < 600:
        prob *= 0.6
    elif cibil < 650:
        prob *= 0.8
    elif cibil < 700:
        prob *= 0.9
    prob = min(prob, 0.95)               # never show 100 %
    # -------------------------------------------------

    pred = 1 if prob >= 0.60 else 0

    st.divider()
    if pred:
        st.markdown('<div class="approved">LOAN APPROVED</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div class="rejected">LOAN REJECTED</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Approval Confidence", f"{prob:.1%}")
    c2.metric("CIBIL Score", cibil)

    st.caption("Powered by Logistic Regression / Decision Tree + SMOTE")