import streamlit as st

# ── Page config (MUST be the very first Streamlit command in the whole file) ──
st.set_page_config(page_title="Adult Income Predictor", page_icon="💰", layout="centered")

import numpy as np
import pandas as pd
import pickle
from keras.models import load_model

# ── Load model and preprocessor ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    model = load_model('adult_ann_model.keras')
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model/preprocessor: {e}")
    st.stop()

st.title("💰 Adult Income Prediction")
st.markdown("Predict whether a person earns **>50K** or **≤50K** per year.")
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("Enter Person Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=35)

    workclass = st.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc',
        'Federal-gov', 'Local-gov', 'State-gov',
        'Without-pay', 'Never-worked'
    ])

    educational_num = st.slider("Education Level (numeric)", min_value=1, max_value=16, value=10,
                                 help="1=Preschool → 16=Doctorate")

    marital_status = st.selectbox("Marital Status", [
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ])

    occupation = st.selectbox("Occupation", [
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
        'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
        'Tech-support', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv'
    ])

    relationship = st.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband',
        'Not-in-family', 'Other-relative', 'Unmarried'
    ])

with col2:
    race = st.selectbox("Race", [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other', 'Black'
    ])

    gender = st.selectbox("Gender", ['Male', 'Female'])

    capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=99999, value=0)

    capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=4356, value=0)

    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99, value=40)

    native_country = st.selectbox("Native Country", [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
        'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
        'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
        'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
        'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
        'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
        'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
        'Peru', 'Hong', 'Holand-Netherlands'
    ])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Income", use_container_width=True):

    input_dict = {
        'age':             [age],
        'workclass':       [workclass],
        'educational-num': [educational_num],
        'marital-status':  [marital_status],
        'occupation':      [occupation],
        'relationship':    [relationship],
        'race':            [race],
        'gender':          [gender],
        'capital-gain':    [capital_gain],
        'capital-loss':    [capital_loss],
        'hours-per-week':  [hours_per_week],
        'native-country':  [native_country]
    }

    input_df = pd.DataFrame(input_dict)

    try:
        # Preprocess
        input_transformed = preprocessor.transform(input_df)

        # Predict
        prob  = model.predict(input_transformed)[0][0]
        label = int(prob > 0.5)

        st.subheader("Prediction Result")

        if label == 1:
            st.success(f"✅ This person likely earns  **> $50K per year**")
        else:
            st.warning(f"⚠️ This person likely earns  **≤ $50K per year**")

        st.metric(label="Confidence",
                  value=f"{prob*100:.1f}%" if label == 1 else f"{(1-prob)*100:.1f}%")

        with st.expander("See input summary"):
            st.dataframe(input_df.T.astype(str).rename(columns={0: 'Value'}))

    except Exception as e:
        st.error(f"Prediction failed: {e}")