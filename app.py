import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

# Same category lists the notebook used to train its LabelEncoders
PCLASS = ["First", "Second", "Third"]
GENDER = ["Male", "Female"]
SIBLING = ["Zero", "One", "Two", "Three"]
EMBARKED = ["Southampton", "Cherbourg", "Queenstown"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_encoders():
    return {
        "PClass": LabelEncoder().fit(PCLASS),
        "Gender": LabelEncoder().fit(GENDER),
        "Sibling": LabelEncoder().fit(SIBLING),
        "Embarked": LabelEncoder().fit(EMBARKED),
    }


model = load_model()
encoders = build_encoders()

st.title("🚢 Titanic Passenger Survival Prediction")

pclass = st.selectbox("PClass", PCLASS)
gender = st.selectbox("Gender", GENDER)
sibling = st.selectbox("Sibling", SIBLING)
embarked = st.selectbox("Embarked", EMBARKED)

if st.button("Predict"):
    # Feature order must match training: PClass, Gender, Sibling, Embarked
    features = pd.DataFrame({
        "PClass": encoders["PClass"].transform([pclass]),
        "Gender": encoders["Gender"].transform([gender]),
        "Sibling": encoders["Sibling"].transform([sibling]),
        "Embarked": encoders["Embarked"].transform([embarked]),
    })
    if model.predict(features)[0] == 1:
        st.success("SURVIVED")
    else:
        st.error("NOT SURVIVED")
