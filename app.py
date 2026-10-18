"""
Heart Disease Prediction System - Web Application
-------------------------------------------------
Course: Machine Learning Fundamentals (AIC354)
Institution: COMSATS University Islamabad, Lahore Campus
Instructor: Dr. Rao Muhammad Adeel Nawab

A Streamlit web application that allows users and clinicians to interact
with the trained Support Vector Classifier (SVC) model for heart disease
prediction without requiring Python or Jupyter installation.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor | AIC354",
    page_icon="🫀",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load Trained Model and Label Encoders
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the training notebook.
    Fitting on identical fixed category lists reproduces the deterministic
    encodings without needing separate encoder pickle files.
    """
    encoders = {}
    encoders["AgeGroup"] = LabelEncoder().fit(["Young", "MiddleAged", "Senior"])
    encoders["Sex"] = LabelEncoder().fit(["Male", "Female"])
    encoders["ChestPain"] = LabelEncoder().fit(["TypicalAngina", "AtypicalAngina", "NonAnginal", "Asymptomatic"])
    encoders["ExerciseAngina"] = LabelEncoder().fit(["Yes", "No"])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# UI Content & Layout
# ---------------------------------------------------------------------------

st.title("🫀 Heart Disease Prediction System")
st.markdown(
    """
    **COMSATS University Islamabad, Lahore Campus — AIC354 (Assignment 2)**  
    **Student Developer:** Abuzar Khan &nbsp;|&nbsp; **Roll Number:** FA24-BED-105  
    *Supervised under Dr. Rao Muhammad Adeel Nawab*

    This end-to-end Machine Learning web application predicts the likelihood of
    coronary heart disease based on **4 cardinal clinical risk attributes** from
    the UCI Heart Disease (Cleveland) dataset using a trained **Support Vector Classifier (SVC)**.
    """
)

st.divider()

st.subheader("📋 Enter Patient Clinical Details")

col1, col2 = st.columns(2)

with col1:
    age_group = st.selectbox(
        "1. Age Group",
        ["Young", "MiddleAged", "Senior"],
        index=1,
        help="Young: <45 years | Middle-Aged: 45–60 years | Senior: >60 years"
    )
    sex = st.selectbox(
        "2. Biological Sex",
        ["Male", "Female"],
        index=0,
        help="Biological sex of the patient"
    )

with col2:
    chest_pain = st.selectbox(
        "3. Chest Pain Type",
        ["TypicalAngina", "AtypicalAngina", "NonAnginal", "Asymptomatic"],
        index=3,
        help="Typical Angina, Atypical Angina, Non-Anginal, or Asymptomatic"
    )
    exercise_angina = st.selectbox(
        "4. Exercise-Induced Angina",
        ["Yes", "No"],
        index=0,
        help="Whether exercise provoked angina (chest pain)"
    )

st.write("")

if st.button("🔍 Predict Heart Disease Status", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "AgeGroup": [age_group],
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
    })

    # Encode user inputs exactly as in training
    encoded = pd.DataFrame({
        "AgeGroup": encoders["AgeGroup"].transform(user_input["AgeGroup"]),
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()

    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE DETECTED (High Risk)")
        st.write(
            "The model predicts the **presence of heart disease** (>50% coronary artery diameter narrowing). "
            "Follow-up medical consultation and diagnostic cardiological evaluation are recommended."
        )
    else:
        st.success("### ✅ NO HEART DISEASE DETECTED (Low Risk)")
        st.write(
            "The model predicts the **absence of significant heart disease**. "
            "Patient displays a lower clinical risk profile based on the selected attributes."
        )

    with st.expander("🔬 View Encoded Feature Vector Sent to SVC Model"):
        st.dataframe(encoded, hide_index=True)
        st.caption("Categorical labels are converted to numerical indices via deterministic label encoding.")

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This system was developed strictly for educational and academic purposes as part of "
    "the AIC354 Machine Learning Fundamentals curriculum. It should not be used as a substitute for professional "
    "clinical medical diagnosis."
)
