# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 23:12:43 2026

@author: HP
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

FEATURES = ["pclass", "sex", "age", "fare"]

# Train and save model if it does not exist
@st.cache_resource
def load_model():
    if not os.path.exists("model.joblib"):
        df = sns.load_dataset("titanic")
        df = df[["survived", "pclass", "sex", "age", "fare"]].dropna()
        df["sex"] = (df["sex"] == "female").astype(int)

        X = df[FEATURES]
        y = df["survived"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, "model.joblib")

    return joblib.load("model.joblib")

# App interface
st.title("🚢 Titanic Survival Predictor")
st.write("Predict the estimated survival probability using a Machine Learning model.")
st.info("This is an educational project using historical Titanic data.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Female", "Male"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare Paid", 0, 500, 30)

if st.button("Predict Survival"):
    try:
        model = load_model()

        row = pd.DataFrame(
            [[pclass, 1 if sex == "Female" else 0, age, fare]],
            columns=FEATURES
        )

        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0][1]

        st.metric("Estimated Survival Probability", f"{probability:.1%}")

        if prediction == 1:
            st.success("Model prediction: Survived")
        else:
            st.warning("Model prediction: Did not survive")

        st.caption("This prediction is educational, not a guarantee of actual survival.")

    except Exception:
        st.error("Unable to make a prediction. Please check the model and try again.")