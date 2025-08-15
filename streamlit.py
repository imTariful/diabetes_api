import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# === API URL ===
API_URL = "https://diabetes-api-20db.onrender.com" 

# === Page Config ===
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# === Title & Intro ===
st.title("🩺 Diabetes Prediction App")
st.markdown("""
Welcome to the **Diabetes Prediction Tool**.  
Fill in the details below to get a **prediction** based on a trained machine learning model.
""")

# === Input form ===
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose", min_value=0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0)
    with col2:
        skin_thickness = st.number_input("Skin Thickness", min_value=0)
        insulin = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
        age = st.number_input("Age", min_value=0)

    submitted = st.form_submit_button("🔍 Predict")

# === Process Prediction ===
if submitted:
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("result", "Unknown")
            confidence = result.get("confidence", 0)

            # === Styled Prediction Output ===
            if prediction.lower() == "positive":
                st.markdown(f"<h2 style='color:red;'>🚨 Prediction: {prediction}</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color:green;'>✅ Prediction: {prediction}</h2>", unsafe_allow_html=True)

            # === Confidence Gauge ===
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Confidence (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightcoral"},
                           {'range': [50, 80], 'color': "lightyellow"},
                           {'range': [80, 100], 'color': "lightgreen"}
                       ]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # === Store history ===
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
                "Prediction": prediction,
                "Confidence": confidence
            })

            # === Show Prediction History Table ===
            st.subheader("📊 Prediction History")
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history)

            # === Plot Confidence by Prediction ===
            fig_bar = px.bar(df_history, x=df_history.index, y="Confidence", color="Prediction",
                             title="Prediction Confidence Over Time",
                             labels={"index": "Prediction #", "Confidence": "Confidence Level"})
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.error("❌ Error in prediction API call.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
