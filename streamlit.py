import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
import altair as alt

# Absolute paths for model and metrics (adjust if needed)
MODEL_PATH = r'C:\Users\TARIF\tariful.py\diabetes_api\diabetes_model.pkl'
METRICS_PATH = r'C:\Users\TARIF\tariful.py\diabetes_api\metrics.json'
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load metrics
with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)

# Load dataset for stats and graphs
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(DATASET_URL, names=names)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# App Title and Description
st.title("Advanced Diabetes Prediction System")
st.markdown("""
This application predicts the likelihood of diabetes based on the Pima Indians Diabetes Dataset. 
It uses a machine learning model trained on historical data to provide predictions with confidence scores.
Explore dataset statistics, model performance metrics, interactive graphs, and more for a comprehensive understanding.
""")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dataset Overview", "Model Metrics", "Visualizations", "Risk Factors & Education"])

# Prediction Page
if page == "Prediction":
    st.header("Make a Prediction")
    st.markdown("Enter patient details below. The model will predict if the patient is diabetic or not, along with a confidence score.")

    # Improved Input Form with Columns and Tooltips
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, value=3, help="Number of times pregnant")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, value=145, help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=70, help="Diastolic blood pressure")
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, value=20, help="Triceps skin fold thickness")

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, value=85, help="2-Hour serum insulin")
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=33.6, help="Body mass index")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.35, help="Diabetes pedigree function (genetic score)")
        age = st.number_input("Age (years)", min_value=0, value=29, help="Age of the patient")

    # Predict Button
    if st.button("Predict Diabetes Risk"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # Display Results with Styling
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"**{result}** - Confidence: {confidence:.2%}")
            st.markdown("**Recommendation:** Consult a healthcare professional for further tests and advice.")
        else:
            st.success(f"**{result}** - Confidence: {confidence:.2%}")
            st.markdown("**Recommendation:** Maintain a healthy lifestyle to prevent risk.")

        # Radar Chart for Input vs. Dataset Average
        st.subheader("Your Input vs. Dataset Averages")
        avg_data = X.mean().values
        features = X.columns
        user_input = input_data[0]

        # Normalize for radar chart
        max_vals = np.maximum(avg_data, user_input)
        norm_avg = avg_data / max_vals
        norm_user = user_input / max_vals

        radar_df = pd.DataFrame({
            'Feature': features,
            'Average': norm_avg,
            'User': norm_user
        }).melt(id_vars='Feature')

        chart = alt.Chart(radar_df).mark_line().encode(
            theta='Feature',
            color='variable',
            radius='value'
        ).properties(width=400, height=400)
        st.altair_chart(chart)

# Dataset Overview Page
elif page == "Dataset Overview":
    st.header("Dataset Overview")
    st.markdown("""
    The Pima Indians Diabetes Dataset includes medical data from 768 female patients of Pima Indian heritage.
    It focuses on predicting diabetes onset within 5 years based on diagnostic measurements.
    """)

    # Display Dataset Sample
    st.subheader("Sample Data")
    st.dataframe(data.head(10))

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())

    # Class Distribution
    st.subheader("Diabetes Outcome Distribution")
    outcome_counts = data['Outcome'].value_counts(normalize=True) * 100
    st.bar_chart(outcome_counts)
    st.markdown(f"**Non-Diabetic:** {outcome_counts[0]:.2f}% | **Diabetic:** {outcome_counts[1]:.2f}%")

# Model Metrics Page
elif page == "Model Metrics":
    st.header("Model Performance Metrics")
    st.markdown("Multiple classification models were trained and evaluated. Below are the metrics for each model on the test set.")

    # Display Metrics in Table
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format("{:.4f}"))

    # Highlight Best Model
    best_model_name = max(metrics, key=lambda k: metrics[k]['f1_score'])
    st.subheader(f"Best Model: {best_model_name}")
    st.markdown(f"F1 Score: {metrics[best_model_name]['f1_score']:.4f}")

    # Confusion Matrix (Recompute for best model, assuming we can refit or approximate)
    st.subheader("Confusion Matrix for Best Model")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = joblib.load(MODEL_PATH)  # Reload if needed
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Visualizations Page
elif page == "Visualizations":
    st.header("Data Visualizations")
    st.markdown("Explore interactive graphs to understand feature distributions and relationships.")

    # Feature Histograms
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", X.columns)
    fig, ax = plt.subplots()
    sns.histplot(data=data, x=feature, hue='Outcome', kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Feature Importance (if model supports)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
        st.bar_chart(feat_imp_df.set_index('Feature'))

# Risk Factors & Education Page
elif page == "Risk Factors & Education":
    st.header("Diabetes Risk Factors & Educational Resources")
    st.markdown("""
    ###
