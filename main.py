import streamlit as st
import joblib
import numpy as np

# Load model & scaler

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details in sidebar and click Predict.")

# Sidebar Inputs

st.sidebar.header("Patient Information")
st.sidebar.info("Fill all details for accurate prediction")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=40, help="Age in years")
cholesterol = st.sidebar.number_input("Cholesterol", min_value=1, max_value=500, value=200, help="Normal: 150–240")
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=1, max_value=300, value=120, help="Normal: ~120")
heart_rate = st.sidebar.number_input("Heart Rate", min_value=1, max_value=200, value=80)
exercise_hours = st.sidebar.number_input("Exercise Hours / Week", min_value=0, max_value=20, value=0)
stress_level = st.sidebar.number_input("Stress Level", max_value=10, value=0)
blood_sugar = st.sidebar.number_input("Blood Sugar", min_value=1, max_value=300, value=100,help="Normal: <100 mg/dL")

gender = st.sidebar.selectbox("Gender", ["Select","Male","Female"])
smoking = st.sidebar.selectbox("Smoking Status", ["Select","Current","Former","Never"])
alcohol = st.sidebar.selectbox("Alcohol Intake", ["Select","Yes","No"])
family_history = st.sidebar.selectbox("Family History", ["Select","Yes","No"])
diabetes = st.sidebar.selectbox("Diabetes", ["Select","Yes","No"])
obesity = st.sidebar.selectbox("Obesity", ["Select","Yes","No"])

chest_pain = st.sidebar.selectbox(
"Chest Pain Type",
["Select","Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"]
)

# Prediction

if st.button("Predict"):
    
    if(gender == "Select" or smoking == "Select" or alcohol == "Select" or family_history == "Select" or 
       diabetes == "Select" or obesity == "Select" or chest_pain == "Select"):
        st.warning("Please fill all fields before predicting.")
        st.stop()
    # Encoding
    else:

        gender_Male = 1 if gender == "Male" else 0
        smoking_Former = 1 if smoking == "Former" else 0
        smoking_Never = 1 if smoking == "Never" else 0
        alcohol_intake_Moderate = 1 if alcohol == "Moderate" else 0
        family_history_Yes = 1 if family_history == "Yes" else 0
        diabetes_Yes = 1 if diabetes == "Yes" else 0
        obesity_Yes = 1 if obesity == "Yes" else 0

        cp_atypical = 1 if chest_pain == "Atypical Angina" else 0
        cp_non = 1 if chest_pain == "Non-Anginal Pain" else 0
        cp_typical = 1 if chest_pain == "Typical Angina" else 0

        # Input array
        input_data = np.array([[
            age, cholesterol, blood_pressure, heart_rate,
            exercise_hours, stress_level, blood_sugar,
            gender_Male, smoking_Former, smoking_Never,
            alcohol_intake_Moderate, family_history_Yes,
            diabetes_Yes, obesity_Yes,
            cp_atypical, cp_non, cp_typical
        ]])

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediction
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.metric("Risk Probability", f"{prob*100:.2f}%")

        st.progress(int(prob * 100))

