import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved models and scaler
dt_model = joblib.load('C:\\Users\\Омар\\PycharmProjects\\PMfinal\\.venv\\Trained models\\decision_tree_model.joblib')
lr_model = joblib.load('C:\\Users\\Омар\\PycharmProjects\\PMfinal\\.venv\\Trained models\\logistic_regression_model.joblib')
rf_model = joblib.load('C:\\Users\\Омар\\PycharmProjects\\PMfinal\\.venv\\Trained models\\random_forest_model.joblib')
scaler = joblib.load('C:\\Users\\Омар\\PycharmProjects\\PMfinal\\.venv\\Trained models\\scaler.joblib')
columns = joblib.load('C:\\Users\\Омар\\PycharmProjects\\PMfinal\\.venv\\Trained models\\columns.joblib')  # Load the columns used during training

# Streamlit app title
st.title('Student Academic Success Prediction')

# Input fields for user to input data
st.header("Enter student details for prediction")

class_input = st.selectbox("Class", [1, 2, 3])
gender_input = st.radio("Gender", ['Male', 'Female'])
age_input = st.slider("Age", min_value=18, max_value=30, value=20)
status_input = st.selectbox("Status", ['Studying', 'Not studying', 'Graduated'])
school_type_input = st.selectbox("School Type", ['School', 'Lycee (KTL)', 'Lyceum'])
gpa_input = st.slider("GPA", 0.0, 4.0, 2.0)  # GPA input
attestat_gpa_input = st.slider("Attestat GPA", 0.0, 5.0, 2.5)  # ATTESTAT GPA input
failed_courses_input = st.text_input("Failed Courses", 'Not Failed:)')
country_input = st.text_input("Country", 'Kazakhstan')
nationality_input = st.text_input("Nationality", 'Kazakh')
grant_input = st.selectbox("Grant Category", ['SG', 'payer'])

# Map categorical inputs to numeric values
gender = 0 if gender_input == 'Male' else 1
status = {'Studying': 0, 'Not studying': 1, 'Graduated': 2}[status_input]
school_type = {'School': 0, 'Lycee (KTL)': 1, 'Lyceum': 1}[school_type_input]
failed_courses = 1 if failed_courses_input != 'Not Failed:)' else 0
country = 1 if country_input == 'Kazakhstan' else 0
nationality = 1 if nationality_input == 'Kazakh' else 0
grant = 1 if grant_input == 'SG' else 0

# Create the input data as a dataframe
input_data = pd.DataFrame({
    'CLASS': [class_input],
    'GENDER': [gender],
    'AGE': [age_input],
    'STATUS': [status],
    'SCHOOL_TYPE': [school_type],
    'GPA': [gpa_input],
    'ATTESTAT_GPA': [attestat_gpa_input],  # Include both GPA and ATTESTAT_GPA
    'FAILED_COURSES': [failed_courses],
    'COUNTRY_Kazakhstan': [country],
    'NATIONALITY_Kazakh': [nationality],
    'GRANT_CATEGORY_SG': [grant]
})

# Ensure the input data has the same columns as the training data (One-Hot Encoding columns)
input_data = input_data.reindex(columns=columns, fill_value=0)  # Reindex and fill missing columns with 0

# Scale the input data using the same scaler fitted on the training data
input_data_scaled = scaler.transform(input_data)

# Add a button to trigger prediction
predict_button = st.button('Predict Academic Success')

if predict_button:
    # Predict the academic success using the trained models
    prediction_dt = dt_model.predict(input_data_scaled)
    prediction_lr = lr_model.predict(input_data_scaled)
    prediction_rf = rf_model.predict(input_data_scaled)

    # Predict probabilities using Logistic Regression and Random Forest
    prob_lr = lr_model.predict_proba(input_data_scaled)[0][1]  # Probability of success for Logistic Regression
    prob_rf = rf_model.predict_proba(input_data_scaled)[0][1]  # Probability of success for Random Forest

    # Output the predictions in the desired format
    st.write(
        f"**Decision Tree**: The student is likely to {'succeed' if prediction_dt[0] == 1 else 'fail'} academically!")
    st.write(f"**Probability of Success**: {prob_rf * 100:.2f}%")

    st.write(
        f"**Logistic Regression**: The student is likely to {'succeed' if prediction_lr[0] == 1 else 'fail'} academically!")
    st.write(f"**Probability of Success**: {prob_lr * 100:.2f}%")

    st.write(
        f"**Random Forest**: The student is likely to {'succeed' if prediction_rf[0] == 1 else 'fail'} academically!")
    st.write(f"**Probability of Success**: {prob_rf * 100:.2f}%")