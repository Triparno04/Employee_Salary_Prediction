import streamlit as st
import joblib
import numpy as np

# Load your model
model = joblib.load('salary_model.pkl')

st.title("💼 Employee Salary Prediction")
st.write("Enter employee details to predict their salary.")

# Input fields
education_level = st.selectbox("Education Level", [
    "High School", "Bachelor's", "Master's", "PhD"
])

# Convert to number like your model was trained on
education_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}
education = education_map[education_level]

experience = st.slider("Years of Experience", 0, 40, 1)
age = st.slider("Age", 18, 70, 25)
job_title = st.selectbox("Job Title", ['Director', 'Engineer', 'Manager'])
location = st.selectbox("Location", ['Rural', 'Suburban', 'Urban'])

# Convert inputs to model format
job_title_director = 1 if job_title == 'Director' else 0
job_title_engineer = 1 if job_title == 'Engineer' else 0
job_title_manager = 1 if job_title == 'Manager' else 0

location_suburban = 1 if location == 'Suburban' else 0
location_urban = 1 if location == 'Urban' else 0

# Create input array
input_data = np.array([[education, experience, age,
                        job_title_director, job_title_engineer, job_title_manager,
                        location_suburban, location_urban]])

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
