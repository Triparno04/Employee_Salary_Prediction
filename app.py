import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from fpdf import FPDF
from io import BytesIO

# Load model and score
model = joblib.load("salary_model.pkl")
model_score = 0.83

# Sample data for demo features
sample_df = pd.DataFrame([
    {"Education": 0, "Experience": 2, "Age": 22, "Job_Title_Director": 0, "Job_Title_Engineer": 1, "Job_Title_Manager": 0, "Location_Suburban": 0, "Location_Urban": 1},
    {"Education": 2, "Experience": 10, "Age": 35, "Job_Title_Director": 0, "Job_Title_Engineer": 0, "Job_Title_Manager": 1, "Location_Suburban": 1, "Location_Urban": 0},
    {"Education": 3, "Experience": 20, "Age": 45, "Job_Title_Director": 1, "Job_Title_Engineer": 0, "Job_Title_Manager": 0, "Location_Suburban": 0, "Location_Urban": 0}
])
feature_columns = sample_df.columns.tolist()

# Page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# --- Header ---
st.title("📊 Employee Salary Prediction")
st.markdown("Enter employee details to predict their salary using a trained ML model.")

# --- Input Section ---
with st.expander("🧾 Fill Employee Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        education_level = st.selectbox("🎓 Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        age = st.slider("🎂 Age", 18, 70, 25)
    with col2:
        experience = st.slider("💼 Years of Experience", 0, 40, 1)
        job_title = st.selectbox("👔 Job Title", ['Director', 'Engineer', 'Manager'])
    location = st.selectbox("📍 Location", ['Rural', 'Suburban', 'Urban'])

# --- Encoding ---
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
education = education_map[education_level]
job_title_director = int(job_title == 'Director')
job_title_engineer = int(job_title == 'Engineer')
job_title_manager = int(job_title == 'Manager')
location_suburban = int(location == 'Suburban')
location_urban = int(location == 'Urban')

input_data = pd.DataFrame([{
    "Education": education,
    "Experience": experience,
    "Age": age,
    "Job_Title_Director": job_title_director,
    "Job_Title_Engineer": job_title_engineer,
    "Job_Title_Manager": job_title_manager,
    "Location_Suburban": location_suburban,
    "Location_Urban": location_urban
}])

# --- Prediction ---
if st.button("🔮 Predict Salary"):
    salary = model.predict(input_data)[0]
    lower = salary * 0.90
    upper = salary * 1.10

    # --- Prediction Results Box ---
    st.subheader("📈 Prediction Results")
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Predicted Salary", f"₹{salary:,.2f}")
        col2.metric("📉 Lower Bound", f"₹{lower:,.0f}")
        col3.metric("📈 Upper Bound", f"₹{upper:,.0f}")
        st.markdown("---")

    st.markdown(f"**✅ Model Accuracy (R²):** {model_score * 100:.2f}%")

    # --- Charts ---
    st.subheader("📊 Salary Visualization")
    with st.container():
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Lower Bound", "Predicted", "Upper Bound"],
                y=[lower, salary, upper],
                marker_color=["#77B6EA", "#3CB371", "#F08080"]
            ))
            fig.update_layout(
                yaxis_title="Salary (₹)",
                xaxis_title="Estimate Type",
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            pie_fig = go.Figure(data=[go.Pie(
                labels=["Lower", "Predicted", "Upper"],
                values=[lower, salary, upper],
                hole=0.3
            )])
            pie_fig.update_traces(marker=dict(colors=["#77B6EA", "#3CB371", "#F08080"]))
            st.plotly_chart(pie_fig, use_container_width=True)

    # --- Input Summary ---
    st.subheader("🧾 Input Summary")
    with st.container():
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎓 Education", education_level)
            st.metric("💼 Experience", f"{experience} years")
            st.metric("🎂 Age", f"{age}")
        with col2:
            st.metric("👔 Job Title", job_title)
            st.metric("📍 Location", location)
        st.markdown("---")

    # --- Feature Importance ---
    st.subheader("🧠 Feature Importance")
    with st.container():
        st.markdown("---")
        try:
            importances = model.coef_
            imp_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": np.abs(importances)
            }).sort_values(by="Importance", ascending=False)
            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation='h',
                marker_color='#4682B4'
            ))
            fig_imp.update_layout(
                height=400,
                margin=dict(l=30, r=20, t=20, b=20),
                xaxis_title="Importance Score"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except:
            st.info("Feature importance not available for this model.")
        st.markdown("---")

    # --- Sample Predictions ---
    st.subheader("📋 Sample Predictions")
    with st.container():
        st.markdown("---")
        sample_preds = model.predict(sample_df)
        sample_display = sample_df.copy()
        sample_display["Predicted Salary (₹)"] = np.round(sample_preds, 2)
        st.dataframe(sample_display.reset_index(drop=True), use_container_width=True)
        st.markdown("---")

    # --- Dataset Insights ---
    st.subheader("📊 Dataset Insights")
    with st.container():
        st.markdown("---")
        st.metric("Sample Avg Salary", f"₹{sample_preds.mean():,.0f}")
        top_title = ["Director", "Engineer", "Manager"][np.argmax([
            sample_df["Job_Title_Director"].sum(),
            sample_df["Job_Title_Engineer"].sum(),
            sample_df["Job_Title_Manager"].sum()
        ])]
        st.metric("Most Common Job Title", top_title)
        st.metric("Avg Experience", f"{sample_df['Experience'].mean():.1f} years")
        st.markdown("---")

    # --- PDF DOWNLOAD ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Employee Salary Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Salary: Rs. {salary:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Lower Bound: Rs. {lower:,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Upper Bound: Rs. {upper:,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Model Accuracy (R²): {model_score * 100:.2f}%", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Input Summary:", ln=True)
    pdf.cell(200, 10, txt=f"Education Level: {education_level}", ln=True)
    pdf.cell(200, 10, txt=f"Experience: {experience} years", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Job Title: {job_title}", ln=True)
    pdf.cell(200, 10, txt=f"Location: {location}", ln=True)

    buffer = BytesIO()
    buffer.write(pdf.output(dest='S').encode('latin1'))
    buffer.seek(0)

    st.download_button(
        label="📄 Download Report as PDF",
        data=buffer,
        file_name="salary_report.pdf",
        mime="application/pdf"
    )
