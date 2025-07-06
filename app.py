import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout='wide')

model = joblib.load("heart_model.pkl")
features = joblib.load("features.pkl")

# Session State
if 'patients' not in st.session_state:
    st.session_state.patients = {}

if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None

if 'form_reset' not in st.session_state:
    st.session_state.form_reset = 0

if 'show_form' not in st.session_state:
    st.session_state.show_form = False

if 'last_submit' not in st.session_state:
    st.session_state.last_submit = None

st.sidebar.title("Patient Management")


search_name = st.sidebar.text_input("🔍Search Patient")
if search_name:
    if search_name in st.session_state.patients:
        st.session_state.current_patient = search_name
    else:
        st.sidebar.warning(f"No patient found with name: {search_name}")

if st.sidebar.button("➕ Add New Patient"):
    st.session_state.show_form = not st.session_state.show_form

# New Patient Form
if st.session_state.show_form:
    with st.sidebar.form("add_patient_form", clear_on_submit=True):
        st.markdown("### Personal Details")
        name = st.text_input("Full Name", key=f"name_{st.session_state.form_reset}")
        pid = st.text_input("Patient ID", key=f"pid_{st.session_state.form_reset}")

        st.markdown("### Clinical Inputs")
        data = {}
        data['age'] = st.slider("Age (for prediction)", 20, 90, 50, key=f"age_{st.session_state.form_reset}")
        data['sex'] = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else "Male", key=f"sex_{st.session_state.form_reset}")
        data['cp'] = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3], key=f"cp_{st.session_state.form_reset}")
        data['exang'] = st.selectbox("Exercise-Induced Angina (exang)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=f"exang_{st.session_state.form_reset}")
        data['oldpeak'] = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1, key=f"oldpeak_{st.session_state.form_reset}")
        data['slope'] = st.selectbox("Slope of the ST segment", [0, 1, 2], key=f"slope_{st.session_state.form_reset}")
        data['ca'] = st.slider("Major vessels colored (ca)", 0, 3, 0, key=f"ca_{st.session_state.form_reset}")
        data['thal'] = st.selectbox("Thalassemia (thal)", [0, 1, 2], key=f"thal_{st.session_state.form_reset}")

        col1, col2 = st.columns(2)
        submit = col1.form_submit_button("Submit")
        generate = col2.form_submit_button("Generate")
        

    # Handle submit / generate
    if (submit or generate) and name:
        st.session_state.patients[name] = {
            "id": pid,
            "clinical": data
        }
        st.session_state.current_patient = name
        st.sidebar.success(f"{'Saved' if submit else 'Predicted for'}: {name}")
        st.session_state.form_reset += 1
        st.session_state.show_form = False



if st.session_state.patients:
    st.sidebar.markdown("### Existing Patients")
    for pname in list(st.session_state.patients.keys())[::-1]:  # latest first
        if st.sidebar.button(pname, key=f"view_{pname}"):
            st.session_state.current_patient = pname


st.title("Heart Disease Clinical Prediction")

st.markdown(
    """
    <span style='color:red'><strong>Note:</strong> This tool is intended for clinical use. Input parameters like thalassemia, ST depression, and vessel count are based on medical test results.</span>
    """,
    unsafe_allow_html=True
)

if st.session_state.current_patient:
    name = st.session_state.current_patient
    patient = st.session_state.patients[name]
    input_df = pd.DataFrame([patient["clinical"]])

    st.subheader(f"Prediction for: **{name}**")

    # Model prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    st.metric(label="Model Confidence", value=f"{proba:.2f}")

    with st.expander("View Input Data"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

    # Model insights
    st.markdown("### Model Insights")
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="Blues_d")
        st.pyplot(fig)
