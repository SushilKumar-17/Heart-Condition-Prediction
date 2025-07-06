import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn

st.set_page_config(page_title="Heart Disease Prediction", layout='wide')

# Load model, features, and scaler
try:
    model = joblib.load("heart_model.pkl")
    features = joblib.load("features.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()

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

def prepare_input_data(patient_data):
    """
    Prepare input data to match the model's expected feature names and order.
    """
    # Create DataFrame with patient data
    input_df = pd.DataFrame([patient_data])
    
    # Reorder columns to match model's expected feature order
    input_df = input_df[features]
    
    return input_df

if st.session_state.current_patient:
    name = st.session_state.current_patient
    patient = st.session_state.patients[name]
    
    st.subheader(f"Prediction for: **{name}**")
    
    # Prepare input data
    input_df = prepare_input_data(patient["clinical"])
    
    if input_df is not None:
        try:
            # Convert DataFrame to numpy array to avoid feature name issues
            input_array = input_df.values
            
            # Scale only the continuous features (age and oldpeak)
            input_scaled = input_df.values.copy()  # Start with all original values
            
            # Apply scaling only to age (index 0) and oldpeak (index 4)
            continuous_features = input_df[['age', 'oldpeak']].values
            scaled_continuous = scaler.transform(continuous_features)
            
            # Replace the continuous features with scaled versions
            input_scaled[:, 0] = scaled_continuous[:, 0]  # age
            input_scaled[:, 4] = scaled_continuous[:, 1]  # oldpeak
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 High Risk of Heart Disease")
                else:
                    st.success("✅ Low Risk of Heart Disease")
            
            with col2:
                st.metric(label="Model Confidence", value=f"{proba:.2%}")
            
            # Risk interpretation
            if proba > 0.7:
                risk_level = "Very High"
                risk_color = "red"
            elif proba > 0.5:
                risk_level = "High"
                risk_color = "orange"
            elif proba > 0.3:
                risk_level = "Moderate"
                risk_color = "yellow"
            else:
                risk_level = "Low"
                risk_color = "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            
            with st.expander("View Input Data"):
                st.dataframe(input_df.T.rename(columns={0: "Value"}))
            
            # Model insights
            st.markdown("### Model Insights")
            
            if hasattr(model, "feature_importances_"):
                # For tree-based models
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="Blues_d")
                ax.set_title("Feature Importance")
                st.pyplot(fig)
                
            elif hasattr(model, "coef_"):
                # For logistic regression
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": np.abs(model.coef_[0])
                }).sort_values(by="Importance", ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="Blues_d")
                ax.set_title("Feature Importance (Logistic Regression Coefficients)")
                st.pyplot(fig)
            
            # Feature contribution for current patient
            if hasattr(model, "coef_"):
                st.markdown("### Feature Contribution for Current Patient")
                
                feature_contrib = pd.DataFrame({
                    "Feature": features,
                    "Value": input_df.iloc[0].values,
                    "Coefficient": model.coef_[0],
                    "Contribution": input_df.iloc[0].values * model.coef_[0]
                }).sort_values(by="Contribution", ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['red' if x < 0 else 'green' for x in feature_contrib['Contribution']]
                sns.barplot(x="Contribution", y="Feature", data=feature_contrib, ax=ax, palette=colors)
                ax.set_title("Feature Contribution to Prediction")
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check that your model files are compatible with the current input format.")
    
else:
    st.info("👈 Please select or add a patient from the sidebar to view predictions.")
    
    # Show some helpful information
    st.markdown("### About This Tool")
    st.markdown("""
    This heart disease prediction tool uses machine learning to assess the risk of heart disease based on clinical parameters.
    
    **Key Features:**
    - Patient management system
    - Clinical parameter input
    - Risk assessment with confidence scores
    - Feature importance analysis
    - Individual feature contribution analysis
    
    **Clinical Parameters:**
    - **Age**: Patient age in years
    - **Sex**: 0 = Female, 1 = Male
    - **Chest Pain Type (cp)**: 0-3 scale
    - **Exercise-Induced Angina (exang)**: 0 = No, 1 = Yes
    - **ST Depression (oldpeak)**: Depression induced by exercise
    - **Slope**: Slope of the peak exercise ST segment
    - **Major Vessels (ca)**: Number of major vessels colored by fluoroscopy
    - **Thalassemia (thal)**: Blood disorder type
    """)