import streamlit as st
import pandas as pd
import joblib

# Load model and expected input features
model = joblib.load("model_lightgbm_v2.pkl")
expected_cols = joblib.load("model_features.pkl")

# Mappings
education_map = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4,
    '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 9,
    'Assoc-acdm': 10, 'Assoc-voc': 11, 'Bachelors': 12, 'Masters': 13,
    'Doctorate': 14, 'Prof-school': 15
}
sex_map = {'Male': 1, 'Female': 0}
income_map = {0: '<=50K', 1: '>50K'}

# Options for categorical fields
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                     'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
marital_options = ['Never-married', 'Married-civ-spouse', 'Divorced',
                   'Separated', 'Married-spouse-absent', 'Widowed']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                      'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                      'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']

# Streamlit UI
st.title("💼 Employee Salary Predictor")

age = st.slider("Age", 18, 80, 30)
education = st.selectbox("Education", list(education_map.keys()))
hours_per_week = st.slider("Hours per week", 1, 99, 40)
sex = st.radio("Sex", list(sex_map.keys()))
workclass = st.selectbox("Workclass", workclass_options)
marital_status = st.selectbox("Marital Status", marital_options)
occupation = st.selectbox("Occupation", occupation_options)

# Prepare input data
input_dict = {
    'age': age,
    'education-num': education_map[education],
    'hours-per-week': hours_per_week,
    'sex': sex_map[sex],
    'workclass': workclass,
    'marital-status': marital_status,
    'occupation': occupation,
}
input_df = pd.DataFrame([input_dict])

# One-hot encode
input_df = pd.get_dummies(input_df)

# Add missing columns all at once (avoids fragmentation warning)
missing_cols = [col for col in expected_cols if col not in input_df.columns]
if missing_cols:
    missing_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)
    input_df = pd.concat([input_df, missing_df], axis=1)

# Reorder to match training
input_df = input_df[expected_cols]

# Predict
if st.button("Predict Income Bracket"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][int(pred)]
    st.success(f"🔍 Predicted Income: {income_map[int(pred)]}")
    st.info(f"Confidence: {proba:.2%}")
