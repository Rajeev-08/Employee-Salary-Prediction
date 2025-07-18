import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("salary_prediction_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")


st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("💼 Employee Salary Classification")
st.markdown("""
This app predicts whether an employee's salary is greater than **$50K** or not, based on their demographic and employment data.
Please provide the input on the left sidebar.
""")

st.sidebar.header("👤 Employee Input Features")


def user_input_features():
    st.sidebar.markdown("---")
    age = st.sidebar.slider("Age", 17, 75, 30)
    workclass = st.sidebar.selectbox("Work Class", ('Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov'))
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 12285, 1490400, 189778)
    educational_num = st.sidebar.slider("Education Level (Numeric)", 5, 16, 10)
    
    st.sidebar.markdown("---")
    marital_status = st.sidebar.selectbox("Marital Status", ('Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'))
    occupation = st.sidebar.selectbox("Occupation", ('Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'))
    relationship = st.sidebar.selectbox("Relationship", ('Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'))
    
    st.sidebar.markdown("---")
    race = st.sidebar.selectbox("Race", ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'))
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", ('United-States', 'Other')) # Simplified for UI, model handles more

    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()


st.subheader("📊 Your Input Data")
st.dataframe(input_df)


if st.button("Predict Salary Class", type="primary"):
    
    prediction_encoded = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)

   
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Prediction Result")
        if prediction_label == '>50K':
            st.success(f"The predicted salary is **{prediction_label}**")
        else:
            st.warning(f"The predicted salary is **{prediction_label}**")

    with col2:
        st.subheader("Confidence Probability")
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=label_encoder.classes_,
            index=['Probability']
        )
        st.dataframe(proba_df)

st.markdown("---")
