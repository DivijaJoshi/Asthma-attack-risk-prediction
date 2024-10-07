import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from streamlit_echarts import st_echarts
import json

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('data with reduced dataset and balanced classes.csv')
    return data

data = load_data()

# Relevant features and target variable
relevant_features = ['work', 'cough', 'chest_tig', 'medc_usage', 'prev_attack', 'smoke', 'fam_asthma', 'AQI']
X = data[relevant_features]
y = data['asthma_attack']

# Split the data and scale the features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the RandomForestClassifier model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

model = train_model()

# Sidebar section for model information and disclaimer
st.sidebar.header('About the Model')
st.sidebar.write("""
This model uses a Random Forest Classifier trained on historical asthma data.
It considers 8 factors to predict the likelihood of an asthma attack.
""")

st.sidebar.header('Disclaimer')
st.sidebar.write("""
This app is for educational purposes only and should not be used as a substitute for professional medical advice.
Always consult your doctor regarding medical conditions.
""")

# App Title and Description
st.title('Asthma Attack Risk Prediction App')
st.write("""
This app predicts the likelihood of an asthma attack based on various factors.
Please input your information below: (0-5 represents rating on a scale of 0 to 5)
""")

# User input section with sliders
st.header("Input Your Information")
work = st.slider('Work environment impact (0-5)', 0.0, 5.0, 0.5)
cough = st.slider('Cough level (0-5)', 0.0, 5.0, 0.5)
chest_tig = st.slider('Chest tightness (0-5)', 0.0, 5.0, 0.5)
medc_usage = st.slider('Medication usage (0-5)', 0.0, 5.0, 0.5)
prev_attack = st.slider('Previous asthma attacks (0-5)', 0.0, 5.0, 0.5)
smoke = st.slider('Smoking status (0-1)', 0.0, 1.0, 0.1)
fam_asthma = st.slider('Family history of asthma (0-1)', 0.0, 1.0, 0.1)
AQI = st.slider('Air Quality Index (0-200)', 0, 200, 20)

# Creating a DataFrame from user inputs
user_data = pd.DataFrame({
    'work': [work],
    'cough': [cough],
    'chest_tig': [chest_tig],
    'medc_usage': [medc_usage],
    'prev_attack': [prev_attack],
    'smoke': [smoke],
    'fam_asthma': [fam_asthma],
    'AQI': [AQI]
})

# Scaling the user input
user_data_scaled = scaler.transform(user_data)

# Session state to store prediction history
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Prediction and result display
if st.button('Predict'):
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)

    # Store current prediction
    result = {
        'work': work,
        'cough': cough,
        'chest_tig': chest_tig,
        'medc_usage': medc_usage,
        'prev_attack': prev_attack,
        'smoke': smoke,
        'fam_asthma': fam_asthma,
        'AQI': AQI,
        'prediction': 'High risk' if prediction[0] == 1 else 'Low risk',
        'probability': f'{probability[0][1]:.2%}'
    }
    st.session_state.predictions.append(result)

    # Display the prediction result
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.warning('High risk of asthma attack')
    else:
        st.success('Low risk of asthma attack')
    
    st.write(f'Probability of asthma attack: {probability[0][1]:.2%}')

# Feature Importance
st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'feature': relevant_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
st.bar_chart(feature_importance.set_index('feature'))

# Prediction History Section
st.subheader('Prediction History')
if len(st.session_state.predictions) > 0:
    st.table(pd.DataFrame(st.session_state.predictions))
    if st.button('Clear All Past Predictions'):
        st.session_state.predictions = []
        st.experimental_set_query_params(dummy=str(np.random.randint(10000)))
else:
    st.write("No predictions made yet.")

# Personalized Advice Section
st.subheader('Personalized Advice')
if smoke > 0.5:
    st.info('Quitting smoking can significantly reduce the risk of asthma attacks.')
if medc_usage > 3:
    st.info('Frequent medication usage suggests you consult a doctor for better management.')
if fam_asthma > 0.5:
    st.info('A family history of asthma warrants regular check-ups.')

# Download Prediction History
if len(st.session_state.predictions) > 0:
    history_json = json.dumps(st.session_state.predictions)
    st.download_button(label="Download Prediction History", data=history_json, file_name='prediction_history.json', mime='application/json')

# Upload Prediction History
uploaded_file = st.file_uploader("Upload Prediction History (JSON)", type="json")
if uploaded_file:
    try:
        predictions_data = json.load(uploaded_file)
        st.session_state.predictions = predictions_data
        st.success("Prediction history loaded!")
        st.subheader("Loaded Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.predictions))
    except Exception as e:
        st.error(f"Error loading prediction history: {e}")

# Symptom Severity Donut Charts
st.subheader("Symptom Severity")
def render_donut(value, title, max_value=5):
    percentage = (value / max_value) * 100
    options = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c} ({d}%)"},
        "series": [{
            "name": title,
            "type": 'pie',
            "radius": ['40%', '70%'],
            "itemStyle": {"borderRadius": 10, "borderColor": '#fff', "borderWidth": 2},
            "label": {"show": True, "formatter": "{b}: {c} ({d}%)", "fontSize": 16, "color": 'white'},
            "data": [{"value": value, "name": title}, {"value": max_value - value, "name": 'Remaining'}]
        }],
        "color": ["#ff6384", "#36a2eb"]
    }
    st_echarts(options)

render_donut(cough, "Cough Level")
render_donut(chest_tig, "Chest Tightness")
