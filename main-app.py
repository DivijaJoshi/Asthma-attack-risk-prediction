import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from streamlit_echarts import st_echarts
import json

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('data with reduced dataset and balanced classes.csv')
    return data

data = load_data()

# Prepare the data
relevant_features = ['work', 'cough', 'chest_tig', 'medc_usage', 'prev_attack', 'smoke', 'fam_asthma', 'AQI']
X = data[relevant_features]
y = data['asthma_attack']

# Split the data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

model = train_model()

# Streamlit app
st.title('Asthma Attack Risk Prediction App')

st.write("""
This app predicts the likelihood of an asthma attack based on various factors.
Please input your information below: Note: (0-5 represents rating on a scale of 0 to 5)
""")

# Create input fields for user data
work = st.slider('Work environment impact (0-5)', 0.0, 5.0, 0.5)
cough = st.slider('Cough level (0-5)', 0.0, 5.0, 0.5)
chest_tig = st.slider('Chest tightness (0-5)', 0.0, 5.0, 0.5)
medc_usage = st.slider('Medication usage (0-5)', 0.0, 5.0, 0.5)
prev_attack = st.slider('Previous asthma attacks (0-5)', 0.0, 5.0, 0.5)
smoke = st.slider('Smoking status (0-1)', 0.0, 1.0, 0.1)
fam_asthma = st.slider('Family history of asthma (0-1)', 0.0, 1.0, 0.1)
AQI = st.slider('Air Quality Index (0-200)', 0, 200, 20)

# Create a dataframe from user input
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

# Scale the user input
user_data_scaled = scaler.transform(user_data)

# Initialize the session state for prediction history
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

    

# Make prediction
if st.button('Predict'):
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)
    
    # Record the current prediction
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
    
    # Append the prediction result to session state
    st.session_state.predictions.append(result)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.warning('High risk of asthma attack')
    else:
        st.success('Low risk of asthma attack')
    
    st.write(f'Probability of asthma attack: {probability[0][1]:.2%}')

# Feature importance
st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'feature': relevant_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

st.bar_chart(feature_importance.set_index('feature'))

# Add a dashboard to show past predictions
st.subheader('Prediction History')

if len(st.session_state.predictions) > 0:
    # Show the past predictions as a table
    st.table(pd.DataFrame(st.session_state.predictions))
    
    # Option to clear all past predictions
    # Option to clear all past predictions
# Option to clear all past predictions
if st.button('Clear All Past Predictions'):
    st.session_state.predictions = []  # Clear the prediction history

    # Use query params to force rerun
    st.experimental_set_query_params(dummy=str(np.random.randint(10000)))  # Add a dummy query param to trigger rerun

# Refresh the page to clear the table
else:
    st.write("No predictions made yet.")
st.subheader('Personalized Advice')

if smoke > 0.5:
    st.info('Quitting smoking can significantly reduce the risk of asthma attacks.')

if medc_usage > 3:
    st.info('It seems like you are using medication frequently. Consider consulting with your doctor for advice on better management.')

if fam_asthma > 0.5:
    st.info('Since you have a family history of asthma, consider regular check-ups to monitor your lung health.')

# Add button to download prediction history as JSON
if len(st.session_state.predictions) > 0:
    history_json = json.dumps(st.session_state.predictions)
    st.download_button(label="Download Prediction History", data=history_json, file_name='prediction_history.json', mime='application/json')

# File uploader to load past prediction history
uploaded_file = st.file_uploader("Upload Prediction History (JSON)", type="json")

if uploaded_file:
    try:
        # Load the JSON data from the uploaded file
        predictions_data = json.load(uploaded_file)
        
        # Store predictions in session state
        st.session_state.predictions = predictions_data
        
        # Convert predictions to DataFrame for better visualization
        predictions_df = pd.DataFrame(st.session_state.predictions)
        
        # Display success message and the loaded DataFrame
        st.success("Prediction history loaded!")
        st.subheader("Loaded Prediction History")
        st.dataframe(predictions_df)  # Display as a DataFrame
    except Exception as e:
        st.error(f"Error loading prediction history: {e}")


from streamlit_echarts import st_echarts

def render_donut(value, title, max_value=5):
    percentage = (value / max_value) * 100  # Calculate percentage
    options = {
        "tooltip": {
            "formatter": "{a} <br/>{b} : {c} ({d}%)"  # Tooltip format for hover
        },
        "series": [{
            "name": title,
            "type": 'pie',
            "radius": ['40%', '70%'],  # Inner radius for the donut effect
            "avoidLabelOverlap": False,
            "itemStyle": {
                "borderRadius": 10,
                "borderColor": '#fff',
                "borderWidth": 2
            },
            "label": {
                "show": True,  # Enable labels
                "position": 'outside',  # Positioning of labels outside the donut
                "formatter": "{b}: {c} ({d}%)",  # Format of labels
                "fontSize": 16,  # Increased font size for better visibility
                "fontFamily": 'Arial',  # Set a clear font family
                "color": 'white'  # Set a color for the labels for contrast
            },
            "emphasis": {
                "label": {
                    "show": True,
                    "fontSize": '30',
                    "fontWeight": 'bold'
                }
            },
            "data": [
                {"value": value, "name": title},  # Value and name for the symptom
                {"value": max_value - value, "name": 'Remaining'}  # Remaining value
            ],
            "silent": True,
        }],
        "color": ["#ff6384", "#36a2eb"]  # Customize colors as needed
    }
    st_echarts(options)

# Donut charts for cough severity and chest tightness severity
st.subheader("Symptom Severity")
render_donut(cough, "Cough Level")  # Updated title
render_donut(chest_tig, "Chest Tightness")  # Updated title



# Add some information about the model
st.sidebar.header('About the Model')
st.sidebar.write("""
This model uses a Random Forest Classifier trained on historical asthma data.
It considers 8 various factors to predict the likelihood of an asthma attack.
""")

st.sidebar.header('Disclaimer')
st.sidebar.write("""
This app is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")
