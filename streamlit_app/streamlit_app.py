import streamlit as st
import requests

# Streamlit page configuration
st.set_page_config(page_title="AI-Doctors", page_icon=":hospital:", layout="wide")

st.title('AI-Doctors')

# Sidebar for input
st.sidebar.header('Input Parameters')
date = st.sidebar.date_input("Select a date")
time = st.sidebar.time_input("Select time")

# Combine date and time into a datetime object
datetime = f"{date}T{time}"

# Endpoint of the Flask server
FLASK_SERVER_URL = "http://localhost:5000/predict"  # Change to your Flask server URL

# Function to send POST request to Flask server
def get_predictions(date):
    response = requests.post(FLASK_SERVER_URL, json={'date': date})
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Button to get data and prediction
if st.sidebar.button('Get ED Data and Prediction'):
    predictions = get_predictions(datetime)
    if predictions:
        st.success(f"Number of people in the ED: {predictions['number_in_ed']}")
        st.success(f"Estimate of people that will be admitted: {predictions['predicted_number_admissions']}")
    else:
        st.error("API connection fail")