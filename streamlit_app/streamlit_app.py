'''A tiny Streamlit application to serve as a front-end'''
import streamlit as st
import requests # type: ignore[import-untyped]

# Streamlit page configuration
st.set_page_config(page_title="AI-Doctors", page_icon=":hospital:", layout="wide")

st.title('AI-Doctors')

# Sidebar for input
st.sidebar.header('Input Parameters')
date = st.sidebar.date_input("Select a date")

# Convert date to a string
date = f"{date}"

# Endpoint of the Flask server
FLASK_SERVER_URL = "http://localhost:5000/predict"  # Change to your Flask server URL

def get_predictions(date1: str) -> requests.Response:
    """Function to send POST request to Flask server

    Args:
        date1 (str): The date as a str (YYYY-MM-DD)
    """
    response = requests.post(FLASK_SERVER_URL, json={'date': date1}, timeout=120)
    if response.status_code == 200:
        return response.json()
    return None

# Button to get data and prediction
if st.sidebar.button('Get ED Data and Prediction'):
    predictions = get_predictions(date)
    if predictions:
        st.success((f"Number of people in the ED:"
                    f"{predictions['number_in_ed']}"))
        st.success((f"Estimate of people that will be admitted:"
                    f"{predictions['predicted_number_admissions']}"))
    else:
        st.error("API connection fail")
