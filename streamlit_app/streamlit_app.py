'''A tiny Streamlit application to serve as a front-end'''
import streamlit as st
import requests # type: ignore[import-untyped]

# Streamlit page configuration
st.set_page_config(page_title="AI-Doctors", page_icon=":hospital:", layout="wide")

st.title('AI-Doctors')

# Sidebar for input
st.sidebar.header('Input Parameters')
date = st.sidebar.date_input("Select a date")

# Combine date and time into a datetime object
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


# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader('Number of people in the ED:')
    if st.button('Get Data'):
        predictions = get_predictions(date)
        if predictions:
            st.success(f"{predictions['number_in_ed']}")
        else:
            st.error("API connection fail")

with col2:
    st.subheader('Estimate of people that will be admitted:')
    if st.button('Get Prediction'):
        predictions = get_predictions(date)
        if predictions:
            st.success(f"{predictions['predicted_number_admissions']}")
        else:
            st.error("API connection fail")
