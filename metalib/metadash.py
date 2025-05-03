import streamlit as st
import requests
import pandas as pd

# Set FastAPI server base URL
API_BASE_URL = "http://localhost:8000"  # Change this if hosted elsewhere

st.title("Strategy Instance Controller Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Start New Strategy")
strategy_type = st.sidebar.text_input("Strategy Type")
params = st.sidebar.text_area("Init Args (key=value, one per line)")

if st.sidebar.button("Start Strategy"):
    param_dict = {}
    for line in params.split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            param_dict[key.strip()] = value.strip()

    if strategy_type:
        param_dict["strategy_type"] = strategy_type
        try:
            response = requests.get(f"{API_BASE_URL}/start", params=param_dict)
            st.sidebar.success(f"Response: {response.json()}")
        except Exception as e:
            st.sidebar.error(f"Failed to start: {e}")
    else:
        st.sidebar.warning("Please enter a strategy type.")

# --- Running Instances ---
st.header("Running Instances")
try:
    if st.button("List Running Instances"):
        try:
            response = requests.get(f"{API_BASE_URL}/list")
            st.success(f"Response: {response.json()}")
        except Exception as e:
            st.error(f"Failed to start stored instances: {e}")

    data = response.json()
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient="index")
        st.dataframe(df)
    else:
        st.write("No running instances.")
except Exception as e:
    st.error(f"Could not fetch instances: {e}")

# --- Stop Instance ---
st.subheader("Stop a Strategy")
tag_to_stop = st.text_input("Enter tag (strategy name) to stop")
if st.button("Stop Strategy"):
    try:
        response = requests.get(f"{API_BASE_URL}/stop/{tag_to_stop}")
        st.success(f"Response: {response.json()}")
    except Exception as e:
        st.error(f"Failed to stop: {e}")

# --- Start All Stored Instances ---
if st.button("Start All Stored Instances"):
    try:
        response = requests.get(f"{API_BASE_URL}/start_stored_instances")
        st.success(f"Response: {response.json()}")
    except Exception as e:
        st.error(f"Failed to start stored instances: {e}")
