import streamlit as st
import requests
import pandas as pd
import h5py
import os
import numpy as np

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
if st.button("List Running Instances"):
    try:
        response = requests.get(f"{API_BASE_URL}/list")
        st.success(f"Response: {response.json()}")

        data = response.json()
        if data:  # Check if data is not empty
            if isinstance(data, dict):
                df = pd.DataFrame.from_dict(data, orient="index")
                st.dataframe(df)
            else:
                st.dataframe(pd.DataFrame({"Instances": data}))
        else:
            st.write("No running instances.")
    except Exception as e:
        st.error(f"Failed to retrieve stored instances: {e}")


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

# --- HDF5 Signals Viewer ---
st.header("Signals HDF5 Viewer")

def find_signals_hdf5():
    """Find signals.hdf5 file in the current directory and subdirectories"""
    possible_locations = [
        "signals.hdf5",
        "../signals.hdf5",
        "data/signals.hdf5",
        "../data/signals.hdf5",
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            return loc
    
    return None

signals_file_path = find_signals_hdf5()
if signals_file_path:
    st.success(f"Found signals file at: {signals_file_path}")
else:
    signals_file_path = st.text_input("Enter path to signals.hdf5 file:", "signals.hdf5")

if os.path.exists(signals_file_path):
    try:
        with h5py.File(signals_file_path, 'r') as f:
            # Get all groups in the file
            groups = list(f.keys())
            
            # Create a dropdown to select a group
            selected_group = st.selectbox("Select a group:", groups)
            
            if selected_group:
                # Get all datasets in the selected group
                datasets = list(f[selected_group].keys())
                
                # Create a dropdown to select a dataset
                selected_dataset = st.selectbox("Select a table:", datasets)
                
                if selected_dataset:
                    # Read the dataset
                    data = f[selected_group][selected_dataset][:]
                    
                    # Convert to DataFrame if possible
                    if isinstance(data, np.ndarray):
                        if data.dtype.names:  # Structured array
                            df = pd.DataFrame(data)
                        else:  # Regular array
                            df = pd.DataFrame(data)
                            
                        # Display the data
                        st.subheader(f"Data from {selected_group}/{selected_dataset}")
                        st.dataframe(df)
                        
                        # Display basic statistics
                        st.subheader("Basic Statistics")
                        st.write(df.describe())
                    else:
                        st.write("Data format not supported for display")
    except Exception as e:
        st.error(f"Error reading HDF5 file: {e}")
else:
    st.warning(f"Signals file not found at: {signals_file_path}")
