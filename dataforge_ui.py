import streamlit as st
import pandas as pd
import pathlib
import os

# Import the core logic (our self-built library) from the backend.
# The app is the frontend that relies entirely on the 'dataforge' library.
from dataforge.data_loader import load_csv 

# --- Configuration & State Management ---

# Streamlit's st.session_state is the app's global memory. 
# We initialize all core data variables here to ensure they persist across reruns.
if 'df_full' not in st.session_state:
    st.session_state.df_full = None # Holds the full DataFrame loaded for EDA/preview
if 'X' not in st.session_state:
    st.session_state.X = None       # Holds features once split
if 'y' not in st.session_state:
    st.session_state.y = None       # Holds target once split
if 'file_name' not in st.session_state:
    st.session_state.file_name = "sample_data.csv" # Default for local testing

# --- UI Layout ---

# Sets up the basic look and title of the browser tab
st.set_page_config(layout="wide", page_title="DATAFORGE - ML Workflow Engine")
st.title("📊 DATAFORGE: Configurable ML Automation")
st.subheader("Day 4: Data Ingestion and Setup")

# The sidebar holds the controls (where the user configures the ML process)
with st.sidebar:
    st.header("1. Upload & Setup")
    
    # File Uploader: The user's main interaction point to get data into the app
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=['csv'], 
        help="Upload the dataset for analysis."
    )
    
    # --- File Path Logic (Handling local vs. uploaded file paths) ---
    if uploaded_file is None:
        # Scenario 1: No file uploaded (Developer is testing locally)
        st.session_state.file_name = "sample_data.csv"
        st.info("Using local 'sample_data.csv' for development.")
        # This path points to the 'sample_data.csv' file in the project root
        data_path = pathlib.Path(st.session_state.file_name)
    else:
        # Scenario 2: A file was uploaded by the user
        temp_dir = pathlib.Path("./temp_upload")
        temp_dir.mkdir(exist_ok=True) # Ensure the temporary directory exists
        data_path = temp_dir / uploaded_file.name
        
        # Write the uploaded file content to the temp path for our load_csv function to access
        data_path.write_bytes(uploaded_file.read())
        
        # --- BUG FIX: Stale Data Reset ---
        # If the user uploads a DIFFERENT file, we must reset the stored data in memory
        if st.session_state.file_name != uploaded_file.name:
            # Setting df_full to None forces the "Data Loading Logic" block below to run
            st.session_state.df_full = None
            st.session_state.X = None
            st.session_state.y = None
            st.session_state.file_name = uploaded_file.name # Save the name of the new file


# --- Data Loading Logic (Initial Load of Full DataFrame) ---
# This block runs only once when the app starts or when a new file is uploaded (df_full is None)
if st.session_state.df_full is None and data_path.exists():
    try:
        # Call our robust backend function (load_csv) without a target (y=None)
        # We use the idiomatic 'df_full, _' to discard the 'None' return value gracefully.
        st.session_state.df_full, _ = load_csv(data_path)
    except Exception as e:
        # If loading fails (e.g., malformed CSV), stop the app and show a clear error
        st.error(f"Error loading initial data: {e}")
        st.stop()


# --- Main Application Logic (Target Selection and X/y Split) ---
# This is the core logic that defines the UI elements for the configured workflow
if st.session_state.df_full is not None:
    # Get all columns dynamically for the dropdown menu
    all_columns = st.session_state.df_full.columns.tolist()
    
    with st.sidebar:
        st.header("2. Configure Target")

        # Set a smart default for the target column if 'Target' exists
        default_index = 0
        if 'Target' in all_columns:
            default_index = all_columns.index('Target') + 1

        target_choice = st.selectbox(
            "Select the Target Column (y):",
            options=['None'] + all_columns,
            index=default_index,
            help="This column will be used as the prediction variable."
        )

        # --- The "Action" Button (Configurable Automation Trigger) ---
        if st.button("Load Data & Configure Pipeline", type="primary"):
            if target_choice == 'None':
                st.warning("Please select a target column to proceed to ML split.")
                # Keep the full DataFrame as X for EDA/Visualization
                st.session_state.X = st.session_state.df_full
                st.session_state.y = None
            else:
                # CALL THE CORE BACKEND FUNCTION (The configurable step)
                try:
                    # Call the load_csv function using the user's selected column name
                    X_loaded, y_loaded = load_csv(data_path, target_column=target_choice)
                    
                    # Store the resulting split dataframes in session state
                    st.session_state.X = X_loaded
                    st.session_state.y = y_loaded
                    st.success(f"Pipeline Configured! X shape: {X_loaded.shape}, Y shape: {y_loaded.shape}")
                except Exception as e:
                    st.error(f"Configuration Error: {e}")

# --- Dashboard Display Tabs (User Feedback) ---

if st.session_state.df_full is not None:
    st.markdown("---")
    st.header(f"Data Source: `{st.session_state.file_name}`")
    
    # Organize output clearly using tabs
    tab1, tab2, tab3 = st.tabs(["Raw Data Preview", "Features (X) Shape", "Target (y) Info"])

    with tab1:
        st.markdown("### Raw Data (First 5 Rows)")
        st.dataframe(st.session_state.df_full.head(), use_container_width=True)

    with tab2:
        if st.session_state.X is not None:
            st.markdown(f"### Features (X) Loaded")
            # Show the dimensions as a clear check for the user
            st.code(f"X.shape: {st.session_state.X.shape}")
            st.dataframe(st.session_state.X.head(), use_container_width=True)
        else:
            st.info("Features (X) not yet split. Select a target column and click the button.")

    with tab3:
        if st.session_state.y is not None:
            st.markdown(f"### Target (y) Loaded")
            # Show the dimensions as a clear check for the user
            st.code(f"y.shape: {st.session_state.y.shape}")
            st.write(st.session_state.y.head())
        else:
            st.info("Target (y) not yet selected.")