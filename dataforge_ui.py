import streamlit as st
import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt # Necessary for rendering plots from backend

# Import core backend library functions
from dataforge.data_loader import load_csv 
from dataforge.eda_utils import (
    get_head_tail, # Added from EDA utilities
    get_column_info, 
    get_descriptive_stats, 
    get_data_quality_report, 
    get_value_counts,
    plot_univariate_distribution,
    plot_bivariate_analysis,
    plot_correlation_matrix
)

# --- Configuration & State Management ---

# Initialize global session state variables (using the standard idiom)
if 'df_full' not in st.session_state:
    st.session_state.df_full = None 
if 'X' not in st.session_state:
    st.session_state.X = None       
if 'y' not in st.session_state:
    st.session_state.y = None       
if 'file_name' not in st.session_state:
    st.session_state.file_name = "sample_data.csv" 
if 'pipeline_configured' not in st.session_state:
    st.session_state.pipeline_configured = False # Flag to track if X/y are split

# --- UI Layout ---

st.set_page_config(layout="wide", page_title="DATAFORGE - ML Workflow Engine")
st.title("📊 DATAFORGE: Configurable ML Automation")
st.subheader("Day 6: EDA and Visualization Dashboard")

# --- FILE HANDLING AND DATA LOADING (Sidebar Section) ---
with st.sidebar:
    st.header("1. Data Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=['csv'], 
        help="Upload the dataset for analysis."
    )
    
    if uploaded_file is None:
        st.session_state.file_name = "sample_data.csv"
        st.info("Using local 'sample_data.csv' for development.")
        data_path = pathlib.Path(st.session_state.file_name)
    else:
        temp_dir = pathlib.Path("./temp_upload")
        temp_dir.mkdir(exist_ok=True)
        data_path = temp_dir / uploaded_file.name
        
        data_path.write_bytes(uploaded_file.read())
        
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.df_full = None
            st.session_state.X = None
            st.session_state.y = None
            st.session_state.file_name = uploaded_file.name


# --- Initial Load of Full DataFrame ---
if st.session_state.df_full is None and data_path.exists():
    try:
        # Call load_csv without target to get full DataFrame for EDA
        # DataFrames are easily serializable, so we keep st.cache_data here (or no cache)
        st.session_state.df_full, _ = load_csv(data_path)
        st.session_state.pipeline_configured = False # Reset flag on new file load
    except Exception as e:
        st.error(f"Error loading initial data: {e}")
        st.stop()


# --- Main Application Logic (Display Dashboard) ---
if st.session_state.df_full is not None:
    # Get all columns dynamically for the dropdown menu
    all_columns = st.session_state.df_full.columns.tolist()

    st.markdown("---")
    st.header(f"Data Source: `{st.session_state.file_name}`")

    # The main tabs for navigating the ML workflow
    tab_eda, tab_cleaning, tab_model = st.tabs([ 
        "🔍 EDA & Analysis", 
        "🧼 Cleaning", 
        "🧠 Modeling"
    ])

    # --- TAB 2: EDA & ANALYSIS ---
    with tab_eda:
        st.header("Exploratory Data Analysis (EDA)")
        
        tab_reports, tab_uni, tab_bi = st.tabs(["Data Reports & Quality", "Univariate Plotting", "Bivariate Plotting"])
        
        # --- TAB 2A: DATA REPORTS & QUALITY ---
        with tab_reports:
            st.subheader("1. Structural & Quality Reports")
            
            st.markdown("#### Head/Tail View")
            head_or_tail = st.radio("View:", ('Head', 'Tail'))
            n_rows_view = st.slider("Rows:", min_value=1, max_value=10, value=5)
            
            if head_or_tail == 'Head':
                st.dataframe(get_head_tail(st.session_state.df_full, n_rows_view, True))
            else:
                st.dataframe(get_head_tail(st.session_state.df_full, n_rows_view, False))

            st.markdown("#### Column Data Types & Missingness")
            info_df = get_column_info(st.session_state.df_full)
            st.dataframe(info_df, use_container_width=True)

            st.markdown("#### Descriptive Statistics (Numerical)")
            stats_df = get_descriptive_stats(st.session_state.df_full)
            st.dataframe(stats_df, use_container_width=True)

            st.markdown("#### Value Counts Report")
            value_col = st.selectbox("Select column for Value Counts:", all_columns)
            if value_col:
                st.dataframe(get_value_counts(st.session_state.df_full, value_col))

            st.markdown("#### Data Quality Overview")
            quality_report = get_data_quality_report(st.session_state.df_full)
            st.json(quality_report)
            
        # --- TAB 2B: UNIVARIATE PLOTTING ---
        with tab_uni:
            st.subheader("2. Univariate Distributions (User Control)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                plot_col_uni = st.selectbox("Select column to plot:", all_columns, key='uni_col')
            
            with col2:
                hue_col_uni = st.selectbox(
                    "Hue (optional):", 
                    ["None"] + all_columns, 
                    key='uni_hue'
                )
                hue_col_uni = None if hue_col_uni == "None" else hue_col_uni
            
            def get_univariate_plot(df, col, hue):
                return plot_univariate_distribution(df, col, hue)

            if plot_col_uni:
                fig = get_univariate_plot(st.session_state.df_full, plot_col_uni, hue_col_uni)
                if fig:
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.warning(f"Cannot generate plot for column {plot_col_uni}.")

        # --- TAB 2C: BIVARIATE PLOTTING ---
        with tab_bi:
            st.subheader("3. Bivariate Relationships & Correlation")

            st.markdown("#### Feature Correlation Matrix")
            
            fig_corr = plot_correlation_matrix(st.session_state.df_full)
            if fig_corr:
                st.pyplot(fig_corr, clear_figure=True)
                plt.close(fig_corr) 
            else:
                st.info("Not enough numerical data to generate correlation matrix.")
            
            st.markdown("#### Customizable Bivariate Plot")
            
            plot_type = st.selectbox("Plot Type:", ('scatter', 'boxplot', 'violin'), key='bi_type')
            col_x = st.selectbox("X-Axis Column:", all_columns, key='bi_x')
            col_y = st.selectbox("Y-Axis Column:", all_columns, key='bi_y')
            
            all_columns_plus_none = ['None'] + all_columns
            col_hue = st.selectbox("Hue/Color Grouping:", all_columns_plus_none, index=0, key='bi_hue')
            
            col_hue = None if col_hue == "None" else col_hue

            if st.button("Generate Bivariate Plot"):
                fig_bi = plot_bivariate_analysis(st.session_state.df_full, col_x, col_y, plot_type, col_hue)
                if fig_bi:
                    st.pyplot(fig_bi, clear_figure=True)
                    plt.close(fig_bi)
                else:
                    st.warning("Could not generate plot. Check column selection and plot type compatibility.")


    # --- TAB 3 & 4 (Cleaning & Modeling - Placeholder for next days) ---
    with tab_cleaning:
        st.info("Day 7: Data Cleaning and Imputation options will be added here.")
    with tab_model:
        st.info("Day 8+: Model Training and Comparison will be added here.")