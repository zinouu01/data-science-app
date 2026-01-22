import streamlit as st
import pandas as pd
import sqlite3
import json
import os

# --- Import Modules ---
# 1. Your existing Data Viewer
from modules import data_viewer, knn_algoo, linear_regression_algoo

# 2. The new Algorithm Modules
from modules import (
    knn_algoo,
    naive_bayes_algo, 
    c45_algo, 
    cart_algo,
    chaid_algo, 
    linear_regression_algoo,
    deep_learning_algo,
    comparison_algo
)

# Page Config
st.set_page_config(layout="wide", page_title="AI Hub: Algorithms", page_icon="🧠")

# --- Sidebar Navigation ---
st.sidebar.title("🧬 Navigation")
app_mode = st.sidebar.selectbox("Choose a Module", ["Data Loader", "Model Training","Compare All Models"])

# --- Step 1: CHARGEMENT DES DONNÉES (CSV, XLSX, SQLITE, JSON) ---
st.sidebar.subheader("📂 Upload Tabular Data")
uploaded_file = st.sidebar.file_uploader("Dataset", type=["csv", "xlsx", "sqlite", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        st.session_state['data'] = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        st.session_state['data'] = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        st.session_state['data'] = pd.DataFrame(json.load(uploaded_file))
    elif uploaded_file.name.endswith(".sqlite"):
        # Temporary save to read via SQLite
        with open("temp.sqlite", "wb") as f:
            f.write(uploaded_file.getbuffer())
        conn = sqlite3.connect("temp.sqlite")
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        # Handle case where database might be empty
        if not tables.empty:
            table_name = st.sidebar.selectbox("Select Table", tables['name'].tolist())
            st.session_state['data'] = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()

# --- Main Page Logic ---

if app_mode == "Data Loader":
    # --- YOUR ORIGINAL CODE RESTORED HERE ---
    if 'data' in st.session_state:
        data_viewer.run() # Calls your existing module
    else:
        st.info("Please upload a dataset in the sidebar to begin.")

elif app_mode == "Model Training":
    st.header("🤖 Algorithm Selection")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state['data']
        
        # --- Algorithm Dropdown ---
        algo_choice = st.selectbox(
            "Select Algorithm", 
            [
                "Naive Bayes", 
                "k-NN", 
                "C4.5", 
                "CART (Decision Tree)",
                "CHAID", 
                "Linear Regression", 
                "Deep Learning"
            ]
        )
        
        st.markdown("---")
        
        # --- Routing to New Modules ---
        
        if algo_choice == "Naive Bayes":
            naive_bayes_algo.run(df)

        elif algo_choice == "k-NN":
            knn_algoo.run(df)
            
        elif algo_choice == "C4.5":
            c45_algo.run(df)

        elif algo_choice == "CART (Decision Tree)":
            cart_algo.run(df)

        elif algo_choice == "CHAID":
            chaid_algo.run(df)
            
        elif algo_choice == "Linear Regression":
            linear_regression_algoo.run(df)
            
        elif algo_choice == "Deep Learning":
            deep_learning_algo.run(df)

elif app_mode == "Compare All Models":
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state['data']
        comparison_algo.run(df)