# modules/knn_algo.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def run(df):
    # --- INIT SESSION STATE ---
    
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = {}

    st.subheader("🟦 k-Nearest Neighbors (k-NN) Analyzer")

    # --- 1. Data Selection ---
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox("Select Target Variable (y)", df.columns)
    
    # Filter out the target column from feature options
    feature_cols = [c for c in df.columns if c != target_col]
    
    with col2:
        selected_features = st.multiselect("Select Features (X)", feature_cols, default=feature_cols)

    # Settings for k
    max_k = st.slider("Maximum k to test", min_value=5, max_value=50, value=10)

    # --- 2. Button to Trigger Training ---
    if st.button("🚀 Train & Evaluate k-NN"):
        if not selected_features:
            st.error("Please select at least one feature.")
            return

        # Prepare Data
        X = df[selected_features]
        y = df[target_col]

        # Handle non-numeric data strictly for the user's snippet logic
        X = pd.get_dummies(X, drop_first=True) # Simple encoding if needed

        # --- CODE FROM YOUR SNIPPET STARTS HERE (Adapted) ---
        
        # Partitionner le jeu de données (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.20, 
            train_size=0.80, 
            random_state=42
        )

        st.write(f"**Data Split:** Training: `{X_train.shape[0]}` samples | Test: `{X_test.shape[0]}` samples")

        # Standardisation (Crucial for k-NN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Listes pour stocker les résultats
        k_values = range(1, max_k + 1)
        performance_metrics = []

        # Progress bar for better UI
        progress_bar = st.progress(0)

        for i, k in enumerate(k_values):
            # Initialiser et entraîner
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            
            # Prédire
            y_pred = knn.predict(X_test_scaled)
            
            # Évaluation
            try:
                cm = confusion_matrix(y_test, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0 
            except:
                tn, fp, fn, tp = 0, 0, 0, 0

            # Mesures
            average_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            
            accuracy = accuracy_score(y_test, y_pred) # Added Accuracy
            precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
            f_measure = f1_score(y_test, y_pred, average=average_method, zero_division=0)

            performance_metrics.append({
                'k': k,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f_measure,
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
            })
            
            # Update progress
            progress_bar.progress((i + 1) / len(k_values))

        # Create DataFrame from results
        results_df = pd.DataFrame(performance_metrics)
        results_df.set_index('k', inplace=True)

        # --- 3. Visualization & Results ---
        
        st.subheader("📊 Performance Metrics Table")
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))

        # Matplotlib Plot
        st.subheader("📈 Precision Curve")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(results_df.index, results_df['Precision'], marker='o', linestyle='-', color='b')
        ax.set_title('Precision vs k Value')
        ax.set_xlabel('k (Number of Neighbors)')
        ax.set_ylabel('Precision')
        ax.set_xticks(results_df.index)
        ax.grid(True)
        st.pyplot(fig)

        # Best k Analysis
        best_k_row = results_df.loc[results_df['Precision'].idxmax()]
        best_k_val = results_df['Precision'].idxmax()
        
        st.success(f"🏆 Best k based on Precision: **k = {best_k_val}**")

        # --- RENDER ACCURACY, F1, MATRIX & SAVE ---
        st.divider()
        st.subheader(f"Results for Best Model (k={best_k_val})")
        
        # 1. Re-generate Confusion Matrix for the best k (to get the object for plotting)
        best_knn = KNeighborsClassifier(n_neighbors=best_k_val)
        best_knn.fit(X_train_scaled, y_train)
        y_pred_best = best_knn.predict(X_test_scaled)
        best_cm = confusion_matrix(y_test, y_pred_best)

        # 2. Render Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{best_k_row['Accuracy']:.4f}")
        m2.metric("F1 Score", f"{best_k_row['F1-Score']:.4f}")
        m3.metric("Precision", f"{best_k_row['Precision']:.4f}")

        # 3. Render Matrix
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

        # 4. Save to Session State
        st.session_state['model_history']['KNN'] = {
            'Algorithm': f'KNN (k={best_k_val})',
            'Accuracy': best_k_row['Accuracy'],
            'F1-Score': best_k_row['F1-Score'],
            'Confusion Matrix': best_cm,
            'Model': best_knn,  
            'Scaler': scaler,   
            'Feature_Names': X.columns.tolist()
            
        }
        st.toast("✅ KNN Results Saved for Comparison!")