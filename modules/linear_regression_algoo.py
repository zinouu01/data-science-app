import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def run(df):
    # --- INIT STATE ---
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = {}

    st.subheader("📈 Linear Regression")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("❌ Your dataset has no numerical columns suitable for Regression.")
        return

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Variable (y) [Numeric Only]", numeric_cols, key="lr_target")
    feature_cols = [c for c in df.columns if c != target_col]
    with col2:
        selected_features = st.multiselect("Select Features (X)", feature_cols, default=feature_cols, key="lr_features")

    if st.button("🚀 Train Linear Regression"):
        if not selected_features:
            st.error("Please select at least one feature.")
            return

        X = df[selected_features]
        y = df[target_col]
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, train_size=0.80, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred = lr.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score", f"{r2:.4f}")
        m2.metric("RMSE", f"{rmse:.4f}")
        m3.metric("MAE", f"{mae:.4f}")
        m4.metric("MSE", f"{mse:.4f}")

        # --- SAVE TO STATE ---
        st.session_state['model_history']['Linear Regression'] = {
            'Algorithm': 'Linear Regression',
            'R2 Score': r2,
            'RMSE': rmse,
            'Confusion Matrix': None, # Not applicable
            'Model': lr,                        # <--- SAVED
            'Scaler': scaler,                   # <--- SAVED
            'Feature_Names': X.columns.tolist() # <--- SAVED
        }
        st.toast("✅ Linear Regression Results Saved!")

        st.subheader("📉 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        st.pyplot(fig)