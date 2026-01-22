import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

try:
    from C45 import C45Classifier
except ImportError:
    C45Classifier = None

def run(df):
    # --- INIT STATE ---
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = {}

    st.subheader("🌳 C4.5 Decision Tree Classifier")

    if C45Classifier is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Variable (y)", df.columns, key="c45_target")
    feature_cols = [c for c in df.columns if c != target_col]
    with col2:
        selected_features = st.multiselect("Select Features (X)", feature_cols, default=feature_cols, key="c45_features")

    if st.button("🚀 Train C4.5 Model"):
        X = df[selected_features]
        y = df[target_col]
        X.columns = X.columns.astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, train_size=0.80, random_state=42
        )
        
        with st.spinner("Training C4.5 Tree..."):
            c45 = C45Classifier()
            c45.fit(X_train, y_train)

        y_pred = c45.predict(X_test)

        avg_method = 'binary' if len(y.unique()) == 2 else 'weighted'

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # --- RENDER RESULTS ---
        st.subheader("📊 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.3f}")
        m2.metric("Precision", f"{precision:.3f}")
        m3.metric("Recall", f"{recall:.3f}")
        m4.metric("F1-Score", f"{f1:.3f}")

        # --- SAVE TO STATE ---
        st.session_state['model_history']['C4.5 Tree'] = {
            'Algorithm': 'C4.5 Tree',
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Confusion Matrix': cm,
            'Model': c45,                       # <--- SAVED
            'Scaler': None,
            'Feature_Names': X.columns.tolist() # <--- SAVED
        }
        st.toast("✅ C4.5 Results Saved!")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax)
        st.pyplot(fig)