import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def run(df):
    # --- INIT STATE ---
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = {}

    st.subheader("🔮 Naïve Bayes (GaussianNB) Classifier")

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Variable (y)", df.columns, key="nb_target")
    feature_cols = [c for c in df.columns if c != target_col]
    with col2:
        selected_features = st.multiselect("Select Features (X)", feature_cols, default=feature_cols, key="nb_features")

    if st.button("🚀 Train Naïve Bayes"):
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
        
        nb = GaussianNB()
        nb.fit(X_train_scaled, y_train)

        y_pred = nb.predict(X_test_scaled)

        # --- Evaluation ---
        unique_classes = len(y.unique())
        avg_method = 'binary' if unique_classes == 2 else 'weighted'

        accuracy = accuracy_score(y_test, y_pred) # Added explicit accuracy var
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # --- RENDER RESULTS ---
        st.subheader("📊 Performance Metrics")
        col_m0, col_m1, col_m2, col_m3 = st.columns(4) # Added col for accuracy
        col_m0.metric("Accuracy", f"{accuracy:.4f}")
        col_m1.metric("Precision", f"{precision:.4f}")
        col_m2.metric("Recall", f"{recall:.4f}")
        col_m3.metric("F1-Score", f"{f1:.4f}")

        # --- SAVE TO STATE ---
        st.session_state['model_history']['Naive Bayes'] = {
            'Algorithm': 'Naive Bayes',
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Confusion Matrix': cm,
            'Model': nb,                        # <--- SAVED
            'Scaler': scaler,                   # <--- SAVED
            'Feature_Names': X.columns.tolist() # <--- SAVED
        }
        st.toast("✅ Naive Bayes Results Saved!")

        st.subheader("Result Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        st.pyplot(fig)