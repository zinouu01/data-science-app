import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from CHAID import Tree
except ImportError:
    Tree = None

def run(df):
    # --- INIT STATE ---
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = {}

    st.subheader("🌳 CHAID (Chi-squared Automatic Interaction Detector)")

    if Tree is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Variable (y)", df.columns, key="chaid_target")
    feature_cols = [c for c in df.columns if c != target_col]
    with col2:
        selected_features = st.multiselect("Select Features (X)", feature_cols, default=feature_cols, key="chaid_features")

    with st.expander("⚙️ CHAID Hyperparameters"):
        max_depth = st.slider("Max Tree Depth", 2, 10, 3)
        min_parent = st.number_input("Min samples in Parent Node", value=30)
        min_child = st.number_input("Min samples in Child Node", value=15)

    if st.button("🚀 Train CHAID Model"):
        df_selected = df[selected_features + [target_col]].copy()
        for col in df_selected.columns:
            df_selected[col] = df_selected[col].astype(str)

        train_data, test_data = train_test_split(df_selected, test_size=0.2, random_state=42)
        
        input_cols_dict = {col: 'nominal' for col in selected_features}
        
        try:
            tree = Tree.from_pandas_df(
                train_data, 
                input_cols_dict, 
                target_col,
                max_depth=max_depth, 
                min_parent_node_size=min_parent, 
                min_child_node_size=min_child
            )
            
            test_predictions = []
            for _, row in test_data.iterrows():
                node = tree.get_node(row)
                if node.members:
                    pred = max(node.members, key=node.members.get)
                else:
                    pred = "Unknown" 
                test_predictions.append(pred)

            y_true = test_data[target_col].tolist()
            
            accuracy = accuracy_score(y_true, test_predictions)
            precision = precision_score(y_true, test_predictions, average='weighted', zero_division=0)
            recall = recall_score(y_true, test_predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_true, test_predictions, average='weighted', zero_division=0)
            
            labels = sorted(list(set(y_true + test_predictions)))
            cm = confusion_matrix(y_true, test_predictions, labels=labels)

            # --- RENDER RESULTS ---
            st.subheader("📊 Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy:.3f}")
            m2.metric("Precision", f"{precision:.3f}")
            m3.metric("Recall", f"{recall:.3f}")
            m4.metric("F1-Score", f"{f1:.3f}")

            # --- SAVE TO STATE ---
            st.session_state['model_history']['CHAID'] = {
                'Algorithm': 'CHAID',
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Confusion Matrix': cm,
                'Model': tree,                   # <--- SAVED
                'Scaler': None,
                'Feature_Names': selected_features # <--- SAVED
            }
            st.toast("✅ CHAID Results Saved!")

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")