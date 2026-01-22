# modules/cart_algo.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def run(df):
    # --- INIT SESSION STATE ---
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = {}

    st.subheader("🌳CART (Classification and Regression Tree)")

    # --- 1. Data Selection ---
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Variable (y)", df.columns, key="cart_target")
    
    feature_cols = [c for c in df.columns if c != target_col]
    with col2:
        selected_features = st.multiselect("Select Features (X)", feature_cols, default=feature_cols, key="cart_features")

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # --- 2. Train Button ---
    if st.button("🌳 Analyze & Train CART"):
        
        # Prepare Data
        X = df[selected_features]
        y = df[target_col]

        # Handle Categorical Data (Encoding)
        X = pd.get_dummies(X, drop_first=True)
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        st.write(f"**Data Split:** Training: `{X_train.shape[0]}` | Test: `{X_test.shape[0]}`")

        # --- PHASE 1: Depth Analysis (1 to 10) ---
        st.subheader("1 Depth Analysis (Pruning)")
        
        depth_values = range(1, 11)
        train_scores = []
        test_scores = []
        
        progress_bar = st.progress(0)
        
        for i, depth in enumerate(depth_values):
            clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
            clf.fit(X_train, y_train)
            
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
            progress_bar.progress((i + 1) / 10)
            
        # Create Results DataFrame
        results_df = pd.DataFrame({
            'Depth': depth_values,
            'Train Accuracy': train_scores,
            'Test Accuracy': test_scores
        })
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(depth_values, train_scores, marker='o', label='Train Accuracy', color='blue')
        ax.plot(depth_values, test_scores, marker='o', label='Test Accuracy', color='orange')
        ax.set_title("Accuracy vs Tree Depth")
        ax.set_xlabel("Max Depth")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(depth_values)
        ax.legend()
        ax.grid(True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(results_df.style.highlight_max(axis=0))
        with c2:
            st.pyplot(fig)

        # --- PHASE 2: Select Optimal Model ---
        st.subheader("2 Final Model Visualization")
        
        # Auto-select best depth based on Test Accuracy
        best_depth_idx = np.argmax(test_scores)
        suggested_depth = depth_values[best_depth_idx]
        
        # NOTE: For the automatic "Compare" flow, we default to the suggested depth if user doesn't interact,
        # but here we keep the slider for interactivity.
        selected_depth = st.slider("Select Max Depth for Final Model", 1, 10, int(suggested_depth))
        
        # Train Final Model
        best_model = DecisionTreeClassifier(criterion='gini', max_depth=selected_depth, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # --- CALCULATE METRICS ---
        avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # --- RENDER METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-Score", f"{f1:.3f}")
        
        # --- RENDER MATRIX ---
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", ax=ax_cm)
        st.pyplot(fig_cm)

        # --- SAVE TO STATE ---
        st.session_state['model_history']['CART (Tree)'] = {
            'Algorithm': 'CART (Tree)',
            'Accuracy': acc,
            'F1-Score': f1,
            'Confusion Matrix': cm,
            'Model': clf,                       # <--- SAVED
            'Scaler': None,                     # No scaler for Trees
            'Feature_Names': X.columns.tolist() # <--- SAVED
        }
        st.toast("✅ CART Results Saved for Comparison!")
        
        # --- Tree Visualization ---
        st.subheader("🌳 Tree Diagram")
        st.info("Visualizing the decision rules...")
        
        fig_tree = plt.figure(figsize=(20, 10))
        plot_tree(
            best_model, 
            feature_names=X.columns.tolist(),
            class_names=[str(c) for c in best_model.classes_],
            filled=True,
            rounded=True,
            fontsize=10
        )
        st.pyplot(fig_tree)