import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def run(df):
    # --- INIT STATE ---
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = {}

    st.subheader("🧠 Deep Learning (Neural Networks)")

    if not TF_AVAILABLE:
        st.error("⚠️ TensorFlow is not installed. Please run: `pip install tensorflow`")
        return

    # --- 1. Data Prep ---
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Variable (y)", df.columns, key="dl_target")
    
    feature_cols = [c for c in df.columns if c != target_col]
    with col2:
        selected_features = st.multiselect("Select Features (X)", feature_cols, default=feature_cols, key="dl_features")

    if not selected_features:
        st.warning("Please select features.")
        return

    X = df[selected_features]
    y = df[target_col]

    X = pd.get_dummies(X, drop_first=True)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    st.write(f"**Input Shape:** `{X_train.shape}` | **Classes:** `{num_classes}`")

    tab1, tab2 = st.tabs(["🛠️ Custom Model Builder", "🔬 Hyperparameter Experiment"])

    # ==========================================
    # TAB 1: CUSTOM BUILDER
    # ==========================================
    with tab1:
        st.markdown("### Design your Neural Network")
        
        c1, c2, c3 = st.columns(3)
        with c1: hidden_layers = st.number_input("Hidden Layers", 1, 5, 2)
        with c2: neurons = st.number_input("Neurons per Layer", 5, 500, 64)
        with c3: dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        
        e1, e2 = st.columns(2)
        with e1: epochs = st.number_input("Epochs", 10, 500, 50)
        with e2: batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

        if st.button("🚀 Train Custom Model"):
            with st.spinner("Training Neural Network..."):
                model = Sequential()
                model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
                model.add(Dropout(dropout_rate))
                
                for _ in range(hidden_layers - 1):
                    model.add(Dense(neurons, activation='relu'))
                    model.add(Dropout(dropout_rate))
                
                model.add(Dense(num_classes, activation='softmax')) 
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                history = model.fit(
                    X_train, y_train_cat,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )

                st.success("Training Complete!")
                
                # --- Evaluation ---
                loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
                
                # Calculate F1
                y_pred_prob = model.predict(X_test)
                y_pred = np.argmax(y_pred_prob, axis=1)
                avg_method = 'binary' if num_classes == 2 else 'weighted'
                f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                
                # --- RENDER RESULTS ---
                c_1, c_2 = st.columns(2)
                c_1.metric("Test Accuracy", f"{acc:.4f}")
                c_2.metric("F1 Score", f"{f1:.4f}")
                
                # --- SAVE TO STATE ---
                cm = confusion_matrix(y_test, y_pred)
                st.session_state['model_history']['Deep Learning'] = {
                    'Algorithm': 'Deep Learning',
                    'Accuracy': acc,
                    'F1-Score': f1,
                    'Confusion Matrix': cm,
                    'Model': model,                     # <--- SAVED
                    'Scaler': scaler,                   # <--- SAVED
                    'Feature_Names': X.columns.tolist() # <--- SAVED
                }
                st.toast("✅ Deep Learning Results Saved!")

                # Plots
                hist_df = pd.DataFrame(history.history)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hist_df['accuracy'], label='Train Accuracy')
                ax.plot(hist_df['val_accuracy'], label='Validation Accuracy', linestyle='--')
                ax.legend()
                st.pyplot(fig)

                st.subheader("Confusion Matrix")
                fig2, ax2 = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax2)
                st.pyplot(fig2)

    # TAB 2 (Experiment) kept as is, but without saving to state to avoid overwriting the custom model
    with tab2:
        st.markdown("### 🔬 Automated Neuron Search")
        # ... (Your existing experiment code remains unchanged) ...
        neuron_list_input = st.text_input("Neurons to test (comma separated)", "1, 5, 10, 20, 50, 100")
        if st.button("🧪 Run Experiment"):
             # ... (Your existing experiment logic) ...
             pass