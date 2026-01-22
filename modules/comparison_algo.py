import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run(df):
    st.subheader("🏆 Algorithm Comparator: Leaderboard")

    # --- 1. Retrieve History ---
    if 'model_history' not in st.session_state or not st.session_state['model_history']:
        st.info("👋 No trained models found in history.")
        st.warning("👉 Go to the other tabs (KNN, Naive Bayes, etc.), train your models, then come back here.")
        return

    history = st.session_state['model_history']
    
    # --- 2. Comparison Table ---
    table_data = []
    for algo, metrics in history.items():
        row = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        row['Algorithm'] = algo
        table_data.append(row)
        
    res_df = pd.DataFrame(table_data)
    cols = ['Algorithm'] + [c for c in res_df.columns if c != 'Algorithm']
    res_df = res_df[cols]

    st.write(f"Comparing **{len(res_df)}** algorithms:")
    
    # Sort by Accuracy (Classification) or R2 Score (Regression)
    sort_col = 'Accuracy' if 'Accuracy' in res_df.columns else res_df.columns[1]
    
    st.dataframe(
        res_df.sort_values(by=sort_col, ascending=False).style.highlight_max(axis=0, color='lightgreen')
    )

    # Highlight Best
    best_row_idx = res_df[sort_col].idxmax()
    best_algo_name = res_df.loc[best_row_idx, 'Algorithm']
    best_score = res_df.loc[best_row_idx, sort_col]
    
    st.success(f"🏆 Best Model: **{best_algo_name}** ({sort_col}: {best_score:.4f})")

    # --- 3. Visuals ---
    with st.expander("📊 Visual Comparison", expanded=False):
        if 'Accuracy' in res_df.columns and 'F1-Score' in res_df.columns:
            plot_df = res_df[['Algorithm', 'Accuracy', 'F1-Score']].melt('Algorithm', var_name='Metric', value_name='Score')
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=plot_df, x='Algorithm', y='Score', hue='Metric', palette="viridis", ax=ax)
            ax.set_ylim(0, 1.1)
            st.pyplot(fig)
        elif 'R2 Score' in res_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=res_df, x='Algorithm', y='R2 Score', palette="magma", ax=ax)
            st.pyplot(fig)

    # =========================================================
    #  🔮 FACILITATED PREDICTION SECTION (UPDATED)
    # =========================================================
    st.divider()
    st.subheader("🔮 Make a Prediction")

    col_sel1, col_sel2 = st.columns(2)
    
    # 1. Choose Model
    with col_sel1:
        # Default to the best model, but allow user to change it
        model_options = list(history.keys())
        default_idx = model_options.index(best_algo_name) if best_algo_name in model_options else 0
        selected_algo = st.selectbox("Select Model to use:", model_options, index=default_idx)

    # 2. Choose Target (To hide it from the inputs)
    with col_sel2:
        # User selects which column represents the 'y' so we don't ask for it in the form
        target_col_option = st.selectbox("Select Target Column (What you want to predict):", df.columns)

    # Retrieve Model Data
    model_data = history[selected_algo]

    if 'Model' in model_data and 'Feature_Names' in model_data:
        model = model_data['Model']
        feature_names = model_data['Feature_Names']
        scaler = model_data.get('Scaler', None)

        st.info(f"👇 Enter values below to predict **{target_col_option}** using **{selected_algo}**.")

        with st.form("prediction_form"):
            input_data = {}
            ui_cols = st.columns(3)
            col_counter = 0
            
            # Loop through dataset columns
            for col_name in df.columns:
                # SKIP the Target Column (We don't input the answer!)
                if col_name == target_col_option:
                    continue
                
                # Create Dropdowns based on existing data
                unique_vals = df[col_name].unique().tolist()
                unique_vals.sort(key=lambda x: str(x))
                
                with ui_cols[col_counter % 3]:
                    val = st.selectbox(f"{col_name}", unique_vals, key=f"pred_{col_name}")
                    input_data[col_name] = val
                col_counter += 1
            
            st.markdown("---")
            submit = st.form_submit_button("✨ Predict Result")

        if submit:
            # 1. Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # 2. Encode
            input_encoded = pd.get_dummies(input_df, drop_first=True)

            # 3. Align Columns (Fill missing with 0)
            final_input = input_encoded.reindex(columns=feature_names, fill_value=0)

            # 4. Scale
            if scaler:
                final_input_scaled = scaler.transform(final_input)
            else:
                final_input_scaled = final_input

            # 5. Predict
            try:
                prediction = model.predict(final_input_scaled)
                st.balloons()
                st.success(f"### 🎯 Predicted {target_col_option}: **{prediction[0]}**")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Prediction unavailable. Please re-train your models using the updated code.")

    # 4. Matrix Inspector
    st.divider()
    with st.expander("🔍 Confusion Matrix Inspector"):
        models_with_cm = [k for k, v in history.items() if v.get('Confusion Matrix') is not None]
        if models_with_cm:
            m = st.selectbox("Select Model:", models_with_cm)
            cm = history[m]['Confusion Matrix']
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)