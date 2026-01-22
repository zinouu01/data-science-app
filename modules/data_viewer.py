import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

# Increase the max elements allowed for styling
pd.set_option("styler.render.max_elements", 1000000)

def run():
    st.header("🔍 Data Exploration & Quality Hub")
    
    # --- 0. INITIALIZATION ---
    if 'data' not in st.session_state:
        st.info("👋 Please upload a dataset in the sidebar to begin.")
        return

    df = st.session_state['data']

    # 🆕 INITIALIZE EDITOR KEY COUNTER
    # This helps us force-reset the editor when needed
    if "editor_key" not in st.session_state:
        st.session_state["editor_key"] = 0

   # --- 0.5 DEFINE MISSING MARKERS (USER INPUT) ---
    # We allow the user to define what looks like a "missing value" via a text box
    
    with st.expander("⚙️ Configure Missing Value Detection", expanded=False):
        st.write("Define what symbols your dataset uses for missing values.")
        
        # 1. User Input for Custom Markers
        default_markers_str = "?, _, !, -, *, @, #, vide, NULL, null, N/A, nan"
        
        user_input_markers = st.text_input(
            "Enter missing value markers (separated by commas):", 
            value=default_markers_str,
            help="Example: ?, N/A, -99"
        )
        
        # 2. Process the string into a Python List
        # Split by comma and remove extra spaces around words
        missing_markers = [x.strip() for x in user_input_markers.split(",")]
        
        # 3. Add Empty Strings (Optional but recommended)
        include_blanks = st.checkbox("Treat empty cells/spaces as missing?", value=True)
        if include_blanks:
            missing_markers.extend(["", " "])

        # 4. Add Zero (Optional)
        check_zero = st.checkbox("Treat '0' (numeric zero) as missing?", value=False)
        if check_zero:
            missing_markers.extend([0, "0", 0.0])

        # Remove duplicates just in case
        missing_markers = list(set(missing_markers))
        
        st.info(f"**Active Missing Markers:** {missing_markers}")

    # --- 1. INTERACTIVE DATA EDITOR & COLUMN MANAGER ---
    with st.expander("✏️ Data Editor & Column Manager", expanded=True):
        
        # 1. INITIALIZE EDITOR KEY (To force reset when needed)
        if "editor_key" not in st.session_state:
            st.session_state["editor_key"] = 0

        # 2. SELECTOR WIDGET (No 'key' argument avoids the API error!)
        # We use the variable 'cols_to_drop' directly.
        all_cols = st.session_state['data'].columns.tolist()
        cols_to_drop = st.multiselect("🗑️ Select columns to remove:", all_cols)
        
        # 3. CONFIRMATION & DELETION
        if cols_to_drop:
            st.warning(f"⚠️ Are you sure you want to permanently delete: {cols_to_drop}?")
            
            if st.button("✅ Yes, Delete Columns", key="btn_confirm_drop"):
                # A. Update the data in session state
                st.session_state['data'] = st.session_state['data'].drop(columns=cols_to_drop)
                
                # B. Force the editor to rebuild (prevents columns reappearing)
                st.session_state["editor_key"] += 1
                
                # C. Rerun immediately to refresh the page
                st.rerun()

        # 4. DATA EDITOR
        # We use the dynamic key. When 'editor_key' increases, this is treated as a brand new widget.
        unique_key = f"main_editor_{st.session_state['editor_key']}"
        
        # Always fetch the latest data before rendering
        current_df = st.session_state['data']
        
        if not current_df.empty:
            edited_df = st.data_editor(
                current_df, 
                use_container_width=True, 
                num_rows="dynamic", 
                key=unique_key
            )
            
            # Save manual edits (like typing in cells)
            if not edited_df.equals(current_df):
                st.session_state['data'] = edited_df
        else:
            st.warning("⚠️ Dataset is empty.")

    # --- 2. DATA OVERVIEW (UPDATED) ---
    st.subheader("Data Overview")
    
    # Calculate Total Missing (Standard NaN + Custom Markers)
    # 1. Standard Python NaNs
    std_missing = df.isnull().sum().sum()
    # 2. Custom Markers (We check object columns for symbols, and all columns for exact matches)
    custom_missing = df.isin(missing_markers).sum().sum()
    total_missing_combined = std_missing + custom_missing

    ov_col1, ov_col2, ov_col3 = st.columns(3)
    with ov_col1:
        st.write("**Shape:**")
        st.code(f"{df.shape}")
    with ov_col2:
        st.write("**Columns:**")
        st.json(list(df.columns))
    with ov_col3:
        st.write("**Total Missing Values:**")
        # Display breakdown
        st.code(f"{total_missing_combined}")
        if custom_missing > 0:
            st.caption(f"(NaN: {std_missing} | Custom: {custom_missing})")

    # --- 3. COLUMN DETAILS ---
    st.subheader("Column Details")
    col_details = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': [str(dtype) for dtype in df.dtypes],
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_details, use_container_width=True)

    # --- 4. DATA QUALITY & AUTO-CONVERT (CORRECTED) ---
    with st.expander("📊 Initial Value Analysis & Data Quality", expanded=True):
        quality_stats = []
        for col in df.columns:
            total_rows = len(df[col])
            
            # 1. Count Standard NaNs (Empty cells, None, NaN)
            std_null_count = df[col].isna().sum()
            
            # 2. Count Custom Markers (e.g., 0, ?, vide)
            # We use isin() to find exact matches for your missing markers
            custom_null_count = df[col].isin(missing_markers).sum()
            
            # 3. Numeric Detection
            # We try to convert the column to numbers to see how many valid numbers exist.
            # 'coerce' turns non-numbers into NaN.
            converted_to_num = pd.to_numeric(df[col], errors='coerce')
            numeric_detected = converted_to_num.notna().sum()
            
            # 4. Validity Logic
            if pd.api.types.is_numeric_dtype(df[col]):
                # FIX: If column is numeric, 'numeric_detected' includes 0.
                # We must subtract the custom markers (like 0) to get the TRUE valid count.
                valid_count = numeric_detected - custom_null_count
            else:
                # If Object, valid = Total - (Real NaNs + Custom Markers + Hidden Numbers)
                pure_object_count = total_rows - std_null_count - custom_null_count - numeric_detected
                pure_object_count = max(0, pure_object_count) # Safety check
                valid_count = pure_object_count + numeric_detected

            # Calculate Percentage
            validity_pct = (valid_count / total_rows) * 100 if total_rows > 0 else 0
            
            quality_stats.append({
                'Column': col, 
                'Type': str(df[col].dtype), 
                'Total Rows': total_rows,
                'Real NaNs': std_null_count,
                'Custom Missing': custom_null_count, # This will now show your 0s
                'Numeric Found': numeric_detected,
                'Validity (%)': f"{validity_pct:.2f}%"
            })
            
        st.dataframe(pd.DataFrame(quality_stats), use_container_width=True)
        
        # Unique key 'btn_auto_convert'
        if st.button("🔄 Auto-Convert Detected Numbers", key="btn_auto_convert"):
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            st.session_state['data'] = df
            st.rerun()

    # --- 4.5 CLEAN NON-STANDARD MISSING VALUES ---
    # This button effectively "commits" the custom missing values to real NaNs
    st.markdown("---")
    if st.button(f"🧼 Replace all {missing_markers} with NaN", key="btn_clean_custom_apply"):
        df = df.replace(missing_markers, np.nan)
        st.session_state['data'] = df
        st.success("Custom missing values have been converted to standard NaNs! Statistics will update.")
        st.rerun()

    # ... (Continue to Step 5: ADVANCED NUMERICAL STATISTICS) ...       

    # --- 5. ADVANCED NUMERICAL STATISTICS ---
    with st.expander("📈 Advanced Numerical Statistics"):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            stats_list = []
            for col in num_cols:
                series = df[col].dropna()
                if not series.empty:
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    stats_list.append({
                        'Column': col, 'Mean': series.mean(), 'Median (Q2)': series.median(),
                        'Q1': q1, 'Q3': q3, 'Min': series.min(), 'Max': series.max(), 'IQR': q3 - q1
                    })
            st.dataframe(pd.DataFrame(stats_list).style.format(precision=2), use_container_width=True)

    # --- 6. CATEGORICAL COLUMNS STATISTICS ---
    st.subheader("🔠 Categorical Columns Statistics")
    cat_cols_list = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols_list:
        cat_stats = []
        for col in cat_cols_list:
            mode_series = df[col].mode()
            mode_val = mode_series[0] if not mode_series.empty else "N/A"
            cat_stats.append({
                'Column': col, 'Mode': mode_val, 
                'Mode Freq': df[col].value_counts().iloc[0] if not df[col].empty else 0,
                'Unique Values': df[col].nunique()
            })
        st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)

    # --- 7. VISUALIZATIONS ---
    st.subheader("🎨 Outlier Analysis & Visualizations")
    plot_num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if plot_num_cols:
        viz_type = st.selectbox("Select Plot Type", ["Boxplot", "Histogram", "Scatter Plot", "Heatmap"], key="viz_selector")
        if viz_type == "Boxplot":
            selected_cols = st.multiselect("Select columns", plot_num_cols, default=plot_num_cols[:1], key="box_cols")
            if selected_cols:
                fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
                ax.boxplot([df[c].dropna() for c in selected_cols], patch_artist=True, labels=selected_cols, 
                           boxprops={'color': '#00FFFF', 'linewidth': 2}, medianprops={'color': '#FFD700', 'linewidth': 3})
                st.pyplot(fig)
        elif viz_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(10, 8)); fig.patch.set_facecolor('#0E1117')
            sns.heatmap(df[plot_num_cols].corr(), annot=True, cmap="mako", ax=ax)
            st.pyplot(fig)

   # --- 7.5 NUMERIC PREDICTIVE IMPUTATION (UPDATED) ---
    st.markdown("---")
    st.header("🩹 Numeric Predictive Imputation")
    
    # 1. Select Numeric Columns
    plot_num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # 2. Create a temporary working copy
    df_working = df.copy()
    
    # 3. CRITICAL: Replace Custom Markers (like 0) with NaN in this copy
    # This ensures the Imputer actually sees them as missing!
    df_working[plot_num_cols] = df_working[plot_num_cols].replace(missing_markers, np.nan)
    
    # 4. Generate Mask (What needs to be fixed?)
    # Now this mask includes your 0s because they are now NaN in df_working
    num_null_mask = df_working[plot_num_cols].isnull()
    
    # Show stats
    total_missing_num = num_null_mask.sum().sum()
    
    if total_missing_num > 0:
        st.warning(f"⚠️ Found {total_missing_num} missing values (including {missing_markers}) to impute.")
        
        impute_method = st.radio("Numeric Method", ["None", "KNN Imputer", "MICE"], horizontal=True, key="num_radio")
        
        if impute_method != "None":
            if impute_method == "KNN Imputer":
                imputer = KNNImputer(n_neighbors=5)
                df_working[plot_num_cols] = imputer.fit_transform(df_working[plot_num_cols])
            else:
                imputer = IterativeImputer(random_state=42)
                df_working[plot_num_cols] = imputer.fit_transform(df_working[plot_num_cols])

            st.write("Preview: Neon Teal = Predicted Values")
            
            # Use the optimized styling (head 50) to prevent crashing
            st.dataframe(df_working.style.apply(lambda x: [
                'background-color: #004d4d; color: #00ffcc;' if (x.name in plot_num_cols and num_null_mask.loc[i, x.name]) else '' 
                for i, v in x.items()
            ], axis=0), use_container_width=True)

            # --- DOWNLOAD ---
            csv_num = df_working.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Dataset with Numeric Imputation",
                data=csv_num,
                file_name="numeric_imputed_data.csv",
                mime="text/csv",
                key="btn_download_num"
            )
    else:
        st.success("✅ No numeric missing values found! (If you have '?' in a column, it might be treated as Text/Object. Clean it in Step 4.5 first!)")

    # --- 7.6 CATEGORICAL DATA IMPUTATION (FIXED: RETURNS TEXT) ---
    st.markdown("---")
    st.header("🔡 Categorical Data Imputation")
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create a working copy
    df_cat_working = df.copy()
    
    # Replace custom missing markers with NaN
    df_cat_working[cat_cols] = df_cat_working[cat_cols].replace(missing_markers, np.nan)
    
    cat_null_mask = df_cat_working[cat_cols].isnull()
    total_missing_cat = cat_null_mask.sum().sum()

    if total_missing_cat > 0:
        st.warning(f"⚠️ Found {total_missing_cat} missing values to impute.")
        cat_method = st.radio("Categorical Method", ["None", "Mode", "KNN"], horizontal=True, key="cat_radio")
        
        if cat_method != "None":
            if cat_method == "Mode":
                for col in cat_cols:
                    if df_cat_working[col].isnull().any():
                        df_cat_working[col] = df_cat_working[col].fillna(df_cat_working[col].mode()[0])
            else:
                st.info("ℹ️ KNN running... (Encoding -> Imputing -> Decoding back to Text)")
                
                from sklearn.preprocessing import LabelEncoder
                
                # Dictionary to save the encoder for each column
                encoders = {}

                # 1. ENCODE (Text -> Number)
                for col in cat_cols:
                    series = df_cat_working[col]
                    # Only fit on valid data (not NaNs)
                    non_nulls = series.dropna()
                    if not non_nulls.empty:
                        le = LabelEncoder()
                        le.fit(non_nulls)
                        # Save the encoder so we can reverse this later!
                        encoders[col] = le
                        
                        # Apply encoding only to non-nulls
                        df_cat_working.loc[series.notna(), col] = le.transform(non_nulls)

                # 2. IMPUTE (Predict Numbers)
                imputer = KNNImputer(n_neighbors=5)
                # KNN outputs float (e.g. 1.2), we round to nearest int (1.0)
                df_cat_working[cat_cols] = imputer.fit_transform(df_cat_working[cat_cols])
                df_cat_working[cat_cols] = df_cat_working[cat_cols].round().astype(int)

                # 3. DECODE (Number -> Text)
                for col in cat_cols:
                    if col in encoders:
                        le = encoders[col]
                        # Inverse transform: Turn 0, 1 back to "Male", "Female"
                        # Clip ensures we don't crash if KNN predicted a number outside the range
                        safe_values = df_cat_working[col].clip(0, len(le.classes_) - 1)
                        df_cat_working[col] = le.inverse_transform(safe_values)

            st.write("Preview (Orange = Predicted):")
            st.dataframe(df_cat_working.style.apply(lambda x: [
                'background-color: #4d1a00; color: #ff9966;' if (x.name in cat_cols and cat_null_mask.loc[i, x.name]) else '' 
                for i, v in x.items()
            ], axis=0), use_container_width=True)

            csv_cat = df_cat_working.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Categorical Imputed Data", csv_cat, "cat_imputed.csv", "text/csv")
    else:
        st.success("✅ No categorical missing values found!")
        
    # --- 8. NORMALIZATION OPTIONS ---
    st.markdown("---")
    st.header("🔧 Data Normalization Options")
    
    # Identify numeric columns for normalization
    norm_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if norm_num_cols:
        # User selection for columns and method
        selected_norm_cols = st.multiselect(
            "Select columns to normalize", 
            norm_num_cols, 
            default=norm_num_cols, 
            key="norm_cols"
        )
        
        norm_method = st.radio(
            "Normalization method", 
            ["Min-Max Scaling", "Z-Score Normalization", "None"], 
            index=2, 
            horizontal=True, 
            key="norm_radio"
        )

        # Create a copy so we don't overwrite the original view unless downloaded
        df_norm = df.copy()

        if norm_method != "None" and selected_norm_cols:
            for col in selected_norm_cols:
                if norm_method == "Min-Max Scaling":
                    # Formula: (x - min) / (max - min)
                    col_min = df_norm[col].min()
                    col_max = df_norm[col].max()
                    if col_max != col_min:
                        df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
                
                elif norm_method == "Z-Score Normalization":
                    # Formula: (x - mean) / std
                    col_mean = df_norm[col].mean()
                    col_std = df_norm[col].std()
                    if col_std != 0:
                        df_norm[col] = (df_norm[col] - col_mean) / col_std

            st.subheader(f"Preview: {norm_method} Applied")
            st.dataframe(df_norm.head(10), use_container_width=True)
            
            # Prepare Download
            csv_norm = df_norm.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Normalized Dataset ({norm_method})",
                data=csv_norm,
                file_name=f"normalized_{norm_method.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key="btn_download_norm"
            )
        else:
            st.info("Select a method and columns to see the normalization preview and download link.")
    else:
        st.warning("No numeric columns available for normalization.")