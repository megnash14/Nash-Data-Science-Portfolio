import streamlit as st
import pandas as pd
import numpy as np

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. PAGE CONFIG
st.set_page_config(
    page_title="ML Playground",
    layout="wide"
)

# 2. DATA CLEANING ENGINE
def preprocess_data(df, target):
    """Handles missing values and converts text to numbers safely."""
    # Remove rows with missing values
    df = df.dropna()
    
    # Identify text columns (excluding the target)
    for col in df.columns:
        if df[col].dtype == 'object' and col != target:
            # Convert text categories to numeric codes (0, 1, 2...)
            df[col] = df[col].astype('category').cat.codes
    
    # If target is text, convert it too
    if df[target].dtype == 'object':
        df[target] = df[target].astype('category').cat.codes
        
    return df

# 3. HEADER
st.title("Machine Learning Playground")
st.markdown("""
This app transforms raw CSV data into a trained Machine Learning model. 
Explore **Feature Importance** to see which variables drive the predictions.
""")

# 4. SIDEBAR CONTROLS
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Initial load
    raw_data = pd.read_csv(uploaded_file)
    
    # User selects target
    target_column = st.sidebar.selectbox("Select Target (Label)", raw_data.columns)
    
    # Clean the data automatically
    df = preprocess_data(raw_data, target_column)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Algorithm Selection
    st.sidebar.divider()
    model_choice = st.sidebar.selectbox("Choose Model", ["Decision Tree", "Logistic Regression", "KNN"])
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

    # Hyperparameters
    if model_choice == "Decision Tree":
        depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=depth)
    elif model_choice == "Logistic Regression":
        c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=c_val, max_iter=1000)
    elif model_choice == "KNN":
        k = st.sidebar.slider("Neighbors (K)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    # 5. MAIN AREA TABS
    tab_preview, tab_results = st.tabs(["Data Preview", "Model Analysis"])

    with tab_preview:
        st.subheader("Processed Dataset Preview")
        st.write("Text categories have been encoded as numbers for the ML model.")
        st.dataframe(df.head(), use_container_width=True)

    with tab_results:
        if st.sidebar.button("Train Model"):
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Fit
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Performance Metric
            acc = accuracy_score(y_test, y_pred)
            st.metric("Model Accuracy", f"{acc:.2%}")

            # --- ROBUST VISUALIZATION SECTION ---
            st.divider()
            
            if model_choice == "Decision Tree":
                st.subheader("Feature Importance")
                st.write("Which columns had the biggest impact on the prediction?")
                
                # Logic to extract importance
                importances = model.feature_importances_
                feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False)
                
                # Plotly-style bar chart (built into Streamlit)
                st.bar_chart(data=feat_df, x='Feature', y='Importance')

            elif model_choice == "Logistic Regression":
                st.subheader("Feature Weights (Coefficients)")
                st.write("Positive weights increase the target value; negative weights decrease it.")
                
                # Logic for weights
                weights = model.coef_[0]
                weight_df = pd.DataFrame({'Feature': X.columns, 'Weight': weights})
                st.bar_chart(data=weight_df, x='Feature', y='Weight')

            elif model_choice == "KNN":
                st.subheader("Model Info")
                st.info("KNN is a distance-based model and does not provide feature importance directly.")
                st.write(f"The model analyzed **{len(X_train)}** samples to find the **{k}** nearest neighbors for each prediction.")

            st.success("Analysis Complete!")
        else:
            st.info("Click 'Train Model' in the sidebar to see results.")

else:
    st.info("Waiting for CSV upload...")