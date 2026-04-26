import streamlit as st
import pandas as pd
import numpy as np
import os

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. This is to basically set up the actual page itself
st.set_page_config(
    page_title="ML Playground",
    layout="wide"
)

# 2. This is to clean the data that is being uploaded by the user  
def preprocess_data(df, target):
    df = df.dropna()
    for col in df.columns:
        if df[col].dtype == 'object' and col != target:
            df[col] = df[col].astype('category').cat.codes
    if df[target].dtype == 'object':
        df[target] = df[target].astype('category').cat.codes
    return df

# 3. This is the header and description for the website
st.title("Machine Learning Playground")
st.markdown("""
This app transforms raw CSV data into a trained Machine Learning model. 
Using **World Happiness Report** as the default dataset.
""")

# 4. This is the sidebar features and the logic for the world happiness dataset I included.
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])

# This finds the exact folder where app.py lives
base_path = os.path.dirname(os.path.abspath(__file__))

# This creates the full path to the CSV file
DEFAULT_FILE = os.path.join(base_path, "world_happiness.csv")

raw_data = None

# Data Loading Logic
if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.sidebar.success("Using uploaded file")
elif os.path.exists(DEFAULT_FILE):
    raw_data = pd.read_csv(DEFAULT_FILE)
    st.sidebar.info("Using default: World Happiness dataset")
else:
    # If it fails, this will show us exactly why in the app
    st.sidebar.error(f"⚠️ File not found at: {DEFAULT_FILE}")
    st.sidebar.write("Files Python sees in this folder:", os.listdir(base_path))

# 5. This is the Machine Learning Engine that only runs for data uploaded or default data
if raw_data is not None:
    target_column = st.sidebar.selectbox("Select Target (Label)", raw_data.columns)
    
    df = preprocess_data(raw_data, target_column)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    st.sidebar.divider()
    model_choice = st.sidebar.selectbox("Choose Model", ["Decision Tree", "Logistic Regression", "KNN"])
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

    if model_choice == "Decision Tree":
        depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=depth)
    elif model_choice == "Logistic Regression":
        c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=c_val, max_iter=1000)
    elif model_choice == "KNN":
        k = st.sidebar.slider("Neighbors (K)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    tab_preview, tab_results = st.tabs(["Data Preview", "Model Analysis"])

    with tab_preview:
        st.subheader("Processed Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

    with tab_results:
        if st.sidebar.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.metric("Model Accuracy", f"{acc:.2%}")
            
            st.divider()
            
            if model_choice == "Decision Tree":
                st.subheader("Feature Importance")
                importances = model.feature_importances_
                feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                st.bar_chart(data=feat_df, x='Feature', y='Importance')

            elif model_choice == "Logistic Regression":
                st.subheader("Feature Weights")
                weights = model.coef_[0]
                weight_df = pd.DataFrame({'Feature': X.columns, 'Weight': weights})
                st.bar_chart(data=weight_df, x='Feature', y='Weight')

            elif model_choice == "KNN":
                st.subheader("Model Info")
                st.info(f"KNN analyzed **{len(X_train)}** samples using **{k}** neighbors.")

            st.success("Analysis Complete!")
        else:
            st.info("Click 'Train Model' to begin.")
else:
    st.info("Waiting for data... Please ensure 'world_happiness.csv' is in the app folder or upload a file.")