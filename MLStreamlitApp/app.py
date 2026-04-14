import streamlit as st
import pandas as pd
import numpy as np

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# PAGE CONFIG (builds layout and title in browser tab)
st.set_page_config(
    page_title="ML Playground",
    layout="wide"
)
# APP TITLE and DESCRIPTION
st.title("Machine Learning Playground")
st.markdown("""
Welcome! Upload your dataset, choose a model, and experiment with hyperparameters  
to see how they impact performance.

This app is designed to make machine learning interactive and intuitive.
""")

# SIDEBAR (User Controls Live Here)
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# MAIN APP LOGIC

if uploaded_file:
    
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Show dataset preview in main area
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # TARGET SELECTION
    target = st.sidebar.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # TRAIN / TEST SPLIT
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # -------------------------------
    # MODEL SELECTION
    # -------------------------------
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Logistic Regression", "KNN", "Decision Tree"]
    )

    # -------------------------------
    # HYPERPARAMETERS (Change based on model)
    # -------------------------------
    if model_choice == "Logistic Regression":
        st.sidebar.markdown("### Logistic Regression Settings")
        C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=1000)

    elif model_choice == "KNN":
        st.sidebar.markdown("### KNN Settings")
        k = st.sidebar.slider("Number of Neighbors", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    elif model_choice == "Decision Tree":
        st.sidebar.markdown("### Decision Tree Settings")
        depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=depth)

    # TRAIN MODEL BUTTON
    if st.sidebar.button("Train Model"):
        
        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # RESULTS DISPLAY (Columns Layout)
        # -------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Performance")

            accuracy = accuracy_score(y_test, predictions)
            st.metric(label="Accuracy", value=f"{accuracy:.2f}")

        with col2:
            st.subheader("Model Info")
            st.write(f"Model: {model_choice}")
            st.write(f"Training Size: {len(X_train)}")
            st.write(f"Test Size: {len(X_test)}")

        # SUCCESS MESSAGE
        st.success("Model trained successfully!")

# DEFAULT MESSAGE (Before Upload)
else:
    st.info("Upload a CSV file from the sidebar to get started!")