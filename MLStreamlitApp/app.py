import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("Machine Learning Playground App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Select target
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Model selection
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "KNN", "Decision Tree"]
    )

    if model_choice == "Logistic Regression":
        C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=1000)

    elif model_choice == "KNN":
        k = st.slider("Number of Neighbors", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    elif model_choice == "Decision Tree":
        depth = st.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=depth)

    # Train model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Performance
    if model_choice == "Logistic Regression" or model_choice == "KNN" or model_choice == "Decision Tree":
        acc = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {acc:.2f}")

    st.success("Model trained successfully!")