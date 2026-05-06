import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page Setup
st.set_page_config(page_title="Supervised ML", layout="wide")

st.title("Supervised Machine Learning Data Modeling ")
st.markdown("""
This application builds regression models with the default graphs showing predicted country happiness scores with the ability to upload your own dataset. 
Adjust the settings in the sidebar to see the model and graphs update automatically as you upload yourw own.
""")

# First load the data from default or have user uplaod their own
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded.")
else:
    try:
        # Defaulting to world_happiness.csv
        df = pd.read_csv("world_happiness.csv")
        st.sidebar.info("Using default: 2026 World Happiness Report")
    except FileNotFoundError:
        st.error("Default data not found. Please upload a CSV file.")
        st.stop()

#  Configure the data header
st.sidebar.header("2. Data Configuration")

# Users can select algorithm
algo = st.sidebar.selectbox("Select Model Algorithm", ["Linear Regression", "Random Forest"])

num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

if len(num_cols) < 2:
    st.error("Dataset requires at least two numerical columns.")
    st.stop()

target_var = st.sidebar.selectbox(
    "Select Target (Y)", 
    num_cols, 
    index=num_cols.index('score') if 'score' in num_cols else 0
)

feature_vars = st.sidebar.multiselect(
    "Select Features (X)", 
    [c for c in num_cols if c != target_var],
    default=[c for c in num_cols if c != target_var][:4]
)

# Parameters for the model
st.sidebar.header("3. Model Hyperparameters")
test_ratio = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)

# Random Forest settings
if algo == "Random Forest":
    n_trees = st.sidebar.slider("Number of Trees", 10, 200, 100)
    tree_depth = st.sidebar.slider("Max Tree Depth", 1, 20, 5)

# Data Preview
with st.expander("Explore Raw Data"):
    st.write(df.head())

# Actual Model logic
# Runs automatically when any changes are made to the sidebar
if target_var and feature_vars:
    # Data Cleaning
    model_df = df[feature_vars + [target_var]].dropna()
    
    if model_df.empty:
        st.error("No data available after filtering missing values.")
    else:
        X = model_df[feature_vars]
        y = model_df[target_var]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio/100, random_state=42
        )

        # Begin Model
        if algo == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=n_trees, max_depth=tree_depth, random_state=42)

        # Train and Predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # 5. Performance Metrics
        st.subheader(f"Results: {algo}")
        m1, m2, m3 = st.columns(3)
        m1.metric("R-Squared (Accuracy)", f"{r2_score(y_test, predictions):.3f}")
        m2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, predictions):.3f}")
        m3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, predictions)):.3f}")

        # Actual Performance Graphs
        st.divider()
        g1, g2 = st.columns(2)

        with g1:
            st.write("#### Actual vs. Predicted")
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
            # Reference line for ideal performance
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual Score")
            ax.set_ylabel("Predicted Score")
            st.pyplot(fig)

        with g2:
            st.write("#### Feature Importance")
            if algo == "Linear Regression":
                importance = pd.DataFrame({'Feature': feature_vars, 'Weight': model.coef_})
            else:
                importance = pd.DataFrame({'Feature': feature_vars, 'Weight': model.feature_importances_})
            
            importance = importance.sort_values(by='Weight', ascending=False)
            st.bar_chart(importance.set_index('Feature'))
else:
    st.info("Select your variables in the sidebar to generate the analysis.")