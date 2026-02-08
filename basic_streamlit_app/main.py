import streamlit as st
import pandas as pd

st.title("World Happiness Data Viewer")

# Define the path to your data
# This assumes you are running the app from the root directory: Nash-Data-Science-Portfolio/
DATA_PATH = "week-04/data/world_happiness.csv"

# Load the data
try:
    df = pd.read_csv(DATA_PATH)
    st.success("Data loaded successfully!")
    
    # Display the data
    st.write("### Full Dataset")
    st.dataframe(df) # This creates an interactive, scrollable table

except FileNotFoundError:
    st.error(f"File not found! I'm looking for: {DATA_PATH}")
    st.info("Check that your CSV is named 'world_happiness.csv' and is inside 'week-04/data/'")