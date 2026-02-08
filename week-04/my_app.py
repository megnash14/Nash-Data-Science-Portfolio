import streamlit as st
import pandas as pd

st.title("My First EDA App")

if st.button("Click me!"):
    st.write("🎉 You clicked the button!")

color = st.color_picker("Pick a color", "#00f900")
st.write(f"You picked: {color}")

st.subheader("Data Explorer")
df = pd.read_csv("data/sample_data-1.csv")

st.write("Full dataset:")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique())
filtered_df = df[df["City"] == city]

st.write(f"People in {city}:")
st.dataframe(filtered_df)

st.subheader("Summary Statistics")
st.write(df.describe())