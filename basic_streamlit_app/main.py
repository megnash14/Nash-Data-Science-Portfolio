import streamlit as st
import pandas as pd

st.title("World Happiness Data Viewer")

st.header("Context")
st.write("The World Happiness Report is a landmark global survey that ranks 155 countries by their happiness levels and has gained significant recognition as a tool for informing government policy-making and assessing national progress. Leading experts across fields like economics, psychology, and public policy use these reports to review the state of global happiness and explain personal and national variations through the lens of well-being science. Happiness rankings are derived from the Gallup World Poll using the Cantril ladder question, which asks respondents to rate their current lives on a scale of 0 to 10. The total happiness score is explained by six key variables including economic production, social support, healthy life expectancy, freedom to make life choices, generosity, and the absence of government corruption. To provide a benchmark for comparison, the report utilizes a hypothetical country called Dystopia which represents the world’s lowest national averages for each of the six key variables. The residuals represent the unexplained components where the six variables either over- or under-explain national life evaluations. This project seeks to identify which regions rank highest in overall happiness and how national scores evolved between the 2015 and 2017 reports.")

st.subheader("Happiness Dataset")
df=pd.read_csv("data/world_happiness.csv")
df_sorted = df.sort_values(by="Happiness Rank", ascending=True)
st.dataframe(df_sorted)


