import streamlit as st
import pandas as pd

st.title("World Happiness Data Viewer")

st.header("Context")
st.write("The World Happiness Report is a landmark global survey that ranks 155 countries by their happiness levels and has gained significant recognition as a tool for informing government policy-making and assessing national progress. Leading experts across fields like economics, psychology, and public policy use these reports to review the state of global happiness and explain personal and national variations through the lens of well-being science. Happiness rankings are derived from the Gallup World Poll using the Cantril ladder question, which asks respondents to rate their current lives on a scale of 0 to 10. The total happiness score is explained by six key variables including economic production, social support, healthy life expectancy, freedom to make life choices, generosity, and the absence of government corruption. To provide a benchmark for comparison, the report utilizes a hypothetical country called Dystopia which represents the world’s lowest national averages for each of the six key variables. The residuals represent the unexplained components where the six variables either over- or under-explain national life evaluations. This project seeks to identify which regions rank highest in overall happiness and how national scores evolved between the 2015 and 2017 reports.")

df=pd.read_csv("data/world_happiness.csv")
df_sorted = df.sort_values(by="Happiness Rank", ascending=True)


st.sidebar.header("Filter the Data")

region_options = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect("Select Regions:", region_options, default=region_options)

country_options = sorted(df["Country"].unique())
selected_countries = st.sidebar.multiselect("Select Countries:", country_options, default=country_options)

filtered_df = df[
    (df["Country"].isin(selected_countries)) & 
    (df["Region"].isin(selected_regions)) 
]
df_sorted = filtered_df.sort_values(by="Happiness Rank", ascending=True)


st.subheader("Key Highlights")

col1, col2, col3, col4 = st.columns(4)

top_happy = filtered_df.sort_values(by="Happiness Score", ascending=False).iloc[0]
col1.metric(
    label="Happiest Country", 
    value=top_happy["Country"], 
    delta=f"Score: {top_happy['Happiness Score']:.2f}",
    delta_color="normal" 
)

top_gdp = filtered_df.sort_values(by="Economy (GDP per Capita)", ascending=False).iloc[0]
col2.metric("Highest GDP", top_gdp["Country"], f"{top_gdp['Economy (GDP per Capita)']:.2f}")

top_health = filtered_df.sort_values(by="Health (Life Expectancy)", ascending=False).iloc[0]
col3.metric("Top Life Expectancy", top_health["Country"], f"{top_health['Health (Life Expectancy)']:.2f}")

top_gen = filtered_df.sort_values(by="Generosity", ascending=False).iloc[0]
col4.metric("Most Generous", top_gen["Country"], f"{top_gen['Generosity']:.3f}")

st.subheader("Happiness Dataset")
st.dataframe(df_sorted, hide_index=True)