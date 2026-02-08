import streamlit as st
import pandas as pd

st.set_page_config(page_title="World Happiness Report", layout="wide")

# LOAD DATA
df = pd.read_csv("data/world_happiness.csv")

# SIDEBAR
st.sidebar.header("Filter the Data")

region_options = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect("Select Regions:", region_options, default=region_options)

country_options = sorted(df["Country"].unique())
selected_countries = st.sidebar.multiselect("Select Countries:", country_options, default=country_options)

# FILTERS
filtered_df = df[
    (df["Country"].isin(selected_countries)) & 
    (df["Region"].isin(selected_regions)) 
]

# MAIN PAGE
st.title("World Happiness Rankings")
st.divider()
st.subheader("Context")
st.text("The World Happiness Report is a landmark survey of the state of global happiness. " \
"The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. " \
"The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. " \
"The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. " \
"Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. " \
"The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.")

st.divider()
st.subheader("App Description")
st.text("This interactive dashboard serves as a dynamic knowledge base for the 2017 World Happiness Report, allowing viewers to go beyond overall happiness rankings to see scores sorted by specific factors."\
        "Some key features are the sidebar to filter the dataset by specific regions or countries, allowing for customized stats based on personal interest."\
        "Additionally, the dashboard displays the leading nations in six categories: Happiness, GDP, Health, Freedom, Trust (Corruption), and Generosity."\
        "With the complete dataset provided below for further exploration.")

st.text("How to Use: Select your desired regions and countries in the sidebar to update the highlights and the data table.")

st.divider()

# HIGHLIGHTS
st.subheader("Top Countries by Category")

with st.container():
    m1, m2, m3 = st.columns(3)
    
    top_happy = filtered_df.sort_values(by="Happiness Score", ascending=False).iloc[0]
    m1.metric("Happiest Country", top_happy["Country"], f"{top_happy['Happiness Score']:.2f}")

    top_gdp = filtered_df.sort_values(by="Economy (GDP per Capita)", ascending=False).iloc[0]
    m2.metric("Highest GDP", top_gdp["Country"], f"{top_gdp['Economy (GDP per Capita)']:.3f}")

    top_health = filtered_df.sort_values(by="Health (Life Expectancy)", ascending=False).iloc[0]
    m3.metric("Top Life Expectancy", top_health["Country"], f"{top_health['Health (Life Expectancy)']:.2f}")

    m4, m5, m6 = st.columns(3)
    
    top_freedom = filtered_df.sort_values(by="Freedom", ascending=False).iloc[0]
    m4.metric("Most Freedom", top_freedom["Country"], f"{top_freedom['Freedom']:.2f}")

    top_trust = filtered_df.sort_values(by="Trust (Government Corruption)", ascending=False).iloc[0]
    m5.metric("Most Trust (Gov)", top_trust["Country"], f"{top_trust['Trust (Government Corruption)']:.2f}")

    top_gen = filtered_df.sort_values(by="Generosity", ascending=False).iloc[0]
    m6.metric("Most Generous", top_gen["Country"], f"{top_gen['Generosity']:.3f}")

st.divider()

# DATASET
st.subheader("Dataset Explorer")

df_sorted = filtered_df.sort_values(by="Happiness Rank", ascending=True)
if "Happiness Rank" in df_sorted.columns:
    cols = df_sorted.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Happiness Rank')))
    df_sorted = df_sorted[cols]

st.dataframe(
    df_sorted, 
    hide_index=True, 
    use_container_width=True
)