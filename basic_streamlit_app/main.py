import streamlit as st
import pandas as pd

st.set_page_config(page_title="World Happiness Report", layout="wide")

# 1. CACHED DATA LOADING
@st.cache_data
def load_data():
    return pd.read_csv("basic_streamlit_app/data/world_happiness.csv")

df = load_data()

# 2. DYNAMIC SIDEBAR
st.sidebar.header("Filter the Data")

region_options = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect("Select Regions:", region_options, default=region_options)

# Filter country options based on selected regions
available_countries = df[df["Region"].isin(selected_regions)]["Country"].unique()
selected_countries = st.sidebar.multiselect("Select Countries:", sorted(available_countries), default=available_countries)

# Apply Filters
filtered_df = df[
    (df["Country"].isin(selected_countries)) & 
    (df["Region"].isin(selected_regions)) 
]

# MAIN PAGE
st.title("World Happiness Rankings")

# Use st.expander to save vertical space for the long text
with st.expander("Show Context & App Description"):
    st.markdown("""
    The **World Happiness Report** is a landmark survey of global happiness. 
    This dashboard allows you to explore the 2017 report by specific factors like GDP, Health, and Trust.
    
    **How to Use:** Adjust the sidebar filters to update the highlights and the data table.
    """)

st.divider()

# 3. METRICS SECTION
st.subheader("Top Countries by Category")

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
else:
    # Dictionary mapping Display Name -> Column Name
    categories = {
        "Happiest Country": "Happiness Score",
        "Highest GDP": "Economy (GDP per Capita)",
        "Top Life Expectancy": "Health (Life Expectancy)",
        "Most Freedom": "Freedom",
        "Most Trust (Gov)": "Trust (Government Corruption)",
        "Most Generous": "Generosity"
    }
    
    # Use a loop to render columns efficiently
    cols = st.columns(3)
    for i, (label, col_name) in enumerate(categories.items()):
        # Select the column (0, 1, or 2) based on index
        target_col = cols[i % 3]
        
        top_row = filtered_df.sort_values(by=col_name, ascending=False).iloc[0]
        val = top_row[col_name]
        
        # Determine formatting (3 decimals for GDP/Generosity, 2 for others)
        fmt = ".3f" if "GDP" in label or "Generous" in label else ".2f"
        
        target_col.metric(label, top_row["Country"], f"{val:{fmt}}")

st.divider()

# 4. CLEANER DATASET EXPLORER
st.subheader("Dataset Explorer")

# Move Rank to the front more cleanly
if "Happiness Rank" in filtered_df.columns:
    cols = ["Happiness Rank"] + [c for c in filtered_df.columns if c != "Happiness Rank"]
    display_df = filtered_df[cols].sort_values("Happiness Rank")
else:
    display_df = filtered_df

st.dataframe(display_df, hide_index=True, use_container_width=True)