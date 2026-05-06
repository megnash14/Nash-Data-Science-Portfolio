import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Page Configuration
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")

st.title("Unsupervised Machine Learning Explorer")
st.markdown("""
Explore **Clustering** and **Dimensionality Reduction** interactively. 
Upload your own data or use the default World Happiness dataset.
""")

# --- 1. DATA SOURCE SELECTION ---
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded: User Dataset")
else:
    try:
        # Defaults to the world_happiness.csv in your folder
        df = pd.read_csv("world_happiness.csv")
        st.sidebar.info("Using default: World Happiness Dataset")
    except FileNotFoundError:
        st.error("Error: 'world_happiness.csv' not found and no file uploaded.")
        st.stop()

# Display Preview
st.write("### Dataset Preview", df.head())

# --- 2. FEATURE SELECTION ---
# Get numerical columns and handle potential missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(num_cols) < 2:
    st.error("Dataset needs at least 2 numerical columns for unsupervised learning.")
    st.stop()

selected_cols = st.multiselect(
    "Select numerical features for modeling", 
    num_cols, 
    default=num_cols[:min(4, len(num_cols))]
)

if len(selected_cols) >= 2:
    # Clean data: Remove rows with missing values for the selected features
    X = df[selected_cols].dropna()
    
    if len(X) < 5:
        st.warning("Not enough data rows after removing missing values.")
        st.stop()

    # Scaling is crucial for distance-based ML
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 3. ALGORITHM CONFIGURATION ---
    st.sidebar.header("2. Configure Model")
    algo = st.sidebar.selectbox(
        "Select Algorithm", 
        ["K-Means", "Hierarchical Clustering", "PCA"]
    )
    
    if algo == "K-Means":
        k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### K-Means Results (k={k})")
            score = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette Score", f"{score:.3f}")
            
            # Scatter Plot of first two selected features
            fig, ax = plt.subplots()
            sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', ax=ax)
            ax.set_xlabel(f"Scaled {selected_cols[0]}")
            ax.set_ylabel(f"Scaled {selected_cols[1]}")
            st.pyplot(fig)
            
        with col2:
            # Elbow Method Plot
            distortions = []
            K_range = range(1, 11)
            for i in K_range:
                km = KMeans(n_clusters=i, random_state=42, n_init=10)
                km.fit(X_scaled)
                distortions.append(km.inertia_)
            
            fig2, ax2 = plt.subplots()
            ax2.plot(K_range, distortions, 'bx-')
            ax2.set_xlabel('Number of clusters (k)')
            ax2.set_ylabel('Inertia')
            ax2.set_title('Elbow Method')
            st.pyplot(fig2)
            
    elif algo == "Hierarchical Clustering":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        linkage_type = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average"])
        
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
        clusters = model.fit_predict(X_scaled)
        
        st.write("### Dendrogram")
        Z = linkage(X_scaled, method=linkage_type)
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        st.pyplot(fig)

    elif algo == "PCA":
        max_pcs = min(len(selected_cols), 10)
        n_components = st.sidebar.slider("PCA Components", 2, max_pcs, 2)
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### PCA - Explained Variance")
            exp_var = pca.explained_variance_ratio_
            st.bar_chart(exp_var)
            st.write(f"Total Explained Variance: {sum(exp_var):.2%}")
            
        with col2:
            st.write("### 2D Projection")
            pca_df = pd.DataFrame(data=components[:, :2], columns=['PC1', 'PC2'])
            fig, ax = plt.subplots()
            sns.scatterplot(data=pca_df, x='PC1', y='PC2')
            st.pyplot(fig)

else:
    st.warning("Please select at least two numerical columns to begin.")