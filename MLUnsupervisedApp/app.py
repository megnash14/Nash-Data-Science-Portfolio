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

# Configure actual streamlit page
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")

st.title("Unsupervised Machine Learning Explorer")
st.markdown("""
Explore **Clustering** and **Dimensionality Reduction** interactively. 
Upload your own data or use the default World Happiness dataset.
""")

# Selecting what data source to use
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded: User Dataset")
else:
    try:
        # defaulting the site to the world happiness csv
        df = pd.read_csv("MLUnsupervisedApp/world_happiness.csv")
        st.sidebar.info("Using default: World Happiness Dataset")
    except FileNotFoundError:
        st.error("Error: 'world_happiness.csv' not found and no file uploaded.")
        st.stop()

# Display Preview
st.write("### Dataset Preview", df.head())

# Selecting the model features
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
    # Clean data
    X = df[selected_cols].dropna()
    
    if len(X) < 5:
        st.warning("Not enough data rows after removing missing values.")
        st.stop()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Configuring the algorithm
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
            
            fig, ax = plt.subplots()
            sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', ax=ax)
            ax.set_xlabel(f"Scaled {selected_cols[0]}")
            ax.set_ylabel(f"Scaled {selected_cols[1]}")
            st.pyplot(fig)
            
        with col2:
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

        with st.expander("Understanding K-Means Metrics"):
            st.markdown("""
            **What is K-Means?**
            K-Means is a centroid-based algorithm that partitions data into *k* non-overlapping subgroups. It tries to make the intra-cluster points as similar as possible while keeping the clusters as far apart as possible.

            **The Silhouette Score:**
            * Ranges from -1 to +1. 
            * A high score (closer to 1) indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
            
            **The Elbow Method (Inertia):**
            * Inertia measures how tightly packed the clusters are (the sum of squared distances to the nearest centroid).
            * We look for the 'elbow' point where the drop in inertia slows down; this usually indicates the optimal number of clusters.
            """)
            
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

        with st.expander("Understanding Hierarchical Clustering"):
            st.markdown("""
            **What is Hierarchical Clustering?**
            Unlike K-Means, this method builds a multi-level hierarchy of clusters. We use **Agglomerative** clustering, which is a "bottom-up" approach where each data point starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

            **The Dendrogram:**
            * A tree-like diagram that records the sequences of merges or splits.
            * The vertical axis represents the distance or dissimilarity between clusters. The higher the horizontal link, the more dissimilar the clusters being joined are.

            **Linkage Methods:**
            * **Ward:** Minimizes the variance of clusters being merged.
            * **Complete:** Uses the maximum distance between points of two clusters.
            * **Average:** Uses the average distance between all points in two clusters.
            """)

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

        with st.expander("Understanding Principal Component Analysis (PCA)"):
            st.markdown("""
            **What is PCA?**
            PCA is a dimensionality reduction technique. It transforms a large set of variables into a smaller one that still contains most of the information (variance) in the large set.

            **Explained Variance Ratio:**
            * This tells you how much information (mathematically, the variance) is captured by each Principal Component. 
            * If PC1 and PC2 cover 80% of the variance, you can effectively visualize your entire dataset in 2D without losing significant patterns.

            **2D Projection:**
            * The scatter plot shows your data transformed into a new coordinate system. This helps identify clusters or trends that were hidden in the original high-dimensional numerical data.
            """)

else:
    st.warning("Please select at least two numerical columns to begin.")