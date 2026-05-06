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

# Modular Functions of the Model

@st.cache_data
def load_and_preprocess(df, selected_cols):
    """
    Cleans and scales data. Standard scaling is essential for distance-based 
    models like K-Means and Hierarchical clustering to ensure all features
    contribute equally to the distance calculations.
    """
    X_clean = df[selected_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    return X_clean, X_scaled

def get_pca_2d(X_scaled):
    """
    Reduces dataset dimensions to 2 Principal Components for visualization.
    This provides a consistent 2D coordinate system for plotting clusters.
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)

# Configuring the streamlit page
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")

st.title("Unsupervised Machine Learning Explorer")
st.write("Analyze hidden patterns and structures within your data using Clustering and PCA.")

# Importing and previewing the raw data
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("MLUnsupervisedApp/world_happiness.csv")
    except FileNotFoundError:
        st.error("Default data not found. Please upload a CSV.")
        st.stop()

# Data preview
with st.expander("Dataset Preview and Raw Data", expanded=True):
    st.write(df.head(10))
    st.info(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Configuring the models
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_cols = st.multiselect("Select numerical features for modeling:", num_cols, default=num_cols[:4])

if len(selected_cols) >= 2:
    X_raw, X_scaled = load_and_preprocess(df, selected_cols)
    X_pca = get_pca_2d(X_scaled)
    
    st.sidebar.header("2. Model Parameters")
    algo = st.sidebar.selectbox("Choose Algorithm", ["K-Means", "Hierarchical Clustering", "PCA"])

    # K Means section
    if algo == "K-Means":
        k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Cluster Visualization")
            fig_km, ax_km = plt.subplots()
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', ax=ax_km)
            ax_km.set_title(f"K-Means (k={k}) PCA Projection")
            st.pyplot(fig_km)
            
        with c2:
            st.subheader("Model Performance")
            st.metric("Silhouette Score", f"{silhouette_score(X_scaled, clusters):.3f}")
            inertias = [KMeans(n_clusters=i, n_init=10, random_state=42).fit(X_scaled).inertia_ for i in range(1, 11)]
            fig_el, ax_el = plt.subplots()
            ax_el.plot(range(1, 11), inertias, 'bx-')
            ax_el.set_title("Elbow Method Plot")
            st.pyplot(fig_el)

        with st.expander("Interpretation of K-Means Metrics and Benchmarks"):
            st.markdown("""
            **Silhouette Score Benchmarks:**
            * **0.71 - 1.0:** Excellent separation; a strong structure has been found.
            * **0.51 - 0.70:** Reasonable separation; the structure is likely valid.
            * **0.26 - 0.50:** Weak separation; the clusters are likely overlapping or noisy.
            * **Less than 0.25:** No substantial structure found.

            **Inertia and the Elbow Method:**
            Inertia measures the sum of squared distances of samples to their closest cluster center. While lower inertia is better, it always decreases as K increases. The **Elbow Point** is the specific value of K where the rate of decrease shifts significantly, representing the optimal balance between cluster tightness and model complexity.
            """)

    # Hierarchial section
    elif algo == "Hierarchical Clustering":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        link_type = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average"])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=link_type)
        clusters = model.fit_predict(X_scaled)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Dendrogram")
            Z = linkage(X_scaled, method=link_type)
            fig_d, ax_d = plt.subplots()
            dendrogram(Z, ax=ax_d, truncate_mode='lastp', p=12)
            ax_d.set_title("Hierarchical Relationship Tree")
            st.pyplot(fig_d)
        with c2:
            st.subheader("Cluster Projection")
            fig_h, ax_h = plt.subplots()
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='magma', ax=ax_h)
            st.pyplot(fig_h)

        with st.expander("Interpretation of Hierarchical Clustering"):
            st.markdown("""
            **Dendrogram Analysis:**
            The dendrogram visualizes the sequence of cluster merges. The vertical axis represents the **Euclidean distance** (or dissimilarity) between clusters. By observing where the vertical lines are longest, you can identify the most distinct groupings in the data.

            **Linkage Method Benchmarks:**
            * **Ward Linkage:** Generally the most effective for creating clusters of similar size by minimizing variance.
            * **Complete Linkage:** Better at finding clusters with clearly defined boundaries (sensitive to outliers).
            * **Average Linkage:** A compromise that balances cluster size and boundary definitions.
            """)

    # PCA section
    elif algo == "PCA":
        n_comp = st.sidebar.slider("PCA Components", 2, len(selected_cols), 2)
        pca_model = PCA(n_components=n_comp)
        pca_model.fit(X_scaled)
        
        st.subheader("Dimensionality Reduction Analysis")
        st.write(f"**Total Variance Explained:** {sum(pca_model.explained_variance_ratio_):.2%}")
        st.bar_chart(pca_model.explained_variance_ratio_)

        with st.expander("Interpretation of PCA Results and Variance"):
            st.markdown("""
            **Explained Variance Benchmarks:**
            In PCA, we aim to retain as much information as possible while reducing the number of features.
            * **80% to 90% Variance:** High-quality reduction; most of the original data's patterns are preserved.
            * **70% Variance:** Acceptable for visualization and general pattern recognition.
            * **Less than 50% Variance:** Significant data loss; the 2D projection may not be a reliable representation of the true data structure.

            **Component Significance:**
            The first component (PC1) always captures the maximum possible variance in the dataset. Subsequent components capture the remaining variance in descending order.
            """)

    # Export results
    st.divider()
    if algo != "PCA":
        export_df = X_raw.copy()
        export_df['Cluster_ID'] = clusters
        st.subheader("Final Clustered Data")
        st.dataframe(export_df.head(), use_container_width=True)
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Dataset", csv, "results.csv", "text/csv")

else:
    st.info("Please select at least two features in the multiselect box to begin modeling.")