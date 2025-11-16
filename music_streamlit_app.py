# =====================================
# Amazon Music Clustering Dashboard (Enhanced)
# =====================================

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Amazon Music Clustering", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("amazon_music_final_clusters.csv")
    summary = pd.read_csv("amazon_music_final_summary.csv")
    return df, summary

df, summary = load_data()

st.title("🎵 Amazon Music Clustering Dashboard")
st.markdown("Explore how songs are grouped into clusters based on audio features like energy, danceability, and tempo.")

# Sidebar controls
st.sidebar.header("Cluster Selection")
cluster_ids = sorted(df['cluster'].unique())
selected_cluster = st.sidebar.selectbox("Select a Cluster:", cluster_ids)

st.sidebar.header("Feature Filters")
features = ['danceability','energy','loudness','speechiness','acousticness',
            'instrumentalness','liveness','valence','tempo','duration_min']
selected_features = st.sidebar.multiselect("Select Features to View:", features, default=['danceability','energy','tempo'])

# Cluster summary
st.subheader(f"Cluster {selected_cluster} Summary")
col1, col2 = st.columns(2)

with col1:
    cluster_data = df[df['cluster'] == selected_cluster]
    st.metric("Number of Songs", len(cluster_data))
    st.metric("Average Energy", f"{cluster_data['energy'].mean():.2f}")
    st.metric("Average Danceability", f"{cluster_data['danceability'].mean():.2f}")
    st.metric("Average Tempo (BPM)", f"{cluster_data['tempo'].mean():.1f}")

with col2:
    st.write("**Feature Means for Cluster:**")
    st.dataframe(summary[summary['cluster'] == selected_cluster].T)

# Cluster distribution chart
st.subheader("Cluster Distribution")
cluster_counts = df['cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
fig_dist = px.bar(cluster_counts, x='Cluster', y='Count', text='Count',
                  title="Number of Songs per Cluster", color='Cluster')
st.plotly_chart(fig_dist, use_container_width=True)

# Heatmap comparison
st.subheader("Feature Comparison Across Clusters")
if 'cluster_label' in summary.columns:
    summary_display = summary.set_index('cluster_label')
else:
    summary_display = summary.set_index('cluster')

numeric_summary = summary_display.select_dtypes(include=['float64', 'int64'])
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_summary, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# PCA visualization
st.subheader("PCA Visualization of Clusters")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig_pca = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=df["cluster"].astype(str),
                     title="PCA 2D Projection of Songs by Cluster",
                     labels={"x":"Principal Component 1", "y":"Principal Component 2"},
                     opacity=0.7)
st.plotly_chart(fig_pca, use_container_width=True)

# Top songs
if 'name_song' in df.columns:
    st.subheader(f"Top 10 Songs in Cluster {selected_cluster}")
    cluster_data = df[df['cluster'] == selected_cluster]

    display_cols = ['name_song']
    if 'genre' in df.columns:
        display_cols.append('genre')
    display_cols += [col for col in selected_features if col in df.columns]

    sort_col = None
    for f in selected_features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            sort_col = f
            break
    if sort_col is None:
        sort_col = 'danceability' if 'danceability' in df.columns else df.columns[-1]

    top_songs = cluster_data.nlargest(10, sort_col)
    st.caption(f"Sorted by {sort_col}")
    st.dataframe(top_songs[display_cols].reset_index(drop=True))

    # Download option
    st.download_button("⬇️ Download Cluster Data",
                       cluster_data.to_csv(index=False),
                       f"cluster_{selected_cluster}_songs.csv",
                       "text/csv")

else:
    st.warning("Song names not found in the dataset. Ensure 'name_song' column exists.")
