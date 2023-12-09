import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("test.csv")

# Select relevant features for clustering
selected_features = ['Age', 'Flight Distance', 'Ease of Online booking', 'Departure/Arrival time convenient',
                      'Seat comfort']


X = df[selected_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels_kmeans = kmeans.fit_predict(X_scaled)

#  Euclidean distance
agglomerative_euclidean = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster_labels_agglomerative_euclidean = agglomerative_euclidean.fit_predict(X_scaled)

# Cosine distance
agglomerative_cosine = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='complete')
cluster_labels_agglomerative_cosine = agglomerative_cosine.fit_predict(X_scaled)

# 
df['Agglomerative_Cluster_Euclidean'] = cluster_labels_agglomerative_euclidean
df['Agglomerative_Cluster_Cosine'] = cluster_labels_agglomerative_cosine
df['KMeans_Cluster'] = cluster_labels_kmeans

# 
print("Number of data points per cluster (K-Means):")
print(pd.Series(cluster_labels_kmeans).value_counts())
print("\nNumber of data points per cluster (Agglomerative - Euclidean):")
print(pd.Series(cluster_labels_agglomerative_euclidean).value_counts())
print("\nNumber of data points per cluster (Agglomerative - Cosine):")
print(pd.Series(cluster_labels_agglomerative_cosine).value_counts())

# Save the dataframe with cluster labels to a new CSV file
df.to_csv("clustered_data.csv", index=False)

# 
plt.figure(figsize=(15, 5))

# 
plt.subplot(1, 3, 1)
sns.scatterplot(data=df, x='Age', y='Flight Distance', hue='KMeans_Cluster', palette='viridis')
plt.title('K-Means Clustering')


plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Age', y='Flight Distance', hue='Agglomerative_Cluster_Euclidean', palette='viridis')
plt.title('Agglomerative Clustering (Euclidean)')


plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Age', y='Flight Distance', hue='Agglomerative_Cluster_Cosine', palette='viridis')
plt.title('Agglomerative Clustering (Cosine)')

plt.tight_layout()
plt.show()
