import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import transform


# Performs K-means-clustering on the swimmers metrics derived in transform.py
def cluster_swimmers(feature_df, max_clusters=10):
    # Seperate 'Name' column
    if 'Name' in feature_df.columns:
        names = feature_df['Name']
        features = feature_df.drop('Name', axis=1)
    else:
        names = feature_df.index
        features = feature_df.copy()
    
    # Use mean in place of missing falues
    features = features.fillna(features.mean())
    
    # Scale all features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Find the optimal number of clusters using both elbow method and silhouette score
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.tight_layout()
    plt.show()
    
    # Get optimal k via the silhouette score
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2 # optimal for this data is six clusters
    
    # Perform final clustering with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(scaled_features)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Name': names,
        'Cluster': cluster_labels
    })
    
    # Add cluster centers 
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(final_kmeans.cluster_centers_),
        columns=features.columns
    )
    
    # Print the characteristics of each cluster
    print("\nClusters:")
    for i in range(optimal_k):
        cluster_size = (cluster_labels == i).sum()
        print(f"\nCluster {i} ({cluster_size} swimmers):")
        
        # Print top characteristics of this cluster
        cluster_center = cluster_centers.iloc[i]
        top_strengths = cluster_center.nlargest(3)
        print("Top strengths:")
        for strength, value in top_strengths.items():
            print(f"-Average {strength}: {value:.2f}")
    
    return results, final_kmeans

# Run the transform file to grab and format the data
feature_df = transform.run()

# Run the model on the current data and print the results
results, model = cluster_swimmers(feature_df)
print(results)
