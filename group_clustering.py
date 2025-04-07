import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap
import transform


# Performs K-means-clustering on the swimmers metrics derived in transform.py
def cluster_swimmers(feature_df, min_clusters=3, max_clusters=8, visualize=True):
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
    
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20) 
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(sil_score)
        print(f"k={k}: silhouette score = {sil_score:.3f}")
    
    # Get optimal k via the silhouette score
    optimal_k = range(min_clusters, max_clusters + 1)[np.argmax(silhouette_scores)]
    print(f"\nSelected k={optimal_k} with silhouette score = {max(silhouette_scores):.3f}")
    
    if visualize:
        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(min_clusters, max_clusters + 1), inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        # Plot silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'rx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Perform final clustering with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
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
        print(f"\nCluster {i+1} ({cluster_size} swimmers):")
        
        # Print top characteristics of this cluster
        cluster_center = cluster_centers.iloc[i]
        top_strengths = cluster_center.nlargest(3)
        print("Top strengths:")
        for strength, value in top_strengths.items():
            print(f"-Average {strength}: {value:.2f}")
    
    # Calculate and print feature importances
    print("\nFeature Importances:")
    
    # Calculate the overall variance of each feature
    feature_variances = np.var(scaled_features, axis=0)
    
    # Calculate the between-cluster variance for each feature
    cluster_variances = np.zeros(scaled_features.shape[1])
    for i in range(optimal_k):
        mask = cluster_labels == i
        cluster_mean = np.mean(scaled_features[mask], axis=0)
        cluster_size = np.sum(mask)
        cluster_variances += cluster_size * (cluster_mean ** 2)
    cluster_variances /= len(scaled_features)
    
    # Calculate feature importance as ratio of between-cluster to total variance
    feature_importance = cluster_variances / feature_variances
    
    # Create and sort feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Print all features and their importance scores
    print("\nFeature Importance Rankings:")
    for _, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.3f}")
    
    print("\n")
    
    # Re-index clusters
    results['Cluster'] = results['Cluster'] + 1
    
    if visualize:
        # Create PCA visualization
        visualize_pca(scaled_features, cluster_labels, optimal_k)
        
        # Create UMAP visualization
        visualize_umap(scaled_features, cluster_labels, optimal_k)

    return results, final_kmeans, importance_df


def visualize_pca(scaled_features, cluster_labels, n_clusters):
    """
    Create a PCA visualization of the clusters.
    """
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot each cluster with a different color
    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            label=f'Cluster {i+1}',
            alpha=0.7
        )
    
    plt.title('PCA Visualization of Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print the total explained variance
    total_var = sum(pca.explained_variance_ratio_)
    print(f"Total variance explained by 2 principal components: {total_var:.2%}")


def visualize_umap(scaled_features, cluster_labels, n_clusters):
    """
    Create a UMAP visualization of the clusters.
    """
    # Apply UMAP 
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(scaled_features)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot each cluster with a different color
    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(
            umap_result[mask, 0], 
            umap_result[mask, 1], 
            label=f'Cluster {i+1}',
            alpha=0.7
        )
    
    plt.title('UMAP Visualization of Clusters')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Run the transform file to grab and format the data
feature_df = transform.run_metrics()

# Run the model on the current data and print the results
results, model, importance_df = cluster_swimmers(feature_df)
print(results)
