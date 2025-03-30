import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import os

print("Starting FCM model training with 4 clusters...")

# Load the dataset
try:
    dataset = pd.read_csv('E-commerce Customer Behavior - Sheet1.csv')
    print(f"Dataset loaded successfully. Shape: {dataset.shape}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit(1)

# Features for clustering - using numerical features only
features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']

# Check if features exist in the dataset
missing_features = [f for f in features if f not in dataset.columns]
if missing_features:
    print(f"Warning: Missing features in dataset: {missing_features}")
    # Use whatever numerical columns are available
    numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    features = numerical_cols
    print(f"Using alternative features: {features}")

# Extract features from dataset
X = dataset[features].values
print(f"Extracted features for clustering: {features}")

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaled successfully")

# Transpose data for skfuzzy (features as rows, samples as columns)
X_scaled_transposed = X_scaled.T

# Set number of clusters
n_clusters = 4
print(f"Training FCM model with {n_clusters} clusters...")

# Use fuzziness parameter of 2.0 (standard for FCM)
m = 2.0

# Train the model
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_scaled_transposed, n_clusters, m, error=0.005, maxiter=1000, init=None
)

print(f"FCM model trained. FPC: {fpc} (higher is better, max is 1.0)")

# Get cluster labels
cluster_labels = np.argmax(u, axis=0)

# Add membership degrees to the dataset
for i in range(n_clusters):
    dataset[f'Membership_Cluster_{i}'] = u[i]

# Add cluster labels to the dataset
dataset['Cluster'] = cluster_labels

# Calculate the cluster sizes
cluster_sizes = dataset['Cluster'].value_counts().sort_index()
print("Cluster sizes:")
for cluster, size in cluster_sizes.items():
    print(f"  Cluster {cluster}: {size} samples ({size/len(dataset)*100:.1f}%)")

# Create the model dictionary
model = {
    'centers': cntr,
    'u': u,
    'features': features,
    'scaler': scaler,
    'n_clusters': n_clusters,
    'm': m,
    'cluster_labels': cluster_labels,
    'fpc': fpc
}

# Save the model
try:
    with open('fcm_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully as fcm_model.pkl")
except Exception as e:
    print(f"Error saving model: {str(e)}")

# Generate visualization of the membership distribution
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    sns.kdeplot(dataset[f'Membership_Cluster_{i}'], label=f'Cluster {i}')
plt.xlabel('Membership Degree')
plt.ylabel('Density')
plt.title('Distribution of Membership Values across Clusters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('membership_distribution.png')
print("Membership distribution visualization saved")

# Generate cluster characteristics visualization
# Compute the mean of each feature for each cluster
cluster_means = dataset.groupby('Cluster')[features].mean()

# Normalize the cluster means for better visualization
cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

# Create a heatmap of the normalized cluster means
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means_normalized, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Normalized Feature Values by Cluster')
plt.savefig('cluster_characteristics.png')
print("Cluster characteristics visualization saved")

print("FCM model training and visualization complete!") 