import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import seaborn as sns

# Step 1: Generate a synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Step 2: Initialize the Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)

# Step 3: Fit the model using the EM algorithm
gmm.fit(X)

# Step 4: Predict cluster assignments
y_gmm = gmm.predict(X)

# Step 5: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, s=40, cmap='viridis')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=300, c='red', marker='X')  # Cluster centers
plt.title('Clusters found by Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Print the parameters of the Gaussian Mixture Model
print('Means of the components:')
print(gmm.means_)
print('\nCovariances of the components:')
print(gmm.covariances_)
