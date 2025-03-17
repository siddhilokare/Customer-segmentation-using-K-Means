import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set relative file paths
file_path = "data/Mall_Customers.csv"
output_path = "data/clustered_customers.csv"

# Load data
df = pd.read_csv(file_path)

# Display basic info about the dataset
print("Dataset Overview:\n", df.head())

# Selecting relevant features: Annual Income & Spending Score
X = df.iloc[:, [3, 4]].values  # Taking 'Annual Income (k$)' & 'Spending Score (1-100)'

# Standardizing the data for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)  # Testing k values from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choosing the optimal k (let’s assume from graph k=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df.iloc[:, 3], y=df.iloc[:, 4], hue=df['Cluster'], palette="viridis", s=100, edgecolor="black")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend(title="Cluster")
plt.show()

# Save clustered data
df.to_csv(output_path, index=False)
print(f"Clustered data saved at: {output_path}")
