import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import random

# K-Means Algorithm Implementation
class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centers = None

    def initialize_centers(self, data):
        return data[random.sample(range(data.shape[0]), self.k)]

    def assign_clusters(self, data, centers):
        clusters = []
        for point in data:
            distances = np.linalg.norm(point - centers, axis=1)
            cluster = np.argmin(distances)
            clusters.append(cluster)
        return np.array(clusters)

    def update_centers(self, data, clusters):
        centers = np.array([data[clusters == i].mean(axis=0) for i in range(self.k)])
        return centers

    def fit(self, data):
        self.centers = self.initialize_centers(data)
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(data, self.centers)
            new_centers = self.update_centers(data, clusters)
            if np.all(new_centers == self.centers):
                break
            self.centers = new_centers
        return clusters

    def predict(self, data):
        return self.assign_clusters(data, self.centers)

# Load and Normalize Datasets
def load_kmtest():
    # Replace with actual loading mechanism of kmtest data
    return pd.read_csv('machinelearning/assignment1/given/kmtest.csv').values

def load_iris_data():
    iris = load_iris()
    return iris.data

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Plotting Function
def plot_clusters(data, clusters, centers, k, title):
    plt.figure()
    for i in range(k):
        plt.scatter(data[clusters == i, 0], data[clusters == i, 1], label=f'Cluster {i+1}')
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centers')
    plt.title(title)
    plt.legend()
    # plt.savefig(f'machinelearning/assignmnet1/plots/kmeans_' + str(k) + '.png')
    # plt.show()
    plt.savefig('machinelearning\\assignment1\plots\kmtestcluster' + str(title) + '.png')
    #save them all onto one large image canvas

# Apply K-Means to kmtest Dataset
def kmeans_kmtest():
    data = load_kmtest()

    for k in [2, 3, 4, 5]:
        kmeans = KMeans(k)
        
        # Without normalization
        clusters = kmeans.fit(data)
        plot_clusters(data, clusters, kmeans.centers, k, f'K={k} without Normalization')

        # With normalization
        norm_data = normalize_data(data)
        norm_clusters = kmeans.fit(norm_data)
        plot_clusters(norm_data, norm_clusters, kmeans.centers, k, f'K={k} with Normalization')

# Apply K-Means to iris Dataset
def kmeans_iris():
    data = load_iris_data()
    
    # K = 3, without normalization
    kmeans = KMeans(3)
    best_clusters, worst_clusters, best_centers, worst_centers = None, None, None, None
    best_dist, worst_dist = float('inf'), float('-inf')
    
    for _ in range(5):
        clusters = kmeans.fit(data)
        center_distances = np.linalg.norm(kmeans.centers, axis=1).sum()
        
        if center_distances < best_dist:
            best_dist = center_distances
            best_clusters, best_centers = clusters, kmeans.centers
            
        if center_distances > worst_dist:
            worst_dist = center_distances
            worst_clusters, worst_centers = clusters, kmeans.centers

    plot_clusters(data[:, 2:4], best_clusters, best_centers[:, 2:4], 3, 'Best Result with K=3')
    plot_clusters(data[:, 2:4], worst_clusters, worst_centers[:, 2:4], 3, 'Worst Result with K=3')
    
    # Plot original data for comparison
    plt.figure()
    plt.scatter(data[:, 2], data[:, 3], c=load_iris().target)
    plt.title('Original Iris Data')
    plt.savefig('machinelearning/assignment1/plots/original_iris_data.png')
    plt.show()
    
    # Calculate distance between centers
    original_centers = np.array([[5.006, 3.428, 1.462, 0.246], [6.262, 2.872, 4.906, 1.676], [7.028, 3.148, 5.660, 2.064]])
    distances = np.linalg.norm(best_centers - original_centers, axis=1)
    print(f'Distances between best centers and original centers: {distances}')

# Main Execution
if __name__ == "__main__":
    kmeans_kmtest()
    kmeans_iris()
