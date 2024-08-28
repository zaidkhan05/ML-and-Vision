# Clustering with K-means algorithm for kmtest dataset.
# a. Without normalization, cluster the dataset by choosing the K value as 2, 3, 4, 5. Plot
# results  for  each  K  values  by  showing  each  cluster  with  different  color  and  cluster
# centers.
# b. With normalization,  cluster the dataset by choosing the  K values as 2, 3, 4, 5.  You
# should create clustering centers and clustering input for normalized data. Use z-score
# normalization  as  the  normalization  method.  First  normalize  the  data  and  apply
# clustering  on  the  normalized  data.  Plot  results  for  each  K  values  by  showing  each
# cluster with different color and cluster centers.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('machinelearning\\assignment1\given\kmtest.csv')
X = data.values
def labelData():
    
# Clustering without normalization
def clusterWithoutNormalization():
    # K values
    K = [2, 3, 4, 5]
    for k in K:
        # K-means clustering
        # kmeans = KMeans(n_clusters=k)
        # kmeans.fit(X)
        # y_kmeans = kmeans.predict(X)
        # centers = kmeans.cluster_centers_
        # # Plot the clusters
        # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.title('Clustering without normalization for K = ' + str(k))
        #save the plots seperately with names based on K values
        plt.savefig('machinelearning\\assignment1\plots\kmtestcluster' + str(k) + '.png')
                # plt.show()


# Clustering with normalization
def clusterWithNormalization():
    # Normalization
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    # K values
    K = [2, 3, 4, 5]
    for k in K:
        # K-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_normalized)
        y_kmeans = kmeans.predict(X_normalized)
        centers = kmeans.cluster_centers_
        # Plot the clusters
        plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y_kmeans, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.title('Clustering with normalization for K = ' + str(k))
        # plt.show()
        #save the plots seperately with names based on K values
        plt.savefig('machinelearning\\assignment1\plots\kmtestnormalcluster' + str(k) + '.png')

# Main
#pip install scikit-learn
clusterWithoutNormalization()
clusterWithNormalization()