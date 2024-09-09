import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Import this for loading saved images

def load_kmtest():
    return pd.read_csv('machinelearning/assignment1/given/kmtest.csv', header=None).values

def zscore(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

def kmeans(data, k, normType, max_iters=100):
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distanceMatrix = np.linalg.norm(data[:, None] - centers, axis=2)
        clusters = np.argmin(distanceMatrix, axis=1)
        new_centers = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        if np.all(new_centers == centers):
            break
        centers = new_centers

    # Plot the clusters
    plt.figure()
    for i in range(k):
        plt.scatter(data[clusters == i, 0], data[clusters == i, 1], label=f'Cluster {i+1}')
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', label='Centers')
    plt.title(f'{normType} K-Means Clustering with {k} clusters')
    plt.legend()

    # Save the figure
    save_path = f'machinelearning/assignment1/results/kmeans_{k}_clusters_{normType}.png'
    plt.savefig(save_path)
    plt.close()

    return save_path  # Return the path to the saved image

def unNormalizedKmeans():
    data = load_kmtest()
    plots = []
    plots.append(kmeans(data, 2, normType='Unnormalized'))
    plots.append(kmeans(data, 3, normType='Unnormalized'))
    plots.append(kmeans(data, 4, normType='Unnormalized'))
    plots.append(kmeans(data, 5, normType='Unnormalized'))
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        img = mpimg.imread(plots[i])  # Load the saved image
        plt.imshow(img)
        plt.axis('off')  # Hide axes for better visualization
        # plt.title(f'K-Means Clustering ({i+2} clusters)')
    plt.tight_layout()
    plt.savefig('machinelearning/assignment1/results/kmeans_all_clusters.png')
    plt.show()

def normalizedKmeans():
    data = load_kmtest()
    norm_data = zscore(data)
    plots = []
    plots.append(kmeans(norm_data, 2, normType='Normalized'))
    plots.append(kmeans(norm_data, 3, normType='Normalized'))
    plots.append(kmeans(norm_data, 4, normType='Normalized'))
    plots.append(kmeans(norm_data, 5, normType='Normalized'))
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        img = mpimg.imread(plots[i])  # Load the saved image
        plt.imshow(img)
        plt.axis('off')  # Hide axes for better visualization
        # plt.title(f'Normalized K-Means Clustering ({i+2} clusters)')
    plt.tight_layout()
    plt.savefig('machinelearning/assignment1/results/kmeans_all_clusters_zscore.png')

    plt.show()
if __name__ == '__main__':
    data = load_kmtest()
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.title('Original Data')
    plt.savefig('machinelearning/assignment1/results/original_data.png')
    plt.show()
    unNormalizedKmeans()
    normalizedKmeans()