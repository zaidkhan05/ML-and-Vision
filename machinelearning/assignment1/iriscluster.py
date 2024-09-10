import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Import this for loading saved images

def load_kmtest():
    return pd.read_csv('machinelearning/assignment1/given/iris.csv', header=None, usecols=(0,1,2,3)).values

def zscore(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)
def kmeans(data, k, run, max_iters=100):
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    # print(centers)
    print(np.random.choice(data.shape[0]))
    for _ in range(max_iters):
        distanceMatrix = np.linalg.norm(data[:, None] - centers, axis=2)
        clusters = np.argmin(distanceMatrix, axis=1)
        new_centers = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        # print(new_centers)
        if np.all(new_centers == centers):
            print(centers)
            break
        centers = new_centers

    # Plot the clusters
    plt.figure()
    for i in range(k):
        plt.scatter(data[clusters == i, 2], data[clusters == i, 3], label=f'Cluster {i+1}')
    plt.scatter(centers[:, 2], centers[:, 3], color='black', marker='x', label='Centers')
    plt.title(f'iris Clustering with 3 clusters. Run {run}')
    plt.legend()

    # Save the figure
    save_path = f'machinelearning/assignment1/results/iris_clusters{run}.png'
    plt.savefig(save_path)
    plt.close()

    return save_path  # Return the path to the saved image

def unNormalizedKmeans():
    data = load_kmtest()
    plots = []
    plots.append('machinelearning/assignment1/results/original_iris_data.png')
    plots.append(kmeans(data, 3, 1))
    plots.append(kmeans(data, 3, 2))
    plots.append(kmeans(data, 3, 3))
    plots.append(kmeans(data, 3, 4))
    plots.append(kmeans(data, 3, 5))
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        img = mpimg.imread(plots[i])  # Load the saved image
        plt.imshow(img)
        plt.axis('off')  # Hide axes for better visualization
        # plt.title(f'K-Means Clustering ({i+2} clusters)')
    plt.tight_layout()
    plt.savefig('machinelearning/assignment1/results/iris_all_clusters.png')
    plt.show()
def originalClusters():
    data = pd.read_csv('machinelearning/assignment1/given/iris.csv', header=None).values
    clusters = data[5]

if __name__ == '__main__':
    data = load_kmtest()
    # print(data)
    plt.figure()
    plt.scatter(data[:, 2], data[:, 3])
    plt.title('Original Data')
    plt.savefig('machinelearning/assignment1/results/original_iris_data.png')
    plt.show()
    unNormalizedKmeans()
    # normalizedKmeans()