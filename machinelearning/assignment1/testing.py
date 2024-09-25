import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Function to load the iris data (excluding species/labels)
def loadIrisData():
    return pd.read_csv('machinelearning/assignment1/given/iris.csv', header=None, usecols=(0, 1, 2, 3)).values


# K-means function to cluster the data and return the centers
def kmeans(data, k, run, max_iters=100):
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
        plt.scatter(data[clusters == i, 2], data[clusters == i, 3], label=f'Cluster {i + 1}')
    plt.scatter(centers[:, 2], centers[:, 3], color='black', marker='x', label='Centers')
    plt.title(f'Iris Clustering with 3 clusters. Run {run}')
    plt.legend()

    # Save the figure
    save_path = f'machinelearning/assignment1/results/iris_clusters{run}.png'
    plt.savefig(save_path)
    plt.close()

    return centers  # Return the final centers for comparison


# Function to find original centers based on species
def findOriginalCenters():
    # Load the dataset (including species/label column)
    data = pd.read_csv('machinelearning/assignment1/given/iris.csv', header=None).values

    # Extract features and species labels
    features = data[:, :4]  # First four columns are features
    species = data[:, 4]  # The last column is species

    # Find unique species (classes)
    unique_species = np.unique(species)

    # Calculate the mean for each species (original centers)
    original_centers = np.array([features[species == sp].mean(axis=0) for sp in unique_species])

    # Print out the original centers for each species
    for i, sp in enumerate(unique_species):
        print(f"Original center for species '{sp}': {original_centers[i]}")

    return original_centers


# Function to compare original centers with the best K-means centers
def compareCenters(original_centers, kmeans_centers):
    print("Comparing Original Centers with K-Means Centers:\n")

    for i in range(len(original_centers)):
        print(f"Original center for species {i + 1}: {original_centers[i]}")
        print(f"Best K-Means center {i + 1}: {kmeans_centers[i]}")
        distance = np.linalg.norm(original_centers[i] - kmeans_centers[i])
        print(f"Euclidean distance between original and K-Means center {i + 1}: {distance}\n")


# Function to plot the original clusters (by species)
def originalClusters():
    # Load the dataset
    data = pd.read_csv('machinelearning/assignment1/given/iris.csv', header=None).values

    # Map species to numerical values for coloring
    species = data[:, 4]
    species_unique = list(set(species))  # Unique species
    species_colors = {species_unique[i]: i for i in range(len(species_unique))}  # Map species to numbers

    # Assign color based on species
    colors = [species_colors[s] for s in species]

    # Create the scatter plot
    plt.figure()
    plt.scatter(data[:, 2], data[:, 3], c=colors, cmap='viridis')
    plt.title('Original Iris Data')

    # Save and show the plot
    plt.savefig('machinelearning/assignment1/results/original_iris_clustered_data.png')
    plt.show()


# Function to run multiple k-means and plot the results
def unNormalizedKmeans():
    data = loadIrisData()
    plots = []
    plots.append('machinelearning/assignment1/results/original_iris_clustered_data.png')
    for run in range(1, 6):
        plots.append(kmeans(data, 3, run))
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        img = mpimg.imread(plots[i])  # Load the saved image
        plt.imshow(img)
        plt.axis('off')  # Hide axes for better visualization
    plt.tight_layout()
    plt.savefig('machinelearning/assignment1/results/iris_all_clusters.png')
    plt.show()


# Main process
if __name__ == '__main__':
    # Find original centers
    original_centers = findOriginalCenters()

    # Run K-Means multiple times and pick the best result (based on total distance)
    data = loadIrisData()
    best_centers = None
    min_distance = float('inf')
    best_run = None

    for run in range(1, 6):
        centers = kmeans(data, 3, run)
        total_distance = np.sum(
            [np.linalg.norm(center - orig_center) for center, orig_center in zip(centers, original_centers)])

        if total_distance < min_distance:
            min_distance = total_distance
            best_centers = centers
            best_run = run

    # Compare original centers with the best K-Means result
    print(f"Best run: {best_run} with total distance: {min_distance}")
    compareCenters(original_centers, best_centers)

    # Continue with original plotting and visualization...
    originalClusters()
    unNormalizedKmeans()
