import sys
import math
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_point(line):
    return np.array([float(coords) for coords in line.split()])


def plot_clusters(points, centroids, iteration):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, centroid in enumerate(centroids):
        cluster_points = np.array(points[points == i])
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}')
        ax.scatter(*centroid, s=200, c='red', marker='x', label=f'Centroid {i + 1}')

    ax.set_title(f'K-Means Clustering - Iteration {iteration}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) != 7:
        print("Usage: ./main.py <inputFile> <K> <outputFolder> <maxIterations> <minShift>")
        sys.exit()

    input_file = sys.argv[1]
    k = int(sys.argv[2])
    output = sys.argv[3]
    max_iterations = int(sys.argv[4])
    min_shift = float(sys.argv[5])

    points = np.loadtxt(input_file)
    
    kmeans = KMeans(n_clusters=k, random_state=0)
    
    shift = math.inf
    iteration = 0

    while iteration < max_iterations and shift > min_shift:
        print("Iteration:", iteration)

        # Fit the KMeans model and get new centroids
        kmeans.fit(points)
        new_centroids = kmeans.cluster_centers_
        print("New centroids:", new_centroids)

        # Compute the shift
        shift = np.linalg.norm(new_centroids - kmeans.cluster_centers_)
        print("Shift:", shift)

        iteration += 1

        # Visualize clusters in 3D for the first three features (assuming at least 3 features)
        if points.shape[1] >= 3:
            plot_clusters(kmeans.labels_, new_centroids, iteration)

    np.savetxt(output, new_centroids)

if __name__ == "__main__":
    main()
