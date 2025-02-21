import math
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    """Load data from csv file.
    1. First row = binary vector indicating which cols to use.
    2. Skip cols with 0 flag
    Return list of data points."""

    with open(filename, "r") as f:
        flags = f.readline().strip().split()
        flags = [int(x) for x in flags]

    data = pd.read_csv(filename, delimiter="\t", skiprows=1)
    kept_cols = []
    for col, fl in zip(data.columns, flags):
        if fl == 0:
            continue
        else:
            kept_cols.append(col)

    data = data[kept_cols]
    return data.values.tolist()

def initialize_centroids(data, k):
    """Initialize k centroids from the dataset. We can either:
    A. Randomly select k points from the data.
    B. implement kmeans++ for better initial centroids.
    Returns:
        list of centroids
    """
    centroids = random.sample(data, k)
    return centroids

def compute_dist(point1, point2):
    """ Compute Euclid dist"""
    dist = 0
    for x,y in zip(point1, point2):
        dist += (float(x) - float(y))**2
    dist = math.sqrt(dist)
    return dist

def assign_clusters(data, centroids):
    """Assign each datapoint to the nearest centroid.
    Returns:
        clusters: dictionary --> key = centroid index, value = list of points in cluster
    """
    # inititalize cluster dict
    clusters = {i: [] for i in range(len(centroids))}  # ex: {0: [], 1: [], 2: []...}

    # for each data point, compute the distance to each centroid.
    # Then, find the closest centroid and add that point to that cluster
    # containing said centroid.
    for point in data:
        # find index of closest centroid
        distances = [compute_dist(point, centroid) for centroid in centroids]
        closest_centroid = distances.index(min(distances))
        clusters[closest_centroid].append(point)

    return clusters

def recompute_centroids(clusters):
    # TODO: handle empty clusters
    """Recompute centroids for each cluster as the mean of all points assigned
    to that cluster.
    Returns a list of new centroids"""
    new_centroids = []
    for index, points in clusters.items():
        if points:
            centroid = np.mean(points, axis=0)
        else:
            centroid = None
        new_centroids.append(centroid)
    return new_centroids

def compute_SSE(clusters, centroids):
    sse = {}
    for i, points in clusters.items():
        error = 0.0
        if centroids[i] is not None:
            for point in points:
                error += compute_dist(point, centroids[i]) ** 2
        sse[i] = error
    return sse


def main():
    # Parse args
    args = len(sys.argv)
    if args < 3: # Add more args if needed and update this
        print("Usage: python kmeans.py <Filename> <k>")
        exit()
    filename = sys.argv[1]
    k = int(sys.argv[2])

    data = load_data(filename)
    centroids = initialize_centroids(data, k)
    clusters = {}
    # Main k-means loop
    max_iterations= 100
    for iteration in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = recompute_centroids(clusters)
        centroids = new_centroids
    sse = compute_SSE(clusters, centroids)

    for i, points in clusters.items():
        print(f"Cluster {i}:")
        print(f"  Center: {centroids[i]}")
        print(f"  Points: {points}")
        print(f"  SSE: {sse[i]}")


    # Visualize the clusters
    visualize_clusters(clusters, centroids, title="K-means Clusters")



def visualize_clusters(clusters, centroids=None, title='Cluster Visualization'):
    """
    Visualizes clusters using a 2D scatter plot.

    Parameters:
      clusters: dict
          Dictionary where keys are cluster IDs and values are lists of data points.
      centroids: list (optional)
          List of centroids (each a list of values) to be plotted.
      title: str
          Title of the plot.

    Note:
      Only the first two dimensions of each data point are plotted.
    """
    # Define a set of colors (extend or modify as needed)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    plt.figure(figsize=(8, 6))

    # Plot each cluster's points
    for cluster_id, points in clusters.items():
        if len(points) == 0:
            continue  # Skip empty clusters
        points = np.array(points)
        # Plot using only the first two dimensions
        plt.scatter(points[:, 0], points[:, 1],
                    color=colors[cluster_id % len(colors)],
                    label=f"Cluster {cluster_id}")

    # Optionally, plot the centroids if provided
    if centroids is not None:
        centroids = np.array(centroids)
        if centroids.shape[1] >= 2:
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        color='black', marker='x', s=100,
                        label='Centroids')

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()