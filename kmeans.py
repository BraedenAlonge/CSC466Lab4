import math
import sys
import pandas as pd
import random
import numpy as np


def load_data(filename):
    """Load data from csv file.
    1. First row = binary vector indicating which cols to use.
    2. Skip cols with 0 flag"""

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
    """Initialize k centroids from the dataset. we can either:
    A. Randomly select k points from the data.
    B. implement kmeans++ for better initial centroids."""
    centroids = random.sample(data, k)
    return centroids

def compute_dist(point1, point2):
    """Compute Euclid dist"""
    dist = 0
    if point1 is None or point2 is None:
        return 0
    for x,y in zip(point1, point2):
        dist += (float(x) - float(y))**2
    dist = math.sqrt(dist)
    return dist

def assign_clusters(data, centroids):
    """Assign each datapoint to the nearest centroid"""
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
    """Recompute centroids for each cluster as the mean of all points assigned
    to that cluster."""
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
        if len(points) > 0:
            distances = [compute_dist(point, centroids[i]) for point in points]
            max_d = max(distances)
            min_d = min(distances)
            avg_d = sum(distances) / len(distances)
        else:
            max_d = "n/a"
            min_d = "n/a"
            avg_d = "n/a"
        print(f"Cluster {i}:")
        print(f"  Center: {centroids[i]}")
        print(f"  Max distance to center: {max_d}")
        print(f"  Min distance to center: {min_d}")
        print(f"  Avg distance to center: {avg_d}")
        print(f"  SSE: {sse[i]}")
        print(f"  {len(points)} Points: {points}")


if __name__ == "__main__":
    main()