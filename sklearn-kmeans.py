#!/usr/bin/env python3
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from sklearn.cluster import KMeans

import csv
import numpy as np


def load_dataset(filename):
    data = []
    with open(filename, 'r', newline='') as f:
        # Read a sample to detect the delimiter
        sample = f.read(1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # fallback to excel dialect if sniffer fails
        reader = csv.reader(f, dialect)

        # Read header (mask) and convert each entry to a boolean
        mask = next(reader)
        mask = [int(x.strip()) == 1 for x in mask]

        for row in reader:
            if not row:
                continue
            # Select only the columns where mask is True, stripping whitespace
            selected = [float(row[i].strip()) for i in range(len(row)) if mask[i]]
            data.append(selected)
    return np.array(data)


def compute_cluster_stats(data, labels, centers):
    stats = {}
    for cluster in np.unique(labels):
        cluster_points = data[labels == cluster]
        center = centers[cluster]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        sse = np.sum(distances ** 2)
        stats[cluster] = {
            'num_points': len(cluster_points),
            'center': center,
            'max_distance': np.max(distances) if len(distances) > 0 else 0,
            'min_distance': np.min(distances) if len(distances) > 0 else 0,
            'avg_distance': np.mean(distances) if len(distances) > 0 else 0,
            'sse': sse,
            'points': cluster_points
        }
    return stats


def visualize_clusters(clusters, centroids=None, title='Cluster Visualization'):
    """
    Visualizes clusters using a scatter plot.

    Parameters:
      clusters: dict
          Dictionary where keys are cluster IDs and values are lists of data points.
      centroids: list (optional)
          List of centroids (each a list of values) to be plotted.
      title: str
          Title of the plot.

    Note:
      If data points have 3 or more dimensions, a 3D scatter plot is used;
      otherwise, only the first two dimensions are plotted.
    """
    # Define a set of colors (extend or modify as needed)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Determine the dimensionality from the first non-empty cluster.
    sample_point = None
    for points in clusters.values():
        if len(points) > 0:
            sample_point = points[0]
            break
    if sample_point is None:
        print("No data points to plot.")
        return

    dim = len(sample_point)

    if dim >= 3:
        # Create a 3D scatter plot.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cluster_id, points in clusters.items():
            if len(points) == 0:
                continue
            pts = np.array(points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       color=colors[cluster_id % len(colors)],
                       label=f"Cluster {cluster_id}")
        if centroids is not None:
            centroids = np.array(centroids)
            if centroids.shape[1] >= 3:
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                           color='black', marker='x', s=100,
                           label='Centroids')
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.legend()
        plt.show()
    else:
        # Create a 2D scatter plot using only the first two dimensions.
        plt.figure(figsize=(8, 6))
        for cluster_id, points in clusters.items():
            if len(points) == 0:
                continue
            pts = np.array(points)
            plt.scatter(pts[:, 0], pts[:, 1],
                        color=colors[cluster_id % len(colors)],
                        label=f"Cluster {cluster_id}")
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


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 kmeans.py <Filename> <k>")
        sys.exit(1)
    filename = sys.argv[1]
    k = int(sys.argv[2])
    data = load_dataset(filename)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    stats = compute_cluster_stats(data, labels, centers)
    for cluster, info in stats.items():
        print(f"Cluster {cluster}:")
        print("  Center:", info['center'])
        print("  Number of points:", info['num_points'])
        print("  Max distance to center:", info['max_distance'])
        print("  Min distance to center:", info['min_distance'])
        print("  Avg distance to center:", info['avg_distance'])
        print("  SSE:", info['sse'])
        print("  Points:")
        for point in info['points']:
            print("   ", point)
        print()

    # Prepare clusters dictionary for visualization.
    clusters_dict = {cluster: info['points'] for cluster, info in stats.items()}
    # Visualize clusters along with centroids.
    visualize_clusters(clusters_dict, centroids=centers, title="K-Means Clustering")


if __name__ == "__main__":
    main()
