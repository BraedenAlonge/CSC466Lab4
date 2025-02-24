#!/usr/bin/env python3
import sys
import csv
import numpy as np
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    data = []
    with open(filename, 'r') as f:
        # Detect the CSV dialect (delimiter, etc.)
        sample = f.read(1024)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.reader(f, dialect)
        # Read header mask and convert each entry to a boolean
        mask = next(reader)
        mask = [int(x.strip()) == 1 for x in mask]
        for row in reader:
            if not row:
                continue
            selected = [float(row[i].strip()) for i in range(len(row)) if mask[i]]
            data.append(selected)
    return np.array(data)

def compute_cluster_stats(data, labels):
    stats = {}
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Skip outliers
        # Select points in the cluster using boolean indexing
        cluster_points = data[labels == label]
        center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        sse = np.sum(distances ** 2)
        stats[label] = {
            'num_points': len(cluster_points),
            'center': center.tolist(),
            'max_distance': float(np.max(distances)) if len(distances) > 0 else 0,
            'min_distance': float(np.min(distances)) if len(distances) > 0 else 0,
            'avg_distance': float(np.mean(distances)) if len(distances) > 0 else 0,
            'sse': float(sse),
            'points': cluster_points.tolist()
        }
    return stats

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 dbscan.py <Filename> <epsilon> <NumPoints>")
        sys.exit(1)
    filename = sys.argv[1]
    epsilon = float(sys.argv[2])
    min_samples = int(sys.argv[3])
    data = load_dataset(filename)

    db = DBSCAN(eps=epsilon, min_samples=min_samples)
    db.fit(data)
    labels = db.labels_

    stats = compute_cluster_stats(data, labels)
    print("DBSCAN Clustering Results:")
    for label, info in stats.items():
        print(f"Cluster {label}:")
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

    # Report outliers (points with label -1)
    outliers = data[labels == -1]
    print("Outliers:")
    print("  Number of outliers:", len(outliers))
    if len(data) > 0:
        print("  Percentage of outliers: {:.2f}%".format(100.0 * len(outliers) / len(data)))
    print("  Outlier points:")
    for point in outliers:
        print("   ", point)

if __name__ == "__main__":
    main()
