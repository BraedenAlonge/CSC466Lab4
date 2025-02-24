#!/usr/bin/env python3
import sys
import csv
import numpy as np
import json
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.spatial.distance import pdist

def load_dataset(filename):
    data = []
    with open(filename, 'r', newline='') as f:
        # Read a sample to detect the delimiter
        sample = f.read(1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # Fallback if detection fails
        reader = csv.reader(f, dialect)

        # Read header (mask) and convert each entry to a boolean
        mask_row = next(reader)
        mask = [int(x.strip()) == 1 for x in mask_row]

        for row in reader:
            if not row:
                continue
            # Ensure the row has at least as many elements as the mask
            if len(row) < len(mask):
                continue
            try:
                # Select only the columns where mask is True, stripping whitespace
                selected = [float(row[i].strip()) for i in range(len(row)) if mask[i]]
            except ValueError as e:
                print(f"Skipping row due to conversion error: {row}\nError: {e}")
                continue
            data.append(selected)
    return np.array(data)

def cluster_stats(data, labels):
    stats = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_points = data[indices]
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

def build_dendrogram_json(node, data, is_root=False):
    if node.is_leaf():
        # For leaf nodes, include the actual data point (as a list)
        return {
            "type": "leaf",
            "height": 0,
            "data": data[node.id].tolist()
        }
    else:
        children = [
            build_dendrogram_json(node.get_left(), data),
            build_dendrogram_json(node.get_right(), data)
        ]
        node_type = "root" if is_root else "node"
        return {
            "type": node_type,
            "height": node.dist,
            "nodes": children
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 hclustering.py <Filename> [<threshold>]")
        sys.exit(1)
    filename = sys.argv[1]
    threshold = None
    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            print("Threshold must be a float.")
            sys.exit(1)

    data = load_dataset(filename)
    if data.size == 0:
        print("No data loaded from file.")
        sys.exit(1)

    # Compute the linkage matrix using Ward's method
    Z = linkage(data, method='ward')

    # Convert the linkage matrix to a tree structure
    tree = to_tree(Z, rd=False)
    dendrogram_json = build_dendrogram_json(tree, data, is_root=True)
    print("Dendrogram (JSON):")
    print(json.dumps(dendrogram_json, indent=2))

    # If a threshold is provided, cut the dendrogram to form clusters
    if threshold is not None:
        labels = fcluster(Z, t=threshold, criterion='distance')
        stats = cluster_stats(data, labels)
        print("\nClusters (cut at threshold {}):".format(threshold))
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

if __name__ == "__main__":
    main()
