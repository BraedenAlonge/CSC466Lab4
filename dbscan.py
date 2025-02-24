import sys
import numpy as np
from kmeans import load_data, compute_dist

def radius_find(data, point_index, epsilon):
    """Find all points within the epsilon radius of the point
    at index point_index."""
    neighbors = []
    for i, point in enumerate(data):
        if compute_dist(data[point_index], point) <= epsilon:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_index, neighbors, cluster_id, epsilon, min_points):
    """Expand new cluster by recursively adding density-reachable points"""
    labels[point_index] = cluster_id
    j = 0
    while j <  len(neighbors):
        n_index = neighbors[j]
        # If neighbor is marked as noise, change to current cluster
        if labels[n_index] == -1:
            labels[n_index] = cluster_id
        # If neighbor is unvisited, assign it to cluster and check neighbors
        elif labels[n_index] == 0:
            labels[n_index] = cluster_id
            n_neighbors = radius_find(data, n_index, epsilon)
            if len(n_neighbors) >= min_points:
                # add any new neighborsto list
                for neigh in n_neighbors:
                    if neigh not in neighbors:
                        neighbors.append(neigh)
        j += 1

def get_clusters_from_labels(data, labels):
    """Returns a dictionary representing the clusters from the points and labels"""
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # noise label
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(data[i])
    return clusters


def dbscan(data, epsilon, min_points):
    """dbscan algo"""
    # 0 inds unvisited. later noise will be -1
    labels = [0] * len(data)
    cluster_id = 0
    for i in range(len(data)):
        if labels[i] != 0:
            continue # already processed!
        neighbors = radius_find(data, i, epsilon)
        if len(neighbors) < min_points:
            labels[i] = -1 # mark it as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, epsilon, min_points)
    return labels

def main():
    args = len(sys.argv)
    if args != 4:
        print("Usage: python dbscan.py <Filename> <epsilon> <NumPoints>")
        exit()
    filename = sys.argv[1]
    epsilon = float(sys.argv[2])
    num_points = int(sys.argv[3])

    data = load_data(filename)
    labels = dbscan(data, epsilon, num_points)

    clusters = get_clusters_from_labels(data, labels)

    for i, points in clusters.items():
        if len(points) > 0:
            centroid = np.mean(points, axis=0)
            distances = [compute_dist(point, centroid) for point in points]
            max_d = max(distances)
            min_d = min(distances)
            avg_d = sum(distances) / len(distances)
            sse = sum(d ** 2 for d in distances)
        else:
            centroid = "n/a"
            max_d = "n/a"
            min_d = "n/a"
            avg_d = "n/a"
            sse = "n/a"
        print(f"Cluster {i}:")
        print(f"  Center: {centroid}")
        print(f"  Max distance to center: {max_d}")
        print(f"  Min distance to center: {min_d}")
        print(f"  Avg distance to center: {avg_d}")
        print(f"  SSE: {sse}")
        print(f"  {len(points)} Points: {points}")

    # report outliers
    outliers = [data[i] for i in range(len(data)) if labels[i] == -1]
    print("Outliers:")
    print(f"  Number of outliers: {len(outliers)}")
    print(f"  Percentage of outliers: {len(outliers) / len(data) * 100}%")
    print(f"  Outlier points:")

    count = 0
    for point in outliers:
        print(f"  {point}")
        count += 1
        if count == 20:  # print only first 20 outliers
            print("  ...")
            break

if __name__ == "__main__":
    main()