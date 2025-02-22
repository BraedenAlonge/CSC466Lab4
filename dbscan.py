import sys
from kmeans import load_data, compute_dist

def radius_find(data, point_index, epsilon):
    """Find all points within the epsilon radius of the point
    at index point_index.
    Return a list of indices representing neighbors"""
    neighbors = []
    for i, point in enumerate(data):
        if compute_dist(data[point_index], point) <= epsilon:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_index, neighbors, cluster_id, epsilon, min_points):
    """
    Expand new cluster by recursively adding density-reachable points
    Params:
    data: list of data points
    labels: current cluster labels for each point (0: unvisited, -1: noise, >0: cluster ID)
    point_index: index of the point we are expanding
    neighbors: list of indices representing neighbors
    cluster_id: current cluster id to assign
    epsilon: radius
    min_points: minimum number of points to form a core point
    """
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

def dbscan(data, epsilon, min_points):
    """dbscan algo.
    Returns labels (list of cluster labels for each datapoint. noise is -1, cluster labels are 1+)
    """
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

    print("Cluster labels:")
    for i, label in enumerate(labels):
        print(f"Point {i}: Cluster {label}")
if __name__ == "__main__":
    main()