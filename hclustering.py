import json
import sys
from kmeans import load_data, compute_dist, visualize_clusters

def compute_distance_matrix(data):
    """ Compute dist matrix for data given (dataset).
        Returs:
        2-D list where [i][j] is dist between data[i] and data[j]"""
    n = len(data)
    distance_matrix = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_dist(data[i], data[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix

def hierarchical_clustering(data, threshold=None):
    """
    Performs agglomerative hierarchical clustering on the data.
    Options:
      - Different linkage methods (e.g., single, complete, average) can be implemented.
    Returns:
      dendrogram: A JSON-compatible structure representing the clustering hierarchy.
      clusters: The final clusters after cutting the dendrogram at the threshold (if provided).
    TODO:
      - Implement the clustering procedure.
      - Create a JSON output for the dendrogram.
      - If threshold is provided, compute and return clusters.
    """
    # TODO: Initialize clusters as individual points
    # TODO: Compute distances and merge clusters iteratively
    # TODO: Build dendrogram structure (each node: type, height, and nodes or data)
    dendrogram = {}  # Placeholder
    clusters = []    # Placeholder (only if threshold is specified)
    return dendrogram, clusters

def output_dendrogram(dendrogram):
    """Output dendrogram as json"""
    json_output = json.dumps(dendrogram, indent=2)
    print(json_output)
    with open("dendrogram.json", "w") as f:
        f.write(json_output)
        f.close()

def main():
    """
    1. Parse args
    2. load data
    3. compute distacnce matrix
    4. run hierarchical clustering algo
    5. output dendrogram and clusters provided a thresh
    """
    args = len(sys.argv)
    if args != 2 or args != 3: # Add more args if needed and update this
        print("Usage: python kmeans.py <Filename> [<Threshold>]")
        exit()
    filename = sys.argv[1]
    if args == 3:
        threshold = float(sys.argv[2])
    else:
        threshold = None

    data = load_data(filename)
    dist_matrix = compute_distance_matrix(data)

    dendrogram, clusters = hierarchical_clustering(data, threshold)
    output_dendrogram(dendrogram)

    if threshold is not None:
        print("Cluster at threshold", threshold)
        for i, cluster  in enumerate(clusters):
            print(f"Cluster {i}: {cluster}")



if __name__ == "__main__":
    main()
