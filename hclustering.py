import json
import sys
from kmeans import load_data, compute_dist

def compute_distance_matrix(data):
    """Compute dist matrix for data given"""
    n = len(data)
    distance_matrix = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_dist(data[i], data[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix


def hierarchical_clustering(data, threshold=None):
    # compute pairwise distance matrix for each individual datapoints
    point_distance_matrix = compute_distance_matrix(data)
    n = len(data)
    clusters = []
    for i in range(n):
        clusters.append({"points": [i], # store indices of datapoints
                         "dendrogram": {"type": "leaf", "height": 0.0, "data": data[i]}
                         })
    # Repeatedly merge the two closest clusters
    while len(clusters) > 1:
        min_dist = float("inf")
        pair_to_merge = (None,None)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_distance(clusters[i], clusters[j], point_distance_matrix)
                if d < min_dist:
                    min_dist = d
                    pair_to_merge = (i, j)
        i, j = pair_to_merge
        cluster_i = clusters[i]
        cluster_j = clusters[j]
        # Create new cluster by merging the two clusters
        new_points = cluster_i["points"] + cluster_j["points"]
        new_dendrogram = {
            "type": "node", "height": min_dist, "nodes": [cluster_i["dendrogram"], cluster_j["dendrogram"]]
        }

        new_cluster = {"points": new_points, "dendrogram": new_dendrogram}

        clusters.pop(j)
        clusters.pop(i)
        clusters.append(new_cluster)

    dendrogram = clusters[0]["dendrogram"]

    # Now, if there exists a threshold, cut the dendrogram into clusters
    clusters_cut = []
    if threshold is not None:
        # Get dendrogram nodes that represent clusters when cut at thresh
        cluster_nodes = get_clusters(dendrogram, threshold)
        # Convert dendro nodes to clusters of datpoints
        clusters_cut = [extract_data_points(node) for node in cluster_nodes]


    return dendrogram, clusters_cut

# Helper func 1 for  hierarchical_clustering
def cluster_distance(cluster1, cluster2, point_distance_matrix):
    min_dist = float("inf")
    for i in cluster1["points"]:
        for j in cluster2["points"]:
            d = point_distance_matrix[i][j]
            if d < min_dist:
                min_dist = d
    return min_dist

# Helper func 2 for  hierarchical_clustering
def get_clusters(node, threshold):
    """Recursively cut dendrogram to threshold"""
    # If leaf, return
    if node["type"] == "leaf":
        return [node]
    # If merge height is greater than thresh, keep cuttin'!
    if node["height"] > threshold:
        clusters_list = []
        for child in node["nodes"]:
            clusters_list.extend(get_clusters(child, threshold)) # extend is like append, but can have more elements
        return clusters_list
    else:
        # Threshold reached!
        return [node]

# Helper func 3 for hierarchical_clustering
def extract_data_points(node):
    """Recursively extract datapoints from dendro node"""
    if node["type"] == "leaf":
        return [node["data"]]
    else:
        points = []
        for child in node["nodes"]:
            points.extend(extract_data_points(child))
        return points

def output_dendrogram(dendrogram):
    """Output dendrogram as json"""
    json_output = json.dumps(dendrogram, indent=2)
    print(json_output)
    with open("dendrogram.json", "w") as f:
        f.write(json_output)
        f.close()

def main():
    args = len(sys.argv)
    if args != 2 and args != 3: # Add more args if needed and update this
        print("Usage: python hclustering.py <Filename> [<Threshold>]")
        exit()
    filename = sys.argv[1]
    if args == 3:
        threshold = float(sys.argv[2])
    else:
        threshold = None

    data = load_data(filename)
    dendrogram, clusters = hierarchical_clustering(data, threshold)
    output_dendrogram(dendrogram)

    if threshold is not None:
        print("Cluster at threshold", threshold)
        for i, cluster  in enumerate(clusters):
            print(f"Cluster {i}: {cluster}")


if __name__ == "__main__":
    main()
