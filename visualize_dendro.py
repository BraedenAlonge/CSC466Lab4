import networkx as nx
import matplotlib.pyplot as plt


def build_graph_from_dendrogram(dendro, graph=None, parent=None, pos=None, level=0, x=0):
    """
    Recursively traverse the dendrogram JSON structure to build a directed graph.

    Parameters:
      dendro: dict
         The current node in the dendrogram.
      graph: networkx.DiGraph
         Graph being built.
      parent: node identifier of the parent (if any)
      pos: dict for storing node positions for plotting.
      level: current depth level (used for vertical spacing)
      x: horizontal position for the current node.

    Returns:
      graph: networkx.DiGraph with all dendrogram nodes added.
      pos: dictionary with node positions for visualization.
    """
    if graph is None:
        graph = nx.DiGraph()
    if pos is None:
        pos = {}

    # Use Python's built-in id() function for a unique node identifier.
    node_id = id(dendro)
    # Create a label with the node type and height (for non-leaf nodes).
    if dendro["type"] == "leaf":
        label = f"Leaf\n{dendro['data']}"
    else:
        label = f"{dendro['type']}\nheight: {dendro['height']:.2f}"

    graph.add_node(node_id, label=label)
    pos[node_id] = (x, -level)

    if parent is not None:
        graph.add_edge(parent, node_id)

    if dendro["type"] == "node":
        # Distribute children horizontally.
        children = dendro["nodes"]
        n_children = len(children)
        # Set a horizontal spread (adjust as needed)
        spread = 1.5
        for i, child in enumerate(children):
            # Compute a new x coordinate for the child.
            child_x = x - spread * (n_children - 1) / 2 + spread * i
            build_graph_from_dendrogram(child, graph, node_id, pos, level + 1, child_x)

    return graph, pos


def plot_dendrogram_from_json(dendro_json):
    """
    Builds and plots a dendrogram from the provided JSON structure.

    Parameters:
      dendro_json: dict
         The dendrogram in JSON format.
    """
    graph, pos = build_graph_from_dendrogram(dendro_json)
    labels = nx.get_node_attributes(graph, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, labels=labels, arrows=False,
            node_size=150, node_color='lightblue', font_size=7, font_weight='bold')
    plt.title("Dendrogram Visualization")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Suppose 'dendrogram' is the JSON structure obtained from your hierarchical_clustering function.
    import json

    with open("dendrogram.json", "r") as f:
        dendro_json = json.load(f)
    plot_dendrogram_from_json(dendro_json)