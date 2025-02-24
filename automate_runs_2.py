#!/usr/bin/env python3
"""
Automate Clustering Runs

This script automatically runs your custom clustering implementations (kmeans.py,
hclustering.py, and dbscan.py) on a predefined set of CSV datasets. For each dataset, the script:
  - Uses default hyperparameters (selected based on the lab description) for each algorithm.
  - Invokes each algorithm via subprocess calls.
  - Captures the output (cluster metrics, printed information, and any error messages).
  - Saves the consolidated output in JSON format to 'consolidated_output.json'.

Datasets are hard-coded (e.g., data/4clusters.csv, data/mammal_milk.csv, data/planets.csv,
data/iris.csv, and data/AccidentsSet03.csv). Please ensure these files are uploaded in a folder
named "data".

After running this script, you can upload the resulting 'consolidated_output.json' back for report
generation. The final lab report will include sections on Study Design, Results, Visualizations,
Discussion/Comparison, and Conclusions/Analysis.

Usage:
    python automate_clustering.py
"""

import subprocess
import os
import json


def run_command(command):
    """
    Runs a shell command and returns its stdout and stderr.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr


def run_kmeans(dataset, k):
    """
    Runs the k-means clustering script with the given dataset and k (number of clusters).
    """
    command = f"python kmeans.py {dataset} {k}"
    print(f"Executing: {command}")
    out, err = run_command(command)
    return out, err


def run_hclustering(dataset, threshold):
    """
    Runs the hierarchical clustering script with the given dataset and threshold.
    """
    command = f"python hclustering.py {dataset} {threshold}"
    print(f"Executing: {command}")
    out, err = run_command(command)
    return out, err


def run_dbscan(dataset, epsilon, min_points):
    """
    Runs the DBSCAN clustering script with the given dataset, epsilon, and min_points.
    """
    command = f"python dbscan.py {dataset} {epsilon} {min_points}"
    print(f"Executing: {command}")
    out, err = run_command(command)
    return out, err


def main():
    # Hard-coded list of dataset filenames (ensure these files are in the 'data' folder)
    datasets = [
        "data/4clusters.csv",
        "data/mammal_milk.csv",
        "data/planets.csv",
        "data/iris.csv",
        "data/AccidentsSet03.csv"
    ]

    # Default hyperparameters for each algorithm.
    # These values can be modified based on your lab description or experimental needs.

    # For k-means: A dictionary mapping dataset filename to a default number of clusters.
    kmeans_params = {
        "data/4clusters.csv": 4,
        "data/mammal_milk.csv": 3,
        "data/planets.csv": 3,
        "data/iris.csv": 3,
        "data/AccidentsSet03.csv": 4
    }

    # For hierarchical clustering, we now use a threshold of 10.0 for 4clusters.csv to force 4 clusters.
    hclustering_params = {
        "data/4clusters.csv": 10.0,  # Increased threshold to merge clusters to yield 4 clusters
        "data/mammal_milk.csv": 5.0,
        "data/planets.csv": 5.0,
        "data/iris.csv": 1.0,  # Lower threshold for iris given its overall low dendrogram height
        "data/AccidentsSet03.csv": 5.0
    }

    # For DBSCAN, revised parameters are set based on dataset scale.
    dbscan_params = {
        "data/4clusters.csv": {"epsilon": 0.8, "min_points": 3},
        "data/mammal_milk.csv": {"epsilon": 1.5, "min_points": 3},
        "data/planets.csv": {"epsilon": 10.0, "min_points": 2},
        "data/iris.csv": {"epsilon": 0.6, "min_points": 5},
        "data/AccidentsSet03.csv": {"epsilon": 0.8, "min_points": 3}
    }

    # Consolidated output dictionary to store results from all runs.
    consolidated_output = {
        "kmeans": {},
        "hclustering": {},
        "dbscan": {}
    }

    # Run k-means for each dataset
    for dataset in datasets:
        k = kmeans_params.get(dataset, 3)
        print(f"\nRunning k-means on {dataset} with k = {k}")
        out, err = run_kmeans(dataset, k)
        consolidated_output["kmeans"][dataset] = {"output": out, "error": err}

    # Run hierarchical clustering for each dataset
    for dataset in datasets:
        threshold = hclustering_params.get(dataset, 5.0)
        print(f"\nRunning hierarchical clustering on {dataset} with threshold = {threshold}")
        out, err = run_hclustering(dataset, threshold)
        consolidated_output["hclustering"][dataset] = {"output": out, "error": err}

    # Run DBSCAN for each dataset
    for dataset in datasets:
        params = dbscan_params.get(dataset, {"epsilon": 1.0, "min_points": 5})
        epsilon = params["epsilon"]
        min_points = params["min_points"]
        print(f"\nRunning DBSCAN on {dataset} with epsilon = {epsilon} and min_points = {min_points}")
        out, err = run_dbscan(dataset, epsilon, min_points)
        consolidated_output["dbscan"][dataset] = {"output": out, "error": err}

    # Save the consolidated output to a JSON file for later report generation.
    output_filename = "consolidated_output.json"
    with open(output_filename, "w") as outfile:
        json.dump(consolidated_output, outfile, indent=4)

    print(f"\nAll clustering runs are complete. Consolidated output has been saved to '{output_filename}'.")
    print("Please upload the 'consolidated_output.json' file for further report generation.")


if __name__ == "__main__":
    main()
