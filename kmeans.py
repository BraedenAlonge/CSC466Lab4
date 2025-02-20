import csv
import sys
import pandas as pd
import random

def load_data(filename):
    """Load data from csv file.
    1. First row = binary vector indicating which cols to use.
    2. Skip cols with 0 flag
    Return list of data points."""
    data = []
    flags = []
    with open(filename, "r") as f:
        flags = f.readline().strip().split(" ")
        flags = [int(x) for x in flags]

    data = pd.read_csv(filename, skiprows=1)
    kept_cols = []
    for col, fl in zip(data.columns, flags):
        if fl == 0:
            continue
        else:
            kept_cols.append(col)

    data = data[kept_cols]
    return data.values.tolist()

def initialize_centroids(data, k):
    """Initialize k centroids from the dataset. We can either:
    A. Randomly select k points from the data.
    B. implement kmeans++ for better initial centroids.
    Returns:
        list of centroids
    """
    centroids = random.sample(data, k)
    return centroids

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




if __name__ == "__main__":
    main()