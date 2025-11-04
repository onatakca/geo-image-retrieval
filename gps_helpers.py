import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple


def cluster_locations(locations: np.ndarray, eps: float = 25, min_samples: int = 2) -> np.ndarray:
    """
    Cluster GPS locations using DBSCAN.

    Args:
        locations: Array of shape (N, 2 or 3) containing GPS coordinates [x, y] or [x, y, heading]
        eps: Maximum distance between samples for clustering (in coordinate units)
        min_samples: Minimum number of samples in a neighborhood for a core point

    Returns:
        cluster_labels: Array of shape (N,) with cluster assignments (-1 for noise)
    """
    # Use only x, y coordinates for clustering (ignore heading if present)
    coords = locations[:, :2]

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clustering.fit_predict(coords)

    return cluster_labels


def compute_location_centroids(features: np.ndarray, cluster_labels: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    Compute centroid features for each location cluster by averaging.

    Args:
        features: Array of shape (N, D) containing image features
        cluster_labels: Array of shape (N,) with cluster assignments

    Returns:
        centroids: Dictionary mapping cluster_id -> centroid feature vector
        cluster_members: Dictionary mapping cluster_id -> list of image indices
    """
    centroids = {}
    cluster_members = {}

    unique_clusters = np.unique(cluster_labels)
    # Remove noise cluster (-1) if present
    unique_clusters = unique_clusters[unique_clusters >= 0]

    for cluster_id in unique_clusters:
        # Get indices of images in this cluster
        member_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_members[cluster_id] = member_indices.tolist()

        # Compute centroid as mean of features
        cluster_features = features[member_indices]
        centroid = cluster_features.mean(axis=0)

        # Normalize centroid
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids[cluster_id] = centroid

    return centroids, cluster_members


def get_cluster_members(cluster_labels: np.ndarray, cluster_id: int) -> np.ndarray:
    return np.where(cluster_labels == cluster_id)[0]


def compute_gps_distances(loc1: np.ndarray, loc2: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances between GPS coordinates.

    Args:
        loc1: Array of shape (N, 2 or 3) with GPS coordinates
        loc2: Array of shape (M, 2 or 3) with GPS coordinates

    Returns:
        distances: Array of shape (N, M) with pairwise distances
    """
    coords1 = loc1[:, :2]
    coords2 = loc2[:, :2]

    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))

    return distances
