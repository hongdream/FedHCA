"""Federated FedHCA clustering pipeline."""

import itertools

import numpy as np

from algorithm.MultiClustering import fedhca_ensemble, fine_cd, multi_ca


def fedClustering(clients, rate, k_0=0.5, k_1=0.5):
    """Run local prototype learning, server clustering, and local label updates.

    Args:
        clients: List of client dictionaries. Each dictionary must contain
            ``data``, ``label``, and ``size`` entries.
        rate: Learning rate used by prototype learning.
        k_0: Local prototype ratio for Fine-CD.
        k_1: Server prototype ratio for Multi-CA.

    Returns:
        Tuple ``(clients, global_centroids, status)`` where status is
        ``"success"`` or ``"error"``.
    """
    true_labels = list(itertools.chain.from_iterable(client["label"] for client in clients))
    n_classes = len(np.unique(true_labels))

    local_prototypes = []
    local_cluster_counts = []
    for client in clients:
        prototypes, labels, n_clusters = fine_cd(client["data"], rate, k_0)
        client["localLabel"] = labels
        client["centroid"] = prototypes
        local_prototypes.append(prototypes)
        local_cluster_counts.append(n_clusters)

    if len(local_prototypes) == 0:
        return [], [], "error"

    all_prototypes = np.concatenate(local_prototypes)
    hierarchical_labels, _ = multi_ca(all_prototypes, rate, k_1)
    label_data = np.stack(hierarchical_labels).T

    prototype_labels, _, _, status = fedhca_ensemble(label_data, n_classes)
    if status == "error":
        return [], [], "error"

    global_centroids = []
    for class_id in range(n_classes):
        members = np.where(prototype_labels == class_id)[0]
        if len(members) == 0:
            return [], [], "error"
        global_centroids.append(np.mean(all_prototypes[members], axis=0))
    global_centroids = np.asarray(global_centroids)

    client_ranges = []
    start = 0
    for n_clusters in local_cluster_counts:
        client_ranges.append(np.arange(start, start + n_clusters))
        start += n_clusters

    for client_id, client in enumerate(clients):
        local_labels = client["localLabel"]
        prototype_index = client_ranges[client_id]
        granular_labels = np.zeros((client["size"], len(hierarchical_labels)), dtype=int)

        for level_id, level_labels in enumerate(hierarchical_labels):
            granular_labels[:, level_id] = np.asarray(level_labels)[prototype_index][local_labels]

        client["GranularLabel"] = granular_labels
        client["predict"] = prototype_labels[prototype_index][local_labels]

    return clients, global_centroids, "success"
