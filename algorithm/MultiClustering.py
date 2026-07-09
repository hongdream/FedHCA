"""Core clustering routines used by FedHCA.

The pipeline has three stages:
1. Fine-CD compresses each client's samples into local prototypes.
2. Multi-CA builds a hierarchy of server-side prototype labels.
3. FedHCA ensembles the hierarchical labels to produce final clusters.
"""

import statistics

import numpy as np

np.seterr(divide="ignore", invalid="ignore")


def similarity(sample, centroids, weights):
    """Return normalized weighted Gaussian similarities to centroids."""
    diff = (centroids - sample) ** 2
    scores = np.exp(-0.5 * np.sum(weights * diff, axis=1))
    scores = np.nan_to_num(scores)

    total = np.sum(scores)
    if total == 0:
        return np.ones_like(scores) / len(scores)
    return scores / total


def _prototype_learning(data, rate, k_ratio):
    """Shared prototype learning loop for Fine-CD and Multi-CA."""
    n_samples, n_features = data.shape
    n_clusters = max(1, int(np.round(k_ratio * n_samples)))
    n_clusters = min(n_clusters, n_samples)

    seed = np.random.choice(range(n_samples), n_clusters, replace=False)
    total_data = data[seed, :].astype(float).copy()
    weights = np.full((n_clusters, n_features), 1 / n_features, dtype=float)
    counts = np.ones(n_clusters, dtype=int)
    beta = np.ones(n_clusters, dtype=float)
    assignment_matrix = np.zeros((n_samples, n_clusters), dtype=int)
    labels = np.full(n_samples, -1, dtype=int)

    labels[seed] = np.arange(n_clusters)
    assignment_matrix[seed, np.arange(n_clusters)] = 1

    for _ in range(30):
        no_change = True
        cluster_sizes = np.sum(assignment_matrix, axis=0)
        centroids = total_data / np.repeat(cluster_sizes[:, np.newaxis], n_features, axis=1)

        for i in range(n_samples):
            scores = similarity(data[i, :], centroids, weights)
            gate = 1 / (1 + np.exp(-10 * beta + 5))
            frequency = counts / np.sum(counts)
            scores = (1 - frequency) * gate * scores

            ranked = np.argsort(-scores)
            best = ranked[0]
            beta[best] += rate
            if len(ranked) > 1 and scores[best] != 0:
                second = ranked[1]
                beta[second] -= rate * (scores[second] / scores[best])

            counts[best] += 1
            old = labels[i]
            labels[i] = best
            if old != best:
                total_data[best, :] += data[i, :]
                assignment_matrix[i, best] = 1
                if old != -1:
                    total_data[old, :] -= data[i, :]
                    assignment_matrix[i, old] = 0
                no_change = False

        if no_change:
            break

        weights = _update_feature_weights(data, labels, assignment_matrix, total_data)

    active = np.where(np.sum(assignment_matrix, axis=0) > 0)[0]
    prototypes = []
    compact_labels = np.full(n_samples, -1, dtype=int)
    for new_label, old_label in enumerate(active):
        members = np.where(assignment_matrix[:, old_label] == 1)[0]
        prototypes.append(total_data[old_label, :] / len(members))
        compact_labels[members] = new_label

    return np.stack(prototypes), compact_labels, len(active)


def _update_feature_weights(data, labels, assignment_matrix, total_data):
    """Update feature weights from within-cluster density and separation."""
    n_samples, n_features = data.shape
    n_clusters = assignment_matrix.shape[1]
    cluster_sizes = np.sum(assignment_matrix, axis=0)
    centroids = total_data / np.repeat(cluster_sizes[:, np.newaxis], n_features, axis=1)

    density = np.zeros((n_clusters, n_features))
    separation = np.zeros((n_clusters, n_features))

    for cluster_id in range(n_clusters):
        inside = np.where(labels == cluster_id)[0]
        outside = np.setdiff1d(range(n_samples), inside)
        count = len(inside)
        if count <= 1 or len(outside) <= 1:
            continue

        density[cluster_id, :] = (
            np.sum(np.exp(-0.5 * ((data[inside, :] - centroids[cluster_id, :]) ** 2)), axis=0)
            / count
        )

        inside_mean = np.sum(data[inside, :], axis=0) / count
        outside_mean = np.sum(data[outside, :], axis=0) / len(outside)
        inside_var = np.sum((data[inside, :] - inside_mean) ** 2, axis=0) / (count - 1)
        outside_var = np.sum((data[outside, :] - outside_mean) ** 2, axis=0) / (len(outside) - 1)
        var_sum = inside_var + outside_var

        separation[cluster_id, :] = np.sqrt(
            1
            - np.sqrt((2 * np.sqrt(inside_var) * np.sqrt(outside_var)) / var_sum)
            * np.exp(-(((inside_mean - outside_mean) ** 2) / (4 * var_sum)))
        )

    combined = density * separation
    row_sums = np.sum(combined, axis=1, keepdims=True)
    return np.divide(
        combined,
        row_sums,
        out=np.full_like(combined, 1 / n_features),
        where=row_sums != 0,
    )


def fine_cd(data, rate, k_0):
    """Run Fine-CD to find local prototypes for one client."""
    return _prototype_learning(data, rate, k_0)


def calWeight(data, centroids, labels):
    """Estimate per-cluster feature weights for categorical label vectors."""
    n_samples, n_features = data.shape
    n_clusters = len(centroids)
    alpha = np.zeros((n_clusters, n_features))
    beta = np.zeros((n_clusters, n_features))

    for cluster_id in range(n_clusters):
        inside_mask = labels == cluster_id
        outside_mask = ~inside_mask
        if np.sum(inside_mask) == 0 or np.sum(outside_mask) == 0:
            continue

        inside = data[inside_mask]
        outside = data[outside_mask]
        for feature_id in range(n_features):
            max_value = int(np.max(data[:, feature_id])) + 1
            inside_hist = np.bincount(inside[:, feature_id], minlength=max_value) / inside.shape[0]
            outside_hist = np.bincount(outside[:, feature_id], minlength=max_value) / outside.shape[0]
            alpha[cluster_id, feature_id] = np.sqrt(np.sum((inside_hist - outside_hist) ** 2) / 2)

            values = inside[:, feature_id]
            counts = np.bincount(values, minlength=max_value)
            beta[cluster_id, feature_id] = np.mean(counts[values] / inside.shape[0])

    scores = alpha * beta
    row_sums = np.sum(scores, axis=1, keepdims=True) + 1e-10
    return scores / row_sums


def labelassign(data, centroids, weights):
    """Assign each label vector to the most similar categorical centroid."""
    same_value = (data[:, np.newaxis, :] == centroids[np.newaxis, :, :]).astype(float)
    scores = np.sum(same_value * weights[np.newaxis, :, :], axis=2)
    return np.argmax(scores, axis=1)


def fedhca_ensemble(label_data, true_k):
    """Cluster prototypes by ensembling Multi-CA hierarchy labels."""
    n_samples, n_levels = label_data.shape
    unique_rows = np.unique(label_data, axis=0)

    if true_k > unique_rows.shape[0]:
        return [], [], [], "error"

    centroids = unique_rows[np.random.choice(unique_rows.shape[0], true_k, replace=False)]
    weights = np.ones((true_k, n_levels)) / n_levels
    result = labelassign(label_data, centroids, weights)

    for _ in range(30):
        old_result = result.copy()
        new_centroids = []
        for cluster_id in range(true_k):
            members = np.where(result == cluster_id)[0]
            if len(members) == 0:
                new_centroids.append(np.full(n_levels, -1))
                continue
            new_centroids.append([statistics.mode(label_data[members, level]) for level in range(n_levels)])

        centroids = np.asarray(new_centroids)
        weights = calWeight(label_data, centroids, old_result)
        result = labelassign(label_data, centroids, weights)
        if np.array_equal(result, old_result):
            break

    return result, weights, centroids, "success"


def multi_ca(data, rate, k_1):
    """Run Multi-CA to build multi-granularity prototype labels on the server."""
    n_samples, _ = data.shape
    k_ratio = k_1
    labels_by_level = []
    cluster_counts = []

    for epoch in range(30):
        _, labels, n_clusters = _prototype_learning(data, rate, k_ratio)

        if n_clusters == int(np.round(k_ratio * n_samples)) and epoch == 0:
            labels_by_level.append(labels)
            cluster_counts.append(n_clusters)
            break

        labels_by_level.append(labels)
        cluster_counts.append(n_clusters)
        if n_clusters < 2 or n_clusters == int(np.round(k_ratio * n_samples)):
            break

        k_ratio = n_clusters / n_samples

    return labels_by_level, cluster_counts
