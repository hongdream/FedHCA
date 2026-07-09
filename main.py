"""Command line entry point for FedHCA experiments."""

import argparse
import time
from pathlib import Path

import numpy as np

from algorithm.experiment import validation
from algorithm.federatedClustering import fedClustering


def load_clients(path):
    """Load client dictionaries from an ``.npz`` dataset file."""
    data = np.load(path, allow_pickle=True)
    return [data[key].item() for key in data.files]


def run(dataset, data_dir, runs, rate, k_0, k_1):
    """Run FedHCA on one dataset."""
    dataset_path = Path(data_dir) / f"{dataset}.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    results = []
    for run_id in range(1, runs + 1):
        clients = load_clients(dataset_path)
        start = time.time()
        clients, centroids, status = fedClustering(clients, rate, k_0, k_1)
        elapsed = time.time() - start

        if status != "success":
            print(f"Run {run_id}: failed to produce valid clusters.")
            continue

        metrics = validation(clients)
        results.append(metrics)
        metric_text = ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
        print(f"Run {run_id}: {elapsed:.4f}s, global clusters={len(centroids)}, {metric_text}")

    if results:
        print(f"Average validation metrics over {len(results)} successful runs:")
        for key in results[0]:
            values = np.asarray([item[key] for item in results])
            print(f"  {key}: {values.mean():.4f} +/- {values.std():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run FedHCA clustering experiments.")
    parser.add_argument("--dataset", default="CA", help="Dataset name without .npz suffix.")
    parser.add_argument("--data-dir", default="dataset", help="Directory containing .npz datasets.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs.")
    parser.add_argument("--rate", type=float, default=0.03, help="Prototype learning rate.")
    parser.add_argument("--k0", type=float, default=0.5, help="Local prototype ratio.")
    parser.add_argument("--k1", type=float, default=0.5, help="Server prototype ratio.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.dataset, args.data_dir, args.runs, args.rate, args.k0, args.k1)
