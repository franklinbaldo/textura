from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage as scipy_linkage


class ClusterAgent:
    """Simple wrapper for hierarchical clustering of embeddings."""

    def __init__(self, embeddings: List[np.ndarray]):
        self.embeddings = np.vstack(embeddings)
        self.labels: np.ndarray | None = None
        self.linkage_matrix: np.ndarray | None = None

    def hierarchical(
        self,
        linkage_method: str = "average",
        distance_threshold: float | None = None,
        n_clusters: int | None = None,
    ) -> Dict[int, List[int]]:
        """Perform agglomerative clustering on stored embeddings."""
        model = AgglomerativeClustering(
            affinity="cosine",
            linkage=linkage_method,
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            compute_full_tree=True,
        ).fit(self.embeddings)
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(model.labels_):
            clusters.setdefault(int(label), []).append(idx)
        self.labels = model.labels_
        self.linkage_matrix = scipy_linkage(
            self.embeddings, method=linkage_method, metric="cosine"
        )
        return clusters
