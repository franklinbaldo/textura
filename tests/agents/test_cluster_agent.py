import numpy as np

from textura.agents.cluster_agent import ClusterAgent


def test_hierarchical_n_clusters():
    rng = np.random.default_rng(0)
    # create three gaussian clusters in 4D
    c1 = rng.normal(0, 0.1, (5, 4))
    c2 = rng.normal(5, 0.1, (5, 4))
    c3 = rng.normal(-5, 0.1, (5, 4))
    data = np.vstack([c1, c2, c3])
    agent = ClusterAgent([d for d in data])
    clusters = agent.hierarchical(n_clusters=3)
    assert len(clusters) == 3
    all_indices = sorted(idx for inds in clusters.values() for idx in inds)
    assert all_indices == list(range(15))
    assert agent.linkage_matrix.shape == (data.shape[0] - 1, 4)
