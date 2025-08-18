import numpy as np
from brainnet.static_analysis import StaticAnalyzer, ConnectivityMatrix


def test_line_graph_metrics():
    mat = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]], dtype=float)
    labels = ['A', 'B', 'C']
    cm = ConnectivityMatrix(matrix=mat, labels=labels)
    analyzer = StaticAnalyzer()
    metrics = analyzer.compute_graph_metrics(cm)
    bet = metrics.node_metrics['betweenness_centrality']
    assert np.allclose(bet, [0.0, 1.0, 0.0])
    assert np.isclose(metrics.global_metrics['average_shortest_path_length'], 4/3)
    assert np.isclose(metrics.global_metrics['modularity'], 0.0)


def test_modularity_two_clusters():
    mat = np.zeros((6, 6))
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    for i, j in edges:
        mat[i, j] = mat[j, i] = 1.0
    labels = [str(i) for i in range(6)]
    cm = ConnectivityMatrix(matrix=mat, labels=labels)
    analyzer = StaticAnalyzer()
    metrics = analyzer.compute_graph_metrics(cm)
    assert metrics.global_metrics['modularity'] > 0.3
