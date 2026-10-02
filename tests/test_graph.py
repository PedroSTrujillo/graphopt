import networkx
import pytest
import scipy.sparse

from graphopt.graph import Graph


def _csr(rows: list[list[float]]) -> scipy.sparse.csr_matrix:
    return scipy.sparse.csr_matrix(rows)


@pytest.fixture
def path_graph() -> Graph:
    # 0 -(2)- 1 -(3)- 2
    return Graph.from_edge_list(3, [(0, 1, 2.0), (1, 2, 3.0)])


def test_valid_construction() -> None:
    g = Graph(adjacency=_csr([[0, 1], [1, 0]]), labels=("a", "b"))
    assert g.num_vertices == 2
    assert g.num_edges == 1
    assert g.labels == ("a", "b")


def test_label_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Number of labels"):
        Graph(adjacency=_csr([[0, 1], [1, 0]]), labels=(0,))


def test_unhashable_labels() -> None:
    with pytest.raises(ValueError, match="hashable"):
        Graph(adjacency=_csr([[0, 1], [1, 0]]), labels=([1], [2]))  # type: ignore[arg-type]


def test_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        Graph(adjacency=_csr([[0, 1, 0], [1, 0, 0]]), labels=(0, 1))


def test_non_csr() -> None:
    with pytest.raises(ValueError, match="CSR"):
        Graph(adjacency=scipy.sparse.coo_matrix([[0, 1], [1, 0]]), labels=(0, 1))  # type: ignore[arg-type]


def test_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="unique"):
        Graph(adjacency=_csr([[0, 1], [1, 0]]), labels=(0, 0))


def test_asymmetric() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        Graph(adjacency=_csr([[0, 1], [0, 0]]), labels=(0, 1))


def test_self_loop() -> None:
    with pytest.raises(ValueError, match="self-loops"):
        Graph(adjacency=_csr([[1, 0], [0, 0]]), labels=(0, 1))


def test_invalid_entries_all_explicit_zeros() -> None:
    adj = scipy.sparse.csr_matrix(([0.0, 0.0], ([0, 1], [1, 0])), shape=(2, 2))
    assert adj.nnz == 2
    with pytest.raises(ValueError, match="invalid entries"):
        Graph(adjacency=adj, labels=(0, 1))


def test_frozen() -> None:
    g = Graph.from_scipy(_csr([[0, 1], [1, 0]]))
    with pytest.raises(AttributeError):
        g.labels = (5, 6)  # type: ignore[misc]


def test_empty_graph() -> None:
    g = Graph.from_edge_list(3, [])
    assert g.num_vertices == 3
    assert g.num_edges == 0
    assert g.degrees == (0, 0, 0)


def test_from_networkx() -> None:
    nx_g = networkx.Graph()
    nx_g.add_edge("a", "b", weight=2.0)
    nx_g.add_edge("b", "c")
    g = Graph.from_networkx(nx_g)
    assert g.labels == ("a", "b", "c")
    assert g.num_edges == 2
    assert g.adjacency[0, 1] == 2.0
    assert g.adjacency[1, 2] == 1.0


def test_from_networkx_custom_weight() -> None:
    nx_g = networkx.Graph()
    nx_g.add_edge(0, 1, cost=5.0)
    g = Graph.from_networkx(nx_g, weight="cost")
    assert g.adjacency[0, 1] == 5.0


def test_from_scipy() -> None:
    g = Graph.from_scipy(_csr([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    assert g.labels == (0, 1, 2)
    assert g.num_edges == 2


def test_from_edge_list_symmetrizes(path_graph: Graph) -> None:
    assert path_graph.num_vertices == 3
    assert path_graph.num_edges == 2
    assert path_graph.adjacency[1, 0] == 2.0
    assert path_graph.adjacency[2, 1] == 3.0


def test_degrees(path_graph: Graph) -> None:
    assert path_graph.degrees == (1, 2, 1)
    assert all(isinstance(d, int) for d in path_graph.degrees)
    assert path_graph.max_degree == 2
    assert path_graph.min_degree == 1


def test_weighted_degrees(path_graph: Graph) -> None:
    assert path_graph.weighted_degrees == (2.0, 5.0, 3.0)
    assert path_graph.max_weighted_degree == 5.0
    assert path_graph.min_weighted_degree == 2.0


def test_cached_properties(path_graph: Graph) -> None:
    assert path_graph.degrees is path_graph.degrees
    assert path_graph.weighted_degrees is path_graph.weighted_degrees
