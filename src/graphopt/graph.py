from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import cached_property

import networkx
import scipy


@dataclass(frozen=True)
class Graph:
    """
    Undirected graph representation using an adjacency matrix and vertex labels. 
    Graphs are required to have symmetric adjacency matrices.
    """

    adjacency: scipy.sparse.csr_matrix
    labels: tuple[Hashable, ...]

    def __post_init__(self):
        if len(self.labels) != self.adjacency.shape[0]:
            raise ValueError("Number of labels must match number of vertices.")
        if not all(isinstance(label, Hashable) for label in self.labels):
            raise ValueError("All labels must be hashable.")
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        if not scipy.sparse.isspmatrix_csr(self.adjacency):
            raise ValueError("Adjacency matrix must be a CSR matrix.")
        if self.adjacency.nnz != 0 and not (self.adjacency != 0).nnz:
            raise ValueError("Adjacency matrix contains invalid entries.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("All labels must be unique.")
        if not (self.adjacency - self.adjacency.T).nnz == 0:
            raise ValueError("Adjacency matrix must be symmetric.")
        if self.adjacency.diagonal().any():
            raise ValueError("Adjacency matrix must not have self-loops.")

    @property
    def num_vertices(self) -> int:
        return self.adjacency.shape[0]

    @property
    def num_edges(self) -> int:
        return self.adjacency.nnz // 2

    @classmethod
    def from_networkx(cls, g: "networkx.Graph", weight: str = "weight") -> "Graph":
        adjacency = networkx.adjacency_matrix(g, weight=weight).tocsr()
        nodes = tuple(g.nodes)
        return cls(adjacency=adjacency, labels=nodes)

    @classmethod
    def from_scipy(cls, adjacency: "scipy.sparse.csr_matrix") -> "Graph":
        return cls(adjacency=adjacency, labels=tuple(range(adjacency.shape[0])))

    @classmethod
    def from_edge_list(
        cls, n: int, edge_list: "Iterable[tuple[int, int, float]]"
    ) -> "Graph":
        rows, cols, vals = zip(*edge_list) if edge_list else ([], [], [])
        adjacency = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        adjacency = adjacency + adjacency.T
        return cls(adjacency=adjacency, labels=tuple(range(n)))

    @cached_property
    def degrees(self) -> tuple[int, ...]:
        return tuple(self.adjacency.getnnz(axis=1))

    @cached_property
    def weighted_degrees(self) -> tuple[float, ...]:
        return tuple(self.adjacency.sum(axis=1).A1)

    @property
    def max_degree(self) -> int:
        return max(self.degrees)

    @property
    def max_weighted_degree(self) -> float:
        return max(self.weighted_degrees)

    @property
    def min_degree(self) -> int:
        return min(self.degrees)

    @property
    def min_weighted_degree(self) -> float:
        return min(self.weighted_degrees)
