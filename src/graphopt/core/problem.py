from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, Generic, TypeVar

from graphopt.graph import Graph


class Sense(str, Enum):
    """Represents the sense of optimization for a problem (e.g., minimization or maximization)."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


S = TypeVar("S")


class Problem(ABC, Generic[S]):
    """A graph problem whose solutions have type S."""

    graph: Graph
    sense: ClassVar[Sense]

    @abstractmethod
    def evaluate(self, solution: S) -> float:
        """Evaluates the given solution and returns its objective value."""
        ...

    @abstractmethod
    def is_feasible(self, solution: S) -> bool:
        """Checks if the given solution is feasible for this problem."""
        ...
