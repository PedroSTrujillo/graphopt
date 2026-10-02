from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from graphopt.core.problem import Problem, Sense

Cut = npt.NDArray[np.int8]  # x[i] in {0, 1}: which side of the cut vertex i belongs to


@dataclass(frozen=True)
class MaxCut(Problem[Cut]):
    sense = Sense.MAXIMIZE

    def is_feasible(self, solution: Cut) -> bool:
        """Checks if a cut is a feasible maximum cut. That is, if it is a one-dimensional array of
        integers (0 or 1) indicating the side of the cut each vertex belongs to.

        Args:
            solution (Cut): The candidate solution representing the cut. Each element should be 0
            or 1, indicating the side of the cut the corresponding vertex belongs to.

        Returns:
            bool: True if the solution is feasible, False otherwise.
        """
        return bool(
            solution.ndim == 1
            and solution.dtype == np.int8
            and np.all((solution == 0) | (solution == 1))
        )

    # TODO: Implement the evaluation function for the Max-Cut problem
    def evaluate(self, solution: Cut) -> float:
        """Evaluates the given solution for the Max-Cut problem."""
        raise NotImplementedError()
