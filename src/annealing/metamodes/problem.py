"""Meta-modes annealing problem."""
import numpy as np

from annealing.metamodes.objectives import Objective
from annealing.metamodes.solution import MetaModeSolution

rng = np.random.default_rng()


class MetaModeProblem:
    """A problem for the meta-modes annealing algorithm.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        The correlation matrix.
    """

    def __init__(
        self: "MetaModeProblem",
        correlation_matrix: np.ndarray,
        n_modes: int,
        n_channels: int,
        objective: str | Objective,
    ) -> None:
        """Initialize a MetaModeProblem object.

        Parameters
        ----------
        correlation_matrix : np.ndarray
            The correlation matrix.
        n_modes : int
            The number of modes.
        n_channels : int
            The number of channels.
        objective : str or Objective
            The objective function to evaluate.
        """
        self.correlation_matrix = correlation_matrix
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.n_metamodes = self.n_modes
        if isinstance(objective, str):
            objective = Objective[objective]
        self.objective = objective

        self.reindexer = np.arange(self.n_channels) * self.n_modes

    def evaluate(
        self: "MetaModeProblem",
        solution: MetaModeSolution,
    ) -> np.ndarray:
        """Permute the correlation matrix.

        Parameters
        ----------
        solution : np.ndarray
            A solution.
        """
        func = self.objective.func
        return func(
            self.correlation_matrix,
            solution.permuted_modes + self.reindexer[:, None],
            self.n_modes,
            self.n_channels,
        )

    def generate(self: "MetaModeProblem", solution: MetaModeSolution) -> np.ndarray:
        """Generate a set of correlation matrices from a solution's permutation.

        Parameters
        ----------
        solution : MetaModeSolution
            A solution.

        Returns
        -------
        np.ndarray
            A set of correlation matrices.
        """
        perm = solution.permuted_modes
        perm = perm + self.reindexer[:, None]
        perm = self.correlation_matrix[perm][:, :, perm]
        return np.diagonal(perm, axis1=1, axis2=3).T
