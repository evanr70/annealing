"""A solver for the meta-modes annealing algorithm."""

import numpy as np
from tqdm.auto import trange

from annealing.metamodes.problem import MetaModeProblem
from annealing.metamodes.solution import MetaModeSolution

rng = np.random.default_rng()


class MetaModeSolver:
    """A solver for the meta-modes annealing algorithm.

    Attributes
    ----------
    problem : MetaModeProblem
        The problem to solve.
    best_solution : MetaModeSolution
        The best solution found so far.
    best_value : float
        The value of the best solution found so far.
    record : list[tuple[int, float]]
        A record of the best value at each iteration
        for which a new best value was found.
    steps : list[np.ndarray]
        A record of the solutions at each iteration
        for which a new best value was found.
    """

    def __init__(
        self: "MetaModeSolver",
        correlation_matrix: np.ndarray,
        n_modes: int,
        n_channels: int,
        objective: str,
    ) -> None:
        """Initialize a MetaModeSolver object.

        Parameters
        ----------
        correlation_matrix : np.ndarray
            The correlation matrix.
        n_modes : int
            The number of modes.
        n_channels : int
            The number of channels.
        objective : str
            The objective function to evaluate.
        """
        self.problem = MetaModeProblem(
            correlation_matrix=correlation_matrix,
            n_modes=n_modes,
            n_channels=n_channels,
            objective=objective,
        )
        solution = MetaModeSolution(
            n_modes=n_modes,
            n_channels=n_channels,
        )
        self.best_solution = solution.random_permutation()
        self.best_value = self.problem.evaluate(self.best_solution)
        self._iteration = 0

        self.record = []
        self.steps = []

        self._tqdm = None

    def step(self: "MetaModeSolver") -> bool:
        """Take a step in the annealing process.

        Returns
        -------
        bool
            Whether the step was accepted.
        """
        solution = self.best_solution.copy()
        solution.step()
        value = self.problem.evaluate(solution)
        if value > self.best_value:
            self.best_solution = solution
            self.best_value = value
            if self._tqdm is not None:
                self._tqdm.set_postfix(best_value=self.best_value)
            return True
        return False

    def solve(self: "MetaModeSolver", n_steps: int, *, progress: bool = True) -> None:
        """Solve the problem.

        Parameters
        ----------
        n_steps : int
            The number of steps to take.
        progress : bool
            Whether to show a progress bar.
        """
        self._tqdm = trange(n_steps, disable=not progress)
        for _ in self._tqdm:
            accepted = self.step()
            if accepted:
                self.record.append((self._iteration, self.best_value))
                self.steps.append(self.problem.generate(self.best_solution))
            self._iteration += 1

        self._tqdm.close()
        self._tqdm = None

    def get_solution(self: "MetaModeSolver") -> MetaModeSolution:
        """Get the best solution.

        Returns
        -------
        MetaModeSolution
            The best solution.
        """
        return self.best_solution

    def get_value(self: "MetaModeSolver") -> float:
        """Get the value of the best solution.

        Returns
        -------
        float
            The value of the best solution.
        """
        return self.best_value

    def generate(self: "MetaModeSolver") -> np.ndarray:
        """Generate a set of covariance matrices from the best solution.

        Returns
        -------
        np.ndarray
            A set of covariance matrices.
        """
        return self.problem.generate(solution=self.best_solution)
