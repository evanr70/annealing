"""Objective functions for the metamodes module."""

from collections.abc import Callable
from enum import Enum

import numba
import numpy as np


class Objective(Enum):
    """Encapsulate objective functions."""

    SUM = "sum"
    ABS_SUM = "abs_sum"
    DYN_SUM = "dyn_sum"

    @property
    def func(self: "Objective") -> Callable:
        """Get the corresponding objective function.

        Returns
        -------
        The objective function.

        Raises
        ------
        KeyError: If the objective is not recognised.
        """
        undefined_msg = f"Objective not recognised. Choose from {self._names}"
        match self:
            case self.SUM:
                return sum_perms
            case self.ABS_SUM:
                return abs_sum_perms
            case _:
                raise KeyError(undefined_msg)

    def _names(self: "Objective") -> list[str]:
        """Return a list of names of all the Objective enum members."""
        return list(Objective.__members__.keys())


@numba.njit
def sum_perms(
    big_corr: np.ndarray,
    perm: np.ndarray,
    n_modes: int,
    n_channels: int,
) -> float:
    """Calculate sum of element-wise sums of permuted matrices.

    Parameters
    ----------
    big_corr : np.ndarray
        The correlation matrix.
    perm : np.ndarray
        The permutation matrix.
    n_modes : int
        The number of modes.
    n_channels : int
        The number of channels.

    Returns
    -------
    The sum of the permuted matrices.

    """
    ret = 0
    for i in range(n_modes):
        first_pass = big_corr[perm[:, i]]
        second_pass = first_pass[:, perm[:, i]]
        ret += second_pass.sum()
    return ret - n_modes * n_channels


@numba.njit
def abs_sum_perms(
    big_corr: np.ndarray,
    perm: np.ndarray,
    n_modes: int,
    n_channels: int,
) -> float:
    """Calculate sum of magnitudes of element-wise sums of permuted matrices.

    Parameters
    ----------
    big_corr : np.ndarray
        The correlation matrix.
    perm : np.ndarray
        The permutation matrix.
    n_modes : int
        The number of modes.
    n_channels : int
        The number of channels.

    Returns
    -------
    The sum of the permuted matrices.

    """
    ret = 0
    for i in range(n_modes):
        first_pass = big_corr[perm[:, i]]
        second_pass = first_pass[:, perm[:, i]]
        ret += abs(second_pass.sum())
    return ret - n_modes * n_channels
