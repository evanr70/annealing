"""Meta-mode solution class."""

import numpy as np

rng = np.random.default_rng()


class MetaModeSolution:
    """A permutation of the modes for each channel.

    Attributes
    ----------
    n_modes : int
        The number of modes.
    n_channels : int
        The number of channels.
    n_metamodes : int
        The number of meta-modes.
    unpermuted_modes : np.ndarray
        The unpermuted modes.
    permuted_modes : np.ndarray
        The permuted modes.
    """

    def __init__(
        self: "MetaModeSolution",
        n_modes: int,
        n_channels: int,
        permutation: np.ndarray | None = None,
    ) -> None:
        """Initialize a MetaModeSolution object.

        Parameters
        ----------
        n_modes : int
            The number of modes.
        n_channels : int
            The number of channels.
        permutation : np.ndarray, optional
            The permutation matrix, by default None.
        """
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.n_metamodes = n_modes

        if permutation is None:
            self.unpermuted_modes = np.tile(
                A=np.arange(self.n_modes),
                reps=(self.n_channels, 1),
            )
        else:
            self.unpermuted_modes = permutation

        self.permuted_modes = None

    def random_permutation(self: "MetaModeSolution") -> "MetaModeSolution":
        """Randomly permute the modes for each channel.

        Returns
        -------
        MetaModeSolution
            A solution with randomly permuted modes.
        """
        self.permuted_modes = np.apply_along_axis(
            func1d=rng.permutation,
            axis=1,
            arr=self.unpermuted_modes,
        )
        return self

    def permute_one_channel(
        self: "MetaModeSolution",
        channel: int,
    ) -> "MetaModeSolution":
        """Permute the modes for one channel.

        Parameters
        ----------
        channel : int
            The channel to permute.

        Returns
        -------
        MetaModeSolution
            The solution with one channel permuted.
        """
        self.permuted_modes = self.unpermuted_modes.copy()
        self.permuted_modes[channel] = rng.permutation(self.n_modes)
        return self

    def step(self: "MetaModeSolution") -> "MetaModeSolution":
        """Take a step in the annealing process.

        Returns
        -------
        MetaModeSolution
            The solution after taking a step.
        """
        self.permute_one_channel(
            channel=rng.integers(self.n_channels),
        )
        return self

    def copy(self: "MetaModeSolution") -> "MetaModeSolution":
        """Copy the solution.

        Returns
        -------
        MetaModeSolution
            A copy of the solution.
        """
        return MetaModeSolution(
            n_modes=self.n_modes,
            n_channels=self.n_channels,
            permutation=self.permuted_modes.copy(),
        )
