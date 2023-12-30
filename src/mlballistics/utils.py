"""Some utility functions for the mechanics module."""

import torch


def _ei(n: int, i: int) -> torch.Tensor:
    """Vector of the standard basis of R^n.

    Parameters
    ----------
    n
        Dimension of the vector space.
    i
        Index of the basis vector.

    Returns
    -------
    torch.Tensor
        Vector of the standard basis of R^n.

    """
    return torch.eye(n, dtype=torch.float)[i]
