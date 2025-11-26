from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "forward_stop",
    "backward_stop",
    "count_transitions",
    "mlp",
    "CommittorOutput",
]


def forward_stop(d: Tensor) -> Tensor:
    """
    Find the first exit time from the domain.

    Parameters
    ----------
    d
        Whether each frame is in the domain.

    Returns
    -------
    t
        First exit time from the domain for each frame.
        A first exit time not within the trajectory is indicated by ``len(d)``.

    """
    if d.ndim != 1:
        raise ValueError(f"d must be 1D (got {d.ndim})")
    if d.dtype != torch.bool:
        raise ValueError(f"d must have dtype bool (got {d.dtype})")

    (t,) = torch.nonzero(torch.logical_not(d), as_tuple=True)
    t = torch.cat((torch.tensor([-1]), t, torch.tensor([len(d)])))
    return torch.repeat_interleave(t[1:], torch.diff(t))[:-1]


def backward_stop(d: Tensor) -> Tensor:
    """
    Find the last entry time into the domain.

    Parameters
    ----------
    d
        Whether each frame is in the domain.

    Returns
    -------
    t
        Last entry time into the domain for each frame.
        A last entry time not within the trajectory is indicated by ``-1``.

    """
    if d.ndim != 1:
        raise ValueError(f"d must be 1D (got {d.ndim})")
    if d.dtype != torch.bool:
        raise ValueError(f"d must have dtype bool (got {d.dtype})")

    (t,) = torch.nonzero(torch.logical_not(d), as_tuple=True)
    t = torch.cat((torch.tensor([-1]), t, torch.tensor([len(d)])))
    return torch.repeat_interleave(t[:-1], torch.diff(t))[1:]


def count_transitions(a: Tensor, b: Tensor, lag: int) -> Tensor:
    """
    Count the number of transition paths within each window.

    Parameters
    ----------
    a
        Whether each frame is in the reactant.
    b
        Whether each frame is in the product.
    lag
        Lag time in frames. The size of each window is ``lag+1``.

    Returns
    -------
        Number of transition paths in each window.

    """
    # check shape
    if not (a.ndim == b.ndim == 1):
        raise ValueError(f"a, b must be 1D (got {a.ndim}, {b.ndim})")
    if not (a.shape == b.shape):
        raise ValueError(f"a, b must have the same shape (got {a.shape}, {b.shape})")

    # check dtype
    if a.dtype != torch.bool:
        raise TypeError(f"a must have dtype bool (got {a.dtype})")
    if b.dtype != torch.bool:
        raise TypeError(f"b must have dtype bool (got {b.dtype})")

    if lag <= 0:
        raise ValueError(f"lag must be > 0 (got {lag})")

    n = len(a)

    (t,) = torch.nonzero(a | b, as_tuple=True)

    is_transition_path = torch.logical_and(a[t[:-1]], b[t[1:]])

    initial_count = torch.zeros(n, dtype=torch.int)
    initial_count[t[:-1][is_transition_path]] = 1
    initial_count = torch.cat(
        [torch.zeros(1, dtype=torch.int), torch.cumsum(initial_count, 0)]
    )

    final_count = torch.zeros(n, dtype=torch.int)
    final_count[t[1:][is_transition_path]] = 1
    final_count = torch.cat(
        [torch.zeros(1, dtype=torch.int), torch.cumsum(final_count, 0)]
    )

    out = final_count[lag + 1 :] - initial_count[: -(lag + 1)]
    out = torch.maximum(out, torch.tensor(0, dtype=torch.int))
    return out


def mlp(
    *sizes: int,
    activation: Callable[[], nn.Module] = nn.ReLU,
    final_activation: Callable[[], nn.Module] | None = None,
) -> nn.Module:
    """
    Create an MLP.

    Parameters
    ----------
    sizes
        Size of each layer (including input and output layers).
    activation
        Activation function between layers.
    final_activation
        Activation function after the last layer.

    Returns
    -------
    model
        MLP

    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        else:
            if final_activation is not None:
                layers.append(final_activation())
    return nn.Sequential(*layers)


class CommittorOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x.reshape(x.shape[:-1]))
