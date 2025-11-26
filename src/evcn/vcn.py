from collections.abc import Iterable
from typing import TypeAlias

import torch
from torch import Tensor, nn, optim
from torch.utils.data import ConcatDataset, Dataset, TensorDataset

from .utils import backward_stop, count_transitions, forward_stop

__all__ = [
    "VCNData",
    "vcn_data",
    "vcn_dataset",
    "vcn_loss",
    "training_step",
    "transform",
]


VCNData: TypeAlias = tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]


def vcn_data(x: Tensor, a: Tensor, b: Tensor, lag: int, exact: bool = True) -> VCNData:
    """
    Preprocess a single trajectory for VCN/EVCN.

    Parameters
    ----------
    x
        Input features at each frame.
    a
        Whether each frame is in the reactant.
    b
        Whether each frame is in the product.
    lag
        Lag time in frames.
    exact
        If True (default), use EVCN. If False, use VCN.

    Returns
    -------
    data
        Preprocessed data.

    """
    # check shape
    if not (a.ndim == b.ndim == 1):
        raise ValueError(f"a, b must be 1D (got {a.ndim}, {b.ndim})")
    if not (len(x) == len(a) == len(b)):
        raise ValueError(
            f"x, a, b must have the same lengths (got {len(x)}, {len(a)}, {len(b)})"
        )

    # check dtype
    if a.dtype != torch.bool:
        raise TypeError(f"a must have dtype bool (got {a.dtype})")
    if b.dtype != torch.bool:
        raise TypeError(f"b must have dtype bool (got {b.dtype})")

    # check `a` and `b` mutually exclusive
    if torch.any(a & b):
        raise ValueError("a and b must not both be True")

    if lag <= 0:
        raise ValueError(f"lag must be > 0 (got {lag})")

    x0 = x[:-lag]
    x1 = x[lag:]

    d = ~(a | b)

    if exact:
        s0 = torch.minimum(torch.arange(lag, len(d)), forward_stop(d)[:-lag])
        s1 = torch.maximum(torch.arange(len(d) - lag), backward_stop(d)[lag:])

        dd = d[s0]  # or, equivalently, dd = d[s1]
        da = a[s0]
        db = b[s0]
        ad = a[s1]
        bd = b[s1]
        ab = count_transitions(a, b, lag)
        ba = count_transitions(b, a, lag)
    else:
        dd = d[:-lag] & d[lag:]
        da = a[:-lag] | (d[:-lag] & a[lag:])
        db = b[:-lag] | (d[:-lag] & b[lag:])
        ad = a[lag:] | (d[lag:] & a[:-lag])
        bd = b[lag:] | (d[lag:] & b[:-lag])
        ab = a[:-lag] & b[lag:]
        ba = b[:-lag] & a[lag:]

    return x0, x1, dd, da, db, ad, bd, ab, ba


def vcn_dataset(
    features: Iterable[Tensor],
    reactant: Iterable[Tensor],
    product: Iterable[Tensor],
    lag: int,
    exact: bool = True,
) -> Dataset:
    """
    Make a dataset for training VCN/EVCN.

    Parameters
    ----------
    features
        Input features at each frame.
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.
    lag
        Lag time in frames.
    exact
        If True (default), use EVCN. If False, use VCN.

    Returns
    -------
    dataset
        VCN/EVCN training dataset.

    """
    datasets = []
    for x, a, b in zip(features, reactant, product, strict=True):
        data = vcn_data(x, a, b, lag, exact=exact)
        datasets.append(TensorDataset(*data))
    return ConcatDataset(datasets)


def vcn_loss(
    q0: Tensor,
    q1: Tensor,
    dd: Tensor,
    da: Tensor,
    db: Tensor,
    ad: Tensor,
    bd: Tensor,
    ab: Tensor,
    ba: Tensor,
    lag: int,
) -> Tensor:
    """
    VCN/EVCN loss function.

    """
    return torch.mean(
        dd * (q0 - q1) ** 2
        + da * q0**2
        + db * (q0 - 1) ** 2
        + ad * q1**2
        + bd * (q1 - 1) ** 2
        + ab
        + ba
    ) / (2 * lag)


def training_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: VCNData,
    lag: int,
    *,
    device: torch.device | None = None,
) -> float:
    """
    Perform one VCN/EVCN training step.

    """
    model.train()
    optimizer.zero_grad()

    x0, x1, *rest = [t.to(device) for t in batch]
    q0 = model(x0)
    q1 = model(x1)
    loss = vcn_loss(q0, q1, *rest, lag=lag)

    loss.backward()
    optimizer.step()
    return loss.item()


def transform(
    model: nn.Module,
    features: Iterable[Tensor],
    reactant: Iterable[Tensor],
    product: Iterable[Tensor],
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> list[Tensor]:
    """

    Parameters
    ----------
    model
        Committor model.
    features
        Input features at each frame.
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.
    batch_size
        Maximum batch size for evaluating the model.
        If None (default), each trajectory is evaluated at once.
    device
        Model device.

    Returns
    -------
    committor
        Committor value at each frame.

    """
    model.eval()
    with torch.no_grad():
        out = []
        for x, a, b in zip(features, reactant, product, strict=True):
            if batch_size is None:
                q = model(x.to(device)).cpu()
            else:
                q = []
                for batch in torch.split(x, batch_size):
                    q.append(model(batch.to(device)).cpu())
                q = torch.cat(q)
            q = torch.where(a, 0, torch.where(b, 1, q))
            out.append(q)
        return out
