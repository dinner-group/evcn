from typing import Iterable

import torch
from torch import Tensor

from .utils import backward_stop, forward_stop
from .vcn import vcn_data

__all__ = [
    "rate_empirical",
    "rate_vcn",
    "rate_vcn2",
    "mse_empirical",
    "mse_vcn2",
    "calibration_curve",
    "histogram_empirical",
    "histogram",
]


def rate_empirical(reactant: Iterable[Tensor], product: Iterable[Tensor]) -> float:
    """
    Estimate the transition rate by counting transitions and dividing by time.

    Parameters
    ----------
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.

    Returns
    -------
    rate
        Transition rate.

    """
    total = 0.0
    count = 0
    for i, (a, b) in enumerate(zip(reactant, product, strict=True)):
        # check shape
        if not (a.ndim == b.ndim == 1):
            raise ValueError(
                f"reactant[{i}], product[{i}] must be 1D (got {a.ndim}, {b.ndim})"
            )
        if not (a.shape == b.shape):
            raise ValueError(
                f"reactant[{i}], product[{i}] must have the same shape (got {a.shape}, {b.shape})"
            )

        # check dtype
        if a.dtype != torch.bool:
            raise TypeError(f"reactant[{i}] must have dtype bool (got {a.dtype})")
        if b.dtype != torch.bool:
            raise TypeError(f"product[{i}] must have dtype bool (got {b.dtype})")

        # check `a` and `b` mutually exclusive
        if torch.any(a & b):
            raise ValueError("a and b must not both be True")

        (t,) = torch.nonzero(a | b, as_tuple=True)
        total += torch.sum(a[t[:-1]] & b[t[1:]]).item()
        total += torch.sum(b[t[:-1]] & a[t[1:]]).item()
        count += 2 * len(a)

    return total / count


def rate_vcn(
    committor: Iterable[Tensor],
    reactant: Iterable[Tensor],
    product: Iterable[Tensor],
    lag: int,
    *,
    exact: bool = True,
) -> float:
    """
    Estimate the transition rate using the VCN/EVCN objective.

    Parameters
    ----------
    committor
        Committor value at each frame.
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
    rate
        Transition rate.

    """
    if lag <= 0:
        raise ValueError(f"lag must be a positive integer (got {lag})")
    total = 0.0
    count = 0
    for i, (q, a, b) in enumerate(zip(committor, reactant, product, strict=True)):
        # check shape
        if not (q.ndim == a.ndim == b.ndim == 1):
            raise ValueError(
                f"committor[{i}], reactant[{i}], product[{i}] must be 1D (got {q.ndim}, {a.ndim}, {b.ndim})"
            )
        if not (q.shape == a.shape == b.shape):
            raise ValueError(
                f"committor[{i}], reactant[{i}], product[{i}] must have the same shape "
                f"(got {q.shape}, {a.shape}, {b.shape})"
            )

        # check dtype
        if a.dtype != torch.bool:
            raise TypeError(f"reactant[{i}] must have dtype bool (got {a.dtype})")
        if b.dtype != torch.bool:
            raise TypeError(f"product[{i}] must have dtype bool (got {b.dtype})")

        # check `a` and `b` mutually exclusive
        if torch.any(a & b):
            raise ValueError(f"reactant[{i}] and product[{i}] must not both be True")

        q = torch.where(a, 0, torch.where(b, 1, q))
        q0, q1, dd, da, db, ad, bd, ab, ba = vcn_data(q, a, b, lag, exact=exact)
        terms = (
            dd * (q0 - q1) ** 2
            + da * q0**2
            + db * (q0 - 1) ** 2
            + ad * q1**2
            + bd * (q1 - 1) ** 2
            + ab
            + ba
        )
        total += torch.sum(terms).item()
        count += len(terms)

    return total / (2 * lag * count)


def rate_vcn2(
    committor: Iterable[Tensor],
    reactant: Iterable[Tensor],
    product: Iterable[Tensor],
    lag1: int,
    lag2: int,
    *,
    exact: bool = True,
) -> float:
    """
    Estimate the transition rate using the difference of VCN/EVCN objectives at two lag times.

    Parameters
    ----------
    committor
        Committor value at each frame.
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.
    lag1, lag2
        Lag times in frames.
    exact
        If True (default), use EVCN. If False, use VCN.

    Returns
    -------
    rate
        Transition rate.

    """
    rate1 = rate_vcn(committor, reactant, product, lag1, exact=exact)
    rate2 = rate_vcn(committor, reactant, product, lag2, exact=exact)
    return (lag1 * rate1 - lag2 * rate2) / (lag1 - lag2)


def mse_empirical(
    committor: Iterable[Tensor], reactant: Iterable[Tensor], product: Iterable[Tensor]
) -> float:
    """
    Estimate the committor MSE using next and last hitting states.

    Parameters
    ----------
    committor
        Committor value at each frame.
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.

    Returns
    -------
    mse
        Committor MSE.

    """
    total = 0.0
    count = 0
    for i, (q, a, b) in enumerate(zip(committor, reactant, product, strict=True)):
        # check shape
        if not (q.ndim == a.ndim == b.ndim == 1):
            raise ValueError(
                f"committor[{i}], reactant[{i}], product[{i}] must be 1D "
                f"(got {q.ndim}, {a.ndim}, {b.ndim})"
            )
        if not (q.shape == a.shape == b.shape):
            raise ValueError(
                f"committor[{i}], reactant[{i}], product[{i}] must have the same shape "
                f"(got {q.shape}, {a.shape}, {b.shape})"
            )

        # check dtype
        if a.dtype != torch.bool:
            raise TypeError(f"reactant[{i}] must have dtype bool (got {a.dtype})")
        if b.dtype != torch.bool:
            raise TypeError(f"product[{i}] must have dtype bool (got {b.dtype})")

        # check `a` and `b` mutually exclusive
        if torch.any(a & b):
            raise ValueError(f"reactant[{i}] and product[{i}] must not both be True")

        di = ~(a | b)
        next_stop = forward_stop(di)
        prev_stop = backward_stop(di)
        mask = (prev_stop >= 0) & (next_stop < len(q))

        q = q[mask]
        q_next = b[next_stop[mask]].float()
        q_prev = b[prev_stop[mask]].float()

        total += torch.sum((q - q_next) * (q - q_prev)).item()
        count += torch.sum(mask).item()

    return total / count


def mse_vcn2(
    committor: Iterable[Tensor],
    reactant: Iterable[Tensor],
    product: Iterable[Tensor],
    lag1: int,
    lag2: int,
    *,
    exact: bool = True,
) -> float:
    """
    Estimate the committor MSE using the difference of VCN/EVCN objectives at two lag times.

    Parameters
    ----------
    committor
        Committor value at each frame.
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.
    lag1, lag2
        Lag times in frames.
    exact
        If True (default), use EVCN. If False, use VCN.

    Returns
    -------
    mse
        Committor MSE.

    """
    rate1 = rate_vcn(committor, reactant, product, lag1, exact=exact)
    rate2 = rate_vcn(committor, reactant, product, lag2, exact=exact)
    return (rate1 - rate2) / (1 / lag1 - 1 / lag2)


def calibration_curve(
    committor: Iterable[Tensor],
    reactant: Iterable[Tensor],
    product: Iterable[Tensor],
    bins,
) -> tuple[Tensor, Tensor]:
    """
    Construct a calibration curve for the committor.

    Parameters
    ----------
    committor
        Committor value at each frame.
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.
    bins
        Number of bins to discretize the [0, 1] interval.

    Returns
    -------
    predicted
        Mean predicted committor in each bin.
    empirical
        Mean empirical committor in each bin.

    """
    edges = torch.linspace(0, 1, bins + 1)

    total = torch.zeros(bins)
    total_ref = torch.zeros(bins)
    count = torch.zeros(bins)

    for i, (q, a, b) in enumerate(zip(committor, reactant, product, strict=True)):
        # check shape
        if not (q.ndim == a.ndim == b.ndim == 1):
            raise ValueError(
                f"committor[{i}], reactant[{i}], product[{i}] must be 1D "
                f"(got {q.ndim}, {a.ndim}, {b.ndim})"
            )
        if not (q.shape == a.shape == b.shape):
            raise ValueError(
                f"committor[{i}], reactant[{i}], product[{i}] must have the same shape "
                f"(got {q.shape}, {a.shape}, {b.shape})"
            )

        # check dtype
        if a.dtype != torch.bool:
            raise TypeError(f"reactant[{i}] must have dtype bool (got {a.dtype})")
        if b.dtype != torch.bool:
            raise TypeError(f"product[{i}] must have dtype bool (got {b.dtype})")

        # check `a` and `b` mutually exclusive
        if torch.any(a & b):
            raise ValueError(f"reactant[{i}] and product[{i}] must not both be True")

        di = ~(a | b)
        next_stop = forward_stop(di)
        prev_stop = backward_stop(di)
        mask = (prev_stop >= 0) & (next_stop < len(q))

        q = q[mask]
        q_ref = 0.5 * (b[next_stop[mask]].float() + b[prev_stop[mask]].float())

        idx = torch.searchsorted(edges[1:-1], q)

        total += torch.bincount(idx, weights=q, minlength=bins)
        total_ref += torch.bincount(idx, weights=q_ref, minlength=bins)
        count += torch.bincount(idx, minlength=bins)

    return total / count, total_ref / count


def histogram_empirical(
    reactant: Iterable[Tensor],
    product: Iterable[Tensor],
    cvs: Iterable[Tensor],
    bins: tuple[Tensor, ...] | list[Tensor],
) -> Tensor:
    """
    Histogram the empirical committor on collective variables.

    Parameters
    ----------
    reactant
        Whether each frame is in the reactant.
    product
        Whether each frame is in the product.
    cvs
        Collective variables at each frame. Each trajectory must be a 2D tensor.
    bins
        Bin edges for each collective variable.

    Returns
    -------
    hist
        Committor histogram.

    """
    shape = [len(bin) - 1 for bin in bins]
    total = torch.zeros(shape)
    count = torch.zeros(shape)
    for i, (a, b, vs) in enumerate(zip(reactant, product, cvs, strict=True)):
        # check shape
        if not (a.ndim == b.ndim == 1):
            raise ValueError(
                f"reactant[{i}], product[{i}] must be 1D (got {a.ndim}, {b.ndim})"
            )
        if vs.ndim != 2:
            raise ValueError(f"cvs[{i}] must be 2D (got {vs.ndim})")
        if not (len(a) == len(b) == len(vs)):
            raise ValueError(
                f"reactant[{i}], product[{i}], cvs[{i}] must have the same length "
                f"(got {len(a)}, {len(b)}, {len(vs)})"
            )

        # check dtype
        if a.dtype != torch.bool:
            raise TypeError(f"reactant[{i}] must have dtype bool (got {a.dtype})")
        if b.dtype != torch.bool:
            raise TypeError(f"product[{i}] must have dtype bool (got {b.dtype})")

        # check `a` and `b` mutually exclusive
        if torch.any(a & b):
            raise ValueError("a and b must not both be True")

        di = ~(a | b)
        next_stop = forward_stop(di)
        prev_stop = backward_stop(di)
        mask = (prev_stop >= 0) & (next_stop < len(di))

        q = 0.5 * (b[next_stop[mask]].float() + b[prev_stop[mask]].float())
        vs = vs[mask]

        total += torch.histogramdd(vs, bins, weight=q)[0]
        count += torch.histogramdd(vs, bins)[0]
    return total / count


def histogram(
    committor: Iterable[Tensor],
    cvs: Iterable[Tensor],
    bins: tuple[Tensor, ...] | list[Tensor],
) -> Tensor:
    """
    Histogram the committor on collective variables.

    Parameters
    ----------
    committor
        Committor value at each frame.
    cvs
        Collective variables at each frame. Each trajectory must be a 2D tensor.
    bins
        Bin edges for each collective variable.

    Returns
    -------
    hist
        Committor histogram.

    """
    shape = [len(bin) - 1 for bin in bins]
    total = torch.zeros(shape)
    count = torch.zeros(shape)
    for i, (q, vs) in enumerate(zip(committor, cvs, strict=True)):
        # check shape
        if q.ndim != 1:
            raise ValueError(f"committor[{i}] must be 1D (got {q.ndim})")
        if vs.ndim != 2:
            raise ValueError(f"cvs[{i}] must be 2D (got {vs.ndim})")
        if len(q) != len(vs):
            raise ValueError(
                f"committor[{i}], cvs[{i}] must have the same length "
                f"(got {len(q)}, {len(vs)})"
            )

        total += torch.histogramdd(vs, bins, weight=q)[0]
        count += torch.histogramdd(vs, bins)[0]
    return total / count
