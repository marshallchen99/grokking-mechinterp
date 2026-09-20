"""Fourier tooling for analysing models trained on Z_p.

Everything the network does is a function of two residues a, b in Z_p, so the
natural coordinate system for looking inside it is the discrete Fourier basis
over Z_p -- and, for quantities that depend on both inputs, the 2D product
basis over (a, b).

We use the real (cos/sin) basis rather than the complex one, because the
network's weights are real and the real basis makes the "the model represents
a number as a point on a circle" story directly visible.

Basis convention
----------------
    index 0          : const                      1 / sqrt(n)
    index 2k - 1     : cos(2 pi k x / n)           normalised to unit norm
    index 2k         : sin(2 pi k x / n)           normalised to unit norm

for k = 1 .. floor((n-1)/2).  When n is even there is one extra vector, the
Nyquist term cos(pi x) = (-1)^x, at the last index; its sine partner is
identically zero and is omitted.  Either way that is exactly n vectors, an
orthonormal basis of R^n.  We call k the *frequency*.

Even n matters here because of modular multiplication.  The nonzero residues
mod p form a cyclic group of order p - 1 under multiplication, so re-indexing
them by discrete logarithm turns multiplication into addition mod p - 1 -- and
p - 1 is even.  If a network trained on multiplication has found that
structure, its spectrum is sparse in the discrete-log coordinate and not in the
ordinary one, which is a sharp, falsifiable prediction.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def make_fourier_basis(n: int, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, List[str]]:
    """Return (F, names).

    F has shape (n, n); row i is the i-th basis vector evaluated at x = 0..n-1.
    F is orthonormal, so projecting is just `F @ v` and reconstructing is
    `F.T @ coeffs`.
    """
    if n < 2:
        raise ValueError(f"need n >= 2, got {n}")

    x = torch.arange(n, dtype=torch.float64)
    rows = [torch.ones(n, dtype=torch.float64)]
    names = ["const"]
    for k in range(1, (n + 1) // 2):
        angle = 2 * torch.pi * k * x / n
        rows.append(torch.cos(angle))
        names.append(f"cos {k}")
        rows.append(torch.sin(angle))
        names.append(f"sin {k}")
    if n % 2 == 0:
        # Nyquist: cos(pi x) = (-1)^x.  Its sine partner is identically zero.
        rows.append(torch.cos(torch.pi * x))
        names.append(f"cos {n // 2}")

    F = torch.stack(rows)                      # (n, n), float64 for accuracy
    F = F / F.norm(dim=1, keepdim=True)        # unit norm -> orthonormal
    assert F.shape == (n, n), (F.shape, n)
    return F.to(dtype), names


def fourier_1d(x: torch.Tensor, F: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Project `x` onto the Fourier basis along `dim`.

    `x` must have size p along `dim`.  Returns a tensor of the same shape whose
    `dim` axis is now indexed by basis element rather than by residue.
    """
    x = x.movedim(dim, 0)
    out = torch.tensordot(F, x, dims=([1], [0]))
    return out.movedim(0, dim)


def fourier_2d(x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Project a tensor whose first two axes are (a, b) onto the 2D basis.

    Element [i, j, ...] is the coefficient of  basis_i(a) * basis_j(b).
    """
    return fourier_1d(fourier_1d(x, F, dim=0), F, dim=1)


def inverse_fourier_2d(coeffs: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Inverse of `fourier_2d`."""
    out = torch.tensordot(F.T, coeffs, dims=([1], [0]))       # undo axis 0
    out = out.movedim(0, 0)
    out = torch.tensordot(F.T, out.movedim(1, 0), dims=([1], [0])).movedim(0, 1)
    return out


def basis_2d_names(names: List[str]) -> List[List[str]]:
    """Human-readable names for the 2D product basis, as a (p, p) nested list."""
    return [[f"{na}(a) {nb}(b)" if na != "const" or nb != "const" else "const"
             for nb in names] for na in names]


def frequency_of_index(i: int) -> int:
    """Which frequency k does basis index i belong to?  const -> 0."""
    return 0 if i == 0 else (i + 1) // 2


def index_frequencies(n: int) -> torch.Tensor:
    """(n,) int tensor giving the frequency of every basis index."""
    return torch.tensor([frequency_of_index(i) for i in range(n)])


def n_frequencies(n: int) -> int:
    """Number of frequency buckets, counting the constant term as frequency 0."""
    return n // 2 + 1


def discrete_log_table(p: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Re-index the nonzero residues mod p by discrete logarithm.

    Returns (dlog, exp_table, g) where g is the smallest primitive root,
    dlog[a] = the unique e in 0..p-2 with g^e = a (mod p) for a = 1..p-1, and
    exp_table[e] = g^e mod p.  dlog[0] is set to -1, since 0 is not in the
    multiplicative group.

    Under this re-indexing, a * b mod p becomes dlog[a] + dlog[b] mod (p - 1):
    multiplication is addition in disguise.
    """
    def is_primitive_root(g: int) -> bool:
        seen, x = set(), 1
        for _ in range(p - 1):
            x = (x * g) % p
            seen.add(x)
        return len(seen) == p - 1

    g = next(c for c in range(2, p) if is_primitive_root(c))
    dlog = torch.full((p,), -1, dtype=torch.long)
    exp_table = torch.zeros(p - 1, dtype=torch.long)
    x = 1
    for e in range(p - 1):
        dlog[x] = e
        exp_table[e] = x
        x = (x * g) % p
    return dlog, exp_table, g


def power_by_frequency(coeffs: torch.Tensor, p: int, dims=(0,)) -> torch.Tensor:
    """Sum of squared coefficients grouped by frequency.

    Returns a tensor indexed by frequency k = 0 .. floor(p/2), where k = 0 is
    the constant term.
    """
    freqs = index_frequencies(p)
    n_freq = p // 2 + 1
    sq = coeffs.pow(2)
    for d in sorted(dims, reverse=True):
        sq = sq.movedim(d, 0)
        grouped = torch.zeros((n_freq,) + sq.shape[1:], dtype=sq.dtype)
        grouped.index_add_(0, freqs, sq)
        sq = grouped.movedim(0, d)
    return sq


def gini(x: torch.Tensor) -> torch.Tensor:
    """Gini coefficient of a non-negative vector: 0 = uniform, ~1 = fully sparse.

    Used as a scalar summary of how concentrated a Fourier spectrum is.
    """
    x = x.flatten().abs().sort().values
    n = x.numel()
    if n == 0 or x.sum() == 0:
        return torch.tensor(0.0)
    idx = torch.arange(1, n + 1, dtype=x.dtype)
    return ((2 * idx - n - 1) * x).sum() / (n * x.sum())
