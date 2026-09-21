"""Modular-arithmetic datasets for grokking experiments.

The whole point of the setup is that the model is never told the rule.  It
only ever sees (a, b) -> c examples for a fixed random subset of the p * p
possible input pairs, and has to recover the rule well enough to fill in the
pairs it has never seen.

Tokenisation follows the convention used by the grokking literature: the input
is the length-3 sequence [a, b, EQ] where EQ is a dedicated token with index p,
so the vocabulary has p + 1 entries.  The loss is taken at the final position
only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch

# --------------------------------------------------------------------------
# Binary operations on Z_p
#
# Each takes integer tensors a, b and the modulus p, and returns op(a, b) mod p.
# The names match the ones used in the grokking papers where they overlap.
# --------------------------------------------------------------------------


def _add(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    return (a + b) % p


def _sub(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    return (a - b) % p


def _mul(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    return (a * b) % p


def _sq_sum(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    return (a * a + b * b) % p


def _sq_sum_cross(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    # a^2 + a*b + b^2, the symmetric quadratic form used by Power et al.
    return (a * a + a * b + b * b) % p


def _cube_add(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    return (a ** 3 + b) % p


def _cube_cross(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    return (a ** 3 + a * b) % p


OPERATIONS: Dict[str, Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = {
    "add": _add,
    "sub": _sub,
    "mul": _mul,
    "sq_sum": _sq_sum,
    "sq_sum_cross": _sq_sum_cross,
    "cube_add": _cube_add,
    "cube_cross": _cube_cross,
}

# Operations whose output does not depend on the order of the operands.  Worth
# tracking because a symmetric task lets the model get away with a symmetric
# circuit, which changes what the attention pattern can look like.
SYMMETRIC_OPS = {"add", "mul", "sq_sum", "sq_sum_cross"}


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModularDataset:
    """The full p * p table, split into train and test by index.

    Attributes
    ----------
    inputs      (p*p, 3) int64 -- [a, b, EQ] for every pair, in row-major
                (a-major) order, so row index i corresponds to a = i // p,
                b = i % p.
    labels      (p*p,)  int64 -- op(a, b) mod p for every pair.
    train_idx   (n_train,) int64 -- indices into inputs/labels.
    test_idx    (n_test,)  int64
    """

    p: int
    op: str
    inputs: torch.Tensor
    labels: torch.Tensor
    train_idx: torch.Tensor
    test_idx: torch.Tensor
    seed: int
    train_frac: float

    # -- convenience views ---------------------------------------------------

    @property
    def eq_token(self) -> int:
        return self.p

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def n_train(self) -> int:
        return int(self.train_idx.numel())

    @property
    def n_test(self) -> int:
        return int(self.test_idx.numel())

    def train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[self.train_idx], self.labels[self.train_idx]

    def test(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[self.test_idx], self.labels[self.test_idx]

    def all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs, self.labels

    def split_hash(self) -> str:
        """Fingerprint of the train/test split.

        The split is not stored; it is rebuilt from (p, op, train_frac, seed)
        through torch.randperm.  If a torch version ever changed randperm's
        output, every analysis of a shipped checkpoint would silently use a
        different split, so the fingerprint is recorded and checked.
        """
        import hashlib
        return hashlib.sha256(self.train_idx.to(torch.int64).numpy().tobytes()).hexdigest()[:16]

    def train_mask(self) -> torch.Tensor:
        """(p, p) bool grid, True where the pair is in the training set."""
        mask = torch.zeros(self.p * self.p, dtype=torch.bool)
        mask[self.train_idx] = True
        return mask.view(self.p, self.p)

    def label_grid(self) -> torch.Tensor:
        """(p, p) int64 grid of correct answers, indexed [a, b]."""
        return self.labels.view(self.p, self.p)


def make_dataset(
    p: int = 113,
    op: str = "add",
    train_frac: float = 0.3,
    seed: int = 0,
) -> ModularDataset:
    """Build the full table for `op` on Z_p and split it.

    The split is a permutation of all p * p pairs under a dedicated generator,
    so it depends only on (p, train_frac, seed) and not on any global RNG state
    or on the order in which other random draws happen.
    """
    if op not in OPERATIONS:
        raise KeyError(f"unknown operation {op!r}; known: {sorted(OPERATIONS)}")
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    a = torch.arange(p).repeat_interleave(p)   # a-major
    b = torch.arange(p).repeat(p)
    eq = torch.full((p * p,), p, dtype=torch.long)
    inputs = torch.stack([a, b, eq], dim=1).long()
    labels = OPERATIONS[op](a, b, p).long()

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=gen)
    n_train = int(train_frac * p * p)
    train_idx = perm[:n_train].sort().values
    test_idx = perm[n_train:].sort().values

    return ModularDataset(
        p=p,
        op=op,
        inputs=inputs,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        seed=seed,
        train_frac=train_frac,
    )
