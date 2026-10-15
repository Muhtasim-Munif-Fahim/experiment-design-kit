"""Thompson sampling adaptive allocation for A/B(/n) experiments.

Instead of a fixed 50/50 split, each new arrival is assigned to an arm by
sampling from the posterior of that arm's conversion rate (Beta-Bernoulli)
or mean reward (Normal with known variance / Normal-Normal). The policy
tracks allocations, observed successes/sums, and can report the probability
that each arm is best (Monte Carlo).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "ThompsonAllocationResult",
    "ThompsonAllocator",
    "simulate_thompson_ab",
]


@dataclass
class ThompsonAllocationResult:
    """Summary of a finished Thompson-sampling allocation run."""

    arms: List[str]
    counts: Dict[str, int]
    successes: Dict[str, float]
    rates: Dict[str, float]
    prob_best: Dict[str, float]
    n: int


class ThompsonAllocator:
    """Online Beta-Bernoulli Thompson sampling allocator.

    Parameters
    ----------
    arms :
        Arm names (e.g. ``["control", "treatment"]``).
    alpha0, beta0 :
        Prior Beta parameters shared by every arm.
    seed :
        RNG seed.
    """

    def __init__(
        self,
        arms: Sequence[str],
        *,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if len(arms) < 2:
            raise ValueError("need at least two arms")
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("alpha0 and beta0 must be positive")
        self.arms = [str(a) for a in arms]
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self._alpha = {a: self.alpha0 for a in self.arms}
        self._beta = {a: self.beta0 for a in self.arms}
        self._counts = {a: 0 for a in self.arms}
        self._successes = {a: 0.0 for a in self.arms}
        self._rng = np.random.default_rng(seed)

    def allocate(self) -> str:
        """Sample one arm from the current posteriors (Thompson step)."""
        samples = {
            a: self._rng.beta(self._alpha[a], self._beta[a]) for a in self.arms
        }
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float) -> None:
        """Update the Beta posterior for ``arm`` with a Bernoulli ``reward``."""
        if arm not in self._alpha:
            raise KeyError(f"unknown arm {arm!r}")
        r = float(reward)
        if not (0.0 <= r <= 1.0):
            raise ValueError("reward must be in [0, 1] for Beta-Bernoulli")
        self._counts[arm] += 1
        self._successes[arm] += r
        self._alpha[arm] += r
        self._beta[arm] += 1.0 - r

    def prob_best(self, n_draws: int = 5000) -> Dict[str, float]:
        """Monte Carlo P(arm is best) under current posteriors."""
        if n_draws < 1:
            raise ValueError("n_draws must be >= 1")
        wins = {a: 0 for a in self.arms}
        for _ in range(int(n_draws)):
            samples = {
                a: self._rng.beta(self._alpha[a], self._beta[a]) for a in self.arms
            }
            winner = max(samples, key=samples.get)
            wins[winner] += 1
        return {a: wins[a] / float(n_draws) for a in self.arms}

    def summary(self, *, n_draws: int = 5000) -> ThompsonAllocationResult:
        rates = {}
        for a in self.arms:
            tot = self._alpha[a] + self._beta[a]
            rates[a] = self._alpha[a] / tot
        return ThompsonAllocationResult(
            arms=list(self.arms),
            counts=dict(self._counts),
            successes=dict(self._successes),
            rates=rates,
            prob_best=self.prob_best(n_draws=n_draws),
            n=int(sum(self._counts.values())),
        )


def simulate_thompson_ab(
    true_rates: Sequence[float],
    n: int,
    *,
    arm_names: Optional[Sequence[str]] = None,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    seed: Optional[int] = None,
) -> ThompsonAllocationResult:
    """Simulate ``n`` arrivals under Thompson allocation with known rates."""
    true_rates = [float(r) for r in true_rates]
    if any(r < 0 or r > 1 for r in true_rates):
        raise ValueError("true_rates must be in [0, 1]")
    if arm_names is None:
        arm_names = [f"arm_{i}" for i in range(len(true_rates))]
    if len(arm_names) != len(true_rates):
        raise ValueError("arm_names and true_rates length mismatch")
    rng = np.random.default_rng(seed)
    alloc = ThompsonAllocator(arm_names, alpha0=alpha0, beta0=beta0, seed=None)
    # Use separate RNG for rewards; allocator has its own
    alloc._rng = np.random.default_rng(None if seed is None else seed + 1)
    rate_map = {a: p for a, p in zip(arm_names, true_rates)}
    for _ in range(int(n)):
        arm = alloc.allocate()
        reward = 1.0 if rng.random() < rate_map[arm] else 0.0
        alloc.update(arm, reward)
    return alloc.summary()
