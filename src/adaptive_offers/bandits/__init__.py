"""Políticas de decisão: baselines de controle e algoritmos adaptativos."""

from .base import BanditPolicy
from .baseline import FixedBaseline, RandomPolicy, SegmentedBaseline
from .epsilon_greedy import ContextualEpsilonGreedy, EpsilonGreedy
from .thompson import (
    ContextualThompsonSampling,
    DiscountedThompsonSampling,
    ThompsonSampling,
)
from .ucb import UCB1, ContextualUCB1

__all__ = [
    "BanditPolicy",
    "FixedBaseline",
    "SegmentedBaseline",
    "RandomPolicy",
    "EpsilonGreedy",
    "ContextualEpsilonGreedy",
    "ThompsonSampling",
    "ContextualThompsonSampling",
    "DiscountedThompsonSampling",
    "UCB1",
    "ContextualUCB1",
]
