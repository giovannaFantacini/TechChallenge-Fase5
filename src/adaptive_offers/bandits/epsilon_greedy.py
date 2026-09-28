"""Epsilon-Greedy: exploração com orçamento fixo ou decrescente.

O algoritmo entra no trabalho por duas razões.

A primeira é didática e de comunicação com a área de negócio: a regra "reserve
epsilon por cento do tráfego para testar e mande o restante para o melhor braço
conhecido" é simples de explicar e de auditar.

A segunda é servir de contraponto ao Thompson Sampling. O epsilon fixo mantém a
mesma fatia de tráfego em braços já identificados como ruins, indefinidamente.
Com 10 braços e epsilon = 0,10, aproximadamente 9% do tráfego vai
permanentemente para braços subótimos. Por isso o arrependimento acumulado
cresce de forma linear, enquanto o do Thompson Sampling cresce de forma
sublinear. As curvas do notebook 02 mostram essa diferença.

A variante com decaimento (`decay=True`) reduz epsilon ao longo do tempo e
atenua o problema, ao custo de um parâmetro adicional a calibrar.
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_EPSILON, RANDOM_SEED
from .base import BanditPolicy


class EpsilonGreedy(BanditPolicy):
    """Explora com probabilidade epsilon; explota o melhor braço caso contrário."""

    name = "epsilon_greedy"

    def __init__(
        self,
        n_arms: int,
        n_segments: int = 1,
        contextual: bool = False,
        epsilon: float = DEFAULT_EPSILON,
        decay: bool = False,
        min_epsilon: float = 0.01,
        random_state: int = RANDOM_SEED,
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon deve estar em [0, 1].")
        self.epsilon = float(epsilon)
        self.decay = decay
        self.min_epsilon = float(min_epsilon)
        super().__init__(n_arms, n_segments, contextual, random_state)

    def current_epsilon(self, ctx: int) -> float:
        """Epsilon efetivo no passo atual.

        Com decaimento, epsilon cai proporcionalmente a 1/sqrt(n), em que n é o
        número de observações do contexto. A taxa 1/sqrt(n) é a usual na
        literatura: reduz o desperdício de tráfego sem zerar a exploração cedo
        demais, o que travaria o algoritmo num ótimo local.
        """
        if not self.decay:
            return self.epsilon
        n = max(1, int(self.counts[ctx].sum()))
        return max(self.min_epsilon, self.epsilon / np.sqrt(n))

    def _select(self, ctx: int) -> int:
        if self.rng.random() < self.current_epsilon(ctx):
            return int(self.rng.integers(self.n_arms))

        counts = self.counts[ctx]
        # Um braço ainda não testado tem média indefinida, e não zero. Tratá-lo
        # como zero faria o algoritmo descartá-lo sem nunca o ter avaliado.
        untried = np.flatnonzero(counts == 0)
        if untried.size > 0:
            return int(self.rng.choice(untried))

        values = np.divide(
            self.rewards[ctx],
            counts,
            out=np.zeros(self.n_arms, dtype=float),
            where=counts > 0,
        )
        return self._argmax_random_tiebreak(values)

    @property
    def params(self) -> dict:
        return {
            **super().params,
            "epsilon": self.epsilon,
            "decay": self.decay,
            "min_epsilon": self.min_epsilon if self.decay else None,
        }


class ContextualEpsilonGreedy(EpsilonGreedy):
    """Epsilon-Greedy com contadores independentes por segmento."""

    name = "epsilon_greedy_contextual"

    def __init__(
        self,
        n_arms: int,
        n_segments: int,
        epsilon: float = DEFAULT_EPSILON,
        decay: bool = False,
        min_epsilon: float = 0.01,
        random_state: int = RANDOM_SEED,
    ):
        super().__init__(
            n_arms,
            n_segments,
            contextual=True,
            epsilon=epsilon,
            decay=decay,
            min_epsilon=min_epsilon,
            random_state=random_state,
        )
