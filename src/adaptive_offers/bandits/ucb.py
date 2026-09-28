"""UCB1: otimismo diante da incerteza.

O Thompson Sampling explora sorteando valores da posterior. O UCB1 explora de
forma determinística, somando à média empírica de cada braço um bônus que cresce
com a incerteza e diminui conforme o braço é observado:

    indice(a) = media(a) + c * sqrt( ln(t) / n(a) )

O bônus representa o quanto a média empírica ainda pode estar subestimando o
braço. Braços pouco testados recebem bônus alto e acabam sendo escolhidos; à
medida que n(a) cresce, o bônus diminui e a decisão passa a ser determinada
pela evidência acumulada.

A vantagem sobre o Thompson Sampling é a auditabilidade, já que a escolha não
depende de sorteio e pode ser reproduzida exatamente ao se explicar uma decisão
passada. A desvantagem aparece com recompensas raras, como a conversão de
aproximadamente 11% desta base: o bônus teórico fica conservador e o algoritmo
explora mais do que seria necessário. É o que se observa nos resultados, em que
o UCB1 fica atrás das demais políticas adaptativas.

Referência: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time
Analysis of the Multiarmed Bandit Problem. Machine Learning, 47, 235-256.
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_UCB_C, RANDOM_SEED
from .base import BanditPolicy


class UCB1(BanditPolicy):
    """Upper Confidence Bound com constante de exploração configurável."""

    name = "ucb1"

    def __init__(
        self,
        n_arms: int,
        n_segments: int = 1,
        contextual: bool = False,
        c: float = DEFAULT_UCB_C,
        random_state: int = RANDOM_SEED,
    ):
        if c < 0:
            raise ValueError("c deve ser >= 0.")
        self.c = float(c)
        super().__init__(n_arms, n_segments, contextual, random_state)

    def _select(self, ctx: int) -> int:
        counts = self.counts[ctx]

        # Fase de inicialização. Para n(a) = 0 o bônus é infinito, de modo que
        # o índice só passa a ser comparável depois que todos os braços tiverem
        # sido testados ao menos uma vez.
        untried = np.flatnonzero(counts == 0)
        if untried.size > 0:
            return int(self.rng.choice(untried))

        total = counts.sum()
        means = self.rewards[ctx] / counts
        bonus = self.c * np.sqrt(np.log(total) / counts)
        return self._argmax_random_tiebreak(means + bonus)

    def confidence_bonus(self, segment: int = 0) -> np.ndarray:
        """Bônus de exploração corrente por braço (diagnóstico)."""
        ctx = self._ctx(segment)
        counts = self.counts[ctx]
        total = max(1, int(counts.sum()))
        return np.divide(
            self.c * np.sqrt(np.log(total) / np.maximum(counts, 1)),
            1.0,
            out=np.full(self.n_arms, np.inf),
            where=counts > 0,
        )

    @property
    def params(self) -> dict:
        return {**super().params, "c": self.c}


class ContextualUCB1(UCB1):
    """UCB1 com índices independentes por segmento."""

    name = "ucb1_contextual"

    def __init__(
        self,
        n_arms: int,
        n_segments: int,
        c: float = DEFAULT_UCB_C,
        random_state: int = RANDOM_SEED,
    ):
        super().__init__(n_arms, n_segments, contextual=True, c=c, random_state=random_state)
