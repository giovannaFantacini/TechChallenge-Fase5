"""Contrato comum das políticas de decisão.

Todas as políticas do trabalho, do baseline determinístico ao Thompson Sampling
contextual, implementam a mesma interface:

    arm = policy.select(segment)      # decide qual braço aplicar
    policy.update(segment, arm, r)    # aprende com a resposta observada

Isso permite avaliar todas elas no mesmo laço de simulação e no mesmo
procedimento de replay, sem código específico por algoritmo.

Sobre a implementação da contextualidade: cada política guarda seu estado numa
matriz de dimensão (n_contexts, n_arms). Com `contextual=False`, n_contexts é 1
e todos os segmentos compartilham o mesmo conjunto de estatísticas, o que
corresponde ao bandit global. Com `contextual=True`, cada segmento possui seu
próprio conjunto. A regra de seleção é a mesma nos dois casos; o que muda é a
granularidade do aprendizado. Essa escolha evita duplicar cada algoritmo em uma
versão global e outra contextual.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import RANDOM_SEED


class BanditPolicy(ABC):
    """Classe base para políticas de seleção de braço."""

    #: Nome curto, usado nos relatórios, nos gráficos e nos runs do MLflow.
    name: str = "policy"

    def __init__(
        self,
        n_arms: int,
        n_segments: int = 1,
        contextual: bool = False,
        random_state: int = RANDOM_SEED,
    ):
        if n_arms < 1:
            raise ValueError("n_arms deve ser >= 1.")
        if n_segments < 1:
            raise ValueError("n_segments deve ser >= 1.")

        self.n_arms = n_arms
        self.n_segments = n_segments
        self.contextual = contextual
        self.random_state = random_state
        self.n_contexts = n_segments if contextual else 1
        self.rng = np.random.default_rng(random_state)
        self.t = 0
        self._init_state()

    # -- estado -------------------------------------------------------------
    def _init_state(self) -> None:
        """Contadores compartilhados por todas as políticas."""
        self.counts = np.zeros((self.n_contexts, self.n_arms), dtype=np.int64)
        self.rewards = np.zeros((self.n_contexts, self.n_arms), dtype=np.float64)

    def reset(self, random_state: int | None = None) -> None:
        """Zera o aprendizado. Usado entre as repetições do experimento."""
        if random_state is not None:
            self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.t = 0
        self._init_state()

    def _ctx(self, segment: int) -> int:
        """Mapeia o segmento para a linha correspondente da matriz de estado."""
        if not self.contextual:
            return 0
        if not 0 <= segment < self.n_segments:
            raise ValueError(
                f"Segmento {segment} fora do intervalo [0, {self.n_segments})."
            )
        return int(segment)

    # -- API pública --------------------------------------------------------
    def select(self, segment: int = 0) -> int:
        """Escolhe o índice do braço a aplicar para o contexto informado."""
        return int(self._select(self._ctx(segment)))

    def update(self, segment: int, arm: int, reward: float) -> None:
        """Incorpora a recompensa observada ao estado da política."""
        ctx = self._ctx(segment)
        self.counts[ctx, arm] += 1
        self.rewards[ctx, arm] += reward
        self.t += 1
        self._update(ctx, arm, reward)

    def expected_rewards(self, segment: int = 0) -> np.ndarray:
        """Estimativa pontual de conversão por braço (média empírica)."""
        ctx = self._ctx(segment)
        counts = self.counts[ctx]
        return np.divide(
            self.rewards[ctx],
            counts,
            out=np.zeros(self.n_arms, dtype=float),
            where=counts > 0,
        )

    # -- pontos de extensão -------------------------------------------------
    @abstractmethod
    def _select(self, ctx: int) -> int:
        """Regra de seleção específica do algoritmo."""

    def _update(self, ctx: int, arm: int, reward: float) -> None:
        """Atualização adicional específica do algoritmo (opcional)."""

    # -- utilidades ---------------------------------------------------------
    def _argmax_random_tiebreak(self, values: np.ndarray) -> int:
        """Argmax com desempate aleatório.

        Empates são frequentes no início da execução, quando todas as médias
        ainda valem zero. O argmax do numpy devolve sempre o menor índice, o que
        introduziria um viés sistemático a favor do primeiro braço do espaço de
        ações. O desempate aleatório elimina esse viés.
        """
        winners = np.flatnonzero(values == values.max())
        return int(self.rng.choice(winners))

    @property
    def params(self) -> dict:
        """Hiperparâmetros da política, para registro no MLflow."""
        return {
            "policy": self.name,
            "contextual": self.contextual,
            "n_arms": self.n_arms,
            "n_segments": self.n_segments if self.contextual else 1,
        }

    def __repr__(self) -> str:
        suffix = " (contextual)" if self.contextual else ""
        return f"<{type(self).__name__}{suffix} arms={self.n_arms} t={self.t}>"
