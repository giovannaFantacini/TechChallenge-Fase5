"""Políticas de controle (não adaptativas).

São os termos de comparação exigidos pela Etapa 3 do enunciado. Uma taxa de
conversão isolada não permite concluir nada sobre a qualidade de uma política;
é preciso compará-la com a alternativa que a instituição usaria hoje.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ARM_COL, RANDOM_SEED, REWARD_COL
from .base import BanditPolicy


class FixedBaseline(BanditPolicy):
    """Regra fixa: aplica sempre o braço de melhor conversão histórica.

    Representa a operação atual da instituição, isto é, uma decisão definida uma
    única vez a partir de um relatório histórico e mantida sem revisão.

    Trata-se de um baseline deliberadamente forte: ele já recebe o melhor braço
    conhecido e não gasta nenhum tráfego explorando alternativas. Escolher um
    baseline fraco tornaria a comparação favorável às políticas adaptativas por
    construção, e a conclusão do trabalho não teria valor.
    """

    name = "baseline_fixo"

    def __init__(
        self,
        n_arms: int,
        best_arm: int = 0,
        n_segments: int = 1,
        random_state: int = RANDOM_SEED,
    ):
        super().__init__(n_arms, n_segments, contextual=False, random_state=random_state)
        if not 0 <= best_arm < n_arms:
            raise ValueError(f"best_arm {best_arm} fora do intervalo [0, {n_arms}).")
        self.best_arm = int(best_arm)

    def _select(self, ctx: int) -> int:
        return self.best_arm

    @classmethod
    def from_events(
        cls, events: pd.DataFrame, arms: list[str], random_state: int = RANDOM_SEED
    ) -> "FixedBaseline":
        """Instancia a regra fixa a partir da conversão histórica observada."""
        rates = events.groupby(ARM_COL)[REWARD_COL].mean()
        best_arm_name = rates.idxmax()
        return cls(
            n_arms=len(arms),
            best_arm=arms.index(best_arm_name),
            random_state=random_state,
        )

    @property
    def params(self) -> dict:
        return {**super().params, "best_arm_index": self.best_arm}


class SegmentedBaseline(BanditPolicy):
    """Regra fixa por segmento: melhor braço histórico dentro de cada segmento.

    Baseline intermediário, que já incorpora personalização mas permanece
    congelado no tempo. Sua função é separar dois efeitos que costumam ser
    confundidos na leitura dos resultados: o ganho que vem de segmentar os
    clientes e o ganho que vem de continuar aprendendo depois da implantação.
    """

    name = "baseline_segmentado"

    def __init__(
        self,
        n_arms: int,
        n_segments: int,
        best_arms: np.ndarray | None = None,
        random_state: int = RANDOM_SEED,
    ):
        super().__init__(n_arms, n_segments, contextual=True, random_state=random_state)
        if best_arms is None:
            best_arms = np.zeros(n_segments, dtype=int)
        self.best_arms = np.asarray(best_arms, dtype=int)
        if self.best_arms.shape != (n_segments,):
            raise ValueError("best_arms deve ter um braço por segmento.")

    def _select(self, ctx: int) -> int:
        return int(self.best_arms[ctx])

    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        arms: list[str],
        segment_col: str,
        n_segments: int,
        random_state: int = RANDOM_SEED,
    ) -> "SegmentedBaseline":
        """Instancia a regra por segmento a partir do histórico observado."""
        rates = events.pivot_table(
            index=segment_col, columns=ARM_COL, values=REWARD_COL, aggfunc="mean"
        ).reindex(columns=arms)

        global_rates = events.groupby(ARM_COL)[REWARD_COL].mean().reindex(arms)
        # Células sem nenhuma observação recebem a taxa global do braço. Sem
        # esse tratamento, um NaN poderia ser selecionado como melhor braço.
        rates = rates.fillna(global_rates)

        best = np.zeros(n_segments, dtype=int)
        for segment in range(n_segments):
            if segment in rates.index:
                best[segment] = arms.index(rates.loc[segment].idxmax())
            else:
                best[segment] = int(np.argmax(global_rates.to_numpy()))
        return cls(len(arms), n_segments, best, random_state=random_state)

    @property
    def params(self) -> dict:
        return {**super().params, "best_arms": self.best_arms.tolist()}


class RandomPolicy(BanditPolicy):
    """Sorteio uniforme entre os braços, usado como piso de referência.

    Equivale a um teste A/B sem realocação de tráfego: todo o volume permanece
    em exploração, sem que o aprendizado acumulado seja aproveitado. Serve para
    delimitar o pior caso na comparação.
    """

    name = "aleatorio"

    def __init__(self, n_arms: int, n_segments: int = 1, random_state: int = RANDOM_SEED):
        super().__init__(n_arms, n_segments, contextual=False, random_state=random_state)

    def _select(self, ctx: int) -> int:
        return int(self.rng.integers(self.n_arms))
