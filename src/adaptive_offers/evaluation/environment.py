"""Ambiente de simulação calibrado com as taxas observadas na base.

Problema metodológico. Um bandit é um algoritmo online: ele escolhe uma ação e
observa a resposta àquela ação. Um conjunto de dados histórico é offline e
registra apenas o desfecho da ação que foi de fato tomada. O contrafactual, isto
é, o que teria acontecido se o contato tivesse sido feito na terça em vez da
quinta, não está no arquivo. Por isso não é possível simplesmente executar o
bandit sobre o CSV.

Solução adotada. Estimamos, a partir dos dados reais, a probabilidade de
conversão de cada par (segmento, braço) e usamos essas probabilidades como o
ambiente Bernoulli contra o qual as políticas competem. Todas as probabilidades
vêm de contagens observadas; nenhum valor é arbitrado.

Encolhimento das estimativas. Cinco das quarenta células têm menos de trinta
eventos, e a média bruta dessas células é dominada por ruído amostral. A célula
(segmento de clientes já convertidos, telephone|tue), por exemplo, tem 21
observações e 17 conversões, o que daria uma taxa de 81%. Para evitar que o
oráculo do simulador persiga esse ruído, aplicamos encolhimento empírico-Bayes:

    p(s,a) = (sucessos(s,a) + k * p_braco(a)) / (n(s,a) + k)

sendo k medido em número de observações equivalentes. Células com muita amostra
praticamente não se alteram; células com pouca amostra são puxadas para o
comportamento médio do braço. A implementação está em `statistics.py` e é
compartilhada com a política servida.

Limitação. As taxas estimam o comportamento da política que gerou os dados, e
não o efeito causal de alterar o canal. O ambiente serve para comparar políticas
sob as mesmas condições, não para projetar receita. A validação complementar,
feita diretamente sobre os eventos registrados, está em `replay.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    ARM_COL,
    ARMS,
    ENVIRONMENT_PATH,
    RANDOM_SEED,
    REWARD_COL,
    SEGMENT_COL,
    SHRINKAGE_STRENGTH,
    ensure_dirs,
)
from ..statistics import shrunk_arm_rates

logger = logging.getLogger(__name__)


class OfflineEnvironment:
    """Ambiente Bernoulli (segmento x braço) calibrado em dados reais."""

    def __init__(
        self,
        rates: np.ndarray,
        segment_probs: np.ndarray,
        arms: list[str],
        support: np.ndarray | None = None,
    ):
        self.rates = np.asarray(rates, dtype=float)
        self.segment_probs = np.asarray(segment_probs, dtype=float)
        self.arms = list(arms)
        self.support = (
            np.asarray(support, dtype=int)
            if support is not None
            else np.zeros_like(self.rates, dtype=int)
        )

        if self.rates.ndim != 2:
            raise ValueError("rates deve ser uma matriz (n_segments, n_arms).")
        if self.rates.shape[1] != len(self.arms):
            raise ValueError("rates e arms têm número de braços incompatível.")
        if self.rates.shape[0] != self.segment_probs.shape[0]:
            raise ValueError("rates e segment_probs têm número de segmentos incompatível.")

        self.n_segments, self.n_arms = self.rates.shape

    # -- calibração ---------------------------------------------------------
    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        arms: list[str] = ARMS,
        shrinkage: float = SHRINKAGE_STRENGTH,
    ) -> "OfflineEnvironment":
        """Calibra o ambiente a partir do log de eventos rotulado por segmento."""
        if SEGMENT_COL not in events.columns:
            raise ValueError(
                f"Coluna '{SEGMENT_COL}' ausente. Rotule os eventos com o Segmenter antes."
            )

        n_segments = int(events[SEGMENT_COL].max()) + 1
        n_arms = len(arms)

        successes = np.zeros((n_segments, n_arms))
        counts = np.zeros((n_segments, n_arms))

        grouped = events.groupby([SEGMENT_COL, ARM_COL])[REWARD_COL].agg(["sum", "count"])
        arm_index = {arm: i for i, arm in enumerate(arms)}
        for (segment, arm), row in grouped.iterrows():
            if arm not in arm_index:
                continue
            successes[int(segment), arm_index[arm]] = row["sum"]
            counts[int(segment), arm_index[arm]] = row["count"]

        # Encolhimento hierárquico em dois níveis: célula -> braço -> carteira.
        # O cálculo do alvo está em `statistics.py` e é o mesmo usado pela
        # política servida, de forma que o experimento e o serviço não possam
        # divergir na maneira de tratar células com pouca amostra.
        global_rates = shrunk_arm_rates(events, arms, shrinkage)

        rates = (successes + shrinkage * global_rates[None, :]) / (counts + shrinkage)

        segment_counts = events[SEGMENT_COL].value_counts().reindex(range(n_segments)).fillna(0)
        segment_probs = (segment_counts / segment_counts.sum()).to_numpy()

        thin = int((counts < 30).sum())
        if thin:
            logger.info(
                "%d de %d células têm menos de 30 eventos e foram fortemente encolhidas.",
                thin,
                counts.size,
            )

        return cls(rates, segment_probs, arms, support=counts.astype(int))

    # -- interação ----------------------------------------------------------
    def sample_segment(self, rng: np.random.Generator) -> int:
        """Sorteia um segmento respeitando a mistura real da base."""
        return int(rng.choice(self.n_segments, p=self.segment_probs))

    def rates_at(self, step: int) -> np.ndarray:
        """Matriz de taxas vigente no passo `step`.

        No ambiente estacionário as taxas são constantes. O método existe como
        ponto de extensão para que `DriftingEnvironment` possa alterar as taxas
        no meio do experimento sem exigir mudanças no laço de simulação.
        """
        return self.rates

    def pull(self, segment: int, arm: int, rng: np.random.Generator, step: int = 0) -> int:
        """Aplica o braço e devolve a conversão observada (0 ou 1)."""
        return int(rng.random() < self.rates_at(step)[segment, arm])

    def expected_reward(self, segment: int, arm: int) -> float:
        return float(self.rates[segment, arm])

    # -- oráculo ------------------------------------------------------------
    @property
    def best_arm_per_segment(self) -> np.ndarray:
        """Melhor braço de cada segmento, usado no cálculo do arrependimento."""
        return np.argmax(self.rates, axis=1)

    @property
    def best_arm_global(self) -> int:
        """Melhor braço na média ponderada da população (o teto de uma regra fixa)."""
        return int(np.argmax(self.segment_probs @ self.rates))

    @property
    def oracle_reward(self) -> float:
        """Conversão esperada de um oráculo que acerta o melhor braço por segmento."""
        best = self.rates.max(axis=1)
        return float(self.segment_probs @ best)

    def oracle_at(self, step: int) -> float:
        """Conversão do oráculo vigente no passo `step`."""
        return float(self.segment_probs @ self.rates_at(step).max(axis=1))

    def mean_oracle(self, horizon: int) -> float:
        """Conversão média do oráculo ao longo de todo o horizonte."""
        return self.oracle_reward

    @property
    def best_fixed_reward(self) -> float:
        """Conversão esperada da melhor regra fixa global possível.

        Corresponde ao teto de desempenho de qualquer política não contextual.
        A diferença entre este valor e `oracle_reward` delimita o ganho máximo
        que a personalização poderia capturar nesta base.
        """
        return float((self.segment_probs @ self.rates).max())

    def contextual_headroom(self) -> float:
        """Ganho relativo máximo da personalização sobre a melhor regra fixa."""
        return self.oracle_reward / self.best_fixed_reward - 1.0

    # -- inspeção e persistência -------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Matriz de taxas calibradas como DataFrame legível."""
        return pd.DataFrame(
            self.rates,
            index=[f"segmento_{i}" for i in range(self.n_segments)],
            columns=self.arms,
        )

    def save(self, path: Path = ENVIRONMENT_PATH) -> Path:
        ensure_dirs()
        Path(path).write_text(
            json.dumps(
                {
                    "rates": self.rates.tolist(),
                    "segment_probs": self.segment_probs.tolist(),
                    "arms": self.arms,
                    "support": self.support.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return Path(path)

    @classmethod
    def load(cls, path: Path = ENVIRONMENT_PATH) -> "OfflineEnvironment":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            np.array(payload["rates"]),
            np.array(payload["segment_probs"]),
            payload["arms"],
            np.array(payload["support"]),
        )

    def __repr__(self) -> str:
        return (
            f"<OfflineEnvironment segmentos={self.n_segments} braços={self.n_arms} "
            f"oráculo={self.oracle_reward:.4f} melhor_fixo={self.best_fixed_reward:.4f}>"
        )


class DriftingEnvironment(OfflineEnvironment):
    """Ambiente com quebra estrutural na metade do horizonte.

    Justificativa do cenário. O enunciado do desafio parte da premissa de que
    regras fixas demoram para reagir a mudanças de contexto. Essa premissa não
    se verifica num ambiente estacionário: se as taxas nunca mudam, a regra fixa
    apontada para o melhor braço é difícil de superar, porque não gasta tráfego
    explorando. O benefício de uma plataforma adaptativa só pode ser medido num
    ambiente que muda, e é isso que esta classe implementa.

    Choque modelado. A partir do passo `change_point`, os braços cujo nome
    começa por `affected_prefix` têm a taxa multiplicada por `factor`. O gatilho
    de negócio correspondente pode ser uma restrição regulatória ao contato por
    celular, uma onda de descadastramento ou a saturação do canal dominante após
    meses de uso intensivo.

    Observação importante. Nenhum dado de cliente é sintetizado. O ponto de
    partida continua sendo a matriz calibrada na base real, e o choque é um
    teste de estresse aplicado sobre ela, com magnitude declarada no `config.py`
    e reprodutível.
    """

    def __init__(
        self,
        base: OfflineEnvironment,
        change_point: int,
        affected_prefix: str = "cellular",
        factor: float = 0.35,
    ):
        super().__init__(base.rates, base.segment_probs, base.arms, base.support)
        self.change_point = int(change_point)
        self.affected_prefix = affected_prefix
        self.factor = float(factor)

        multiplier = np.array(
            [factor if a.startswith(affected_prefix) else 1.0 for a in self.arms]
        )
        self.affected_arms = [a for a in self.arms if a.startswith(affected_prefix)]
        self.rates_after = self.rates * multiplier[None, :]

    def rates_at(self, step: int) -> np.ndarray:
        return self.rates if step < self.change_point else self.rates_after

    # -- oráculo sensível ao tempo -----------------------------------------
    def best_arm_per_segment_at(self, step: int) -> np.ndarray:
        return np.argmax(self.rates_at(step), axis=1)

    @property
    def best_arm_per_segment_after(self) -> np.ndarray:
        return np.argmax(self.rates_after, axis=1)

    @property
    def oracle_reward_after(self) -> float:
        return float(self.segment_probs @ self.rates_after.max(axis=1))

    def mean_oracle(self, horizon: int) -> float:
        """Oráculo médio do horizonte, ponderado pelo tempo em cada regime."""
        if horizon <= self.change_point:
            return self.oracle_reward
        share_before = self.change_point / horizon
        return (
            share_before * self.oracle_reward
            + (1.0 - share_before) * self.oracle_reward_after
        )

    def describe(self) -> str:
        before = self.arms[int(np.argmax(self.segment_probs @ self.rates))]
        after = self.arms[int(np.argmax(self.segment_probs @ self.rates_after))]
        return (
            f"Choque no passo {self.change_point}: braços '{self.affected_prefix}*' "
            f"multiplicados por {self.factor:.2f}. "
            f"Melhor braço global antes: {before} | depois: {after}."
        )


def make_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
