"""Avaliação off-policy por replay, usando diretamente os eventos registrados.

O módulo `environment.py` compara as políticas num ambiente modelado. Aqui a
avaliação é feita sobre os registros reais, sem passar por um modelo de
recompensa, o que serve como validação independente do simulador.

Método do replay (Li, Chu, Langford & Schapire, 2011). Os eventos são
percorridos em ordem aleatória. Para cada evento a política escolhe um braço:

- se o braço escolhido coincide com o braço que foi de fato aplicado, a
  recompensa real é revelada, contabilizada, e a política aprende com ela;
- caso contrário o evento é descartado, porque o contrafactual é desconhecido e
  qualquer valor imputado seria arbitrário.

A estimativa da conversão é a média das recompensas dos eventos aproveitados.

Viés do método e tratamento adotado. A garantia de não-viés do replay exige que
o log tenha sido gerado por uma política uniformemente aleatória. A base Bank
Marketing não atende a essa condição, pois a instituição contatou muito mais
clientes por celular do que por telefone fixo. Aplicar o replay sobre o log
bruto favoreceria políticas que por acaso concordam com a política histórica.

A correção implementada é o parâmetro `uniformize=True`, que subamostra os
eventos até que todos os braços tenham o mesmo número de registros, recriando a
condição exigida pelo método. O custo é o descarte de dados: o tamanho efetivo
passa a ser n_braços x menor_braço, cerca de 2.800 eventos por política. Os dois
modos ficam disponíveis para que a diferença entre eles possa ser inspecionada.

Consequência para a leitura dos resultados. Com esse volume, os intervalos de
confiança de quase todas as políticas se sobrepõem. O replay confirma que
nenhuma política está grosseiramente incorreta, mas não tem poder estatístico
para ordená-las.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..bandits.base import BanditPolicy
from ..config import ARM_COL, ARMS, RANDOM_SEED, REWARD_COL, SEGMENT_COL

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Resultado de uma avaliação por replay."""

    policy_name: str
    contextual: bool
    #: Eventos efetivamente aproveitados (braço escolhido == braço registrado)
    n_matched: int
    #: Eventos percorridos no log
    n_total: int
    conversions: int
    #: Recompensa acumulada ao longo dos eventos aproveitados
    reward_curve: np.ndarray

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.n_matched if self.n_matched else 0.0

    @property
    def match_rate(self) -> float:
        """Fração do log aproveitada, indicador da confiabilidade da estimativa."""
        return self.n_matched / self.n_total if self.n_total else 0.0

    @property
    def standard_error(self) -> float:
        """Erro padrão binomial da estimativa de conversão."""
        if self.n_matched < 2:
            return float("nan")
        p = self.conversion_rate
        return float(np.sqrt(p * (1.0 - p) / self.n_matched))

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Intervalo de confiança pela aproximação normal."""
        se = self.standard_error
        if np.isnan(se):
            return (float("nan"), float("nan"))
        p = self.conversion_rate
        return (max(0.0, p - z * se), min(1.0, p + z * se))

    def to_row(self) -> dict:
        low, high = self.confidence_interval()
        return {
            "politica": self.policy_name,
            "contextual": self.contextual,
            "conversao_replay": self.conversion_rate,
            "ic95_inferior": low,
            "ic95_superior": high,
            "eventos_aproveitados": self.n_matched,
            "taxa_aproveitamento": self.match_rate,
        }


def uniformize_log(
    events: pd.DataFrame, arms: list[str] = ARMS, random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """Subamostra o log para que todos os braços tenham o mesmo volume.

    Recria a condição de log uniforme exigida pelo método do replay, ao custo de
    descartar eventos dos braços mais frequentes.
    """
    present = events[events[ARM_COL].isin(arms)]
    per_arm = present[ARM_COL].value_counts()
    target = int(per_arm.min())

    rng = np.random.default_rng(random_state)
    picks = [
        rng.choice(group.index.to_numpy(), size=target, replace=False)
        for _, group in present.groupby(ARM_COL, sort=True)
    ]
    balanced = present.loc[np.concatenate(picks)].reset_index(drop=True)
    logger.info(
        "Log uniformizado: %d eventos (%d por braço), de %d originais.",
        len(balanced),
        target,
        len(events),
    )
    return balanced


def replay_evaluate(
    policy: BanditPolicy,
    events: pd.DataFrame,
    arms: list[str] = ARMS,
    uniformize: bool = True,
    random_state: int = RANDOM_SEED,
    learn: bool = True,
) -> ReplayResult:
    """Avalia uma política pelo método do replay sobre os eventos reais.

    Args:
        policy: política a avaliar (é resetada antes de começar).
        events: log com colunas de braço, recompensa e segmento.
        arms: espaço de ações, na ordem canônica.
        uniformize: equilibra o log entre braços antes de avaliar.
        random_state: semente do embaralhamento e da política.
        learn: quando False, a política não aprende, o que permite medir uma
            política congelada.
    """
    if SEGMENT_COL not in events.columns:
        raise ValueError(
            f"Coluna '{SEGMENT_COL}' ausente. Rotule os eventos com o Segmenter antes."
        )

    log = uniformize_log(events, arms, random_state) if uniformize else events
    log = log.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    policy.reset(random_state=random_state)
    arm_index = {arm: i for i, arm in enumerate(arms)}

    logged_arms = log[ARM_COL].map(arm_index).to_numpy()
    rewards = log[REWARD_COL].to_numpy()
    segments = log[SEGMENT_COL].to_numpy(dtype=int)

    matched_rewards: list[float] = []
    for logged_arm, reward, segment in zip(logged_arms, rewards, segments):
        chosen = policy.select(segment)
        if chosen != logged_arm:
            continue  # contrafactual desconhecido, o evento é descartado
        matched_rewards.append(float(reward))
        if learn:
            policy.update(segment, chosen, float(reward))

    curve = np.cumsum(matched_rewards) if matched_rewards else np.zeros(0)

    result = ReplayResult(
        policy_name=policy.name,
        contextual=policy.contextual,
        n_matched=len(matched_rewards),
        n_total=len(log),
        conversions=int(sum(matched_rewards)),
        reward_curve=curve,
    )
    logger.info(
        "%-32s replay=%.4f (n=%d, aproveitamento=%.1f%%)",
        policy.name,
        result.conversion_rate,
        result.n_matched,
        result.match_rate * 100,
    )
    return result


def compare_replay(
    policies: list[BanditPolicy],
    events: pd.DataFrame,
    arms: list[str] = ARMS,
    uniformize: bool = True,
    random_state: int = RANDOM_SEED,
) -> tuple[dict[str, ReplayResult], pd.DataFrame]:
    """Avalia várias políticas por replay e monta o quadro comparativo."""
    results = {
        p.name: replay_evaluate(p, events, arms, uniformize, random_state) for p in policies
    }
    table = pd.DataFrame([r.to_row() for r in results.values()])

    baseline = table.loc[table["politica"] == "baseline_fixo", "conversao_replay"]
    if not baseline.empty and baseline.iloc[0] > 0:
        table["uplift_vs_baseline_pct"] = (
            table["conversao_replay"] / baseline.iloc[0] - 1.0
        ) * 100.0

    return results, table.sort_values("conversao_replay", ascending=False).reset_index(drop=True)
