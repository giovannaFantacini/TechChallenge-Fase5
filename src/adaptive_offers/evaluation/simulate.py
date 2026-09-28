"""Laço de simulação online e agregação dos resultados.

Cada passo do laço reproduz um atendimento:

1. chega um cliente elegível, cujo segmento é sorteado seguindo a distribuição
   real observada na base;
2. a política escolhe o próximo passo, ou seja, o par canal e dia;
3. o ambiente devolve a conversão observada;
4. a política atualiza seu estado, etapa que só tem efeito nas políticas
   adaptativas.

O experimento executa `n_runs` repetições com sementes diferentes. Uma única
execução de um algoritmo estocástico não permite afirmar que uma política supera
outra, já que a diferença observada pode vir da sequência sorteada. Reportamos a
média e o desvio-padrão entre repetições.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..bandits.base import BanditPolicy
from ..config import DEFAULT_HORIZON, DEFAULT_N_RUNS, RANDOM_SEED
from .environment import OfflineEnvironment

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Resultado agregado de uma política ao longo de várias repetições."""

    policy_name: str
    contextual: bool
    horizon: int
    n_runs: int
    #: (n_runs, horizon): recompensa observada a cada passo
    rewards: np.ndarray
    #: (n_runs, horizon): arrependimento instantâneo em relação ao oráculo
    regrets: np.ndarray
    #: (n_runs, n_arms): número de vezes que cada braço foi escolhido
    arm_counts: np.ndarray
    params: dict = field(default_factory=dict)

    # -- métricas centrais --------------------------------------------------
    @property
    def conversion_rate(self) -> float:
        """Taxa média de conversão, métrica de negócio pedida na Etapa 3."""
        return float(self.rewards.mean())

    @property
    def conversion_std(self) -> float:
        """Desvio-padrão da conversão entre repetições, não entre passos."""
        return float(self.rewards.mean(axis=1).std(ddof=1)) if self.n_runs > 1 else 0.0

    @property
    def total_conversions(self) -> float:
        """Conversões acumuladas por execução, em média."""
        return float(self.rewards.sum(axis=1).mean())

    @property
    def cumulative_regret(self) -> float:
        """Conversões perdidas em relação ao oráculo ao fim do horizonte."""
        return float(self.regrets.sum(axis=1).mean())

    @property
    def cumulative_regret_std(self) -> float:
        return float(self.regrets.sum(axis=1).std(ddof=1)) if self.n_runs > 1 else 0.0

    @property
    def mean_cumulative_reward(self) -> np.ndarray:
        return self.rewards.cumsum(axis=1).mean(axis=0)

    @property
    def mean_cumulative_regret(self) -> np.ndarray:
        return self.regrets.cumsum(axis=1).mean(axis=0)

    @property
    def arm_distribution(self) -> np.ndarray:
        """Fração do tráfego destinada a cada braço."""
        counts = self.arm_counts.mean(axis=0)
        return counts / counts.sum()

    def exploration_share(self, best_arm_global: int) -> float:
        """Fatia do tráfego enviada a braços diferentes do melhor braço global.

        Mede o custo de exploração, isto é, quanto do tráfego a política aplicou
        em alternativas em vez de usar o melhor braço médio. Numa política
        contextual, parte desse volume corresponde a personalização e não a
        desperdício, motivo pelo qual a métrica deve ser lida em conjunto com o
        arrependimento acumulado.
        """
        return float(1.0 - self.arm_distribution[best_arm_global])

    def final_conversion_rate(self, last_fraction: float = 0.2) -> float:
        """Conversão observada no trecho final do horizonte.

        A média sobre todo o horizonte inclui a fase de aprendizado, cujo custo é
        pago uma única vez. Este recorte mostra o comportamento já convergido,
        que é o que a operação observaria em produção após a fase inicial.
        """
        cut = max(1, int(self.horizon * (1.0 - last_fraction)))
        return float(self.rewards[:, cut:].mean())

    def final_expected_conversion(
        self, final_oracle: float, last_fraction: float = 0.2
    ) -> float:
        """Conversão esperada da política já convergida.

        É esta a métrica usada para afirmar que uma política supera outra no
        regime final, e não `final_conversion_rate`.

        A razão é estatística. A conversão observada carrega o ruído do sorteio
        Bernoulli: com taxa em torno de 14%, o desvio-padrão da média é da ordem
        de 0,1 ponto percentual mesmo com centenas de milhares de amostras, ou
        seja, a mesma ordem de grandeza da diferença que se pretende medir. Como
        o arrependimento é calculado sobre as probabilidades do ambiente, e não
        sobre o resultado sorteado, a expressão (oráculo menos arrependimento
        médio) estima a qualidade da decisão sem esse ruído.
        """
        cut = max(1, int(self.horizon * (1.0 - last_fraction)))
        return float(final_oracle - self.regrets[:, cut:].mean())

    def to_row(
        self, best_arm_global: int, mean_oracle: float, final_oracle: float
    ) -> dict:
        """Linha do quadro comparativo final."""
        return {
            "politica": self.policy_name,
            "contextual": self.contextual,
            "conversao_media": self.conversion_rate,
            "conversao_desvio": self.conversion_std,
            "conversao_esperada_regime_final": self.final_expected_conversion(final_oracle),
            "conversao_observada_regime_final": self.final_conversion_rate(),
            "conversoes_totais": self.total_conversions,
            "regret_acumulado": self.cumulative_regret,
            "regret_desvio": self.cumulative_regret_std,
            "exploracao_pct": self.exploration_share(best_arm_global),
        }


def run_policy(
    policy: BanditPolicy,
    env: OfflineEnvironment,
    horizon: int = DEFAULT_HORIZON,
    n_runs: int = DEFAULT_N_RUNS,
    base_seed: int = RANDOM_SEED,
) -> SimulationResult:
    """Executa uma política contra o ambiente por `n_runs` repetições."""
    rewards = np.zeros((n_runs, horizon))
    regrets = np.zeros((n_runs, horizon))
    arm_counts = np.zeros((n_runs, policy.n_arms))

    # As taxas são resolvidas fora do laço. Num ambiente com quebra estrutural
    # existem apenas dois regimes, de modo que basta alternar a referência no
    # ponto de mudança. Recalcular o ótimo a cada passo custaria um argmax por
    # iteração, o que é significativo em 100 mil passos por repetição.
    change_point = getattr(env, "change_point", None)
    rates_before = env.rates_at(0)
    best_before = np.argmax(rates_before, axis=1)
    optimum_before = rates_before[np.arange(env.n_segments), best_before]

    if change_point is not None and change_point < horizon:
        rates_after = env.rates_at(change_point)
        best_after = np.argmax(rates_after, axis=1)
        optimum_after = rates_after[np.arange(env.n_segments), best_after]
    else:
        change_point = horizon  # nunca alterna
        rates_after, optimum_after = rates_before, optimum_before

    segment_probs = env.segment_probs
    n_segments = env.n_segments

    for run in range(n_runs):
        seed = base_seed + run
        policy.reset(random_state=seed)
        # Semente do ambiente deslocada da semente da política: assim a sequência
        # de clientes é a mesma para todas as políticas na mesma repetição
        # (comparação pareada), mas independente do sorteio interno da política.
        rng = np.random.default_rng(10_000 + seed)
        segment_stream = rng.choice(n_segments, size=horizon, p=segment_probs)
        noise = rng.random(horizon)

        for step in range(horizon):
            if step < change_point:
                rates, optimum = rates_before, optimum_before
            else:
                rates, optimum = rates_after, optimum_after

            segment = int(segment_stream[step])
            arm = policy.select(segment)
            reward = int(noise[step] < rates[segment, arm])
            policy.update(segment, arm, reward)

            rewards[run, step] = reward
            # Arrependimento esperado, e não realizado: usa as probabilidades do
            # ambiente em vez do resultado sorteado. Isso remove o ruído
            # Bernoulli da curva e deixa visível a qualidade da decisão.
            regrets[run, step] = optimum[segment] - rates[segment, arm]
            arm_counts[run, arm] += 1

    logger.info(
        "%-32s conversão=%.4f regret=%7.1f",
        policy.name,
        rewards.mean(),
        regrets.sum(axis=1).mean(),
    )

    return SimulationResult(
        policy_name=policy.name,
        contextual=policy.contextual,
        horizon=horizon,
        n_runs=n_runs,
        rewards=rewards,
        regrets=regrets,
        arm_counts=arm_counts,
        params=policy.params,
    )


def compare_policies(
    policies: list[BanditPolicy],
    env: OfflineEnvironment,
    horizon: int = DEFAULT_HORIZON,
    n_runs: int = DEFAULT_N_RUNS,
    base_seed: int = RANDOM_SEED,
) -> tuple[dict[str, SimulationResult], pd.DataFrame]:
    """Roda todas as políticas e devolve os resultados e o quadro comparativo."""
    results: dict[str, SimulationResult] = {}
    for policy in policies:
        results[policy.name] = run_policy(policy, env, horizon, n_runs, base_seed)

    best_global = env.best_arm_global
    mean_oracle = env.mean_oracle(horizon)
    final_oracle = env.oracle_at(horizon - 1)

    table = pd.DataFrame(
        [r.to_row(best_global, mean_oracle, final_oracle) for r in results.values()]
    )

    # O ganho é sempre calculado em relação ao baseline fixo, que representa a
    # política em produção hoje. São reportadas duas leituras, porque respondem
    # a perguntas diferentes:
    #   uplift_vs_baseline_pct    -> compensa migrar, considerando o custo de
    #                                aprendizado pago uma única vez?
    #   uplift_regime_final_pct   -> uma vez convergida, a política é melhor?
    baseline = table.loc[table["politica"] == "baseline_fixo"]
    if not baseline.empty:
        base_mean = baseline["conversao_media"].iloc[0]
        base_final = baseline["conversao_esperada_regime_final"].iloc[0]
        if base_mean > 0:
            table["uplift_vs_baseline_pct"] = (
                table["conversao_media"] / base_mean - 1.0
            ) * 100.0
        if base_final > 0:
            table["uplift_regime_final_pct"] = (
                table["conversao_esperada_regime_final"] / base_final - 1.0
            ) * 100.0

    table["conversao_oraculo"] = mean_oracle
    table["pct_do_oraculo"] = table["conversao_media"] / mean_oracle * 100.0

    return results, table.sort_values("conversao_media", ascending=False).reset_index(drop=True)
