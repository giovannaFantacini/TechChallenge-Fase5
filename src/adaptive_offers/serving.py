"""Política treinada, serializada e pronta para servir (Etapa 5).

Como a política vai para produção. O experimento do módulo `evaluation` compara
os algoritmos partindo do zero, que é a forma correta de medir qual deles
aprende melhor. Colocar no ar um bandit com posteriors vazias, porém, faria com
que toda a fase de exploração já paga no histórico fosse repetida com clientes
reais.

Como a posterior Beta é conjugada da Bernoulli, as contagens históricas são a
estatística suficiente do problema. A posterior após observar todo o log tem
forma fechada:

    alpha(s,a) = prior_alpha(a) + conversões(s,a)
    beta(s,a)  = prior_beta(a)  + não-conversões(s,a)

Não há aproximação nem etapa de retreino: é a posterior exata dado o histórico.
Esse procedimento é chamado de warm start. A partir daí, cada resposta observada
em produção, recebida pelo endpoint `/feedback`, atualiza a posterior de forma
incremental em O(1), e o modelo continua aprendendo após a implantação.

O prior usado aqui é informativo e calibrado por braço (ver `statistics.py`), ao
contrário do prior uniforme adotado no experimento. A justificativa da diferença
está documentada naquele módulo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ARM_COL,
    ARMS,
    N_SEGMENTS,
    POLICY_PATH,
    RANDOM_SEED,
    REWARD_COL,
    SEGMENT_COL,
    SEGMENTER_PATH,
    SHRINKAGE_STRENGTH,
    describe_arm,
    ensure_dirs,
)
from .segmentation import Segmenter
from .statistics import arm_priors

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Decisão devolvida pela política para um cliente."""

    arm: str
    arm_label: str
    channel: str
    timing: str
    segment: int
    segment_label: str
    expected_conversion: float
    uncertainty: float
    credible_interval: tuple[float, float]
    probability_best: float
    #: True quando a decisão veio de exploração (não é o braço de maior média).
    exploring: bool
    ranking: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "arm_label": self.arm_label,
            "channel": self.channel,
            "timing": self.timing,
            "segment": self.segment,
            "segment_label": self.segment_label,
            "expected_conversion": self.expected_conversion,
            "uncertainty": self.uncertainty,
            "credible_interval": list(self.credible_interval),
            "probability_best": self.probability_best,
            "exploring": self.exploring,
            "ranking": self.ranking,
        }


def fit_posterior_from_events(
    events: pd.DataFrame,
    arms: list[str] = ARMS,
    n_segments: int = N_SEGMENTS,
    prior_alpha: float | None = None,
    prior_beta: float | None = None,
    shrinkage: float = SHRINKAGE_STRENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior Beta exata por (segmento, braço) dada a base histórica.

    Por padrão usa um prior informativo por braço, calibrado no próprio
    histórico (`statistics.arm_priors`). Isso ancora células com pouca amostra
    no comportamento conhecido do braço, em vez de deixá-las cravar uma opinião
    extrema a partir de algumas dezenas de registros. É o mesmo encolhimento que
    o ambiente de simulação aplica, pela mesma razão.

    Passar `prior_alpha`/`prior_beta` força um prior uniforme e explícito, útil
    para reproduzir o comportamento "aprender do zero" do experimento.
    """
    if SEGMENT_COL not in events.columns:
        raise ValueError(f"Coluna '{SEGMENT_COL}' ausente no log de eventos.")

    if prior_alpha is None or prior_beta is None:
        alpha0, beta0 = arm_priors(events, arms, shrinkage)
    else:
        alpha0 = np.full(len(arms), float(prior_alpha))
        beta0 = np.full(len(arms), float(prior_beta))

    alpha = np.tile(alpha0, (n_segments, 1))
    beta = np.tile(beta0, (n_segments, 1))
    arm_index = {arm: i for i, arm in enumerate(arms)}

    grouped = events.groupby([SEGMENT_COL, ARM_COL])[REWARD_COL].agg(["sum", "count"])
    for (segment, arm), row in grouped.iterrows():
        if arm not in arm_index or not 0 <= int(segment) < n_segments:
            continue
        s, a = int(segment), arm_index[arm]
        alpha[s, a] += float(row["sum"])
        beta[s, a] += float(row["count"]) - float(row["sum"])

    return alpha, beta


class Recommender:
    """Serve recomendações e incorpora o retorno observado.

    Mantém em memória as posteriors por segmento e o segmentador treinado. É o
    mesmo objeto usado pela API (Etapa 5) e pelo Golden Set (Etapa 4), o que
    assegura que a demonstração e o serviço executem o mesmo código de decisão.
    """

    def __init__(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        arms: list[str],
        segmenter: Segmenter,
        segment_labels: dict[int, str] | None = None,
        metadata: dict | None = None,
        prior_mass: np.ndarray | float = 2.0,
        random_state: int = RANDOM_SEED,
    ):
        self.alpha = np.asarray(alpha, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.arms = list(arms)
        self.segmenter = segmenter
        self.segment_labels = segment_labels or {}
        self.metadata = metadata or {}
        # Massa do prior por braço, armazenada para que `observations` possa
        # reportar apenas a evidência real. Sem descontá-la, um prior de força
        # 50 seria exibido como 50 observações que nunca ocorreram.
        self.prior_mass = np.broadcast_to(
            np.asarray(prior_mass, dtype=float), (self.alpha.shape[1],)
        ).copy()
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if self.alpha.shape != self.beta.shape:
            raise ValueError("alpha e beta devem ter o mesmo formato.")
        if self.alpha.shape[1] != len(self.arms):
            raise ValueError("Número de braços incompatível com as posteriors.")
        self.n_segments, self.n_arms = self.alpha.shape

    # -- decisão ------------------------------------------------------------
    def segment_of(self, client: dict | pd.DataFrame) -> int:
        """Determina o segmento do cliente a partir do contexto informado.

        Aceita tanto o payload bruto, com `pdays`, como chega da API, quanto um
        contexto já processado. A engenharia de atributos é delegada a
        `context_from_client`, a mesma função usada no treino, o que impede que
        as duas etapas derivem as features de formas diferentes.
        """
        from .data.prepare import context_from_client

        frame = client if isinstance(client, pd.DataFrame) else pd.DataFrame([client])
        if "pdays" in frame.columns:
            frame = context_from_client(frame.to_dict(orient="records"))
        return int(self.segmenter.predict(frame)[0])

    def posterior_mean(self, segment: int) -> np.ndarray:
        return self.alpha[segment] / (self.alpha[segment] + self.beta[segment])

    def posterior_std(self, segment: int) -> np.ndarray:
        a, b = self.alpha[segment], self.beta[segment]
        return np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1.0)))

    def observations(self, segment: int) -> np.ndarray:
        """Evidência efetivamente observada por braço, sem a massa do prior."""
        total = self.alpha[segment] + self.beta[segment] - self.prior_mass
        return np.maximum(total, 0.0)

    def probability_best(self, segment: int, n_samples: int = 5_000) -> np.ndarray:
        """Probabilidade de o braço ser o melhor do segmento, via Monte Carlo.

        Usa um gerador próprio com semente fixa, e não o gerador da decisão. O
        motivo é o contrato do modo `explore=False`, documentado como
        determinístico para fins de auditoria: se a resposta mudasse entre duas
        chamadas idênticas, esse contrato seria violado, ainda que o braço
        escolhido permanecesse o mesmo. Como esta métrica é diagnóstico e não
        fonte de exploração, fixar a semente não tem efeito colateral.
        """
        rng = np.random.default_rng(self.random_state + 1_000 + segment)
        draws = rng.beta(
            self.alpha[segment][None, :], self.beta[segment][None, :],
            size=(n_samples, self.n_arms),
        )
        return np.bincount(np.argmax(draws, axis=1), minlength=self.n_arms) / n_samples

    def recommend(
        self, client: dict, explore: bool = True, top_k: int = 3
    ) -> Recommendation:
        """Recomenda o próximo passo para um cliente.

        Args:
            explore: quando True, aplica Thompson Sampling amostrando da
                posterior. É o modo de produção, que mantém o aprendizado ativo.
                Quando False, devolve o braço de maior média posterior, usado em
                demonstrações e na auditoria da decisão.
            top_k: número de alternativas retornadas no ranking explicativo.
        """
        from scipy.stats import beta as beta_dist

        segment = self.segment_of(client)
        means = self.posterior_mean(segment)
        stds = self.posterior_std(segment)

        if explore:
            samples = self.rng.beta(self.alpha[segment], self.beta[segment])
            chosen = int(np.argmax(samples))
        else:
            chosen = int(np.argmax(means))

        greedy = int(np.argmax(means))
        prob_best = self.probability_best(segment)

        lower = beta_dist.ppf(0.025, self.alpha[segment], self.beta[segment])
        upper = beta_dist.ppf(0.975, self.alpha[segment], self.beta[segment])

        order = np.argsort(means)[::-1][:top_k]
        ranking = [
            {
                "arm": self.arms[i],
                "arm_label": describe_arm(self.arms[i]),
                "expected_conversion": float(means[i]),
                "probability_best": float(prob_best[i]),
                "observations": self.observations(segment)[i],
            }
            for i in order
        ]

        channel, timing = self.arms[chosen].split("|")
        return Recommendation(
            arm=self.arms[chosen],
            arm_label=describe_arm(self.arms[chosen]),
            channel=channel,
            timing=timing,
            segment=segment,
            segment_label=self.segment_labels.get(segment, f"Segmento {segment}"),
            expected_conversion=float(means[chosen]),
            uncertainty=float(stds[chosen]),
            credible_interval=(float(lower[chosen]), float(upper[chosen])),
            probability_best=float(prob_best[chosen]),
            exploring=chosen != greedy,
            ranking=ranking,
        )

    # -- aprendizado online -------------------------------------------------
    def update(self, segment: int, arm: str, reward: float) -> None:
        """Incorpora uma resposta observada em produção à posterior."""
        if arm not in self.arms:
            raise ValueError(f"Braço desconhecido: {arm}")
        if not 0 <= segment < self.n_segments:
            raise ValueError(f"Segmento {segment} fora do intervalo.")
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError("A recompensa deve ser 0 ou 1 (conversão binária).")

        index = self.arms.index(arm)
        self.alpha[segment, index] += reward
        self.beta[segment, index] += 1.0 - reward

    # -- persistência -------------------------------------------------------
    def save(self, path: Path = POLICY_PATH) -> Path:
        ensure_dirs()
        Path(path).write_text(
            json.dumps(
                {
                    "arms": self.arms,
                    "alpha": self.alpha.tolist(),
                    "beta": self.beta.tolist(),
                    "prior_mass": self.prior_mass.tolist(),
                    "segment_labels": {str(k): v for k, v in self.segment_labels.items()},
                    "metadata": {
                        **self.metadata,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("Política salva em %s", path)
        return Path(path)

    @classmethod
    def load(
        cls,
        path: Path = POLICY_PATH,
        segmenter_path: Path = SEGMENTER_PATH,
        random_state: int = RANDOM_SEED,
    ) -> "Recommender":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Política não encontrada em {path}. Rode `make train` antes de servir."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            alpha=np.array(payload["alpha"]),
            beta=np.array(payload["beta"]),
            arms=payload["arms"],
            segmenter=Segmenter.load(segmenter_path),
            segment_labels={int(k): v for k, v in payload.get("segment_labels", {}).items()},
            metadata=payload.get("metadata", {}),
            prior_mass=np.array(payload.get("prior_mass", [2.0] * len(payload["arms"]))),
            random_state=random_state,
        )

    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        segmenter: Segmenter,
        arms: list[str] = ARMS,
        n_segments: int = N_SEGMENTS,
        segment_labels: dict[int, str] | None = None,
        metadata: dict | None = None,
        random_state: int = RANDOM_SEED,
    ) -> "Recommender":
        """Warm-start: constrói a política a partir do histórico completo."""
        alpha0, beta0 = arm_priors(events, arms, SHRINKAGE_STRENGTH)
        alpha, beta = fit_posterior_from_events(events, arms, n_segments)
        return cls(
            alpha,
            beta,
            arms,
            segmenter,
            segment_labels=segment_labels,
            metadata={
                **(metadata or {}),
                "n_eventos_historicos": int(len(events)),
                "forca_prior": SHRINKAGE_STRENGTH,
            },
            prior_mass=alpha0 + beta0,
            random_state=random_state,
        )

    def __repr__(self) -> str:
        return f"<Recommender segmentos={self.n_segments} braços={self.n_arms}>"
