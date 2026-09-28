"""Segmentação de clientes, que é a forma como o contexto entra na decisão.

Um bandit global aprende uma única resposta para toda a base, do tipo "qual o
melhor canal em média". O desafio pede que o contexto participe da decisão, e a
forma mais simples e auditável de fazer isso é discretizar o espaço de clientes
em poucos segmentos comportamentais, mantendo uma posterior independente por
segmento. Dentro de um segmento o problema volta a ser um bandit estacionário
clássico. Essa construção é conhecida como bandit contextual por discretização.

Escolhas de projeto.

KMeans sobre features padronizadas e codificadas em one-hot. É interpretável,
já que é possível descrever o perfil de cada cluster; é estável sob semente
fixa; e o custo de servir é baixo, pois `predict` calcula apenas a distância a K
centróides.

K igual a 4. Cada célula (segmento x braço) precisa de amostra suficiente para a
posterior Beta convergir. Com 10 braços, um K maior fragmentaria o aprendizado.
Os perfis obtidos estão documentados na seção 3 do README.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    N_SEGMENTS,
    RANDOM_SEED,
    SEGMENT_COL,
    SEGMENTER_PATH,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def _make_encoder() -> OneHotEncoder:
    """OneHotEncoder compatível com scikit-learn >= 1.2 e versões anteriores."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class Segmenter:
    """Agrupa clientes em segmentos comportamentais.

    Encapsula o pré-processamento e o KMeans num único objeto serializável, de
    forma que o treino, os notebooks e a API apliquem exatamente a mesma
    transformação sobre o contexto.
    """

    def __init__(self, n_segments: int = N_SEGMENTS, random_state: int = RANDOM_SEED):
        self.n_segments = n_segments
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    # -- ciclo de vida ------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        # O one-hot também passa pelo StandardScaler, e essa decisão é
        # necessária, não cosmética. O KMeans agrupa por distância euclidiana,
        # de modo que a influência de cada feature é proporcional à sua
        # variância. Uma coluna one-hot de categoria rara tem variância baixa:
        # `poutcome=success` aparece em 3,3% da base e tem variância em torno de
        # 0,03, contra 1,0 das numéricas padronizadas. Sem padronizá-la, o sinal
        # mais preditivo da base, que é o desfecho da campanha anterior, fica
        # diluído, e clientes que aceitaram a oferta caem no mesmo cluster de
        # clientes que a recusaram. O notebook 02 demonstra o efeito.
        categorical = Pipeline(
            [("onehot", _make_encoder()), ("scale", StandardScaler())]
        )
        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), CONTEXT_NUMERIC),
                ("cat", categorical, CONTEXT_CATEGORICAL),
            ]
        )
        return Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "kmeans",
                    KMeans(
                        n_clusters=self.n_segments,
                        n_init=10,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "Segmenter":
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df[CONTEXT_NUMERIC + CONTEXT_CATEGORICAL])
        logger.info("Segmentador treinado com %d segmentos.", self.n_segments)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Segmenter não treinado. Chame fit() ou load().")
        return self.pipeline.predict(df[CONTEXT_NUMERIC + CONTEXT_CATEGORICAL])

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).predict(df)

    # -- persistência -------------------------------------------------------
    def save(self, path: Path = SEGMENTER_PATH) -> Path:
        import joblib

        ensure_dirs()
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "n_segments": self.n_segments,
                "random_state": self.random_state,
            },
            path,
        )
        logger.info("Segmentador salvo em %s", path)
        return path

    @classmethod
    def load(cls, path: Path = SEGMENTER_PATH) -> "Segmenter":
        import joblib

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Segmentador não encontrado em {path}. Rode o treino antes de servir."
            )
        payload = joblib.load(path)
        segmenter = cls(
            n_segments=payload["n_segments"], random_state=payload["random_state"]
        )
        segmenter.pipeline = payload["pipeline"]
        return segmenter


def profile_segments(df: pd.DataFrame, segments: np.ndarray) -> pd.DataFrame:
    """Descreve cada segmento em linguagem de negócio.

    Sem essa descrição, o rótulo "segmento 2" não informa nada. Com ela, o
    segmento passa a ser identificado como "clientes com campanha anterior
    bem-sucedida", o que permite explicar a decisão para a área de negócio e
    para uma eventual auditoria.
    """
    work = df.copy()
    work[SEGMENT_COL] = segments

    rows = []
    for segment, group in work.groupby(SEGMENT_COL):
        rows.append(
            {
                "segmento": int(segment),
                "clientes": len(group),
                "share": len(group) / len(work),
                "idade_media": group["age"].mean(),
                "contato_previo_pct": group["previously_contacted"].mean(),
                "contatos_campanha_atual": group["campaign"].mean(),
                "poutcome_sucesso_pct": (group["poutcome"] == "success").mean(),
                "poutcome_falha_pct": (group["poutcome"] == "failure").mean(),
                "profissao_predominante": group["job"].mode().iloc[0],
                "escolaridade_predominante": group["education"].mode().iloc[0],
            }
        )

    profile = pd.DataFrame(rows).sort_values("segmento").reset_index(drop=True)
    if "reward" in work.columns:
        conv = work.groupby(SEGMENT_COL)["reward"].mean()
        profile["conversao_observada"] = profile["segmento"].map(conv)
    return profile


def label_segment(row: pd.Series) -> str:
    """Rótulo curto derivado do perfil estatístico do segmento.

    A ordem dos testes segue a hierarquia de informação observada na EDA: o
    desfecho da última campanha é o atributo mais preditivo da base e por isso é
    avaliado antes dos demais.
    """
    if row["poutcome_sucesso_pct"] > 0.5:
        return "Reengajamento quente (converteu na campanha anterior)"
    if row["poutcome_falha_pct"] > 0.5:
        return "Reengajamento morno (recusou na campanha anterior)"
    if row["contatos_campanha_atual"] > 4:
        return "Alta pressão de contato (fadiga de campanha)"
    return "Prospecção fria (sem histórico de campanha)"


def make_segment_labels(profile: pd.DataFrame) -> dict[int, str]:
    """Gera rótulos únicos por segmento.

    Dois clusters podem compartilhar o mesmo estado de relacionamento, situação
    que ocorre nesta base porque o KMeans divide a massa sem histórico em mais de
    um grupo. Nesse caso o rótulo é complementado com a profissão predominante e
    a idade média, para que nenhuma resposta da API saia com rótulo ambíguo.
    """
    base = profile.apply(label_segment, axis=1)
    counts = base.value_counts()

    labels: dict[int, str] = {}
    for position, segment in enumerate(profile["segmento"]):
        text = base.iloc[position]
        if counts[text] > 1:
            row = profile.iloc[position]
            text = (
                f"{text} — {row['profissao_predominante']}, "
                f"{row['idade_media']:.0f} anos"
            )
        labels[int(segment)] = text

    # Segunda passada. Dois clusters podem coincidir também na profissão e na
    # faixa etária. Nesse caso o índice do segmento é o único desempate
    # disponível: atribuir um rótulo descritivo diferente sugeriria uma
    # distinção de perfil que o modelo não encontrou.
    final_counts = pd.Series(list(labels.values())).value_counts()
    return {
        segment: (f"{text} (grupo {segment})" if final_counts[text] > 1 else text)
        for segment, text in labels.items()
    }
