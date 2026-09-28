"""Etapa 2: preparação da base.

A base Bank Marketing já traz o alvo de conversão (`y`, assinatura do depósito a
prazo), de modo que nenhum dado sintético precisa ser gerado. O log de eventos
usado pelo bandit é construído diretamente das colunas originais.

Cada linha da base é tratada como um evento de decisão:

    contexto do cliente  ->  braço aplicado (canal | dia)  ->  recompensa (0/1)

Transformações aplicadas:

1. remoção de duplicatas exatas (12 linhas);
2. descarte de `duration`, que constitui vazamento temporal por só ser conhecida
   após o término da ligação;
3. descarte dos atributos sensíveis e das variáveis macroeconômicas do contexto
   de decisão, conforme a política de features do `config.py`;
4. conversão de `pdays == 999`, que na base significa "nunca contatado", em uma
   flag binária e um score de recência.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import (
    ACTION_CHANNEL_COL,
    ACTION_TIMING_COL,
    ARM_COL,
    ARM_SEPARATOR,
    ARMS,
    CONTEXT_FEATURES,
    LEAKAGE_COLUMNS,
    PDAYS_NEVER_CONTACTED,
    PROCESSED_CSV,
    PROCESSED_PARQUET,
    REWARD_COL,
    SOURCE_CONTEXT_COLUMNS,
    TARGET_COL,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def engineer_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Converte `pdays` em duas features derivadas.

    Na base original, `pdays == 999` codifica "cliente nunca contatado em
    campanha anterior", e não "contatado há 999 dias". Mantida como número, essa
    coluna posicionaria tais clientes a uma distância muito grande de todos os
    demais no espaço métrico usado pela clusterização, distorcendo os clusters.

    A codificação adotada é:

    - `previously_contacted`: 1 se houve contato anterior, 0 caso contrário;
    - `recency_score`: 1/(1 + dias), no intervalo [0, 1], crescente conforme o
      último contato é mais recente, e igual a 0 para quem nunca foi contatado.
    """
    out = df.copy()
    contacted = out["pdays"] != PDAYS_NEVER_CONTACTED
    out["previously_contacted"] = contacted.astype(int)
    out["recency_score"] = np.where(contacted, 1.0 / (1.0 + out["pdays"]), 0.0)
    return out


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma a base bruta no log de eventos (contexto, braço, recompensa)."""
    required = set(SOURCE_CONTEXT_COLUMNS) | {
        ACTION_CHANNEL_COL,
        ACTION_TIMING_COL,
        TARGET_COL,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes na base bruta: {sorted(missing)}")

    n_before = len(df)
    work = df.drop_duplicates().reset_index(drop=True)
    if len(work) < n_before:
        logger.info("Removidas %d linhas duplicadas.", n_before - len(work))

    # `duration` só é conhecida depois que a ligação termina, enquanto a decisão
    # é tomada antes do contato. Mantê-la caracterizaria vazamento temporal e
    # produziria métricas altas sem valor preditivo real.
    present_leakage = [c for c in LEAKAGE_COLUMNS if c in work.columns]
    if present_leakage:
        work = work.drop(columns=present_leakage)
        logger.info("Descartadas por vazamento temporal: %s", present_leakage)

    work = engineer_recency(work)

    work[ARM_COL] = (
        work[ACTION_CHANNEL_COL].astype(str)
        + ARM_SEPARATOR
        + work[ACTION_TIMING_COL].astype(str)
    )
    work[REWARD_COL] = (work[TARGET_COL].astype(str).str.strip() == "yes").astype(int)

    unknown_arms = set(work[ARM_COL].unique()) - set(ARMS)
    if unknown_arms:
        raise ValueError(f"Braços fora do espaço de ações declarado: {sorted(unknown_arms)}")

    events = work[CONTEXT_FEATURES + [ARM_COL, REWARD_COL]].reset_index(drop=True)

    logger.info(
        "Log de eventos: %d eventos, %d braços, conversão global %.4f",
        len(events),
        events[ARM_COL].nunique(),
        events[REWARD_COL].mean(),
    )
    return events


def context_from_client(raw: dict | list[dict]) -> pd.DataFrame:
    """Monta o contexto de decisão a partir dos dados brutos de um cliente.

    Aplica a mesma engenharia de atributos usada no treino. Concentrar esse
    cálculo numa única função evita que o serviço e o pipeline derivem
    `recency_score` de formas diferentes, divergência que não produziria erro
    algum e se manifestaria apenas como perda silenciosa de qualidade nas
    recomendações.
    """
    rows = raw if isinstance(raw, list) else [raw]
    frame = pd.DataFrame(rows)

    if "pdays" not in frame.columns:
        raise ValueError("Campo 'pdays' é obrigatório para derivar a recência.")

    frame = engineer_recency(frame)

    missing = [c for c in CONTEXT_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Campos de contexto ausentes: {missing}")

    return frame[CONTEXT_FEATURES]


def arm_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Volume e taxa de conversão observada por braço, em ordem decrescente."""
    summary = (
        events.groupby(ARM_COL)[REWARD_COL]
        .agg(eventos="count", conversoes="sum", taxa_conversao="mean")
        .sort_values("taxa_conversao", ascending=False)
    )
    return summary.reset_index()


def save_events(events: pd.DataFrame) -> None:
    """Persiste o log de eventos em parquet (com fallback para CSV)."""
    ensure_dirs()
    events.to_csv(PROCESSED_CSV, index=False)
    try:
        events.to_parquet(PROCESSED_PARQUET, index=False)
    except Exception as exc:  # sem engine parquet; o CSV já garante o fluxo
        logger.info("Parquet indisponível (%s); log salvo apenas em CSV.", exc)


def load_events() -> pd.DataFrame:
    """Carrega o log de eventos processado, preferindo parquet."""
    if PROCESSED_PARQUET.exists():
        return pd.read_parquet(PROCESSED_PARQUET)
    if PROCESSED_CSV.exists():
        return pd.read_csv(PROCESSED_CSV)
    raise FileNotFoundError(
        "Log de eventos não encontrado. Rode `make data` ou "
        "`python -m adaptive_offers.cli data` primeiro."
    )


def prepare(force_download: bool = False) -> pd.DataFrame:
    """Pipeline completo: baixa a base, constrói e salva o log de eventos."""
    from .download import load_raw

    raw = load_raw(force_download=force_download)
    events = build_events(raw)
    save_events(events)
    return events


if __name__ == "__main__":  # pragma: no cover - utilitário de linha de comando
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    events = prepare()
    print(f"\nEventos processados: {len(events):,}".replace(",", "."))
    print(f"\nConversão por braço:\n{arm_summary(events).to_string(index=False)}")
