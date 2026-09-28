"""Testes do serviço de decisão (Etapa 5).

A política usada nos testes é construída dentro do próprio arquivo, a partir de
uma base sintética com o mesmo esquema da base real. Assim a suíte não depende
de o `make train` ter sido executado antes e não altera os artefatos gravados em
`artifacts/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from adaptive_offers.config import ARMS, PDAYS_NEVER_CONTACTED, SEGMENT_COL
from adaptive_offers.data.prepare import build_events
from adaptive_offers.segmentation import Segmenter
from adaptive_offers.serving import Recommender

RNG = np.random.default_rng(17)

VALID_CLIENT = {
    "age": 58,
    "job": "retired",
    "education": "professional.course",
    "housing": "no",
    "loan": "no",
    "campaign": 1,
    "pdays": 3,
    "previous": 2,
    "poutcome": "success",
}


@pytest.fixture(scope="module")
def recommender() -> Recommender:
    """Política treinada sobre base sintética com o mesmo esquema da base real."""
    n = 4000
    raw = pd.DataFrame(
        {
            "age": RNG.integers(20, 75, n),
            "job": RNG.choice(["admin.", "blue-collar", "retired", "technician"], n),
            "education": RNG.choice(["basic.9y", "university.degree", "high.school"], n),
            "housing": RNG.choice(["yes", "no"], n),
            "loan": RNG.choice(["yes", "no"], n),
            "contact": RNG.choice(["cellular", "telephone"], n),
            "day_of_week": RNG.choice(["mon", "tue", "wed", "thu", "fri"], n),
            "duration": RNG.integers(10, 900, n),
            "campaign": RNG.integers(1, 8, n),
            "pdays": RNG.choice([PDAYS_NEVER_CONTACTED, 3, 7], n),
            "previous": RNG.integers(0, 4, n),
            "poutcome": RNG.choice(["nonexistent", "failure", "success"], n),
            "y": RNG.choice(["yes", "no"], n, p=[0.15, 0.85]),
        }
    )
    events = build_events(raw)
    segmenter = Segmenter(n_segments=3, random_state=0).fit(events)
    events[SEGMENT_COL] = segmenter.predict(events)

    return Recommender.from_events(
        events,
        segmenter,
        ARMS,
        n_segments=3,
        segment_labels={0: "Segmento A", 1: "Segmento B", 2: "Segmento C"},
        metadata={"saved_at": "2026-01-01T00:00:00Z"},
    )


@pytest.fixture
def client(recommender, monkeypatch) -> TestClient:
    """Cliente de teste com a política injetada no estado do serviço."""
    from adaptive_offers.api import main as api_main

    monkeypatch.setitem(api_main._state, "recommender", recommender)
    return TestClient(api_main.app)


@pytest.fixture
def offline_client(monkeypatch) -> TestClient:
    """Serviço sem política carregada, para verificar a degradação controlada."""
    from adaptive_offers.api import main as api_main

    monkeypatch.setitem(api_main._state, "recommender", None)
    monkeypatch.setitem(api_main._state, "error", "artefato ausente")
    return TestClient(api_main.app)


# --------------------------------------------------------------------------
# Metadados
# --------------------------------------------------------------------------
def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert "/recommend" in " ".join(body["endpoints"].keys())


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_arms"] == len(ARMS)


def test_health_answers_even_without_model(offline_client):
    """O `/health` deve responder mesmo degradado, pois é lido pelo orquestrador."""
    response = offline_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_recommend_returns_503_without_model(offline_client):
    response = offline_client.post("/recommend", json={"client": VALID_CLIENT})
    assert response.status_code == 503
    assert "make train" in response.json()["detail"]


# --------------------------------------------------------------------------
# Recomendação
# --------------------------------------------------------------------------
def test_recommend_returns_a_valid_decision(client):
    body = client.post("/recommend", json={"client": VALID_CLIENT}).json()

    assert body["arm"] in ARMS
    assert body["channel"] in ("cellular", "telephone")
    assert body["timing"] in ("mon", "tue", "wed", "thu", "fri")
    assert 0.0 <= body["expected_conversion"] <= 1.0
    assert 0.0 <= body["probability_best"] <= 1.0
    assert body["arm_label"]


def test_recommend_reports_calibrated_uncertainty(client):
    body = client.post("/recommend", json={"client": VALID_CLIENT}).json()

    low, high = body["credible_interval"]
    assert 0.0 <= low <= high <= 1.0
    assert body["uncertainty"] > 0


def test_greedy_mode_is_deterministic(client):
    payload = {"client": VALID_CLIENT, "explore": False}
    decisions = {client.post("/recommend", json=payload).json()["arm"] for _ in range(15)}
    assert len(decisions) == 1


def test_greedy_mode_never_flags_exploration(client):
    body = client.post("/recommend", json={"client": VALID_CLIENT, "explore": False}).json()
    assert body["exploring"] is False


def test_explore_mode_eventually_varies(client):
    """O Thompson Sampling precisa variar as escolhas, senão não está amostrando."""
    payload = {"client": VALID_CLIENT, "explore": True}
    decisions = {client.post("/recommend", json=payload).json()["arm"] for _ in range(60)}
    assert len(decisions) > 1


def test_ranking_is_sorted_and_respects_top_k(client):
    body = client.post(
        "/recommend", json={"client": VALID_CLIENT, "explore": False, "top_k": 4}
    ).json()

    scores = [item["expected_conversion"] for item in body["ranking"]]
    assert len(scores) == 4
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# Validação de entrada
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,value",
    [
        ("job", "astronauta"),
        ("education", "doutorado"),
        ("poutcome", "talvez"),
        ("housing", "quem sabe"),
        ("age", 7),
        ("age", 200),
        ("campaign", 0),
        ("previous", -1),
        ("pdays", -5),
    ],
)
def test_invalid_field_is_rejected(client, field, value):
    payload = {"client": {**VALID_CLIENT, field: value}}
    assert client.post("/recommend", json=payload).status_code == 422


def test_missing_field_is_rejected(client):
    payload = {"client": {k: v for k, v in VALID_CLIENT.items() if k != "poutcome"}}
    assert client.post("/recommend", json=payload).status_code == 422


def test_sensitive_fields_are_ignored_not_absorbed(client):
    """Campos sensíveis enviados por engano não podem influenciar a decisão."""
    poisoned = {**VALID_CLIENT, "renda": 250000, "genero": "F", "marital": "single"}
    a = client.post("/recommend", json={"client": poisoned, "explore": False}).json()
    b = client.post("/recommend", json={"client": VALID_CLIENT, "explore": False}).json()

    assert a["arm"] == b["arm"]
    assert a["segment"] == b["segment"]


# --------------------------------------------------------------------------
# Feedback: o ciclo adaptativo
# --------------------------------------------------------------------------
def test_feedback_updates_the_posterior(client):
    payload = {"client": VALID_CLIENT, "arm": ARMS[0], "reward": 1}

    first = client.post("/feedback", json=payload).json()
    second = client.post("/feedback", json=payload).json()

    assert first["status"] == "atualizado"
    assert second["observations"] == pytest.approx(first["observations"] + 1)


def test_feedback_moves_conversion_in_the_right_direction(client):
    payload = {"client": VALID_CLIENT, "arm": ARMS[3], "reward": 1}
    before = client.post("/feedback", json=payload).json()["expected_conversion"]

    for _ in range(40):
        client.post("/feedback", json=payload)

    after = client.post("/feedback", json=payload).json()["expected_conversion"]
    assert after > before


def test_feedback_rejects_unknown_arm(client):
    payload = {"client": VALID_CLIENT, "arm": "pombo-correio|sab", "reward": 1}
    assert client.post("/feedback", json=payload).status_code == 422


def test_feedback_rejects_non_binary_reward(client):
    payload = {"client": VALID_CLIENT, "arm": ARMS[0], "reward": 7}
    assert client.post("/feedback", json=payload).status_code == 422


# --------------------------------------------------------------------------
# Diagnóstico
# --------------------------------------------------------------------------
def test_segments_endpoint_describes_every_segment(client, recommender):
    body = client.get("/segments").json()

    assert len(body) == recommender.n_segments
    for item in body:
        assert item["best_arm"] in ARMS
        assert item["label"]
        assert 0.0 <= item["expected_conversion"] <= 1.0


def test_greedy_mode_is_fully_reproducible(client):
    """O modo `explore=False` é documentado como determinístico para auditoria.

    Teste de regressão. A `probability_best` era calculada por Monte Carlo com o
    mesmo gerador usado na decisão, de modo que duas chamadas idênticas
    devolviam valores diferentes. O braço escolhido não mudava, mas a resposta
    sim, o que já viola o contrato do endpoint.
    """
    payload = {"client": VALID_CLIENT, "explore": False}
    respostas = [client.post("/recommend", json=payload).json() for _ in range(5)]

    for campo in ("arm", "expected_conversion", "probability_best", "credible_interval"):
        assert len({str(r[campo]) for r in respostas}) == 1, (
            f"campo '{campo}' variou entre chamadas idênticas em modo determinístico"
        )


def test_health_lists_every_context_feature(client):
    """O `/health` deve declarar todas as features de contexto, não só as categóricas."""
    from adaptive_offers.config import CONTEXT_FEATURES

    assert client.get("/health").json()["context_features"] == list(CONTEXT_FEATURES)
