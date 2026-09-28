"""Etapa 5: serviço de decisão em tempo real com FastAPI.

Expõe a política treinada como um endpoint de próxima melhor ação. Três
características distinguem este serviço de um invólucro genérico de modelo.

Primeiro, ele decide sob incerteza em vez de apenas prever. A resposta traz a
conversão esperada, o intervalo de credibilidade e a probabilidade de o braço
escolhido ser o melhor do segmento. Esses valores permitem que o canal defina
quando agir automaticamente e quando encaminhar o caso a um analista.

Segundo, ele continua aprendendo após a implantação. O endpoint `/feedback`
incorpora a resposta observada à posterior em tempo O(1), fechando o ciclo
adaptativo. Sem esse endpoint, a solução seria um modelo estático.

Terceiro, ele informa quando está explorando. O campo `exploring` indica que a
decisão foi uma tentativa deliberada, e não a melhor aposta corrente. Isso é
relevante para a atribuição de resultados, já que o tráfego de exploração não
deve ser contabilizado como tráfego otimizado na avaliação da campanha.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..config import POLICY_PATH, SEGMENTER_PATH, describe_arm
from ..serving import Recommender
from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    SegmentInfo,
)

logger = logging.getLogger(__name__)

#: Estado do processo. A política é carregada uma única vez na inicialização e
#: mantida em memória. As posteriors são a única estrutura consultada para
#: decidir e ocupam poucos kilobytes.
_state: dict = {"recommender": None, "error": None}


def get_recommender() -> Recommender:
    """Devolve a política carregada ou explica por que ela não está disponível."""
    recommender = _state.get("recommender")
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Política não carregada. Rode `make train` (ou "
                "`python -m adaptive_offers.train`) para gerar "
                f"{POLICY_PATH.name} e {SEGMENTER_PATH.name}. "
                f"Erro original: {_state.get('error')}"
            ),
        )
    return recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega a política na inicialização do serviço.

    Uma falha no carregamento não interrompe o processo. O serviço sobe e passa
    a responder 503 com a instrução de correção, mantendo o `/health`
    acessível. Essa escolha evita que um orquestrador entre em ciclo de
    reinicialização sem conseguir diagnosticar a causa.
    """
    try:
        _state["recommender"] = Recommender.load()
        logger.info("Política carregada: %s", _state["recommender"])
    except Exception as exc:
        _state["error"] = str(exc)
        logger.warning("Política indisponível na subida: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="Plataforma de Ofertas Adaptativas",
    description=(
        "Serviço de próxima melhor ação baseado em Thompson Sampling contextual. "
        "Datathon POSTECH — Machine Learning Engineering (Fase 5)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict:
    """Índice navegável do serviço."""
    return {
        "servico": "Plataforma de Ofertas Adaptativas",
        "versao": "1.0.0",
        "documentacao": "/docs",
        "endpoints": {
            "POST /recommend": "recomenda o próximo passo para um cliente",
            "POST /feedback": "registra a resposta observada e atualiza a política",
            "GET /segments": "segmentos aprendidos e melhor ação de cada um",
            "GET /health": "prontidão do serviço",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Prontidão do serviço. Responde mesmo quando a política não foi carregada."""
    recommender = _state.get("recommender")
    return HealthResponse(
        status="ok" if recommender else "degraded",
        model_loaded=recommender is not None,
        n_arms=recommender.n_arms if recommender else 0,
        n_segments=recommender.n_segments if recommender else 0,
        model_version=(
            recommender.metadata.get("saved_at", "desconhecida")
            if recommender
            else "não carregada"
        ),
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["decisão"])
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Recomenda o próximo passo (canal e dia) para um cliente elegível."""
    recommender = get_recommender()

    try:
        result = recommender.recommend(
            request.client.model_dump(), explore=request.explore, top_k=request.top_k
        )
    except Exception as exc:  # contexto válido no schema, porém não processável
        logger.exception("Falha ao recomendar")
        raise HTTPException(status_code=422, detail=f"Não foi possível decidir: {exc}")

    payload = result.to_dict()
    payload["model_version"] = recommender.metadata.get("saved_at", "desconhecida")
    return RecommendResponse(**payload)


@app.post("/feedback", response_model=FeedbackResponse, tags=["decisão"])
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Registra a resposta observada e atualiza a posterior do par (segmento, braço).

    Observação sobre a implantação: nesta implementação a atualização ocorre em
    memória, dentro do processo. Numa implantação real a posterior precisa ficar
    em armazenamento compartilhado, conforme descrito na seção 8 do README. Com
    várias réplicas do serviço e estado local, cada réplica aprenderia de forma
    independente e as políticas divergiriam entre si.
    """
    recommender = get_recommender()

    segment = recommender.segment_of(request.client.model_dump())
    recommender.update(segment, request.arm, float(request.reward))

    index = recommender.arms.index(request.arm)
    return FeedbackResponse(
        status="atualizado",
        segment=segment,
        arm=request.arm,
        observations=float(recommender.observations(segment)[index]),
        expected_conversion=float(recommender.posterior_mean(segment)[index]),
    )


@app.get("/segments", response_model=list[SegmentInfo], tags=["diagnóstico"])
def segments() -> list[SegmentInfo]:
    """Segmentos aprendidos e a melhor ação corrente de cada um.

    Endpoint de diagnóstico, destinado à área de negócio auditar o comportamento
    da política sem precisar executar os notebooks.
    """
    recommender = get_recommender()

    output = []
    for segment in range(recommender.n_segments):
        means = recommender.posterior_mean(segment)
        best = int(means.argmax())
        output.append(
            SegmentInfo(
                segment=segment,
                label=recommender.segment_labels.get(segment, f"Segmento {segment}"),
                best_arm=recommender.arms[best],
                best_arm_label=describe_arm(recommender.arms[best]),
                expected_conversion=float(means[best]),
                observations=float(recommender.observations(segment).sum()),
            )
        )
    return output


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:
    """Converte erro de domínio em 422, evitando um 500 genérico."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})
