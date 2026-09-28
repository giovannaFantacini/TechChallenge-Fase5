"""Contratos de entrada e saída da API.

Os schemas constituem a fronteira de validação do serviço, já que nenhum payload
alcança o modelo sem passar por eles. A validação impede que um campo fora do
domínio esperado, como uma profissão inexistente ou um `pdays` negativo, seja
atribuído a um cluster incorreto e gere uma recomendação sem sentido, sem que
nenhum erro seja levantado.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..config import (
    ARMS,
    CONTEXT_FEATURES,
    PDAYS_NEVER_CONTACTED,
)

# Domínios válidos, extraídos da base Bank Marketing.
JOBS = (
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
)
EDUCATIONS = (
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
)
YES_NO_UNKNOWN = ("yes", "no", "unknown")
POUTCOMES = ("failure", "nonexistent", "success")


class ClientRequest(BaseModel):
    """Contexto de um cliente elegível.

    O schema aceita apenas atributos comportamentais e de relacionamento. Renda,
    patrimônio, gênero, raça, estado civil e situação de inadimplência não
    possuem campo correspondente, de modo que não podem ser enviados nem
    utilizados na decisão. Ver a política de minimização de dados na seção 10 do
    README.
    """

    age: int = Field(..., ge=18, le=120, description="Idade do cliente")
    job: str = Field(..., description="Ocupação declarada")
    education: str = Field(..., description="Escolaridade")
    housing: str = Field(..., description="Possui financiamento imobiliário")
    loan: str = Field(..., description="Possui empréstimo pessoal")
    campaign: int = Field(
        ..., ge=1, le=100, description="Contatos já feitos nesta campanha"
    )
    pdays: int = Field(
        ...,
        ge=0,
        description=f"Dias desde o último contato ({PDAYS_NEVER_CONTACTED} = nunca contatado)",
    )
    previous: int = Field(
        ..., ge=0, le=100, description="Contatos em campanhas anteriores"
    )
    poutcome: str = Field(..., description="Desfecho da campanha anterior")

    @field_validator("job")
    @classmethod
    def _check_job(cls, value: str) -> str:
        if value not in JOBS:
            raise ValueError(f"job inválido. Valores aceitos: {sorted(JOBS)}")
        return value

    @field_validator("education")
    @classmethod
    def _check_education(cls, value: str) -> str:
        if value not in EDUCATIONS:
            raise ValueError(f"education inválida. Valores aceitos: {sorted(EDUCATIONS)}")
        return value

    @field_validator("housing", "loan")
    @classmethod
    def _check_yes_no(cls, value: str) -> str:
        if value not in YES_NO_UNKNOWN:
            raise ValueError(f"valor inválido. Aceitos: {sorted(YES_NO_UNKNOWN)}")
        return value

    @field_validator("poutcome")
    @classmethod
    def _check_poutcome(cls, value: str) -> str:
        if value not in POUTCOMES:
            raise ValueError(f"poutcome inválido. Aceitos: {sorted(POUTCOMES)}")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class RecommendRequest(BaseModel):
    """Pedido de recomendação."""

    client: ClientRequest
    explore: bool = Field(
        True,
        description=(
            "Quando True, valor padrão, a decisão é amostrada da posterior, o "
            "que mantém o aprendizado ativo em produção. Quando False, devolve a "
            "decisão de maior média posterior, de forma determinística, para "
            "auditoria e demonstração."
        ),
    )
    top_k: int = Field(3, ge=1, le=len(ARMS), description="Alternativas no ranking")


class ArmScore(BaseModel):
    """Alternativa avaliada, incluída na resposta para explicar a decisão."""

    arm: str
    arm_label: str
    expected_conversion: float
    probability_best: float
    observations: float


class RecommendResponse(BaseModel):
    """Decisão devolvida ao canal, acompanhada dos elementos que a justificam."""

    arm: str = Field(..., description="Braço escolhido (canal|dia)")
    arm_label: str = Field(..., description="Descrição legível do próximo passo")
    channel: str
    timing: str
    segment: int
    segment_label: str
    expected_conversion: float = Field(..., description="Média da posterior do braço")
    uncertainty: float = Field(..., description="Desvio-padrão da posterior")
    credible_interval: list[float] = Field(..., description="IC de 95% da conversão")
    probability_best: float = Field(..., description="P(ser o melhor braço do segmento)")
    exploring: bool = Field(
        ..., description="True quando a decisão veio de exploração, não de explotação"
    )
    ranking: list[ArmScore]
    model_version: str


class FeedbackRequest(BaseModel):
    """Resposta observada, enviada pelo canal após o contato.

    É a informação que fecha o ciclo adaptativo. Sem esse retorno, a política
    permaneceria estática após a implantação.
    """

    client: ClientRequest
    arm: str = Field(..., description="Braço efetivamente aplicado")
    reward: int = Field(..., ge=0, le=1, description="1 = converteu, 0 = não converteu")

    @field_validator("arm")
    @classmethod
    def _check_arm(cls, value: str) -> str:
        if value not in ARMS:
            raise ValueError(f"arm inválido. Valores aceitos: {ARMS}")
        return value


class FeedbackResponse(BaseModel):
    """Confirmação da atualização da posterior."""

    status: str
    segment: int
    arm: str
    observations: float = Field(..., description="Observações acumuladas nesse par")
    expected_conversion: float = Field(..., description="Posterior após a atualização")


class HealthResponse(BaseModel):
    """Prontidão do serviço."""

    status: str
    model_loaded: bool
    n_arms: int
    n_segments: int
    model_version: str
    context_features: list[str] = Field(
        default_factory=lambda: list(CONTEXT_FEATURES),
        description="Features que efetivamente entram na decisão",
    )


class SegmentInfo(BaseModel):
    """Descrição de um segmento e sua melhor ação corrente."""

    segment: int
    label: str
    best_arm: str
    best_arm_label: str
    expected_conversion: float
    observations: float
