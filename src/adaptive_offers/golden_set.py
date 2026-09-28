"""Etapa 4: Golden Set com cinco clientes de referência.

Métricas agregadas não revelam o comportamento em casos específicos. Uma
política pode apresentar boa conversão média e ainda assim tomar decisões sem
sentido para determinados perfis, que é justamente o que uma banca avaliadora,
uma área de risco ou uma auditoria precisam verificar.

Os cinco perfis a seguir foram definidos manualmente para cobrir os regimes de
decisão relevantes nesta base: cliente sem histórico, cliente que já converteu,
cliente que recusou, cliente saturado de contatos e cliente sênior sem
histórico. Cada caso registra a decisão esperada e a justificativa, de modo que
a saída possa ser avaliada quanto ao sentido, e não apenas comparada a um número.

O conjunto funciona também como teste de regressão de comportamento: se uma
alteração no modelo mudar qualquer uma dessas recomendações, a mudança precisa
ser intencional e justificada.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PDAYS_NEVER_CONTACTED
from .serving import Recommender


@dataclass
class GoldenCase:
    """Um cliente de referência e a expectativa de negócio sobre a decisão."""

    case_id: str
    persona: str
    client: dict
    expectation: str


GOLDEN_SET: list[GoldenCase] = [
    GoldenCase(
        case_id="GS-01",
        persona="Prospecção fria — profissional urbano, primeiro contato",
        client={
            "age": 34,
            "job": "admin.",
            "education": "university.degree",
            "housing": "yes",
            "loan": "no",
            "campaign": 1,
            "pdays": PDAYS_NEVER_CONTACTED,
            "previous": 0,
            "poutcome": "nonexistent",
        },
        expectation=(
            "Cliente sem nenhum histórico de campanha. A decisão esperada é o canal "
            "de maior conversão média da base, o celular. Neste caso o contexto "
            "informa pouco, e o comportamento correto da política é coincidir com o "
            "melhor braço global."
        ),
    ),
    GoldenCase(
        case_id="GS-02",
        persona="Reengajamento quente — converteu na campanha anterior",
        client={
            "age": 58,
            "job": "retired",
            "education": "professional.course",
            "housing": "no",
            "loan": "no",
            "campaign": 1,
            "pdays": 3,
            "previous": 2,
            "poutcome": "success",
        },
        expectation=(
            "Caso central do conjunto. Este segmento converte cerca de seis vezes a "
            "média da base, e a conversão esperada deve refletir essa diferença, o "
            "que evidencia que o contexto participa da decisão. O braço escolhido "
            "pode coincidir com o do GS-01, porque nesta base a personalização se "
            "manifesta na expectativa calibrada e no dia da abordagem, e não na troca "
            "de canal. Uma recomendação de telefone fixo aqui indicaria falha do "
            "encolhimento, pois essa célula tem apenas 21 observações."
        ),
    ),
    GoldenCase(
        case_id="GS-03",
        persona="Reengajamento morno — recusou na campanha anterior",
        client={
            "age": 45,
            "job": "blue-collar",
            "education": "basic.9y",
            "housing": "yes",
            "loan": "yes",
            "campaign": 2,
            "pdays": 10,
            "previous": 1,
            "poutcome": "failure",
        },
        expectation=(
            "Cliente com histórico de campanha, porém negativo. A conversão esperada "
            "deve ficar bem abaixo da do GS-02, o que verifica se a política distingue "
            "o cliente que aceitou do cliente que recusou, em vez de tratar qualquer "
            "contato anterior como sinal positivo."
        ),
    ),
    GoldenCase(
        case_id="GS-04",
        persona="Fadiga de campanha — 12 contatos no ciclo atual",
        client={
            "age": 29,
            "job": "technician",
            "education": "high.school",
            "housing": "no",
            "loan": "no",
            "campaign": 12,
            "pdays": PDAYS_NEVER_CONTACTED,
            "previous": 0,
            "poutcome": "nonexistent",
        },
        expectation=(
            "Cliente sob pressão elevada de contato. A conversão esperada deve ser "
            "baixa, indicando pouco valor em insistir. Este é o sinal sobre o qual "
            "atua a regra de supressão descrita no README: abaixo de um limiar, a "
            "decisão correta é não contatar."
        ),
    ),
    GoldenCase(
        case_id="GS-05",
        persona="Sênior aposentado — sem histórico de campanha",
        client={
            "age": 67,
            "job": "retired",
            "education": "basic.4y",
            "housing": "no",
            "loan": "no",
            "campaign": 1,
            "pdays": PDAYS_NEVER_CONTACTED,
            "previous": 0,
            "poutcome": "nonexistent",
        },
        expectation=(
            "Aposentados convertem acima da média nesta base, mas este cliente não "
            "possui histórico de campanha. O caso verifica se a política distingue um "
            "perfil demográfico favorável de um relacionamento já estabelecido."
        ),
    ),
]


def evaluate_golden_set(
    recommender: Recommender,
    cases: list[GoldenCase] | None = None,
    explore: bool = False,
) -> pd.DataFrame:
    """Executa o Golden Set e devolve o quadro de decisões.

    Args:
        recommender: política treinada.
        cases: conjunto de casos; usa o padrão do módulo se omitido.
        explore: mantido em False por padrão, usando o modo determinístico,
            porque a inspeção exige que a decisão seja reprodutível entre
            execuções.
    """
    cases = cases or GOLDEN_SET
    rows = []

    for case in cases:
        rec = recommender.recommend(case.client, explore=explore)
        rows.append(
            {
                "caso": case.case_id,
                "persona": case.persona,
                "segmento": rec.segment,
                "segmento_label": rec.segment_label,
                "recomendacao": rec.arm_label,
                "braco": rec.arm,
                "conversao_esperada": round(rec.expected_conversion, 4),
                "ic95": (
                    f"[{rec.credible_interval[0]:.3f}, {rec.credible_interval[1]:.3f}]"
                ),
                "prob_melhor_braco": round(rec.probability_best, 3),
                "expectativa": case.expectation,
            }
        )

    return pd.DataFrame(rows)


def format_golden_report(table: pd.DataFrame) -> str:
    """Relatório textual do Golden Set, usado no README e na apresentação."""
    lines = ["=" * 78, "GOLDEN SET — 5 CASOS DE REFERÊNCIA", "=" * 78]

    for _, row in table.iterrows():
        lines += [
            "",
            f"[{row['caso']}] {row['persona']}",
            f"  Segmento atribuído .... {row['segmento']} — {row['segmento_label']}",
            f"  Próximo passo ......... {row['recomendacao']}",
            f"  Conversão esperada .... {row['conversao_esperada']:.2%}  IC95 {row['ic95']}",
            f"  P(ser o melhor braço) . {row['prob_melhor_braco']:.1%}",
            f"  Leitura ............... {row['expectativa']}",
        ]

    lines += ["", "=" * 78]
    return "\n".join(lines)
