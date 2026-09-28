"""Estimação empírica-Bayes compartilhada.

O módulo existe para garantir que o ambiente de simulação e a política servida
tratem células com pouca amostra exatamente da mesma forma. Se cada um tivesse
sua própria regra, o experimento mediria uma política e a produção executaria
outra, sem que nada no código sinalizasse a diferença. Foi o que ocorreu numa
versão anterior do trabalho, em que o simulador encolhia as estimativas e o
recomendador não, levando o Golden Set a sugerir um canal apoiado em apenas 21
observações.

A hierarquia de encolhimento é a mesma nos dois usos:

    célula (segmento x braço)  ->  braço  ->  carteira inteira

Uma célula com pouca amostra é puxada para a taxa do seu braço, e um braço com
pouca amostra, como ocorreria com um canal recém-introduzido, é puxado para a
taxa média da carteira. O parâmetro `strength` é expresso em número de
observações equivalentes: uma célula com `strength` observações próprias fica a
meio caminho entre sua média bruta e o alvo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ARM_COL, REWARD_COL, SHRINKAGE_STRENGTH


def shrunk_arm_rates(
    events: pd.DataFrame,
    arms: list[str],
    strength: float = SHRINKAGE_STRENGTH,
) -> np.ndarray:
    """Taxa de conversão por braço, encolhida para a taxa geral da carteira.

    Braços com histórico extenso ficam praticamente inalterados. Braços novos
    herdam o comportamento médio da carteira até acumular evidência própria.
    """
    overall = float(events[REWARD_COL].mean())

    stats = events.groupby(ARM_COL)[REWARD_COL].agg(["sum", "count"]).reindex(arms)
    successes = stats["sum"].fillna(0.0).to_numpy(dtype=float)
    counts = stats["count"].fillna(0.0).to_numpy(dtype=float)

    return (successes + strength * overall) / (counts + strength)


def arm_priors(
    events: pd.DataFrame,
    arms: list[str],
    strength: float = SHRINKAGE_STRENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior Beta informativo por braço, calibrado no histórico.

    Devolve `(alpha0, beta0)` tais que `alpha0 / (alpha0 + beta0)` é a taxa
    encolhida do braço e `alpha0 + beta0 == strength`.

    Motivo de usar um prior informativo em vez de Beta(1,1). No experimento o
    prior uniforme é adequado, porque o objetivo é medir quanto o algoritmo
    aprende sozinho, e um prior informativo adiantaria parte da resposta.

    Na política servida o objetivo é decidir bem desde a primeira chamada. Com
    prior uniforme, uma célula com 21 registros e 17 conversões produz uma média
    posterior de 78%, valor que não se sustenta na amostra seguinte e que levaria
    o serviço a escolher o braço errado. Com o prior calibrado por braço, essa
    mesma célula permanece ancorada no comportamento conhecido do braço até
    acumular evidência suficiente para se afastar dele.
    """
    rates = shrunk_arm_rates(events, arms, strength)
    return strength * rates, strength * (1.0 - rates)
