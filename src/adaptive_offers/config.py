"""Configuração central do projeto.

Concentra caminhos, a definição do espaço de ações, as features de contexto e os
hiperparâmetros padrão.

Toda alteração de escopo de dados deve passar por este arquivo. Ele funciona
como ponto único de governança sobre quais colunas entram na decisão, o que
permite auditar a política de minimização de dados em um só lugar, conforme a
seção 10 do README.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = ARTIFACTS_DIR / "models"

RAW_CSV = RAW_DIR / "bank-additional-full.csv"
PROCESSED_PARQUET = PROCESSED_DIR / "events.parquet"
PROCESSED_CSV = PROCESSED_DIR / "events.csv"
POLICY_PATH = MODELS_DIR / "policy.json"
ENVIRONMENT_PATH = MODELS_DIR / "environment.json"
SEGMENTER_PATH = MODELS_DIR / "segmenter.joblib"

# --------------------------------------------------------------------------
# Base de dados
# --------------------------------------------------------------------------
KAGGLE_DATASET = "henriqueyamahata/bank-marketing"
KAGGLE_URL = "https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing"
# Espelho público e estável da mesma base (UCI Bank Marketing, Moro et al. 2014),
# usado quando não há credenciais Kaggle na máquina.
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CSV_SEPARATOR = ";"

# --------------------------------------------------------------------------
# Espaço de ações (braços do bandit)
# --------------------------------------------------------------------------
# A decisão modelada é o próximo passo da abordagem: por qual canal e em qual
# dia da semana contatar o cliente elegível. As duas são variáveis de ação
# registradas na base e sob controle da instituição. Nenhuma delas é atributo do
# cliente, distinção necessária para não tratar uma característica pessoal como
# se fosse uma alavanca de decisão.
ACTION_CHANNEL_COL = "contact"
ACTION_TIMING_COL = "day_of_week"
ARM_SEPARATOR = "|"

CHANNELS = ("cellular", "telephone")
TIMINGS = ("mon", "tue", "wed", "thu", "fri")

CHANNEL_LABELS_PT = {"cellular": "Celular", "telephone": "Telefone fixo"}
TIMING_LABELS_PT = {
    "mon": "segunda-feira",
    "tue": "terça-feira",
    "wed": "quarta-feira",
    "thu": "quinta-feira",
    "fri": "sexta-feira",
}


def build_arms() -> list[str]:
    """Espaço de ações completo, em ordem determinística."""
    return [f"{c}{ARM_SEPARATOR}{t}" for c in CHANNELS for t in TIMINGS]


ARMS: list[str] = build_arms()
N_ARMS: int = len(ARMS)


def describe_arm(arm: str) -> str:
    """Rótulo em português, usado na API e nos relatórios."""
    channel, timing = arm.split(ARM_SEPARATOR)
    return f"{CHANNEL_LABELS_PT.get(channel, channel)} na {TIMING_LABELS_PT.get(timing, timing)}"


# --------------------------------------------------------------------------
# Features de contexto
# --------------------------------------------------------------------------
# Política de minimização de dados: entram na decisão apenas atributos
# comportamentais e de relacionamento. Ver README (seção Governança) para a
# justificativa de cada exclusão.

# Colunas lidas da base bruta para montar o contexto.
SOURCE_CONTEXT_COLUMNS = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "job",
    "education",
    "housing",
    "loan",
    "poutcome",
]

# Contexto efetivamente usado na decisão, após engenharia de atributos.
# `pdays` é substituída por `previously_contacted` + `recency_score` (ver
# `data/prepare.py`) para não injetar a distância numérica falsa de 999 dias.
CONTEXT_NUMERIC = [
    "age",
    "campaign",
    "previous",
    "previously_contacted",
    "recency_score",
]
CONTEXT_CATEGORICAL = ["job", "education", "housing", "loan", "poutcome"]
CONTEXT_FEATURES = CONTEXT_NUMERIC + CONTEXT_CATEGORICAL

# Excluídas deliberadamente do contexto de decisão:
#   duration        -> vazamento temporal (só é conhecida DEPOIS da ligação)
#   marital, default-> atributos sensíveis / estado civil e inadimplência
#   emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
#                   -> macroeconômicas: contexto ambiente, não do cliente
#   month           -> sazonalidade do histórico, não é decisão por cliente
LEAKAGE_COLUMNS = ["duration"]
EXCLUDED_SENSITIVE = ["marital", "default"]
MACRO_COLUMNS = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

TARGET_COL = "y"
REWARD_COL = "reward"
SEGMENT_COL = "segment"
ARM_COL = "arm"

# Na base original, pdays == 999 codifica "nunca contatado antes". O valor é
# convertido em flag binária e score de recência (ver data/prepare.py) para não
# introduzir uma distância numérica de 999 dias que não existe de fato.
PDAYS_NEVER_CONTACTED = 999

# --------------------------------------------------------------------------
# Hiperparâmetros padrão
# --------------------------------------------------------------------------
N_SEGMENTS = 4
RANDOM_SEED = 42

# Horizonte e número de repetições do experimento.
# São necessárias cerca de 100 mil decisões para que as políticas contextuais
# convirjam: há 4 segmentos x 10 braços = 40 células a estimar, e o segmento
# menor recebe apenas 3,4% do tráfego. Horizontes mais curtos medem a fase de
# aprendizado e não a qualidade da política já convergida, o que foi verificado
# na varredura de horizontes documentada no notebook 02.
DEFAULT_HORIZON = 100_000
DEFAULT_N_RUNS = 10

# Beta(1,1) equivale à distribuição uniforme, ou seja, nenhuma crença inicial
# sobre a conversão. É o prior usado no experimento; a política servida usa um
# prior informativo calibrado no histórico (ver statistics.py).
DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0
DEFAULT_EPSILON = 0.10
DEFAULT_UCB_C = 2.0

# Fator de desconto do Thompson Sampling adaptativo. Corresponde a uma memória
# efetiva de 1/(1-gama) = 5.000 observações por contexto. O valor foi escolhido
# por varredura entre 0,999 e 0,9998: precisa ser longo o suficiente para
# distinguir 10 braços com conversão em torno de 14%, o que exige centenas de
# observações por braço, e curto o suficiente para detectar uma quebra
# estrutural em milhares de eventos.
DEFAULT_GAMMA = 0.9998

# --------------------------------------------------------------------------
# Cenário de estresse (regime não estacionário)
# --------------------------------------------------------------------------
# Choque aplicado na metade do horizonte: os braços de celular passam a
# converter 20% do que convertiam. O gatilho de negócio correspondente pode ser
# uma restrição regulatória ao contato por celular, uma onda de descadastramento
# ou a saturação do canal dominante. Trata-se de um teste de estresse sobre a
# matriz calibrada em dados reais; nenhum dado de cliente é sintetizado.
SHOCK_FACTOR = 0.20
SHOCK_CHANNEL_PREFIX = "cellular"

# Força do encolhimento empírico-Bayes, expressa em número de observações
# equivalentes. Define o peso dado à taxa global do braço quando uma célula
# (segmento x braço) tem poucas amostras. Usada tanto na calibração do simulador
# quanto no prior da política servida.
SHRINKAGE_STRENGTH = 50.0

# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
MLFLOW_EXPERIMENT = "datathon-ofertas-adaptativas"

# Backend SQLite local. A partir do MLflow 3, o backend de arquivo (./mlruns)
# entrou em modo de manutenção e passa a levantar exceção. O SQLite é o
# substituto recomendado pela documentação, permanece local, ou seja, um único
# arquivo no repositório e sem servidor, e habilita a interface completa por
# meio de `mlflow ui`.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"


def ensure_dirs() -> None:
    """Garante a existência das pastas de saída."""
    for path in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
