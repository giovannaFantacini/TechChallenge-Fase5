# Plataforma de Experimentação Adaptativa — Datathon POSTECH MLET Fase 5
#
# Use `make help` para ver os alvos disponíveis.

PYTHON ?= python
VENV   := .venv
BIN    := $(VENV)/bin
PORT   ?= 8000
MLPORT ?= 5000

export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help setup data train golden serve mlflow test notebooks clean all

help:  ## Mostra esta ajuda
	@echo "Plataforma de Experimentação Adaptativa — alvos disponíveis:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Cria o ambiente virtual e instala as dependências
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

data:  ## Baixa a base (Kaggle ou espelho UCI) e prepara o log de eventos
	$(BIN)/python -m adaptive_offers.cli data

train:  ## Pipeline completo: experimento, MLflow e artefatos (~4 min)
	$(BIN)/python -m adaptive_offers.train

train-fast:  ## Versão rápida do pipeline, para demonstração
	$(BIN)/python -m adaptive_offers.train --horizon 20000 --runs 3

golden:  ## Roda o Golden Set — 5 casos de referência
	$(BIN)/python -m adaptive_offers.cli golden

serve:  ## Sobe a API em http://127.0.0.1:$(PORT)/docs
	$(BIN)/python -m adaptive_offers.cli serve --port $(PORT)

mlflow:  ## Abre a UI do MLflow em http://127.0.0.1:$(MLPORT)
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLPORT)

test:  ## Roda a suíte de testes
	$(BIN)/python -m pytest tests/ -q

notebooks:  ## Executa os notebooks de ponta a ponta
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace \
		notebooks/01_eda.ipynb notebooks/02_baseline_e_bandits.ipynb

clean:  ## Remove artefatos gerados (mantém a base bruta)
	rm -rf artifacts/figures/*.png artifacts/models/*.json artifacts/models/*.csv \
	       artifacts/models/*.joblib data/processed/* mlflow.db mlruns mlartifacts \
	       .pytest_cache **/__pycache__
	@echo "Artefatos removidos. Rode 'make train' para regerá-los."

all: data train golden  ## Pipeline completo de ponta a ponta
