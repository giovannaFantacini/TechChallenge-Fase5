"""Ingestão da base Bank Marketing.

O download é tentado em duas vias, nesta ordem:

1. Kaggle (`henriqueyamahata/bank-marketing`) por meio do `kagglehub`, quando há
   credenciais configuradas na máquina. Esta é a fonte citada no enunciado.
2. UCI Machine Learning Repository, que disponibiliza publicamente o mesmo
   arquivo (`bank-additional-full.csv`, de Moro et al., 2014) sem exigir
   autenticação.

As duas vias entregam o mesmo arquivo, de modo que o pipeline é reprodutível em
qualquer máquina, com ou sem conta no Kaggle. A referência ao Kaggle é mantida
no README e no `config.py`, conforme pedido no enunciado.
"""

from __future__ import annotations

import io
import logging
import shutil
import zipfile
from pathlib import Path

import requests

from ..config import CSV_SEPARATOR, KAGGLE_DATASET, RAW_CSV, RAW_DIR, UCI_URL, ensure_dirs

logger = logging.getLogger(__name__)

TARGET_FILENAME = "bank-additional-full.csv"
_DOWNLOAD_TIMEOUT = 120


def _try_kagglehub() -> Path | None:
    """Baixa a base pelo Kaggle. Devolve None se faltar a biblioteca ou as credenciais."""
    try:
        import kagglehub  # importado aqui por ser dependência opcional
    except ImportError:
        logger.info("kagglehub não instalado; usando o espelho da UCI.")
        return None

    try:
        cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:  # credenciais ausentes, falha de rede ou mudança do slug
        logger.info("Download via Kaggle indisponível (%s); usando o espelho da UCI.", exc)
        return None

    for candidate in cache_dir.rglob(TARGET_FILENAME):
        logger.info("Base obtida do Kaggle: %s", candidate)
        return candidate

    logger.warning("Kaggle respondeu mas %s não foi encontrado no pacote.", TARGET_FILENAME)
    return None


def _download_from_uci() -> Path:
    """Baixa o zip da UCI e extrai o CSV alvo (zip aninhado)."""
    logger.info("Baixando a base do espelho UCI: %s", UCI_URL)
    response = requests.get(UCI_URL, timeout=_DOWNLOAD_TIMEOUT)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as outer:
        # O pacote da UCI contém dois zips aninhados, bank.zip e
        # bank-additional.zip. O segundo é o que corresponde à base do Kaggle.
        inner_name = next(
            (n for n in outer.namelist() if n.endswith("bank-additional.zip")), None
        )
        if inner_name is None:
            raise RuntimeError("bank-additional.zip não encontrado no pacote da UCI.")

        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            member = next(
                (n for n in inner.namelist() if n.endswith(TARGET_FILENAME)), None
            )
            if member is None:
                raise RuntimeError(f"{TARGET_FILENAME} não encontrado no pacote da UCI.")
            RAW_CSV.write_bytes(inner.read(member))

    logger.info("Base salva em %s", RAW_CSV)
    return RAW_CSV


def download_dataset(force: bool = False) -> Path:
    """Garante o CSV bruto em `data/raw/` e devolve seu caminho.

    Args:
        force: quando True, rebaixa mesmo se o arquivo já existir localmente.
    """
    ensure_dirs()

    if RAW_CSV.exists() and not force:
        logger.info("Base já presente em %s (use force=True para rebaixar).", RAW_CSV)
        return RAW_CSV

    kaggle_path = _try_kagglehub()
    if kaggle_path is not None:
        shutil.copyfile(kaggle_path, RAW_CSV)
        return RAW_CSV

    return _download_from_uci()


def load_raw(force_download: bool = False):
    """Carrega o CSV bruto como DataFrame, baixando-o se necessário."""
    import pandas as pd

    path = download_dataset(force=force_download)
    df = pd.read_csv(path, sep=CSV_SEPARATOR)
    logger.info("Base bruta carregada: %d linhas x %d colunas", *df.shape)
    return df


if __name__ == "__main__":  # pragma: no cover - utilitário de linha de comando
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    path = download_dataset()
    print(f"Base disponível em: {path}")
    print(f"Tamanho: {path.stat().st_size / 1024:.0f} KB")
    print(f"Diretório: {RAW_DIR}")
