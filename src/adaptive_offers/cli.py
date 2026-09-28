"""Interface de linha de comando do projeto.

Reúne num único ponto de entrada as tarefas do ciclo de vida (preparar dados,
treinar, servir, rodar o Golden Set), de modo que os comandos citados no README
sejam curtos e não exijam conhecer os caminhos internos dos módulos.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _cmd_data(args: argparse.Namespace) -> int:
    from .data.prepare import arm_summary, prepare

    events = prepare(force_download=args.force)
    print(f"Eventos processados: {len(events):,}".replace(",", "."))
    print(f"\nConversão por braço:\n{arm_summary(events).to_string(index=False)}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import main as train_main

    forwarded: list[str] = []
    if args.horizon is not None:
        forwarded += ["--horizon", str(args.horizon)]
    if args.runs is not None:
        forwarded += ["--runs", str(args.runs)]
    if args.no_mlflow:
        forwarded.append("--no-mlflow")
    return train_main(forwarded)


def _cmd_recommend(args: argparse.Namespace) -> int:
    """Gera uma recomendação diretamente no terminal, sem subir o servidor."""
    import json

    from .serving import Recommender

    recommender = Recommender.load()
    client = json.loads(args.client) if args.client else _example_client()
    result = recommender.recommend(client, explore=args.explore)

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_golden(args: argparse.Namespace) -> int:
    from .golden_set import evaluate_golden_set, format_golden_report
    from .serving import Recommender

    recommender = Recommender.load()
    print(format_golden_report(evaluate_golden_set(recommender, explore=False)))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "adaptive_offers.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _example_client() -> dict:
    return {
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptive-offers",
        description="Plataforma de experimentação adaptativa para próxima melhor ação.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="baixa e prepara a base Kaggle")
    p_data.add_argument("--force", action="store_true", help="rebaixa a base")
    p_data.set_defaults(func=_cmd_data)

    p_train = sub.add_parser("train", help="roda o pipeline completo e registra no MLflow")
    p_train.add_argument("--horizon", type=int, default=None)
    p_train.add_argument("--runs", type=int, default=None)
    p_train.add_argument("--no-mlflow", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_rec = sub.add_parser("recommend", help="recomenda o próximo passo para um cliente")
    p_rec.add_argument("--client", type=str, default=None, help="JSON com o contexto")
    p_rec.add_argument("--explore", action="store_true", help="amostra da posterior")
    p_rec.set_defaults(func=_cmd_recommend)

    p_golden = sub.add_parser("golden", help="roda o Golden Set de 5 casos")
    p_golden.set_defaults(func=_cmd_golden)

    p_serve = sub.add_parser("serve", help="sobe a API FastAPI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
