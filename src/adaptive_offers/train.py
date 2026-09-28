"""Pipeline completo: dados, segmentação, experimento, MLflow e artefatos.

Este script reproduz o trabalho de ponta a ponta e gera, numa única execução,
todos os resultados que a avaliação precisa inspecionar:

- o quadro comparativo entre baseline e políticas adaptativas (Etapa 3);
- as métricas de avaliação e o Golden Set (Etapa 4);
- a política serializada que a API consome (Etapa 5);
- os runs registrados no MLflow (Etapa 7).

O experimento é executado em dois regimes. O regime A é estacionário e mede o
custo do aprendizado frente a uma regra fixa. O regime B aplica uma quebra
estrutural na metade do horizonte e mede a capacidade de reação de cada
política. Os dois são necessários: avaliar apenas o regime A levaria à conclusão
incorreta de que a plataforma adaptativa não se justifica.

No MLflow, cada política gera um run aninhado com seus hiperparâmetros, suas
métricas nos dois regimes e suas figuras. O run pai guarda o contexto do
experimento, de modo que execuções futuras com outros parâmetros sejam
comparáveis sem sobrescrever as anteriores.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from .bandits import (
    ContextualEpsilonGreedy,
    ContextualThompsonSampling,
    ContextualUCB1,
    DiscountedThompsonSampling,
    EpsilonGreedy,
    FixedBaseline,
    RandomPolicy,
    SegmentedBaseline,
    ThompsonSampling,
    UCB1,
)
from .config import (
    ARMS,
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_HORIZON,
    DEFAULT_N_RUNS,
    DEFAULT_UCB_C,
    KAGGLE_URL,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    FIGURES_DIR,
    MODELS_DIR,
    N_SEGMENTS,
    RANDOM_SEED,
    SEGMENT_COL,
    SHOCK_CHANNEL_PREFIX,
    SHOCK_FACTOR,
    SHRINKAGE_STRENGTH,
    ensure_dirs,
)
from .data.prepare import arm_summary, prepare
from .evaluation.environment import DriftingEnvironment, OfflineEnvironment
from .evaluation.plots import (
    plot_arm_distribution,
    plot_cumulative_regret,
    plot_cumulative_reward,
    plot_environment_heatmap,
    plot_posterior,
)
from .evaluation.replay import compare_replay
from .evaluation.simulate import compare_policies
from .golden_set import evaluate_golden_set, format_golden_report
from .segmentation import Segmenter, make_segment_labels, profile_segments
from .serving import Recommender

logger = logging.getLogger(__name__)


def build_policies(
    events: pd.DataFrame,
    n_segments: int,
    epsilon: float = DEFAULT_EPSILON,
    ucb_c: float = DEFAULT_UCB_C,
    gamma: float = DEFAULT_GAMMA,
    seed: int = RANDOM_SEED,
) -> list:
    """Monta o conjunto de políticas comparadas.

    A ordem segue do controle mais simples, o sorteio aleatório, ao mais
    elaborado, o Thompson Sampling contextual com desconto, passando pelos dois
    baselines não adaptativos.

    Comparar apenas bandit contra sorteio aleatório seria um teste pouco
    informativo. São os baselines fixo e segmentado que tornam a conclusão
    defensável, em especial o `baseline_fixo`, que já recebe o melhor braço
    histórico e não gasta tráfego explorando.
    """
    n_arms = len(ARMS)
    return [
        RandomPolicy(n_arms, random_state=seed),
        FixedBaseline.from_events(events, ARMS, random_state=seed),
        SegmentedBaseline.from_events(events, ARMS, SEGMENT_COL, n_segments, random_state=seed),
        EpsilonGreedy(n_arms, epsilon=epsilon, random_state=seed),
        EpsilonGreedy(n_arms, epsilon=epsilon, decay=True, random_state=seed),
        UCB1(n_arms, c=ucb_c, random_state=seed),
        ThompsonSampling(n_arms, random_state=seed),
        ContextualEpsilonGreedy(n_arms, n_segments, epsilon=epsilon, random_state=seed),
        ContextualUCB1(n_arms, n_segments, c=ucb_c, random_state=seed),
        ContextualThompsonSampling(n_arms, n_segments, random_state=seed),
        DiscountedThompsonSampling(n_arms, n_segments, gamma=gamma, random_state=seed),
    ]


def _fix_decay_names(policies: list) -> None:
    """Diferencia as duas variantes de Epsilon-Greedy no relatório e no MLflow."""
    for policy in policies:
        if getattr(policy, "decay", False):
            policy.name = f"{policy.name}_decay"


def run_pipeline(
    horizon: int = DEFAULT_HORIZON,
    n_runs: int = DEFAULT_N_RUNS,
    n_segments: int = N_SEGMENTS,
    epsilon: float = DEFAULT_EPSILON,
    ucb_c: float = DEFAULT_UCB_C,
    gamma: float = DEFAULT_GAMMA,
    seed: int = RANDOM_SEED,
    use_mlflow: bool = True,
    force_download: bool = False,
) -> dict:
    """Executa o pipeline completo e devolve os principais resultados."""
    ensure_dirs()

    # -- Etapas 1 e 2: base e preparação ------------------------------------
    logger.info("[1/7] Preparando a base...")
    events = prepare(force_download=force_download)

    # -- Contexto: segmentação ---------------------------------------------
    logger.info("[2/7] Segmentando clientes (k=%d)...", n_segments)
    segmenter = Segmenter(n_segments=n_segments, random_state=seed).fit(events)
    events[SEGMENT_COL] = segmenter.predict(events)
    segmenter.save()

    profile = profile_segments(events, events[SEGMENT_COL].to_numpy())
    segment_labels = make_segment_labels(profile)
    profile["rotulo"] = profile["segmento"].map(segment_labels)
    logger.info("\n%s", profile.to_string(index=False))

    # -- Ambiente calibrado -------------------------------------------------
    logger.info("[3/7] Calibrando o ambiente de simulação...")
    env = OfflineEnvironment.from_events(events, ARMS, shrinkage=SHRINKAGE_STRENGTH)
    env.save()
    logger.info("%s", env)
    logger.info(
        "Espaço para personalização: oráculo %.4f vs. melhor regra fixa %.4f (+%.1f%%)",
        env.oracle_reward,
        env.best_fixed_reward,
        env.contextual_headroom() * 100,
    )

    # -- Etapa 3, regime A: mundo estacionário ------------------------------
    logger.info(
        "[4/7] Experimento A — regime estacionário (%d passos x %d repetições)...",
        horizon,
        n_runs,
    )
    policies = build_policies(events, n_segments, epsilon, ucb_c, gamma, seed)
    _fix_decay_names(policies)
    results, table = compare_policies(policies, env, horizon, n_runs, seed)

    # -- Etapa 3, regime B: ambiente com quebra estrutural ------------------
    # O enunciado do desafio parte da premissa de que regras fixas demoram para
    # reagir a mudanças de contexto. Essa premissa não se verifica num ambiente
    # estacionário, de forma que medir apenas o regime A produziria uma
    # conclusão incompleta sobre o valor da plataforma adaptativa.
    drift_env = DriftingEnvironment(env, horizon // 2, SHOCK_CHANNEL_PREFIX, SHOCK_FACTOR)
    logger.info("[5/7] Experimento B — %s", drift_env.describe())
    drift_policies = build_policies(events, n_segments, epsilon, ucb_c, gamma, seed)
    _fix_decay_names(drift_policies)
    drift_results, drift_table = compare_policies(
        drift_policies, drift_env, horizon, n_runs, seed
    )

    # -- Validação complementar: replay off-policy --------------------------
    logger.info("[6/7] Validando por replay off-policy sobre os eventos reais...")
    replay_policies = build_policies(events, n_segments, epsilon, ucb_c, gamma, seed)
    _fix_decay_names(replay_policies)
    _, replay_table = compare_replay(replay_policies, events, ARMS, uniformize=True)

    # -- Figuras ------------------------------------------------------------
    logger.info("[7/7] Gerando figuras, política servível e Golden Set...")
    best_policy = next(p for p in policies if p.name == "thompson_sampling_contextual")
    figures = [
        plot_cumulative_reward(results),
        plot_cumulative_regret(results),
        plot_arm_distribution(results, ARMS),
        plot_environment_heatmap(env),
        plot_posterior(best_policy, segment=int(profile["segmento"].iloc[0]), arms=ARMS),
        plot_cumulative_regret(
            drift_results, path=FIGURES_DIR / "regret_acumulado_com_choque.png"
        ),
        plot_cumulative_reward(
            drift_results, path=FIGURES_DIR / "conversoes_acumuladas_com_choque.png"
        ),
    ]

    # -- Etapa 5: política servível + Golden Set ----------------------------
    recommender = Recommender.from_events(
        events,
        segmenter,
        ARMS,
        n_segments,
        segment_labels=segment_labels,
        metadata={
            "base_kaggle": KAGGLE_URL,
            "horizonte_experimento": horizon,
            "repeticoes": n_runs,
            "semente": seed,
            "conversao_oraculo": env.oracle_reward,
        },
    )
    recommender.save()
    golden = evaluate_golden_set(recommender, explore=False)

    # -- Tabelas em disco ---------------------------------------------------
    table.to_csv(MODELS_DIR / "comparativo_politicas.csv", index=False)
    drift_table.to_csv(MODELS_DIR / "comparativo_com_choque.csv", index=False)
    replay_table.to_csv(MODELS_DIR / "comparativo_replay.csv", index=False)
    profile.to_csv(MODELS_DIR / "perfil_segmentos.csv", index=False)
    golden.to_csv(MODELS_DIR / "golden_set.csv", index=False)
    (MODELS_DIR / "taxas_ambiente.csv").write_text(
        env.to_frame().to_csv(), encoding="utf-8"
    )

    if use_mlflow:
        _log_to_mlflow(
            table=table,
            drift_table=drift_table,
            replay_table=replay_table,
            results=results,
            drift_results=drift_results,
            env=env,
            drift_env=drift_env,
            profile=profile,
            golden=golden,
            figures=figures,
            horizon=horizon,
            n_runs=n_runs,
            n_segments=n_segments,
            seed=seed,
            n_events=len(events),
        )

    return {
        "events": events,
        "segmenter": segmenter,
        "environment": env,
        "drift_environment": drift_env,
        "results": results,
        "drift_results": drift_results,
        "table": table,
        "drift_table": drift_table,
        "replay_table": replay_table,
        "profile": profile,
        "golden": golden,
        "recommender": recommender,
        "arm_summary": arm_summary(events),
    }


def _log_to_mlflow(
    table: pd.DataFrame,
    drift_table: pd.DataFrame,
    replay_table: pd.DataFrame,
    results: dict,
    drift_results: dict,
    env: OfflineEnvironment,
    drift_env: DriftingEnvironment,
    profile: pd.DataFrame,
    golden: pd.DataFrame,
    figures: list,
    horizon: int,
    n_runs: int,
    n_segments: int,
    seed: int,
    n_events: int,
) -> None:
    """Registra o experimento no MLflow (Etapa 7).

    A estrutura é um run pai com o contexto do experimento e um run filho por
    política. Isso permite ordenar as políticas por qualquer métrica na interface
    do MLflow e comparar execuções feitas com hiperparâmetros diferentes sem
    perder o histórico anterior.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    replay_by_policy = replay_table.set_index("politica")
    drift_by_policy = drift_table.set_index("politica")

    with mlflow.start_run(run_name="experimento_completo"):
        mlflow.log_params(
            {
                "base": "Kaggle bank-marketing (bank-additional-full)",
                "n_eventos": n_events,
                "n_bracos": len(ARMS),
                "n_segmentos": n_segments,
                "horizonte": horizon,
                "repeticoes": n_runs,
                "semente": seed,
                "encolhimento_eb": SHRINKAGE_STRENGTH,
                "choque_canal": drift_env.affected_prefix,
                "choque_fator": drift_env.factor,
                "choque_passo": drift_env.change_point,
            }
        )
        mlflow.log_metrics(
            {
                "conversao_oraculo": env.oracle_reward,
                "conversao_melhor_regra_fixa": env.best_fixed_reward,
                "espaco_personalizacao_pct": env.contextual_headroom() * 100,
                "conversao_oraculo_pos_choque": drift_env.oracle_reward_after,
            }
        )
        mlflow.log_text(profile.to_string(index=False), "perfil_segmentos.txt")
        mlflow.log_text(table.to_string(index=False), "comparativo_estacionario.txt")
        mlflow.log_text(drift_table.to_string(index=False), "comparativo_com_choque.txt")
        mlflow.log_text(replay_table.to_string(index=False), "comparativo_replay.txt")
        mlflow.log_text(format_golden_report(golden), "golden_set.txt")
        mlflow.log_text(env.to_frame().to_csv(), "taxas_ambiente.csv")
        mlflow.log_text(drift_env.describe(), "cenario_choque.txt")
        for figure in figures:
            mlflow.log_artifact(str(figure), artifact_path="figuras")

        for _, row in table.iterrows():
            name = row["politica"]
            result = results[name]
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({str(k): v for k, v in result.params.items()})
                metrics = {
                    "conversao_media": row["conversao_media"],
                    "conversao_desvio": row["conversao_desvio"],
                    "conversao_esperada_regime_final": row["conversao_esperada_regime_final"],
                    "conversao_observada_regime_final": row["conversao_observada_regime_final"],
                    "conversoes_totais": row["conversoes_totais"],
                    "regret_acumulado": row["regret_acumulado"],
                    "regret_desvio": row["regret_desvio"],
                    "exploracao_pct": row["exploracao_pct"],
                    "pct_do_oraculo": row["pct_do_oraculo"],
                }
                for column in ("uplift_vs_baseline_pct", "uplift_regime_final_pct"):
                    if column in row:
                        metrics[column] = row[column]
                if name in replay_by_policy.index:
                    metrics["conversao_replay"] = replay_by_policy.loc[name, "conversao_replay"]
                    metrics["replay_eventos"] = replay_by_policy.loc[
                        name, "eventos_aproveitados"
                    ]
                # As métricas do regime com choque ficam no mesmo run da
                # política. Essa é a comparação relevante para a decisão de
                # arquitetura: quanto cada política custa em um ambiente estável
                # e quanto ela recupera quando o ambiente muda.
                if name in drift_by_policy.index:
                    metrics["choque_conversao_media"] = drift_by_policy.loc[
                        name, "conversao_media"
                    ]
                    metrics["choque_conversao_pos_choque"] = drift_by_policy.loc[
                        name, "conversao_esperada_regime_final"
                    ]
                    metrics["choque_regret_acumulado"] = drift_by_policy.loc[
                        name, "regret_acumulado"
                    ]
                    for column, metric in (
                        ("uplift_vs_baseline_pct", "choque_uplift_vs_baseline_pct"),
                        ("uplift_regime_final_pct", "choque_uplift_pos_choque_pct"),
                    ):
                        if column in drift_by_policy.columns:
                            metrics[metric] = drift_by_policy.loc[name, column]
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

                # Curvas de arrependimento amostradas em 50 pontos. Permitem
                # verificar na interface do MLflow se a curva se achata, o que
                # indica aprendizado, ou se permanece linear.
                for metric_name, source in (
                    ("regret_acumulado_curva", result),
                    ("regret_acumulado_curva_choque", drift_results.get(name)),
                ):
                    if source is None:
                        continue
                    curve = source.mean_cumulative_regret
                    for step in range(0, len(curve), max(1, len(curve) // 50)):
                        mlflow.log_metric(metric_name, float(curve[step]), step=step)

    logger.info("Runs registrados no MLflow em %s", MLFLOW_TRACKING_URI)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline completo do Datathon — ofertas adaptativas."
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--segments", type=int, default=N_SEGMENTS)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--ucb-c", type=float, default=DEFAULT_UCB_C)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-mlflow", action="store_true", help="pula o tracking")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    output = run_pipeline(
        horizon=args.horizon,
        n_runs=args.runs,
        n_segments=args.segments,
        epsilon=args.epsilon,
        ucb_c=args.ucb_c,
        gamma=args.gamma,
        seed=args.seed,
        use_mlflow=not args.no_mlflow,
        force_download=args.force_download,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("\n" + "=" * 78)
    print("CONVERSÃO POR BRAÇO (histórico observado)")
    print("=" * 78)
    print(output["arm_summary"].to_string(index=False))

    print("\n" + "=" * 78)
    print("PERFIL DOS SEGMENTOS")
    print("=" * 78)
    print(output["profile"].to_string(index=False))

    print("\n" + "=" * 78)
    print("EXPERIMENTO A — REGIME ESTACIONÁRIO")
    print("=" * 78)
    print(output["table"].to_string(index=False))
    print(
        "\nLeitura: em ambiente estacionário, a regra fixa apontada para o melhor\n"
        "braço não gasta tráfego explorando e é difícil de superar no acumulado. O\n"
        "ganho da política adaptativa aparece na coluna `uplift_regime_final_pct`:\n"
        "após a convergência ela supera a regra fixa, tendo identificado o melhor\n"
        "braço por conta própria."
    )

    print("\n" + "=" * 78)
    print("EXPERIMENTO B — REGIME COM QUEBRA ESTRUTURAL")
    print("=" * 78)
    print(output["drift_environment"].describe())
    print(output["drift_table"].to_string(index=False))
    print(
        "\nLeitura: quando o canal dominante perde eficácia, a regra fixa continua\n"
        "aplicando o braço que deixou de ser o melhor até que uma revisão manual\n"
        "seja feita. O Thompson Sampling com desconto identifica a mudança e\n"
        "realoca o tráfego automaticamente."
    )

    print("\n" + "=" * 78)
    print("VALIDAÇÃO COMPLEMENTAR — REPLAY OFF-POLICY (eventos reais)")
    print("=" * 78)
    print(output["replay_table"].to_string(index=False))

    print("\n" + format_golden_report(output["golden"]))

    print(json.dumps({"artefatos": str(MODELS_DIR)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
