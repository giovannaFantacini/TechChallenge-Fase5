"""Figuras de avaliação salvas em `artifacts/figures/` e anexadas ao MLflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem display, para rodar em servidor e em CI
import matplotlib.pyplot as plt  # noqa: E402

from ..config import FIGURES_DIR, describe_arm, ensure_dirs  # noqa: E402
from .environment import OfflineEnvironment  # noqa: E402
from .simulate import SimulationResult  # noqa: E402

_STYLE = {"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10}


def plot_cumulative_reward(
    results: dict[str, SimulationResult], path: Path | None = None
) -> Path:
    """Conversões acumuladas por política, que é a leitura de negócio do experimento."""
    ensure_dirs()
    path = path or FIGURES_DIR / "conversoes_acumuladas.png"

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for name, result in results.items():
            ax.plot(result.mean_cumulative_reward, label=name, linewidth=1.8)
        ax.set_xlabel("Clientes atendidos")
        ax.set_ylabel("Conversões acumuladas")
        ax.set_title("Conversões acumuladas por política (média de repetições)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_cumulative_regret(
    results: dict[str, SimulationResult], path: Path | None = None
) -> Path:
    """Arrependimento acumulado, que é a leitura técnica do experimento.

    A forma da curva permite o diagnóstico. Políticas que aprendem apresentam
    arrependimento sublinear, e a curva se achata porque os erros se tornam cada
    vez mais raros. Políticas congeladas ou com exploração de taxa fixa
    apresentam arrependimento linear, e a curva se aproxima de uma reta porque a
    proporção de erros permanece constante.
    """
    ensure_dirs()
    path = path or FIGURES_DIR / "regret_acumulado.png"

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for name, result in results.items():
            ax.plot(result.mean_cumulative_regret, label=name, linewidth=1.8)
        ax.set_xlabel("Clientes atendidos")
        ax.set_ylabel("Regret acumulado (conversões perdidas)")
        ax.set_title("Arrependimento acumulado frente ao oráculo por segmento")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_arm_distribution(
    results: dict[str, SimulationResult], arms: list[str], path: Path | None = None
) -> Path:
    """Distribuição do tráfego entre os braços, por política."""
    ensure_dirs()
    path = path or FIGURES_DIR / "distribuicao_bracos.png"

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        n = len(results)
        width = 0.8 / max(n, 1)
        positions = range(len(arms))

        for i, (name, result) in enumerate(results.items()):
            offset = (i - (n - 1) / 2) * width
            ax.bar(
                [p + offset for p in positions],
                result.arm_distribution,
                width=width,
                label=name,
            )

        ax.set_xticks(list(positions))
        ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Fração do tráfego")
        ax.set_title("Alocação de tráfego por braço")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_environment_heatmap(env: OfflineEnvironment, path: Path | None = None) -> Path:
    """Mapa de calor segmento x braço.

    Figura de apoio à decisão de usar um bandit contextual: se todas as linhas
    apresentassem o máximo na mesma coluna, uma regra fixa seria suficiente e a
    personalização não se justificaria.
    """
    ensure_dirs()
    path = path or FIGURES_DIR / "mapa_calor_conversao.png"

    frame = env.to_frame()
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 4 + 0.4 * env.n_segments))
        im = ax.imshow(frame.to_numpy(), aspect="auto", cmap="YlGnBu")

        ax.set_xticks(range(len(frame.columns)))
        ax.set_xticklabels(frame.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(frame.index)))
        ax.set_yticklabels(frame.index, fontsize=9)

        best = env.best_arm_per_segment
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                marker = "*" if j == best[i] else ""
                ax.text(
                    j,
                    i,
                    f"{frame.iat[i, j]:.3f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

        ax.set_title("Conversão calibrada por segmento x braço (* = melhor braço do segmento)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Probabilidade de conversão")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_posterior(policy, segment: int, arms: list[str], path: Path | None = None) -> Path:
    """Posteriors Beta de um segmento, mostrando a incerteza que gera a exploração."""
    import numpy as np
    from scipy.stats import beta as beta_dist

    ensure_dirs()
    path = path or FIGURES_DIR / f"posteriors_segmento_{segment}.png"

    grid = np.linspace(0, 0.6, 600)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, arm in enumerate(arms):
            a = policy.alpha[policy._ctx(segment), i]
            b = policy.beta[policy._ctx(segment), i]
            ax.plot(grid, beta_dist.pdf(grid, a, b), label=f"{describe_arm(arm)}", linewidth=1.5)
        ax.set_xlabel("Probabilidade de conversão")
        ax.set_ylabel("Densidade da posterior")
        ax.set_title(f"Posteriors aprendidas — segmento {segment}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path
