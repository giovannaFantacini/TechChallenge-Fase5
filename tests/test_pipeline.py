"""Testes de dados, segmentação, ambiente e avaliação.

Estes testes protegem invariantes que, se quebradas, produzem resultados
plausíveis mas incorretos. Essa é a falha mais difícil de detectar num projeto
de aprendizado de máquina, porque não levanta exceção e não interrompe a
execução.

São exemplos do que é verificado aqui: o retorno de `duration` ao contexto de
decisão, a imputação de contrafactual no replay e a perda de efeito do
encolhimento. Nenhum desses casos seria detectado por um teste que apenas
verifica se o código executa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_offers.bandits import FixedBaseline, RandomPolicy, ThompsonSampling
from adaptive_offers.config import (
    ARMS,
    CONTEXT_FEATURES,
    LEAKAGE_COLUMNS,
    PDAYS_NEVER_CONTACTED,
    SEGMENT_COL,
)
from adaptive_offers.data.prepare import build_events, context_from_client, engineer_recency
from adaptive_offers.evaluation.environment import DriftingEnvironment, OfflineEnvironment
from adaptive_offers.evaluation.replay import replay_evaluate, uniformize_log
from adaptive_offers.evaluation.simulate import run_policy
from adaptive_offers.segmentation import Segmenter, make_segment_labels, profile_segments
from adaptive_offers.serving import Recommender, fit_posterior_from_events
from adaptive_offers.statistics import arm_priors, shrunk_arm_rates

RNG = np.random.default_rng(3)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Base sintética com o mesmo esquema da Bank Marketing, para isolar os testes."""
    n = 3000
    return pd.DataFrame(
        {
            "age": RNG.integers(20, 75, n),
            "job": RNG.choice(["admin.", "blue-collar", "retired", "technician"], n),
            "marital": RNG.choice(["married", "single"], n),
            "education": RNG.choice(["basic.9y", "university.degree", "high.school"], n),
            "default": RNG.choice(["no", "unknown"], n),
            "housing": RNG.choice(["yes", "no"], n),
            "loan": RNG.choice(["yes", "no"], n),
            "contact": RNG.choice(["cellular", "telephone"], n),
            "day_of_week": RNG.choice(["mon", "tue", "wed", "thu", "fri"], n),
            "duration": RNG.integers(10, 900, n),
            "campaign": RNG.integers(1, 8, n),
            "pdays": RNG.choice([PDAYS_NEVER_CONTACTED, 3, 7, 14], n),
            "previous": RNG.integers(0, 4, n),
            "poutcome": RNG.choice(["nonexistent", "failure", "success"], n),
            "y": RNG.choice(["yes", "no"], n, p=[0.12, 0.88]),
        }
    )


@pytest.fixture
def events(raw_frame) -> pd.DataFrame:
    frame = build_events(raw_frame)
    segmenter = Segmenter(n_segments=3, random_state=0).fit(frame)
    frame[SEGMENT_COL] = segmenter.predict(frame)
    return frame


# --------------------------------------------------------------------------
# Preparação da base
# --------------------------------------------------------------------------
def test_leakage_column_never_reaches_the_context(raw_frame):
    """`duration` só é conhecida após a ligação e não pode entrar na decisão."""
    frame = build_events(raw_frame)
    for column in LEAKAGE_COLUMNS:
        assert column not in frame.columns


def test_sensitive_columns_are_absent_from_the_context(raw_frame):
    frame = build_events(raw_frame)
    for column in ("marital", "default"):
        assert column not in frame.columns


def test_events_carry_arm_and_binary_reward(raw_frame):
    frame = build_events(raw_frame)
    assert set(frame["arm"].unique()).issubset(set(ARMS))
    assert set(frame["reward"].unique()).issubset({0, 1})


def test_reward_matches_the_original_target(raw_frame):
    frame = build_events(raw_frame.drop_duplicates())
    expected = (raw_frame.drop_duplicates()["y"] == "yes").astype(int).to_numpy()
    assert np.array_equal(frame["reward"].to_numpy(), expected)


def test_recency_encodes_never_contacted_as_zero():
    frame = pd.DataFrame({"pdays": [PDAYS_NEVER_CONTACTED, 0, 9]})
    out = engineer_recency(frame)

    assert out.loc[0, "previously_contacted"] == 0
    assert out.loc[0, "recency_score"] == 0.0
    # Contato mais recente => score maior.
    assert out.loc[1, "recency_score"] > out.loc[2, "recency_score"]


def test_build_events_rejects_missing_columns(raw_frame):
    with pytest.raises(ValueError):
        build_events(raw_frame.drop(columns=["poutcome"]))


def test_context_from_client_matches_training_features():
    client = {
        "age": 40, "job": "admin.", "education": "high.school", "housing": "yes",
        "loan": "no", "campaign": 2, "pdays": 5, "previous": 1, "poutcome": "failure",
    }
    frame = context_from_client(client)
    assert list(frame.columns) == CONTEXT_FEATURES


def test_context_from_client_requires_pdays():
    with pytest.raises(ValueError):
        context_from_client({"age": 40})


# --------------------------------------------------------------------------
# Segmentação
# --------------------------------------------------------------------------
def test_segmenter_is_deterministic(events):
    a = Segmenter(n_segments=3, random_state=1).fit_predict(events)
    b = Segmenter(n_segments=3, random_state=1).fit_predict(events)
    assert np.array_equal(a, b)


def test_segmenter_roundtrips_through_disk(events, tmp_path):
    segmenter = Segmenter(n_segments=3, random_state=1).fit(events)
    path = segmenter.save(tmp_path / "seg.joblib")

    reloaded = Segmenter.load(path)
    assert np.array_equal(segmenter.predict(events), reloaded.predict(events))


def test_segment_labels_are_unique(events):
    profile = profile_segments(events, events[SEGMENT_COL].to_numpy())
    labels = make_segment_labels(profile)

    assert len(labels) == profile.shape[0]
    assert len(set(labels.values())) == len(labels), "rótulos de segmento colidiram"


def test_untrained_segmenter_refuses_to_predict(events):
    with pytest.raises(RuntimeError):
        Segmenter().predict(events)


# --------------------------------------------------------------------------
# Ambiente calibrado
# --------------------------------------------------------------------------
def test_environment_rates_are_probabilities(events):
    env = OfflineEnvironment.from_events(events, ARMS)
    assert np.all((env.rates >= 0) & (env.rates <= 1))
    assert np.isclose(env.segment_probs.sum(), 1.0)


def test_environment_requires_segment_column(events):
    with pytest.raises(ValueError):
        OfflineEnvironment.from_events(events.drop(columns=[SEGMENT_COL]), ARMS)


def test_shrinkage_pulls_thin_cells_toward_the_global_rate():
    """Uma célula com amostra muito pequena não pode determinar o ótimo do segmento.

    No cenário construído, um braço aparece 3 vezes e converte nas 3. Sem
    encolhimento sua taxa estimada seria 1,0 e ele passaria a ser o melhor braço
    do segmento, com base apenas em ruído amostral.
    """
    rows = []
    for _ in range(2000):
        rows.append({SEGMENT_COL: 0, "arm": ARMS[0], "reward": 0})
    for _ in range(200):
        rows.append({SEGMENT_COL: 0, "arm": ARMS[0], "reward": 1})
    for _ in range(3):
        rows.append({SEGMENT_COL: 0, "arm": ARMS[1], "reward": 1})

    env = OfflineEnvironment.from_events(pd.DataFrame(rows), ARMS, shrinkage=50.0)
    assert env.rates[0, 1] < 0.5, "a célula com pouca amostra não foi encolhida"


def test_oracle_is_at_least_the_best_fixed_rule(events):
    """A personalização nunca pode resultar pior do que a melhor regra fixa."""
    env = OfflineEnvironment.from_events(events, ARMS)
    assert env.oracle_reward >= env.best_fixed_reward - 1e-12
    assert env.contextual_headroom() >= -1e-12


def test_environment_roundtrips_through_disk(events, tmp_path):
    env = OfflineEnvironment.from_events(events, ARMS)
    reloaded = OfflineEnvironment.load(env.save(tmp_path / "env.json"))
    assert np.allclose(env.rates, reloaded.rates)


# --------------------------------------------------------------------------
# Ambiente com quebra estrutural
# --------------------------------------------------------------------------
def test_drift_changes_rates_only_after_the_change_point(events):
    base = OfflineEnvironment.from_events(events, ARMS)
    drift = DriftingEnvironment(base, change_point=100, affected_prefix="cellular", factor=0.2)

    assert np.allclose(drift.rates_at(99), base.rates)
    assert not np.allclose(drift.rates_at(100), base.rates)


def test_drift_only_penalises_the_affected_channel(events):
    base = OfflineEnvironment.from_events(events, ARMS)
    drift = DriftingEnvironment(base, change_point=10, affected_prefix="cellular", factor=0.2)

    after = drift.rates_at(10)
    for index, arm in enumerate(ARMS):
        if arm.startswith("cellular"):
            assert np.allclose(after[:, index], base.rates[:, index] * 0.2)
        else:
            assert np.allclose(after[:, index], base.rates[:, index])


# --------------------------------------------------------------------------
# Simulação
# --------------------------------------------------------------------------
def test_simulation_shapes_and_non_negative_regret(events):
    env = OfflineEnvironment.from_events(events, ARMS)
    result = run_policy(ThompsonSampling(len(ARMS)), env, horizon=500, n_runs=3)

    assert result.rewards.shape == (3, 500)
    assert result.arm_counts.sum(axis=1).tolist() == [500, 500, 500]
    assert np.all(result.regrets >= -1e-12), (
        "arrependimento negativo indica que o oráculo não é realmente ótimo"
    )


def test_fixed_baseline_uses_a_single_arm_in_simulation(events):
    env = OfflineEnvironment.from_events(events, ARMS)
    result = run_policy(FixedBaseline(len(ARMS), best_arm=2), env, horizon=300, n_runs=2)
    assert (result.arm_counts[:, 2] == 300).all()


def test_random_policy_has_the_largest_regret(events):
    """Verificação de sanidade: a escolha aleatória deve ser o pior caso."""
    env = OfflineEnvironment.from_events(events, ARMS)
    random = run_policy(RandomPolicy(len(ARMS)), env, horizon=3000, n_runs=3)
    thompson = run_policy(ThompsonSampling(len(ARMS)), env, horizon=3000, n_runs=3)

    assert random.cumulative_regret > thompson.cumulative_regret


# --------------------------------------------------------------------------
# Replay off-policy
# --------------------------------------------------------------------------
def test_uniformize_balances_the_log(events):
    balanced = uniformize_log(events, ARMS)
    counts = balanced["arm"].value_counts()
    assert counts.nunique() == 1, "log uniformizado ficou desbalanceado"


def test_replay_only_counts_matching_events(events):
    """O replay não pode imputar contrafactual: eventos discordantes são descartados."""
    result = replay_evaluate(ThompsonSampling(len(ARMS)), events, ARMS, uniformize=True)

    assert result.n_matched <= result.n_total
    assert 0.0 <= result.conversion_rate <= 1.0
    assert result.conversions <= result.n_matched


def test_replay_of_fixed_baseline_matches_that_arm_only(events):
    """Uma política determinística só aproveita eventos do seu próprio braço.

    Como o log uniformizado tem o mesmo volume por braço, a taxa de
    aproveitamento deve ficar próxima de 1/n_braços.
    """
    policy = FixedBaseline(len(ARMS), best_arm=0)
    result = replay_evaluate(policy, events, ARMS, uniformize=True)
    assert abs(result.match_rate - 1 / len(ARMS)) < 0.01


def test_replay_requires_segment_column(events):
    with pytest.raises(ValueError):
        replay_evaluate(ThompsonSampling(len(ARMS)), events.drop(columns=[SEGMENT_COL]), ARMS)


# --------------------------------------------------------------------------
# Política servida
# --------------------------------------------------------------------------
def test_warm_start_with_uniform_prior_equals_observed_counts(events):
    """Com prior uniforme explícito, a posterior é exatamente prior mais contagens."""
    alpha, beta = fit_posterior_from_events(
        events, ARMS, n_segments=3, prior_alpha=1.0, prior_beta=1.0
    )

    grouped = events.groupby([SEGMENT_COL, "arm"])["reward"].agg(["sum", "count"])
    for (segment, arm), row in grouped.iterrows():
        index = ARMS.index(arm)
        assert alpha[segment, index] == pytest.approx(1.0 + row["sum"])
        assert beta[segment, index] == pytest.approx(1.0 + row["count"] - row["sum"])


def test_warm_start_default_uses_informative_prior(events):
    """Por padrão, a posterior soma as contagens a um prior calibrado por braço."""
    strength = 50.0
    alpha0, beta0 = arm_priors(events, ARMS, strength)
    alpha, beta = fit_posterior_from_events(events, ARMS, n_segments=3, shrinkage=strength)

    # A massa do prior é exatamente `strength`, e sua média é a taxa do braço.
    assert np.allclose(alpha0 + beta0, strength)

    grouped = events.groupby([SEGMENT_COL, "arm"])["reward"].agg(["sum", "count"])
    for (segment, arm), row in grouped.iterrows():
        index = ARMS.index(arm)
        assert alpha[segment, index] == pytest.approx(alpha0[index] + row["sum"])
        assert beta[segment, index] == pytest.approx(
            beta0[index] + row["count"] - row["sum"]
        )


def test_informative_prior_tames_a_thin_cell():
    """Uma célula pequena e extrema não pode se tornar o melhor braço servido.

    Teste de regressão do defeito que levava o Golden Set a recomendar um canal
    com base em cerca de 20 observações. Sem prior informativo, 17 conversões em
    21 tentativas produzem uma média posterior de 78%, que supera qualquer
    alternativa estimada com muito mais dados.
    """
    rows = [{SEGMENT_COL: 0, "arm": ARMS[0], "reward": r} for r in [1] * 183 + [0] * 92]
    rows += [{SEGMENT_COL: 0, "arm": ARMS[5], "reward": r} for r in [1] * 17 + [0] * 4]
    # Contexto global: o braço 5 tem desempenho ruim na carteira inteira.
    rows += [{SEGMENT_COL: 1, "arm": ARMS[5], "reward": r} for r in [1] * 150 + [0] * 2850]
    rows += [{SEGMENT_COL: 1, "arm": ARMS[0], "reward": r} for r in [1] * 400 + [0] * 2600]

    frame = pd.DataFrame(rows)
    alpha, beta = fit_posterior_from_events(frame, ARMS, n_segments=2, shrinkage=50.0)
    means = alpha[0] / (alpha[0] + beta[0])

    assert means[0] > means[5], (
        "a célula pequena do braço 5 superou a célula com muitos dados do "
        "braço 0, indicando que o prior informativo não conteve o ruído"
    )


def test_recommender_roundtrips_and_keeps_decisions(events, tmp_path):
    segmenter = Segmenter(n_segments=3, random_state=0).fit(events)
    recommender = Recommender.from_events(events, segmenter, ARMS, n_segments=3)

    segmenter.save(tmp_path / "seg.joblib")
    recommender.save(tmp_path / "policy.json")
    reloaded = Recommender.load(tmp_path / "policy.json", tmp_path / "seg.joblib")

    client = {
        "age": 45, "job": "technician", "education": "high.school", "housing": "no",
        "loan": "no", "campaign": 1, "pdays": PDAYS_NEVER_CONTACTED,
        "previous": 0, "poutcome": "nonexistent",
    }
    assert (
        recommender.recommend(client, explore=False).arm
        == reloaded.recommend(client, explore=False).arm
    )


def test_greedy_recommendation_is_deterministic(events):
    segmenter = Segmenter(n_segments=3, random_state=0).fit(events)
    recommender = Recommender.from_events(events, segmenter, ARMS, n_segments=3)

    client = {
        "age": 30, "job": "admin.", "education": "university.degree", "housing": "yes",
        "loan": "no", "campaign": 1, "pdays": PDAYS_NEVER_CONTACTED,
        "previous": 0, "poutcome": "nonexistent",
    }
    decisions = {recommender.recommend(client, explore=False).arm for _ in range(20)}
    assert len(decisions) == 1


def test_feedback_moves_the_posterior(events):
    segmenter = Segmenter(n_segments=3, random_state=0).fit(events)
    recommender = Recommender.from_events(events, segmenter, ARMS, n_segments=3)

    before = recommender.posterior_mean(0)[0]
    for _ in range(500):
        recommender.update(0, ARMS[0], 1.0)
    assert recommender.posterior_mean(0)[0] > before


def test_feedback_rejects_bad_input(events):
    segmenter = Segmenter(n_segments=3, random_state=0).fit(events)
    recommender = Recommender.from_events(events, segmenter, ARMS, n_segments=3)

    with pytest.raises(ValueError):
        recommender.update(0, "canal_inexistente", 1.0)
    with pytest.raises(ValueError):
        recommender.update(0, ARMS[0], 0.5)
    with pytest.raises(ValueError):
        recommender.update(99, ARMS[0], 1.0)
