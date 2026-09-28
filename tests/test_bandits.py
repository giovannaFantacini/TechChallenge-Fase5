"""Testes das políticas de decisão.

Os testes são comportamentais. Um bandit funciona quando converge para o melhor
braço, respeita o orçamento de exploração declarado e reage a mudanças no
ambiente. Verificar apenas se o código executa sem exceção não avaliaria nenhuma
dessas propriedades.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_offers.bandits import (
    ContextualThompsonSampling,
    DiscountedThompsonSampling,
    EpsilonGreedy,
    FixedBaseline,
    RandomPolicy,
    ThompsonSampling,
    UCB1,
)

N_ARMS = 4
N_SEGMENTS = 3


def run_bernoulli(policy, probs, steps=8000, seed=0, segment=0):
    """Executa a política contra braços Bernoulli fixos e devolve as escolhas."""
    rng = np.random.default_rng(seed)
    choices = []
    for _ in range(steps):
        arm = policy.select(segment)
        reward = int(rng.random() < probs[arm])
        policy.update(segment, arm, reward)
        choices.append(arm)
    return np.array(choices)


# --------------------------------------------------------------------------
# Contrato comum
# --------------------------------------------------------------------------
ALL_POLICIES = [
    lambda: RandomPolicy(N_ARMS),
    lambda: EpsilonGreedy(N_ARMS),
    lambda: UCB1(N_ARMS),
    lambda: ThompsonSampling(N_ARMS),
    lambda: ContextualThompsonSampling(N_ARMS, N_SEGMENTS),
    lambda: DiscountedThompsonSampling(N_ARMS, N_SEGMENTS),
]


@pytest.mark.parametrize("factory", ALL_POLICIES)
def test_select_returns_valid_arm(factory):
    policy = factory()
    for segment in range(policy.n_segments if policy.contextual else 1):
        for _ in range(50):
            assert 0 <= policy.select(segment) < N_ARMS


@pytest.mark.parametrize("factory", ALL_POLICIES)
def test_reset_clears_learning(factory):
    policy = factory()
    run_bernoulli(policy, [0.1, 0.9, 0.2, 0.3], steps=200)
    assert policy.t == 200

    policy.reset()
    assert policy.t == 0
    assert policy.counts.sum() == 0
    assert policy.rewards.sum() == 0


@pytest.mark.parametrize("factory", ALL_POLICIES)
def test_same_seed_reproduces_run(factory):
    a, b = factory(), factory()
    assert np.array_equal(
        run_bernoulli(a, [0.2, 0.5, 0.3, 0.1], steps=500),
        run_bernoulli(b, [0.2, 0.5, 0.3, 0.1], steps=500),
    )


# --------------------------------------------------------------------------
# Convergência
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory",
    [
        lambda: ThompsonSampling(N_ARMS),
        lambda: EpsilonGreedy(N_ARMS, epsilon=0.05),
        lambda: UCB1(N_ARMS, c=0.5),
    ],
)
def test_converges_to_best_arm(factory):
    """A política deve concentrar as escolhas finais no braço de maior taxa."""
    probs = [0.10, 0.15, 0.45, 0.12]
    choices = run_bernoulli(factory(), probs, steps=8000)

    tail = choices[-2000:]
    share_best = (tail == 2).mean()
    assert share_best > 0.75, f"braço ótimo escolhido em apenas {share_best:.1%} do final"


def test_thompson_beats_random_on_reward():
    probs = [0.05, 0.05, 0.40, 0.05]
    ts = run_bernoulli(ThompsonSampling(N_ARMS), probs, steps=5000)
    rnd = run_bernoulli(RandomPolicy(N_ARMS), probs, steps=5000)

    assert np.mean([probs[a] for a in ts]) > np.mean([probs[a] for a in rnd])


# --------------------------------------------------------------------------
# Comportamento específico de cada algoritmo
# --------------------------------------------------------------------------
def test_epsilon_greedy_respects_exploration_budget():
    """Com epsilon fixo, a fração de tráfego em braços subótimos é previsível.

    Com K braços e epsilon E, a exploração aleatória envia E*(K-1)/K do tráfego
    para braços não gulosos. Esse desperdício permanente é a causa do
    arrependimento linear do Epsilon-Greedy, e o teste fixa esse comportamento
    para que ele não mude sem que se perceba.
    """
    epsilon = 0.20
    probs = [0.02, 0.02, 0.60, 0.02]
    choices = run_bernoulli(EpsilonGreedy(N_ARMS, epsilon=epsilon), probs, steps=20000)

    suboptimal = (choices[-10000:] != 2).mean()
    expected = epsilon * (N_ARMS - 1) / N_ARMS
    assert abs(suboptimal - expected) < 0.03


def test_epsilon_decay_explores_less_over_time():
    probs = [0.05, 0.05, 0.50, 0.05]
    choices = run_bernoulli(
        EpsilonGreedy(N_ARMS, epsilon=0.5, decay=True), probs, steps=20000
    )

    early = (choices[:2000] != 2).mean()
    late = (choices[-2000:] != 2).mean()
    assert late < early / 2


def test_ucb_tries_every_arm_before_exploiting():
    """O bônus do UCB1 é infinito para braços ainda não testados."""
    policy = UCB1(N_ARMS)
    choices = run_bernoulli(policy, [0.1, 0.1, 0.9, 0.1], steps=N_ARMS)
    assert sorted(choices) == list(range(N_ARMS))


def test_thompson_posterior_tracks_true_rate():
    probs = [0.10, 0.80, 0.30, 0.05]
    policy = ThompsonSampling(N_ARMS)
    run_bernoulli(policy, probs, steps=20000)

    # O braço ótimo é o mais amostrado, logo sua posterior é a mais precisa.
    assert abs(policy.posterior_mean()[1] - 0.80) < 0.05


def test_thompson_probability_best_concentrates():
    policy = ThompsonSampling(N_ARMS)
    run_bernoulli(policy, [0.05, 0.05, 0.50, 0.05], steps=10000)
    assert policy.probability_best()[2] > 0.90


def test_credible_interval_brackets_the_mean():
    policy = ThompsonSampling(N_ARMS)
    run_bernoulli(policy, [0.2, 0.4, 0.6, 0.3], steps=4000)

    lower, upper = policy.credible_interval()
    mean = policy.posterior_mean()
    assert np.all(lower <= mean) and np.all(mean <= upper)


# --------------------------------------------------------------------------
# Contextualidade
# --------------------------------------------------------------------------
def test_contextual_learns_different_arms_per_segment():
    """Verifica se a política contextual aprende um ótimo distinto por segmento."""
    best_by_segment = {0: 0, 1: 2, 2: 3}
    probs = {
        0: [0.60, 0.10, 0.10, 0.10],
        1: [0.10, 0.10, 0.55, 0.10],
        2: [0.10, 0.10, 0.10, 0.50],
    }

    policy = ContextualThompsonSampling(N_ARMS, N_SEGMENTS)
    rng = np.random.default_rng(7)
    for _ in range(30000):
        segment = int(rng.integers(N_SEGMENTS))
        arm = policy.select(segment)
        policy.update(segment, arm, int(rng.random() < probs[segment][arm]))

    for segment, expected in best_by_segment.items():
        assert int(np.argmax(policy.posterior_mean(segment))) == expected


def test_non_contextual_shares_state_across_segments():
    """Sem contextualidade, todos os segmentos alimentam a mesma posterior."""
    policy = ThompsonSampling(N_ARMS, n_segments=N_SEGMENTS, contextual=False)
    policy.update(segment=0, arm=1, reward=1)
    policy.update(segment=2, arm=1, reward=1)

    assert policy.counts.shape[0] == 1
    assert policy.counts[0, 1] == 2


def test_contextual_rejects_out_of_range_segment():
    policy = ContextualThompsonSampling(N_ARMS, N_SEGMENTS)
    with pytest.raises(ValueError):
        policy.select(N_SEGMENTS)


# --------------------------------------------------------------------------
# Não estacionariedade
# --------------------------------------------------------------------------
def test_discounted_adapts_faster_than_plain_thompson():
    """Justifica a inclusão do Thompson Sampling com desconto.

    A formulação da hipótese precisa ser exata. O TS clássico não fica
    permanentemente preso, porque sua exploração nunca chega a zero: mais cedo ou
    mais tarde ele amostra o braço que melhorou, recebe recompensa e uma
    realimentação positiva o traz de volta. O problema é o tempo que isso leva,
    e em produção o intervalo entre a mudança e a reação corresponde a conversões
    perdidas.

    Por isso o teste mede latência de adaptação, e não capacidade de adaptação.
    O cenário escolhido é o pior caso para a memória longa: o braço 0 permanece
    bom (0,50) e o braço 2, abandonado cedo com 0,05, passa a valer 0,90. O TS
    clássico mantém uma posterior estreita e baixa no braço 2 e raramente o
    reconsidera. O desconto devolve incerteza a braços não testados e reduz esse
    tempo de espera.
    """
    before = [0.50, 0.10, 0.05, 0.10]
    after = [0.50, 0.10, 0.90, 0.10]
    phase, new_best = 15000, 2

    def adaptation_latency(policy) -> int:
        """Número de passos, após a mudança, até o novo ótimo dominar as escolhas."""
        rng = np.random.default_rng(11)
        window, latency = [], None

        for step in range(phase * 2):
            probs = before if step < phase else after
            arm = policy.select(0)
            policy.update(0, arm, int(rng.random() < probs[arm]))

            if step >= phase:
                window.append(arm == new_best)
                if len(window) > 500:
                    window.pop(0)
                if latency is None and len(window) == 500 and np.mean(window) >= 0.5:
                    latency = step - phase
        return latency if latency is not None else phase

    discounted = adaptation_latency(
        DiscountedThompsonSampling(N_ARMS, 1, contextual=True, gamma=0.999)
    )
    plain = adaptation_latency(ThompsonSampling(N_ARMS))

    assert discounted < phase, "TS com desconto não se adaptou dentro do horizonte"
    assert discounted * 3 < plain, (
        "o desconto deveria reagir muito mais rápido que o TS clássico "
        f"(desconto={discounted} passos, clássico={plain} passos)"
    )


def test_discount_widens_posterior_of_untouched_arms():
    """Braços não testados precisam recuperar incerteza para voltar a ser considerados."""
    policy = DiscountedThompsonSampling(N_ARMS, 1, contextual=True, gamma=0.99)
    for _ in range(500):
        policy.update(0, 0, 1)

    # O braço 1 nunca foi jogado: sua posterior deve permanecer larga (perto do
    # prior), e não colapsar junto com a do braço 0.
    assert policy.posterior_std()[1] > policy.posterior_std()[0]


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def test_fixed_baseline_never_deviates():
    policy = FixedBaseline(N_ARMS, best_arm=2)
    choices = run_bernoulli(policy, [0.1, 0.1, 0.5, 0.1], steps=500)
    assert set(choices.tolist()) == {2}


def test_fixed_baseline_rejects_invalid_arm():
    with pytest.raises(ValueError):
        FixedBaseline(N_ARMS, best_arm=N_ARMS)


def test_random_policy_is_uniform():
    choices = run_bernoulli(RandomPolicy(N_ARMS), [0.1] * N_ARMS, steps=20000)
    shares = np.bincount(choices, minlength=N_ARMS) / len(choices)
    assert np.all(np.abs(shares - 1 / N_ARMS) < 0.02)


# --------------------------------------------------------------------------
# Validação de parâmetros
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory",
    [
        lambda: ThompsonSampling(N_ARMS, prior_alpha=0),
        lambda: ThompsonSampling(N_ARMS, prior_beta=-1),
        lambda: EpsilonGreedy(N_ARMS, epsilon=1.5),
        lambda: UCB1(N_ARMS, c=-1),
        lambda: DiscountedThompsonSampling(N_ARMS, 1, gamma=0),
        lambda: DiscountedThompsonSampling(N_ARMS, 1, gamma=1.5),
        lambda: ThompsonSampling(0),
    ],
)
def test_invalid_parameters_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
