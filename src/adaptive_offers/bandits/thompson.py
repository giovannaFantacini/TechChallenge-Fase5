"""Thompson Sampling com posteriors Beta-Bernoulli.

Este é o algoritmo adaptativo principal do trabalho. A escolha se apoia em três
propriedades do problema.

1. A recompensa é binária (o cliente converteu ou não), então o par
   Beta-Bernoulli é conjugado. A posterior de cada braço tem forma fechada e a
   atualização após cada evento é apenas um incremento em alpha ou em beta. O
   custo por decisão é O(1) e não existe etapa de retreino, o que atende o
   requisito de latência de um canal digital.

2. A exploração é proporcional à probabilidade de o braço ser o melhor. O
   Epsilon-Greedy reserva uma fatia fixa do tráfego para explorar, inclusive em
   braços já descartados. No Thompson Sampling, um braço com poucas observações
   tem posterior larga e continua sendo sorteado, enquanto um braço com
   evidência consistente de baixa conversão deixa de ser escolhido sem que
   nenhum parâmetro precise ser ajustado.

3. A largura da posterior quantifica a incerteza da decisão. Esse valor é
   exposto na resposta da API (seção 7 do README) e permite definir uma regra
   de negócio para escalar casos incertos a um analista humano.

Referência: Chapelle, O., & Li, L. (2011). An Empirical Evaluation of Thompson
Sampling. NIPS.
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_GAMMA, DEFAULT_PRIOR_ALPHA, DEFAULT_PRIOR_BETA, RANDOM_SEED
from .base import BanditPolicy


class ThompsonSampling(BanditPolicy):
    """Bandit bayesiano com prior Beta(alpha, beta) por braço.

    O prior padrão é Beta(1, 1), equivalente à distribuição uniforme em [0, 1],
    ou seja, nenhuma crença inicial sobre a conversão de nenhum braço. Usamos
    esse prior no experimento porque o objetivo ali é medir quanto o algoritmo
    consegue aprender sozinho a partir dos dados; um prior informativo
    adiantaria parte da resposta e contaminaria a comparação com o baseline.

    Em produção a situação é outra, e o prior calibrado no histórico encurta a
    fase inicial de exploração. Por isso os parâmetros ficam expostos no
    construtor. A política que de fato é servida usa essa alternativa (ver
    `serving.py` e `statistics.py`).
    """

    name = "thompson_sampling"

    def __init__(
        self,
        n_arms: int,
        n_segments: int = 1,
        contextual: bool = False,
        prior_alpha: float = DEFAULT_PRIOR_ALPHA,
        prior_beta: float = DEFAULT_PRIOR_BETA,
        random_state: int = RANDOM_SEED,
    ):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Os parâmetros do prior Beta devem ser positivos.")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        super().__init__(n_arms, n_segments, contextual, random_state)

    def _init_state(self) -> None:
        super()._init_state()
        shape = (self.n_contexts, self.n_arms)
        self.alpha = np.full(shape, self.prior_alpha, dtype=np.float64)
        self.beta = np.full(shape, self.prior_beta, dtype=np.float64)

    def _select(self, ctx: int) -> int:
        # Sorteia um valor da posterior de cada braço e escolhe o maior.
        # A exploração vem desse sorteio: não há parâmetro de temperatura nem
        # cronograma de decaimento a calibrar.
        samples = self.rng.beta(self.alpha[ctx], self.beta[ctx])
        return int(np.argmax(samples))

    def _update(self, ctx: int, arm: int, reward: float) -> None:
        # Conjugação Beta-Bernoulli: sucesso incrementa alpha, falha incrementa beta.
        self.alpha[ctx, arm] += reward
        self.beta[ctx, arm] += 1.0 - reward

    # -- leitura da posterior ----------------------------------------------
    def posterior_mean(self, segment: int = 0) -> np.ndarray:
        """Média da posterior: alpha / (alpha + beta)."""
        ctx = self._ctx(segment)
        return self.alpha[ctx] / (self.alpha[ctx] + self.beta[ctx])

    def posterior_std(self, segment: int = 0) -> np.ndarray:
        """Desvio-padrão da posterior, usado como medida de incerteza."""
        ctx = self._ctx(segment)
        a, b = self.alpha[ctx], self.beta[ctx]
        return np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1.0)))

    def credible_interval(
        self, segment: int = 0, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intervalo de credibilidade central por braço."""
        from scipy.stats import beta as beta_dist

        ctx = self._ctx(segment)
        tail = (1.0 - level) / 2.0
        lower = beta_dist.ppf(tail, self.alpha[ctx], self.beta[ctx])
        upper = beta_dist.ppf(1.0 - tail, self.alpha[ctx], self.beta[ctx])
        return lower, upper

    def probability_best(self, segment: int = 0, n_samples: int = 10_000) -> np.ndarray:
        """Probabilidade de cada braço ser o melhor, estimada por Monte Carlo.

        Serve como critério de parada da exploração: quando um braço ultrapassa
        cerca de 95%, a evidência já é suficiente para tratar a decisão como
        resolvida naquele segmento.
        """
        ctx = self._ctx(segment)
        rng = np.random.default_rng(self.random_state)
        draws = rng.beta(
            self.alpha[ctx][None, :], self.beta[ctx][None, :], size=(n_samples, self.n_arms)
        )
        winners = np.argmax(draws, axis=1)
        return np.bincount(winners, minlength=self.n_arms) / n_samples

    @property
    def params(self) -> dict:
        return {
            **super().params,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
        }


class DiscountedThompsonSampling(ThompsonSampling):
    """Thompson Sampling com esquecimento geométrico (caso não estacionário).

    Motivação. No Thompson Sampling clássico as posteriors acumulam evidência
    indefinidamente. Após 50 mil observações um braço tem alpha + beta na casa
    dos milhares e a posterior fica muito concentrada. Se a conversão daquele
    braço cair pela metade, seriam necessárias dezenas de milhares de novas
    observações para a média posterior acompanhar a mudança. O algoritmo passa
    a ser tão lento para reagir quanto a regra fixa que ele deveria substituir.

    Solução adotada. A cada passo as posteriors do contexto atual são deslocadas
    na direção do prior por um fator gama:

        alpha <- gama * alpha + (1 - gama) * prior_alpha
        beta  <- gama * beta  + (1 - gama) * prior_beta

    Isso equivale a manter uma janela de memória de aproximadamente 1/(1 - gama)
    observações por contexto.

    Calibração de gama. Valores baixos esquecem rápido demais e o algoritmo não
    chega a distinguir dez braços cuja conversão difere por poucos pontos
    percentuais. Valores próximos de 1 não esquecem nada e reproduzem a rigidez
    do TS clássico. Testamos 0,999, 0,9995 e 0,9998 (resultados na seção 5 do
    README) e adotamos 0,9998, que corresponde a cerca de 5.000 observações de
    memória.

    Custo da escolha. Esquecer reduz a precisão quando o ambiente é estável: no
    regime estacionário essa política perde 4,83% de conversão frente ao
    baseline, enquanto no regime com quebra estrutural ganha 78,57%. O
    experimento reporta os dois números para deixar o compromisso explícito.

    Referência: Raj, V., & Kalyani, S. (2017). Taming Non-stationary Bandits: A
    Bayesian Approach. arXiv:1707.09727.
    """

    name = "thompson_sampling_discounted"

    def __init__(
        self,
        n_arms: int,
        n_segments: int = 1,
        contextual: bool = True,
        gamma: float = DEFAULT_GAMMA,
        prior_alpha: float = DEFAULT_PRIOR_ALPHA,
        prior_beta: float = DEFAULT_PRIOR_BETA,
        random_state: int = RANDOM_SEED,
    ):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma deve estar em (0, 1].")
        self.gamma = float(gamma)
        super().__init__(
            n_arms, n_segments, contextual, prior_alpha, prior_beta, random_state
        )

    def _update(self, ctx: int, arm: int, reward: float) -> None:
        # O desconto incide sobre todos os braços do contexto, e não apenas
        # sobre o braço jogado. Um braço que deixou de ser testado precisa
        # recuperar incerteza ao longo do tempo, caso contrário sua posterior
        # permanece estreita e ele nunca volta a ser considerado.
        g = self.gamma
        self.alpha[ctx] = g * self.alpha[ctx] + (1.0 - g) * self.prior_alpha
        self.beta[ctx] = g * self.beta[ctx] + (1.0 - g) * self.prior_beta

        self.alpha[ctx, arm] += reward
        self.beta[ctx, arm] += 1.0 - reward

    @property
    def effective_memory(self) -> float:
        """Número aproximado de observações que o algoritmo mantém em memória."""
        return float("inf") if self.gamma >= 1.0 else 1.0 / (1.0 - self.gamma)

    @property
    def params(self) -> dict:
        return {
            **super().params,
            "gamma": self.gamma,
            "memoria_efetiva": self.effective_memory,
        }


class ContextualThompsonSampling(ThompsonSampling):
    """Thompson Sampling com uma posterior independente por segmento.

    Implementa o bandit contextual por discretização: o segmento do cliente
    seleciona qual conjunto de posteriors será consultado. A hipótese testada é
    que o melhor braço varia entre segmentos. Um bandit global converge para o
    braço de melhor desempenho médio e, com isso, atende mal os segmentos cujo
    comportamento se afasta da média.

    No experimento esta é a única política com ganho positivo sobre o baseline
    no regime estacionário após a convergência (+0,89%).
    """

    name = "thompson_sampling_contextual"

    def __init__(
        self,
        n_arms: int,
        n_segments: int,
        prior_alpha: float = DEFAULT_PRIOR_ALPHA,
        prior_beta: float = DEFAULT_PRIOR_BETA,
        random_state: int = RANDOM_SEED,
    ):
        super().__init__(
            n_arms,
            n_segments,
            contextual=True,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            random_state=random_state,
        )
