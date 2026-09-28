# Plataforma de Experimentação Adaptativa para Próxima Melhor Ação

**Datathon POSTECH — Machine Learning Engineering (Fase 5)**

Uma instituição financeira digital precisa definir, para cada cliente elegível, o
próximo passo da abordagem comercial. Este trabalho formula essa decisão como um
problema de *multi-armed bandit* e a implementa como uma plataforma de
experimentação adaptativa: em lugar de fixar a escolha em uma regra estática ou
submetê-la a um teste A/B prolongado, o sistema equilibra exploração e
explotação, atualiza-se a cada resposta observada e permanece em aprendizado
após a implantação.

---

## Índice

1. [O problema de negócio](#1-o-problema-de-negócio)
2. [A base de dados (Etapa 1)](#2-a-base-de-dados-etapa-1)
3. [Modelagem da decisão (Etapa 2)](#3-modelagem-da-decisão-etapa-2)
4. [Algoritmos implementados (Etapa 3)](#4-algoritmos-implementados-etapa-3)
5. [Resultados e avaliação (Etapas 3 e 4)](#5-resultados-e-avaliação-etapas-3-e-4)
6. [Golden Set — 5 casos de teste (Etapa 4)](#6-golden-set--5-casos-de-teste-etapa-4)
7. [Serviço de decisão (Etapa 5)](#7-serviço-de-decisão-etapa-5)
8. [Arquitetura-alvo em nuvem (Etapa 6)](#8-arquitetura-alvo-em-nuvem-etapa-6)
9. [Ciclo de vida MLOps (Etapa 7)](#9-ciclo-de-vida-mlops-etapa-7)
10. [Governança, ética e limitações](#10-governança-ética-e-limitações)
11. [Como executar](#11-como-executar)
12. [Estrutura do repositório](#12-estrutura-do-repositório)
13. [Roteiro do vídeo pitch (Etapa 8)](#13-roteiro-do-vídeo-pitch-etapa-8)

---

## 1. O problema de negócio

Em canais digitais, cada contato com um cliente constitui uma decisão sob
incerteza. As duas abordagens tradicionais apresentam limitações de naturezas
opostas:

| Abordagem | Problema |
|---|---|
| **Regra fixa** (contato sempre pelo mesmo canal) | Não personaliza e não reage a mudanças de comportamento. Alterado o ambiente, a regra permanece subótima até que uma revisão manual a atualize. |
| **Teste A/B prolongado** | Aloca tráfego de forma ineficiente: mantém metade dos clientes na variante inferior por semanas, mesmo após a evidência acumulada já indicar a variante superior. |

A abordagem adaptativa (*multi-armed bandit*) contorna ambas as limitações:
realoca o tráfego continuamente à medida que a evidência se acumula, preservando
apenas a exploração que ainda possui valor informativo.

**Decisão modelada.** Para cada cliente elegível, o próximo passo da abordagem:
o canal e o dia da semana do contato.

**Recompensa.** Conversão binária — a adesão ou não do cliente ao depósito a
prazo.

---

## 2. A base de dados (Etapa 1)

**Base Kaggle:** [Bank Marketing — henriqueyamahata](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
(arquivo `bank-additional-full.csv`)

**Origem primária:** UCI Machine Learning Repository — *Bank Marketing Data Set*,
Moro, Cortez & Rita (2014). Licença CC BY 4.0.

| Característica | Valor |
|---|---|
| Eventos (após remoção de 12 duplicatas) | 41.176 |
| Colunas originais | 21 |
| Conversão global | 11,27% |
| Período | maio/2008 – novembro/2010 |

A ingestão foi implementada em duas vias, consultadas nesta ordem: o Kaggle, por
meio da biblioteca `kagglehub`, e o espelho público da UCI. Ambas as vias
fornecem o mesmo arquivo (`bank-additional-full.csv`), o que assegura a
reprodutibilidade do pipeline.

**EDA completa:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

### Achados que orientaram a modelagem

**1. A variável `duration` constitui vazamento temporal e foi descartada.** A
duração da ligação só é conhecida após o seu término, ao passo que a decisão é
tomada antes do contato. A correlação com a conversão é de 0,405 e as
distribuições das duas classes apresentam sobreposição mínima: um modelo que a
utilizasse obteria métricas elevadas e careceria de validade operacional. A
exclusão reduz as métricas aparentes e é a única alternativa metodologicamente
defensável.

**2. Há uma ação dominante.** Todos os braços de telefonia celular apresentam
conversão próxima ao triplo da observada em qualquer braço de telefonia fixa
(2,8× no agregado):

| Braço | Eventos | Conversão |
|---|---:|---:|
| `cellular\|tue` | 5.104 | 15,77% |
| `cellular\|thu` | 5.802 | 15,36% |
| `cellular\|wed` | 5.052 | 15,28% |
| `cellular\|fri` | 4.644 | 14,56% |
| `cellular\|mon` | 5.533 | 12,80% |
| `telephone\|wed` | 3.082 | 5,74% |
| `telephone\|thu` | 2.816 | 5,43% |
| `telephone\|fri` | 3.182 | 5,34% |
| `telephone\|tue` | 2.982 | 4,96% |
| `telephone\|mon` | 2.979 | 4,67% |

Disso decorre uma implicação relevante para a avaliação: uma regra fixa que
selecione sempre o braço `cellular|tue` já é quase ótima, e qualquer algoritmo
que precise explorar apresentará desempenho inferior a ela em um ambiente
estacionário. Este trabalho quantifica explicitamente esse custo.

**3. O efeito contextual observado incide sobre custo, não sobre conversão.** Na
base completa, a telefonia fixa converte cerca de um terço do observado na
telefonia celular. No segmento de clientes que converteram em campanha anterior,
os dois canais são estatisticamente indistinguíveis:

| Grupo | Celular | Telefone fixo | Diferença |
|---|---:|---:|---|
| Base inteira | 14,74% (n=26.135) | 5,23% (n=15.041) | +9,51 p.p. (z = 29,4) |
| Já converteu antes (`poutcome=success`) | 65,20% (n=1.270) | 64,08% (n=103) | +1,12 p.p. (z = 0,23; **p = 0,82**) |

Na base completa a diferença é expressiva; no segmento de reengajamento, o
intervalo de 95% da diferença é [−8,5 p.p.; +10,8 p.p.] e contém o zero. Para
esses clientes, portanto, a penalidade associada ao canal não se verifica, o que
permite empregar o canal de menor custo sem perda de conversão. Trata-se de um
achado de custo operacional, não de conversão.

> **Nota metodológica.** A célula `telephone|tue` desse segmento apresenta taxa
> de conversão de 81%, o que sugeriria uma inversão de canal. A célula reúne,
> contudo, apenas 21 observações, e a diferença desaparece na agregação por
> canal. A taxa de 81% é ruído amostral e não é reportada como achado.

**4. A variável `poutcome` (desfecho da campanha anterior) é o preditor mais
forte da base:**

| poutcome | Clientes | Conversão |
|---|---:|---:|
| `success` | 1.373 | 65,11% |
| `failure` | 4.252 | 14,23% |
| `nonexistent` | 35.551 | 8,83% |

---

## 3. Modelagem da decisão (Etapa 2)

A base registra a conversão observada (`y`), de modo que nenhum rótulo é
sintetizado. Cada linha corresponde a um evento de decisão:

```
contexto do cliente  →  braço aplicado (canal | dia)  →  recompensa (0/1)
```

### Espaço de ações — 10 braços

| Coluna | Papel | Valores |
|---|---|---|
| `contact` | canal do contato | `cellular`, `telephone` |
| `day_of_week` | momento da abordagem | `mon`–`fri` |

Ambas são variáveis de ação, sob controle da instituição e registradas na base,
e não atributos do cliente. A distinção evita o erro de tratar uma característica
pessoal como alavanca de decisão.

### Contexto — como a personalização entra

O bandit contextual opera por discretização: os clientes são agrupados em
segmentos comportamentais (KMeans, k=4) e cada segmento mantém uma distribuição
posterior independente. No interior de um segmento, o problema se reduz a um
bandit clássico.

| Seg | Clientes | Share | Conversão | Perfil |
|---:|---:|---:|---:|---|
| 0 | 29.304 | 71,2% | 8,9% | Prospecção fria (sem histórico) — grupo 0 |
| 1 | 1.402 | 3,4% | 65,0% | **Reengajamento quente** (converteu antes) |
| 2 | 6.247 | 15,2% | 8,5% | Prospecção fria (sem histórico) — grupo 2 |
| 3 | 4.223 | 10,3% | 13,9% | **Reengajamento morno** (recusou antes) |

Os agrupamentos recuperam, essencialmente, o estado do relacionamento com o
cliente.

> **Limitação registrada.** Os segmentos 0 e 2 apresentam perfis e taxas de
> conversão próximos: o KMeans dividiu a população sem histórico em dois grupos
> que não correspondem a comportamentos distintos. O efeito sobre a política é
> reduzido, uma vez que ela estima parâmetros semelhantes em ambos, mas `k=4`
> sugere uma estrutura de quatro grupos genuinamente diferentes que a base não
> sustenta.

### Padronização das variáveis indicadoras na segmentação

O KMeans agrupa por distância euclidiana, de modo que a influência de cada
atributo é proporcional à sua variância. Uma variável indicadora de categoria
rara — `poutcome=success` ocorre em 3,3% da base — apresenta variância ≈0,03,
ante 1,0 das variáveis numéricas padronizadas.

Sem a padronização das indicadoras, o atributo mais preditivo da base é diluído a
ponto de não influenciar o agrupamento, e clientes que aceitaram a oferta
anterior são alocados ao mesmo agrupamento de clientes que a recusaram. O sintoma
observado foi a atribuição, no Golden Set, de 78% de conversão esperada a um
cliente que havia recusado a oferta anterior.

A correção adotada — padronizar também as variáveis indicadoras — encontra-se em
[`segmentation.py`](src/adaptive_offers/segmentation.py), com a justificativa
registrada no código.

---

## 4. Algoritmos implementados (Etapa 3)

Foram implementadas onze políticas, que abrangem desde controles elementares até
algoritmos adaptativos contextuais.

### Baselines de controle

| Política | O que faz | Por que existe |
|---|---|---|
| `aleatorio` | Sorteia uniformemente | Limite inferior de referência: exploração permanente, sem explotação |
| **`baseline_fixo`** | Sempre o melhor braço histórico | Referência principal de comparação. Representa a operação vigente: decisão fixa, definida uma única vez a partir de relatório histórico |
| `baseline_segmentado` | Melhor braço histórico por segmento | Isola o ganho atribuível à segmentação do ganho atribuível ao aprendizado contínuo |

> O `baseline_fixo` é deliberadamente exigente: recebe sem custo o melhor braço
> conhecido e não incorre em custo de exploração. A adoção de uma referência
> menos exigente tornaria pouco informativo qualquer ganho reportado.

### Políticas adaptativas

**Thompson Sampling** (`thompson.py`) — política principal. A recompensa é
binária, o que torna exata a conjugação Beta-Bernoulli: a posterior tem forma
fechada e a atualização é um incremento inteiro. O custo é O(1) por evento, sem
reotimização, requisito para decidir dentro do orçamento de latência de um canal
digital. A exploração é proporcional à probabilidade de o braço ser o melhor:
braços com poucas observações apresentam posterior dispersa e permanecem sob
teste, enquanto braços consistentemente inferiores deixam de ser selecionados.

O prior padrão é Beta(1,1), uniforme. A escolha é deliberada: permite que o
algoritmo derive toda a informação dos dados e evita incorporar ao prior a
própria quantidade sob medição.

**Epsilon-Greedy** (`epsilon_greedy.py`) — política de interpretação mais direta,
na qual uma fração ε do tráfego é reservada à exploração. Serve de contraponto
técnico: com 10 braços e ε=0,10, cerca de 9% do tráfego é permanentemente
alocado a braços subótimos, mesmo após milhares de observações, o que produz
crescimento linear do regret. Foi implementada também a variante com decaimento
de ε.

**UCB1** (`ucb.py`) — otimismo diante da incerteza: `média(a) + c·√(ln t / n(a))`.
Em relação ao Thompson Sampling, apresenta a vantagem de produzir escolhas
determinísticas e reprodutíveis, o que facilita a auditoria posterior de uma
decisão específica. Em contrapartida, sob recompensas raras (~11%) o bônus
teórico é conservador e induz exploração superior à necessária, efeito
observável nos resultados.

**Thompson Sampling com desconto** (`DiscountedThompsonSampling`) — destinado a
ambientes não estacionários. O Thompson Sampling clássico acumula evidência
indefinidamente: após 50 mil observações, a posterior é altamente concentrada e o
algoritmo torna-se tão rígido quanto a regra fixa que deveria substituir. A
variante aplica esquecimento geométrico a cada passo:

```
alpha ← γ·alpha + (1-γ)·prior_alpha
beta  ← γ·beta  + (1-γ)·prior_beta
```

A memória efetiva é de aproximadamente `1/(1-γ)` observações por contexto. Com
γ=0,9998, são cerca de 5.000 observações, valor calibrado para ser
suficientemente longo para distinguir 10 braços com conversão em torno de 14% e
suficientemente curto para detectar uma quebra estrutural na ordem de milhares, e
não de dezenas de milhares, de eventos.

O desconto é aplicado a todos os braços do contexto, e não apenas ao braço
selecionado: a incerteza sobre um braço não utilizado também deve voltar a
crescer, sob pena de ele jamais ser reconsiderado.

**Referências dos algoritmos:** Chapelle & Li (2011), *An Empirical Evaluation of Thompson
Sampling*; Auer, Cesa-Bianchi & Fischer (2002), *Finite-time Analysis of the
Multiarmed Bandit Problem*; Raj & Kalyani (2017), *Taming Non-stationary Bandits*.

---

## 5. Resultados e avaliação (Etapas 3 e 4)

### O problema do contrafactual

Um bandit é um algoritmo online: seleciona uma ação e observa a resposta àquela
ação. Uma base histórica é offline e registra apenas o desfecho da ação
efetivamente executada. O desfecho de uma ação alternativa — o contrafactual —
não está registrado, de modo que não é possível executar o algoritmo diretamente
sobre a base.

Foram implementadas duas estratégias complementares:

**1. Simulador calibrado** ([`environment.py`](src/adaptive_offers/evaluation/environment.py)) —
a probabilidade de conversão de cada célula (segmento × braço) é estimada dos
dados observados e utilizada como ambiente Bernoulli, o que permite calcular o
regret em relação a um oráculo conhecido. Todas as probabilidades derivam de
contagens observadas.

**2. Replay off-policy** ([`replay.py`](src/adaptive_offers/evaluation/replay.py)) —
método de Li et al. (2011): percorre os eventos observados e contabiliza apenas
aqueles em que a política selecionou o braço efetivamente aplicado. O método não
pressupõe modelo do ambiente.

### Encolhimento empírico-Bayes

Cinco das 40 células apresentam menos de 30 eventos. A célula `telephone|tue` no
segmento de reengajamento quente reúne 21 observações e 17 conversões, taxa bruta
de 81%. Sem encolhimento, essa célula seria identificada como o melhor braço do
segmento e o oráculo passaria a perseguir ruído amostral.

O encolhimento é **hierárquico**: célula → braço → carteira.

```
p(s,a) = (sucessos(s,a) + k·p_braço(a)) / (n(s,a) + k)
p_braço(a) = (sucessos(a) + k·p_geral) / (n(a) + k)
```

O segundo nível é necessário porque um braço recém-introduzido — um canal novo —
também possui estimativa ruidosa, de modo que encolher apenas em sua direção
reproduziria o ruído. A implementação encontra-se em
[`statistics.py`](src/adaptive_offers/statistics.py) e é compartilhada pelo
simulador e pela política servida, o que impede divergência entre experimento e
produção.

### Limite superior do ganho por personalização

```
Oráculo (melhor braço por segmento) ...... 0,1466
Melhor regra fixa possível ............... 0,1433
Espaço para personalização ............... +2,34%
```

Mesmo um oráculo perfeito superaria a melhor regra fixa em apenas 2,34%. Trata-se
de consequência direta da ação dominante: à personalização resta apenas a escolha
do dia, e os dias diferem pouco entre si. O registro desse limite estabelece a
referência correta para a interpretação dos ganhos reportados adiante.

### Regime A — mundo estacionário

*100.000 decisões × 10 repetições com sementes distintas.*

| Política | Conversão média | Conversão convergida | Regret | Exploração | Uplift acum. | **Uplift convergido** |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_fixo` | 0,1439 | 0,1433 | 336 | 0,0% | — | — |
| **`thompson_sampling_contextual`** | 0,1422 | **0,1445** | 501 | 66,3% | −1,20% | **+0,89%** |
| `thompson_sampling` | 0,1416 | 0,1420 | 565 | 39,3% | −1,59% | −0,90% |
| `epsilon_greedy_decay` | 0,1403 | 0,1416 | 697 | 65,4% | −2,53% | −1,16% |
| `epsilon_greedy_contextual` | 0,1384 | 0,1391 | 865 | 78,5% | −3,83% | −2,89% |
| `epsilon_greedy` | 0,1376 | 0,1373 | 960 | 60,9% | −4,38% | −4,15% |
| `baseline_segmentado` | 0,1357 | 0,1351 | 1.151 | 18,6% | −5,73% | −5,69% |
| `thompson_sampling_discounted` | 0,1356 | 0,1363 | 1.153 | 74,2% | −5,80% | −4,83% |
| `ucb1` | 0,1281 | 0,1335 | 1.911 | 77,3% | −10,97% | −6,82% |
| `ucb1_contextual` | 0,1227 | 0,1285 | 2.443 | 82,4% | −14,71% | −10,30% |
| `aleatorio` | 0,0965 | 0,0966 | 5.018 | 90,0% | −32,96% | −32,58% |

**Interpretação das duas colunas de uplift.**

O *uplift acumulado* incorpora a fase de aprendizado, durante a qual a política
ainda identifica os braços de maior retorno. Nessa métrica o baseline apresenta
desempenho superior, o que não constitui artefato de medição: em um ambiente
estacionário, uma política que já dispõe da resposta não incorre em custo de
busca.

O *uplift convergido* mede o desempenho da política após o aprendizado, que
corresponde ao regime observado em operação. Nessa métrica o Thompson Sampling
contextual supera o baseline, tendo identificado a melhor ação a partir dos
dados, e é a única política com uplift convergido positivo.

> **Nota metodológica.** A coluna convergida emprega a conversão esperada
> (oráculo menos regret médio), e não a observada. Com conversão em torno de 14%,
> o ruído Bernoulli é de aproximadamente 0,1 p.p. mesmo em centenas de milhares
> de amostras, mesma ordem de
> grandeza da diferença medida. Como o regret é calculado sobre as probabilidades
> do ambiente, e não sobre as realizações amostrais, ele mede a qualidade da
> decisão sem esse ruído. Sem essa correção, o ganho de +0,89% permaneceria
> dentro da margem de erro e não poderia ser afirmado.

A forma da curva de regret constitui o principal diagnóstico
([`regret_acumulado.png`](artifacts/figures/regret_acumulado.png)): as políticas
`aleatorio` e `epsilon_greedy` exibem crescimento linear, o que indica taxa de
erro constante ao longo de todo o horizonte; o Thompson Sampling exibe
crescimento sublinear, com achatamento progressivo da curva — comportamento
característico de uma política que aprende.

### Regime B — quebra estrutural

Na metade do horizonte, os braços de telefonia celular passam a converter 20% do
valor anterior. O cenário corresponde a eventos plausíveis de negócio: restrição
regulatória ao contato por celular, elevação da taxa de opt-out ou saturação do
canal após uso intensivo prolongado.

*Nenhum dado de cliente é sintetizado: trata-se de um teste de estresse aplicado
à matriz calibrada em dados observados, com magnitude declarada e auditável.*

| Política | Conversão média | **Conversão pós-choque** | Uplift acum. | **Uplift pós-choque** |
|---|---:|---:|---:|---:|
| **`thompson_sampling_discounted`** | **0,0901** | **0,0512** | **+4,56%** | **+78,57%** |
| `baseline_fixo` | 0,0862 | 0,0287 | — | — |
| `thompson_sampling_contextual` | 0,0859 | 0,0340 | −0,36% | +18,53% |
| `baseline_segmentado` | 0,0850 | 0,0345 | −1,34% | +20,35% |
| `thompson_sampling` | 0,0848 | 0,0297 | −1,55% | +3,45% |
| `ucb1_contextual` | 0,0795 | 0,0422 | −7,74% | +47,20% |
| `aleatorio` | 0,0693 | 0,0420 | −19,61% | +46,51% |

**Interpretação.** Após o choque, o `baseline_fixo` mantém o contato pelo canal
degradado até que a perda seja detectada, analisada e a regra reimplantada,
processo que consome semanas em operação. O Thompson Sampling com desconto
detecta a mudança e realoca o tráfego automaticamente, na ordem de milhares de
eventos.

Registre-se que a política `aleatorio` supera o `baseline_fixo` após o choque. O
resultado não indica mérito da seleção aleatória, mas dimensiona o custo de uma
regra fixa em um ambiente alterado: a alocação uniforme torna-se preferível à
persistência em um braço que deixou de ser o melhor.

O desconto tem custo, observável no regime A (−4,83% convergido): a memória curta
reduz a precisão em ambientes estáveis. O mecanismo opera como um seguro, cujo
prêmio constante é pago para preservar a capacidade de adaptação.

### Validação complementar — replay off-policy

| Política | Conversão | IC 95% | Eventos aproveitados |
|---|---:|---|---:|
| `epsilon_greedy_decay` | 0,1617 | [0,148 – 0,175] | 2.832 |
| `baseline_fixo` | 0,1520 | [0,139 – 0,165] | 2.816 |
| `thompson_sampling` | 0,1505 | [0,137 – 0,164] | 2.810 |
| `thompson_sampling_contextual` | 0,1251 | [0,112 – 0,138] | 2.557 |
| `aleatorio` | 0,0978 | [0,087 – 0,109] | 2.813 |

O método pressupõe que o log tenha sido gerado por política uniformemente
aleatória, condição não satisfeita nesta base, dado o predomínio do contato por
celular. A correção adotada consistiu em subamostrar o log até equilibrar os
braços, ao custo de descartar cerca de 90% dos eventos.

> **Interpretação.** Com aproximadamente 2.800 eventos aproveitados, os
> intervalos de 95% de quase todas as políticas se sobrepõem. O replay não dispõe
> de poder estatístico para ordenar as políticas: sua função é verificar que
> nenhuma apresenta falha grosseira e servir de checagem de consistência ao
> simulador, não de evidência de ordenação. Qualquer ordenação extraída desta
> tabela refletiria ruído amostral.

---

## 6. Golden Set — 5 casos de teste (Etapa 4)

Métricas agregadas não revelam o comportamento em casos individuais: uma política
pode apresentar conversão média elevada e, ainda assim, produzir decisões
inadequadas em situações específicas. Cinco perfis construídos manualmente cobrem
os principais regimes de decisão:

| Caso | Persona | Seg | Recomendação | Conversão esperada | IC 95% | P(melhor) |
|---|---|---:|---|---:|---|---:|
| **GS-01** | Prospecção fria — 34a, admin., 1º contato | 0 | Celular na terça | 12,45% | [0,114–0,136] | 34,8% |
| **GS-02** | Reengajamento quente — converteu antes | 1 | Celular na quinta | **61,44%** | [0,563–0,665] | 44,4% |
| **GS-03** | Reengajamento morno — recusou antes | 3 | Celular na terça | **16,23%** | [0,137–0,189] | 68,9% |
| **GS-04** | Fadiga — 12 contatos no ciclo | 0 | Celular na terça | 12,45% | [0,114–0,136] | 34,8% |
| **GS-05** | Sênior aposentado, sem histórico | 0 | Celular na terça | 12,45% | [0,114–0,136] | 34,8% |

**Análise dos casos**

- O par **GS-02 / GS-03** é o mais informativo: ambos possuem histórico de
  campanha, mas um aceitou e o outro recusou a oferta anterior. A conversão
  esperada difere por um fator de 4, o que evidencia a incorporação do contexto à
  decisão. Era nesse par que se manifestava a falha de padronização das variáveis
  indicadoras, que atribuía 78% a ambos.
- **GS-02** recomenda o canal celular, e não a telefonia fixa. Uma recomendação
  de telefonia fixa, apoiada nas 21 observações daquela célula, indicaria falha
  do encolhimento e retorno ao ajuste sobre ruído amostral.
- **GS-04** (12 contatos no ciclo) recebe conversão esperada baixa, o que
  sinaliza retorno reduzido para contatos adicionais. A regra de supressão que
  atua sobre esse sinal é decisão de negócio, externa ao bandit.
- **GS-01** e **GS-05** recebem o mesmo braço, resultado esperado: na ausência de
  histórico, o contexto é pouco informativo e a decisão adequada é o melhor braço
  global. Uma política que diferenciasse esses casos estaria sobreajustando.

O conjunto opera como teste de regressão de comportamento: qualquer alteração de
modelo que modifique uma dessas decisões deve ser deliberada e justificada, e não
um efeito colateral não detectado.

```bash
make golden
```

---

## 7. Serviço de decisão (Etapa 5)

A política é exposta como serviço de próxima melhor ação por meio de uma API
implementada em FastAPI.

```bash
make serve          # http://127.0.0.1:8000/docs
```

### Inicialização informada (*warm start*)

O experimento compara os algoritmos a partir de posteriors não informativas,
condição adequada para medir capacidade de aprendizado. A implantação de um
bandit nessas condições, contudo, repetiria com clientes reais toda a exploração
já custeada no histórico.

Como a distribuição Beta é conjugada da Bernoulli, as contagens históricas
constituem estatística suficiente. A posterior condicionada a todo o log tem
forma fechada, sem aproximação nem retreino. A partir daí, cada resposta
observada em produção atualiza a posterior em O(1).

### Endpoints

| Método | Rota | Função |
|---|---|---|
| `POST` | `/recommend` | Recomenda o próximo passo |
| `POST` | `/feedback` | Registra a resposta observada e atualiza a política |
| `GET` | `/segments` | Segmentos aprendidos e melhor ação de cada um |
| `GET` | `/health` | Prontidão (responde mesmo sem política carregada) |

### Exemplo

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"client": {"age": 58, "job": "retired", "education": "professional.course",
                  "housing": "no", "loan": "no", "campaign": 1, "pdays": 3,
                  "previous": 2, "poutcome": "success"}}'
```

```json
{
  "arm": "cellular|thu",
  "arm_label": "Celular na quinta-feira",
  "segment": 1,
  "segment_label": "Reengajamento quente (converteu na campanha anterior)",
  "expected_conversion": 0.6144,
  "uncertainty": 0.0261,
  "credible_interval": [0.563, 0.665],
  "probability_best": 0.444,
  "exploring": false,
  "ranking": [ "..." ]
}
```

### Características do serviço

1. **Decisão sob incerteza, e não apenas predição.** O serviço retorna o
   intervalo de credibilidade e a probabilidade de o braço ser o melhor,
   informações que permitem ao canal decidir entre atuar automaticamente ou
   encaminhar o caso a um operador humano.
2. **Aprendizado após a implantação.** O endpoint `/feedback` fecha o ciclo
   adaptativo; sem ele, a plataforma equivaleria a um modelo estático.
3. **Sinalização de exploração.** O campo `exploring` indica quando a decisão
   corresponde a uma tentativa deliberada de exploração, e não à melhor
   estimativa corrente. A distinção é relevante para a atribuição de resultados:
   o tráfego de exploração não deve ser imputado à campanha como tráfego
   otimizado.

---

## 8. Arquitetura-alvo em nuvem (Etapa 6)

A implantação-alvo na **AWS** separa três caminhos com requisitos distintos de
latência e custo.

**Caminho de decisão (online, < 50 ms).** A requisição do canal digital é
recebida por um **API Gateway**, que aciona a aplicação FastAPI empacotada em
contêiner no **ECS Fargate** (ou em **Lambda**, dado que o payload é reduzido e a
inferência consiste na amostragem de distribuições Beta, sem carregamento de
modelo). As posteriors ocupam alguns kilobytes e são armazenadas em um
**DynamoDB** com chave `(segmento, braço)`: leitura
de milissegundos e, sobretudo, estado compartilhado entre réplicas. Este é o
ponto arquitetural mais relevante: a implementação local atualiza a posterior em
memória, o que pressupõe processo único; com múltiplas réplicas, cada uma
estimaria parâmetros distintos e a política divergiria entre elas. O DynamoDB
suporta incremento atômico, que é exatamente a operação de atualização
Beta-Bernoulli.

**Caminho de feedback (assíncrono).** A resposta observada no canal (conversão ou
não) entra por um tópico **Kinesis Data Streams**, é consumida por uma Lambda que
aplica o incremento atômico na posterior e persiste o evento bruto no **S3** (zona
raw, particionada por data). O desacoplamento entre feedback e decisão impede que
uma indisponibilidade do caminho de aprendizado interrompa o caminho de
decisão.

**Caminho de treino e governança (batch).** O pipeline completo — preparação,
segmentação, experimento, calibração — é executado em **SageMaker Pipelines**,
com agendamento semanal e leitura do S3 via **Glue Catalog**. O rastreamento de
experimentos migra do MLflow local para o **MLflow gerenciado no SageMaker**, e o
segmentador versionado é publicado no **SageMaker Model Registry**, com promoção
a produção condicionada a aprovação manual. **CloudWatch** monitora latência, taxa de exploração e
conversão por segmento; um alarme sobre a **queda de conversão de um braço**
dispara a revisão do choque estrutural descrito no Regime B. Os segredos são
mantidos no **Secrets Manager**, o isolamento de rede é feito por **VPC** e a
trilha de auditoria combina o **CloudTrail** com a tabela de decisões no S3,
necessária para justificar, meses depois, a abordagem atribuída a um cliente
específico.

**Custo.** O desenho é de baixo custo por construção: não há GPU, não há endpoint
de inferência permanentemente ativo com modelo de grande porte e o modelo cabe em
uma tabela DynamoDB. O custo concentra-se no Fargate do caminho online e no
Kinesis.

---

## 9. Ciclo de vida MLOps (Etapa 7)

O rastreamento de experimentos é feito em **MLflow**, com backend SQLite local.

```bash
make train          # executa o pipeline e registra o experimento
make mlflow         # abre a UI em http://127.0.0.1:5000
```

**Estrutura:** um run pai (`experimento_completo`), que registra o contexto do
experimento, e um run aninhado por política.

| Nível | Registrado |
|---|---|
| **Run pai** — parâmetros | base, nº de eventos, braços, segmentos, horizonte, repetições, semente, força do encolhimento, parâmetros do choque |
| **Run pai** — métricas | conversão do oráculo, melhor regra fixa, espaço de personalização, oráculo pós-choque |
| **Run pai** — artefatos | perfil dos segmentos, comparativos (estacionário / choque / replay), Golden Set, matriz de taxas, 7 figuras |
| **Run filho** — parâmetros | hiperparâmetros da política (ε, c, γ, priors, contextual) |
| **Run filho** — métricas | conversão média e desvio, conversão convergida, regret e desvio, exploração, uplift (acumulado e convergido), replay, **e as mesmas métricas no regime com choque** |
| **Run filho** — séries | curva de regret amostrada (50 pontos) nos dois regimes |

As métricas dos dois regimes são registradas no mesmo run, uma vez que essa é a
comparação que fundamenta a escolha arquitetural: o custo de cada política em
ambiente estável e o ganho obtido quando o ambiente se altera.

> **Nota técnica.** O backend de arquivo (`./mlruns`) entrou em modo de manutenção
> no MLflow 3 e levanta exceção. Adotou-se o backend SQLite (`mlflow.db`),
> substituto recomendado, que permanece integralmente local e habilita a
> interface completa.

### Testes

São 104 testes, voltados à verificação de comportamento e não apenas à ausência
de erros de execução:

```bash
make test
```

Os testes protegem invariantes cuja violação produz resultado plausível, porém
incorreto — modo de falha particularmente crítico em aprendizado de máquina, por
não gerar exceção:

- `duration` nunca retorna ao contexto e as colunas sensíveis permanecem ausentes
- o replay nunca imputa contrafactual
- o encolhimento contém células de baixa frequência (teste de regressão da falha
  identificada no Golden Set)
- o oráculo nunca é inferior à melhor regra fixa e o regret nunca é negativo
- há convergência ao melhor braço e o ε-greedy respeita o orçamento de exploração
- o TS com desconto adapta-se mais rapidamente que o clássico (medido em
  latência de adaptação)

> **Nota.** A primeira versão desse teste afirmava que o Thompson Sampling
> clássico não se adapta, o que é incorreto: sua taxa de exploração nunca se
> anula, de modo que ele eventualmente redescobre o melhor braço. A formulação
> correta, verificada pelo teste atual, é que sua adaptação exige um número
> substancialmente maior de eventos.

---

## 10. Governança, ética e limitações

### Minimização de dados

Entram na decisão apenas atributos comportamentais e de relacionamento. Cada
exclusão tem justificativa registrada em
[`config.py`](src/adaptive_offers/config.py):

| Variável | Decisão | Justificativa |
|---|---|---|
| `age`, `job`, `education` | **Usa** | Padrões de conversão bem definidos; monitoradas contra viés de proxy |
| `housing`, `loan` | **Usa** | Relacionamento com a instituição, não capacidade de pagamento |
| `campaign`, `pdays`, `previous`, `poutcome` | **Usa** | Recência, frequência e desfecho — o sinal mais forte da base |
| `duration` | **Fora** | Vazamento temporal: só é conhecida após a ligação |
| `marital` | **Fora** | Estado civil — sensível, sem justificativa de necessidade |
| `default` | **Fora** | Inadimplência — sensível; decisão de crédito exige humano no loop |
| macroeconômicas | **Fora** | Contexto ambiente, não atributo do cliente |
| `month` | **Fora** | Sazonalidade do histórico; não é decisão por cliente |

Renda, patrimônio, gênero e raça não constam da base e não possuem campo
correspondente na API: além de não serem utilizados, não há estrutura que os
comporte.

### Base legal e retenção

- **Finalidade:** otimizar a abordagem comercial de produto já contratado ou
  elegível. **Base legal:** legítimo interesse (LGPD art. 7º, IX), com opt-out.
- **Minimização:** apenas os 9 atributos listados acima.
- **Retenção:** eventos de decisão por 24 meses; posteriors agregadas
  indefinidamente (não contêm dado pessoal — são contagens por segmento).
- **Humano no loop:** o bandit decide *canal e momento*. Qualquer decisão de
  crédito, limite ou precificação está fora do escopo e exige análise humana.
- **Supressão:** abaixo de um limiar de conversão esperada, ou acima de um teto de
  contatos no ciclo, o próximo passo correto é **não contatar** — regra de negócio
  que age sobre o sinal do modelo (ver GS-04).

### Limitações declaradas

1. **Associação, não causalidade.** A instituição não atribuiu o canal de forma
   aleatória. É plausível que clientes com telefone celular cadastrado sejam
   sistematicamente mais jovens e mais digitalizados, de modo que parte da
   vantagem atribuída ao canal decorra do perfil de quem o possui. A medição
   causal exigiria randomização — precisamente o que uma plataforma de bandit
   produz ao explorar.
2. **O simulador é uma aproximação.** As taxas estimam a política de log que
   gerou os dados, e não o efeito causal da alteração de canal. Prestam-se à
   comparação de políticas em condições equivalentes, não à previsão de receita.
3. **O replay não tem poder estatístico** para ordenar políticas com
   aproximadamente 2.800 eventos aproveitados.
4. **O espaço de personalização é reduzido (+2,34%)**, em decorrência da ação
   dominante presente na base. Ganhos superiores exigiriam um espaço de ações com
   ofertas efetivamente distintas, o que esta base, restrita a um único produto,
   não contempla.
5. **O cenário de choque é um teste de estresse**, com magnitude arbitrada
   (×0,20) e declarada, e não uma previsão.
6. **Os segmentos 0 e 2 não são comportamentalmente distintos** (ver seção 3).
7. **Os dados abrangem o período de 2008 a 2010**, marcado pela crise financeira
   global. Os níveis absolutos de conversão não são transferíveis ao contexto
   atual; a estrutura do problema, sim.

---

## 11. Como executar

### Requisitos

Python ≥ 3.10. A obtenção da base dispensa credenciais de autenticação.

### Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Pipeline completo

```bash
make all
```

Ou passo a passo:

```bash
make data       # baixa e prepara a base
make train      # experimento completo + MLflow + artefatos  (~4 min)
make golden     # Golden Set — 5 casos
make serve      # API em http://127.0.0.1:8000/docs
make mlflow     # UI do MLflow
make test       # 104 testes
make notebooks  # executa os notebooks de ponta a ponta
```

### Execução reduzida (para demonstração)

```bash
python -m adaptive_offers.train --horizon 20000 --runs 3
```

### Recomendação direto do terminal

```bash
python -m adaptive_offers.cli recommend
```

---

## 12. Estrutura do repositório

```
├── README.md                     # este documento (governança consolidada)
├── requirements.txt / pyproject.toml
├── Makefile
├── notebooks/
│   ├── 01_eda.ipynb              # Etapa 1 — EDA e tratamento
│   └── 02_baseline_e_bandits.ipynb  # Etapas 2–4
├── src/adaptive_offers/
│   ├── config.py                 # ponto único de governança de features
│   ├── statistics.py             # encolhimento empírico-Bayes compartilhado
│   ├── segmentation.py           # KMeans — o contexto da decisão
│   ├── serving.py                # política servida + warm start
│   ├── golden_set.py             # Etapa 4 — 5 casos de referência
│   ├── train.py                  # pipeline completo + MLflow
│   ├── cli.py
│   ├── data/                     # ingestão Kaggle/UCI e preparação
│   ├── bandits/                  # baselines, ε-greedy, UCB1, Thompson
│   ├── evaluation/               # ambiente, simulação, replay, figuras
│   └── api/                      # Etapa 5 — FastAPI
├── tests/                        # 104 testes
└── artifacts/
    ├── figures/                  # 7 figuras
    └── models/                   # policy.json, segmenter, comparativos
```
