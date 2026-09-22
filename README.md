# Monte Carlo — NFL

Projeto pessoal. Estima o resultado final da temporada da NFL simulando os
jogos que faltam, e projeta a corrida aos playoffs.

Irmão do [Brasileirão](https://github.com/danguibes/monte-carlo-brasileirao), e
herda dele o método — medir antes de afirmar, mercado como régua, e registrar o
que foi descartado e por quê. Mas o modelo é **outro**, por três razões que a
medição deixou claras logo no começo.

```bash
python dados.py --refresh    # baixa o games.csv do nflverse
python regua.py              # caminhada para a frente contra o spread
python regua.py --buscar     # refaz a busca de λ e encolhimento
```

## Por que não dá para copiar o modelo do futebol

**1. A margem é feita de 3 e 7.** Medido em 6.999 jogos de 2002 a 2026:

| \|margem\| | frequência |
|---|---|
| **3** | **15,0%** |
| 7 | 9,1% |
| 6 | 6,0% |
| 10 | 5,6% |
| 5 | 3,6% |

Margem 3 é quatro vezes mais comum que margem 5. Nenhuma Poisson reproduz isso
— a estrutura não está na média, está na aritmética do esporte. Por isso aqui
não se modela "pontos por time": modela-se a **margem** diretamente, e o
desvio do erro contra o spread é de **13,19 pontos**.

**2. A amostra nunca fica boa.** São 17 jogos por time na temporada inteira,
contra 38 do Brasileirão — e na semana 3 são dois. Ajustar 32 ratings livres em
32 jogos é ruído puro. A saída é o **prior**: cada time começa com a força que
terminou a temporada anterior, encolhida para a média, e os jogos novos puxam
dali. Formalmente, uma crista centrada no prior:

```
minimiza  Σ(margem − Xr)²  +  λ‖r − prior‖²
```

λ e o encolhimento **não são escolhidos** — saem de busca em grade sobre 24
temporadas.

**3. A vantagem de casa tem um experimento natural.** Medida por era:

| era | vantagem |
|---|---|
| 1999–2009 | +2,58 |
| 2010–2019 | +2,19 |
| **2020, sem torcida** | **+0,05** |
| 2021–2026 | +2,02 |

Praticamente toda a vantagem de casa some sem público. Isso não é teoria: é
2020 medido.

## A régua vem de graça

Diferente do futebol, onde as odds eram históricas e difíceis de achar, aqui o
`spread_line` do nflverse cobre 97% dos jogos desde 1999, vale também para os
jogos **futuros**, e é praticamente não-enviesado (viés medido: +0,055 ponto).
Então a régua existe desde a primeira linha de código.

**Caminhada para a frente, 5.999 jogos previstos (2003–2026):**

| | RMSE | MAE | log-loss | acerto |
|---|---|---|---|---|
| só vantagem de casa | 14,667 | 11,384 | 0,6860 | 55,9% |
| **este modelo** | **13,839** | **10,801** | **0,6461** | **62,9%** |
| mercado (spread) | 13,203 | 10,271 | 0,6098 | 66,9% |

**O modelo andou 57% do caminho** entre o baseline e o mercado em RMSE, e 52%
em log-loss. Falta 0,64 ponto de RMSE.

Isso é com ratings de margem apenas — sem EPA, sem QB, sem viagem. Os três
entram depois, e cada um só fica se **melhorar a régua**.

## Dados

Tudo do [nflverse](https://github.com/nflverse), sem chave, atualizado
diariamente. Um arquivo resolve quase tudo:

`games.csv` — 7.548 jogos desde 1999, com agenda, placar, **descanso de cada
lado**, spread e total do mercado, clima, superfície, jogo de divisão, e o **QB
titular de cada jogo**. É o `home_qb_name` histórico que vai permitir *medir*
quanto vale um QB, em vez de chutar.

O projeto começa em **2002**, primeiro ano com 32 times e 8 divisões — o
formato de hoje. Antes disso a estrutura de conferência e os desempates eram
outros.

## O que ainda não existe

- **EPA como entrada do rating.** É o próximo passo, e a razão é conhecida:
  EPA/play estabiliza muito mais rápido que pontos, porque usa centenas de
  jogadas por jogo em vez de um placar. Mas entra medindo, não por fé.
- **QB e lesões.** O caminho é medir o valor do titular contra o reserva no
  histórico, não arbitrar "vale 5 pontos".
- **Viagem e fuso.** O descanso já está no modelo; distância e fuso ainda não.
- **Motor de playoffs.** Sete por conferência, quatro campeões de divisão nas
  cabeças 1–4, e uma cascata de doze critérios de desempate com regras
  diferentes para divisão e wild card — incluindo a regra de **recomeçar do
  passo 1** quando um time é eliminado de um empate múltiplo. O teste de
  aceitação é absoluto: reproduzir a semeadura real de 2002 a 2025, time por
  time.
- **Simulação e página.** Depois que as peças acima estiverem medidas.

## Mercado: régua, com mistura opcional

O spread não entra no modelo. Fica como régua, para medir quanto do erro é
irredutível. Haverá um parâmetro de mistura, **desligado por padrão**, para
medir exatamente quanto o mercado adiciona — e aí a decisão de usá-lo ou não
passa a ser de propósito, não de suposição.
