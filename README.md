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
temporadas. Medidos (RMSE fora da amostra):

| λ \ encolhimento | 0,0 | 0,3 | 0,5 | **0,7** | 0,9 |
|---|---|---|---|---|---|
| 4 | 14,010 | 13,879 | 13,839 | 13,839 | 13,877 |
| **8** | 14,024 | 13,863 | 13,803 | **13,783** | 13,801 |
| 12 | 14,104 | 13,933 | 13,862 | 13,825 | 13,822 |
| 20 | 14,256 | 14,088 | 14,007 | 13,952 | 13,922 |
| 50 | 14,556 | 14,432 | 14,361 | 14,299 | 14,248 |

O ótimo é **interior** nas duas dimensões, não na borda — a grade cobriu o
mínimo de verdade.

E a coluna que mais informa é a primeira: **encolhimento 0,0 é o pior valor em
toda linha.** Ignorar a temporada anterior custa cerca de 0,22 ponto de RMSE,
que é mais de um terço da distância inteira até o mercado. O prior não é um
detalhe de regularização — é a peça que mais carrega o modelo enquanto a
amostra é pequena.

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
| **este modelo** | **13,783** | **10,800** | **0,6426** | **62,7%** |
| mercado (spread) | 13,203 | 10,271 | 0,6098 | 66,9% |

**O modelo andou 60% do caminho** entre o baseline e o mercado em RMSE, e 57%
em log-loss. Falta 0,58 ponto de RMSE.

Uma ressalva que a tabela esconde: ao ajustar λ e o encolhimento, o RMSE
melhorou (13,839 → 13,783) e o **acerto piorou** (62,9% → 62,7%). Não é
contradição — otimizar erro quadrático não é otimizar quantas vezes se acerta o
vencedor, e as duas medidas podem andar em sentidos opostos. A escolha aqui foi
pelo RMSE, porque é ele que alimenta a simulação.

Isso é com ratings de margem apenas — sem EPA, sem QB, sem viagem. Os três
entram depois, e cada um só fica se **melhorar a régua**.

## EPA: medido, e não entrou

A afirmação corrente é que EPA estabiliza mais rápido que o placar. Testada
aqui, **não se sustentou** — e o caminho até essa conclusão ensinou mais que
ela.

`regua_epa.py` ajusta o rating sobre uma resposta misturada,
`y = α·margem_real + (1−α)·margem_de_EPA`, medindo sempre a previsão da margem
**real** futura. α = 1 é só placar; α = 0 é só EPA.

| α (peso do placar) | RMSE | acerto |
|---|---|---|
| 1,00 (só placar) | 13,7816 | 62,74% |
| 0,85 | 13,7749 | 62,94% |
| **0,70** | **13,7742** | 63,11% |
| 0,50 | 13,7826 | **63,38%** |
| 0,30 | 13,8016 | 63,16% |
| 0,00 (só EPA) | 13,8501 | 62,81% |

O mínimo de RMSE cai em α = 0,70, mas o ganho é de **0,0075 ponto, com t =
0,92** — nada. O acerto melhora mais (+0,63 p.p. em α = 0,5, t = 2,30), só que
esse α foi escolhido *depois* de ver sete valores em duas métricas; corrigindo
para a seleção, o limiar subiria para perto de 2,7. Não é evidência.

**EPA fica desligado.** Continua no repositório, medido e disponível.

### Dois erros silenciosos encontrados no caminho

**1. Códigos de franquia.** O `games.csv` usa `OAK`, `SD` e `STL`; o
play-by-play usa `LV`, `LAC` e `LA`. Sem a ponte, **714 jogos — 11% da base**
perdiam o EPA, e o único sintoma era um `NaN` que ninguém olha. Normalizado em
`dados.py`: a franquia é a mesma, e o rating deve atravessar a mudança de
cidade.

**2. A primeira versão do teste reprovava o EPA por escala, não por
informação.** A margem de EPA de ataque tem desvio **17,3** contra 14,6 da
margem real, e média −0,55 contra +2,18. Misturadas cruas, a escala da resposta
mudava junto com α — o rating inflava 18% em α = 0 — e a previsão saía
sistematicamente grande demais. Com esse defeito, α = 1 vencia por larga
margem e o EPA teria sido descartado por um motivo que não é o dele. Corrigido
trazendo as duas séries para a mesma escala, com referência sempre na
temporada anterior, para não haver vazamento.

### Por que o teste usa EPA só de ataque

A margem de EPA **total** correlaciona **0,992** com a margem real — o EPA
telescopa ao longo do jogo, e a diferença entre os dois lados reconstrói o
placar final. Misturar placar com isso seria misturar um número com ele mesmo.

Uma intuição que se corrige junto: EPA total **de um time** não é o placar dele
(correlação 0,78 com os pontos marcados, média −0,64 contra 22,35). É a
*margem* que reconstrói o resultado, não a soma de um lado.

Por isso o teste usa EPA de passe e corrida, que correlaciona 0,936 e deixa de
fora retorno, jogada especial e touchdown de defesa — as partes de maior
variância e menor repetibilidade.

### O que ainda não foi tentado

Misturar respostas não é a única forma de usar EPA, e talvez não seja a melhor.
O desenho que falta testar é separar ataque e defesa em dois ratings de EPA e
somá-los ao rating de pontos, em vez de diluir a resposta. Fica registrado como
**não tentado**, não como descartado.

## QB: o efeito é enorme, e quase impossível de antecipar

O titular vale **−4,37 pontos** quando falta (t = −13,07). Isso põe número no
folclore de "uns 5 pontos" e é, de longe, o maior efeito medido neste projeto —
o dobro da vantagem de casa.

Mas efeito grande não é previsão melhor, e a distância entre os dois é a lição
inteira.

| cenário | RMSE | log-loss | acerto |
|---|---|---|---|
| sem QB | 13,7816 | 0,64265 | 62,74% |
| corrige o passado, assume titular de volta | 13,8286 | 0,64574 | 62,51% |
| **situação persiste** | 13,7922 | **0,64005** | **63,79%** |
| sabe quem começa (irreal) | 13,6615 | 0,63552 | 63,88% |
| mercado (spread) | 13,2025 | 0,60975 | 66,86% |

**Corrigir o passado e assumir o titular de volta PIORA** (t = −3,29), e o
mecanismo é direto: a correção infla o rating justamente do time desfalcado, e
depois prevê como se o titular tivesse voltado. Lesão de quarterback não
evapora de uma semana para a outra.

Trocando a premissa para "a situação do último jogo continua", o sinal inverte:
acerto sobe **+1,05 p.p., t = 2,95**. Em log-loss o ganho é menor e não
significante (t = 1,45), e em RMSE é nulo — o ajuste conserta o **lado** da
previsão sem encurtar o erro quadrático.

E a conta que mais orienta o próximo passo: a premissa de persistência captura
só **37%** do benefício de saber de fato quem começa. O valor está menos no
modelo e mais em ligar uma fonte de escalação — boletim de lesões ou depth
chart, que o nflverse também publica.

Uma restrição que decidiu o desenho, medida antes de tudo: dos **240 jogos
ainda não disputados de 2026, só 16 têm o QB preenchido**. O titular de um jogo
futuro não vem desta fonte.

### Por que a definição de "reserva" mudou no meio

A primeira definição, sem vazamento, marcava como reserva todo QB que ainda não
tinha acumulado jogos pelo time. Medido: **37,6% dos jogos assim marcados foram
começados por alguém com 8 ou mais jogos naquela temporada** — o novato ou o
recém-contratado que ERA o titular do ano.

Com essa diluição o efeito aparecia como **−2,35** em vez de −4,37: quase
metade dele escondida por uma escolha de rótulo. A medição usa a definição
retrospectiva, que olha a temporada inteira e por isso **só serve para medir**;
qualquer uso preditivo fica com a versão sem vazamento.

## Mando por time: medido, e não sobrou nada para medir

A intuição é forte — Green Bay em dezembro, o barulho de Seattle, a altitude de
Denver — e a resposta do modelo é um número só, **2,22 pontos para os 32 times**.
A pergunta merecia medição, e `casa.py` a faz.

O método evita o confundidor óbvio (time bom ganha em casa por ser bom) usando
o **resíduo fora da amostra**: a previsão já embute força dos dois lados, mando
de liga e descanso. Mas o resíduo médio em casa ainda mistura o mando extra do
time (h) com o erro do rating dele (b). A separação sai da própria definição da
margem, que é (casa − fora):

    resíduo médio em CASA  =  +h + b
    resíduo médio FORA     =       −b

Somar cancela o rating e deixa h; subtrair deixa b, que serve de **controle**.

Sobre 5.999 jogos previstos, 2003–2026:

| | pontos |
|---|---|
| desvio observado de h entre os 32 times | 1,33 |
| desvio que o puro acaso produziria | **1,40** |
| dispersão verdadeira que sobra | **zero** |

A dispersão observada é **menor que o ruído esperado**. Ela não é pequena: ela
é indistinguível de nenhuma. O ranking existe e é convincente de ler —
Pittsburgh +1,79, Green Bay +1,73, Tampa −2,73 — e é exatamente o que 32 médias
de 185 jogos fazem quando o efeito verdadeiro é o mesmo para todos.

A persistência confirma, e é o teste que decide:

| | correlação |
|---|---|
| h dos anos pares × h dos ímpares | **−0,005** |
| h de 2003–2014 × h de 2015–2026 | −0,345 |

Zero e negativo. O time que foi forte em casa nos anos pares não é o forte nos
ímpares.

E a régua, em `regua_casa.py`, fecha: acrescentar à previsão o h estimado só com
o passado, em grade de janela (3, 5, 10, todas) e de encolhimento, **piora o
RMSE em todas as 24 combinações menos três**, e essas três são o canto da grade
onde h já foi encolhido a quase zero (−0,0009 ponto, com |h| médio de 0,24).
Uma grade monótona na direção de "desligue isto" é a forma mais limpa de um
teste dizer não.

Vale a ressalva sobre o que **não** foi medido: isto é o mando de um time ao
longo de 24 anos. Não responde se um estádio específico, numa janela específica
— Seattle no auge do barulho, digamos —, teve vantagem real por alguns anos. O
que a medição responde é a pergunta que o modelo precisa fazer: **dá para saber
de antemão?** Não dá.

## Modelo × mercado, e por que a divergência não é oportunidade

A página tem uma seção que põe lado a lado a margem do modelo e a linha das
casas de aposta, ordenável pela diferença. Hoje são **31 jogos de 240** — a
linha só existe para as semanas próximas, e essa escassez é a razão de o spread
nunca entrar no modelo: um modelo que dependesse dela não simularia a temporada
até o fim, que é a única coisa que este projeto faz.

A tentação de ler a coluna `Diferença` como lista de apostas é óbvia, e por isso
a legenda dela é medida em `discordancia.py`, não escrita de cabeça. Nos 5.999
jogos previstos fora da amostra entre 2003 e 2026:

| discordância | n | RMSE modelo | RMSE linha | diferença |
|---|---|---|---|---|
| 0–1 pts | 1.778 | 13,32 | 13,31 | +0,01 |
| 1–2 pts | 1.485 | 13,32 | 13,23 | +0,09 |
| 2–3 pts | 1.098 | 13,17 | 12,94 | +0,23 |
| 3–5 pts | 1.165 | 13,75 | 13,35 | +0,40 |
| **5+ pts** | 473 | **15,14** | **12,97** | **+2,17** |

A desvantagem do modelo **cresce com a discordância**. Onde os dois concordam,
empatam; onde o modelo diverge muito, ele erra 2,2 pontos a mais. É o retrato de
quem discorda por ruído, não por informação.

O teste direto confirma: nos 1.638 jogos em que a discordância passou de 3
pontos, o placar caiu do lado apontado pelo modelo em **49,0%** das vezes
(±2,4 p.p., t = −0,79 contra 50%). Moeda.

Isso não invalida a seção — invalida a leitura fácil dela. A divergência é
interessante como **diagnóstico do modelo**: onde ele discorda da linha, ou o
mercado sabe de algo que o placar não conta (lesão, escalação, motivação), ou o
rating de alguém ainda está preso ao ano anterior. Os dois casos apontam para
onde olhar, e nenhum aponta para onde apostar.

A probabilidade do mercado na tabela passa pela **mesma** nuvem de resíduos e
pelo mesmo sorteio do modelo; só o centro muda. Assim a diferença entre as duas
colunas de porcentagem é a diferença entre as duas margens, e não entre dois
métodos.

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

- **Viagem e fuso.** O descanso já está no modelo; distância e fuso ainda não.
- **Motor de playoffs.** Sete por conferência, quatro campeões de divisão nas
  cabeças 1–4, e uma cascata de doze critérios de desempate com regras
  diferentes para divisão e wild card — incluindo a regra de **recomeçar do
  passo 1** quando um time é eliminado de um empate múltiplo. O teste de
  aceitação é absoluto: reproduzir a semeadura real de 2002 a 2025, time por
  time.
- **Simulação e página.** Depois que as peças acima estiverem medidas.

## O spread não pode entrar no modelo, e não é questão de princípio

A pergunta natural, depois de validar o spread como régua, é por que não usá-lo
como ingrediente. A resposta é uma medição, não uma opinião:

**Dos 240 jogos que faltam em 2026, só 16 têm spread** — exatamente a semana
seguinte. As casas só abrem linha para a rodada iminente.

| semana | jogos com spread |
|---|---|
| 3 | 16 de 16 |
| 4 | 0 de 16 |
| 5 | 0 de 15 |
| 6+ | 0 |

Uma projeção de temporada precisa de probabilidade para **todos** os jogos
restantes, inclusive os de dezembro. O spread não existe para eles hoje, e não
existirá até a véspera. Então ele pode melhorar a previsão *da próxima rodada*,
e não tem como alimentar a simulação que a página faz.

Isso fecha a questão por viabilidade, antes de chegar ao mérito. Se um dia a
página ganhar uma aba "jogos desta semana", aí o spread entra — e aí sim vale
discutir se o projeto quer ser um invólucro do mercado.

## Jogo em andamento não conta

Não consegui observar se o nflverse publica placar parcial — precisaria de uma
partida rolando no instante da conferência, e não havia nenhuma quando fui
olhar. Mas o projeto irmão do Brasileirão ensinou o custo de descobrir isso no
dia da rodada: lá as **duas** fontes publicavam parcial e **concordavam entre
si**, então a conferência cruzada — que era a defesa — passava com nota máxima
enquanto o placar de um jogo no intervalo entrava no ajuste, na tabela e na
projeção.

A defesa que não depende da fonte é o relógio: só vale como encerrado o jogo
que começou há mais de **seis horas**. Jogo de NFL dura umas 3h10, então seis é
folgado, e errar para o lado de não contar é barato — o jogo entra na
atualização seguinte.

O horário do `games.csv` é de **Nova York**, e o relógio usado é o de lá, não o
da máquina. Usar `datetime.now()` cru foi um bug que no Brasileirão **só
aparecia no CI**, onde o runner roda em UTC: um jogo das 20:30 parecia ter
começado há três horas, e na máquina do dono o mesmo código acertava.

E o placar parcial é **apagado da linha**, não só marcado. Marcar sem apagar
seria pior que não ter guarda: qualquer código que leia `home_score` sem olhar
`disputado` continuaria usando o parcial, em silêncio.

`testes.py` trava tudo isso com relógio congelado, porque um bug que só se
manifesta com jogo rolando não pode depender de alguém estar olhando na hora.
Roda como primeiro passo de `atualizar.py`, sem tolerância a falha.

## O que está no modelo, e o que não está

| | no modelo? | valor medido |
|---|---|---|
| vantagem de casa | **sim** | +2,24 pontos (t = 12,0) |
| dias de descanso | **sim** | +0,17 ponto por dia (t = 2,2) |
| campo neutro (Londres, Super Bowl) | **sim** | zera o mando |
| prior da temporada anterior | **sim** | encolhimento 0,7 |
| horário do jogo | não | — |
| distância de viagem e fuso | não | — |
| clima, vento, superfície | não | — |
| quarterback | medido, fora | −4,37 pontos (ver acima) |
| EPA | medido, fora | ver acima |

Os pesos do prior de liga **não são arbitrados**: numa crista, `λ = σ²/τ²`,
onde σ é o erro por jogo (13,2 pontos) e τ é a incerteza do próprio prior — o
erro-padrão da estimativa histórica. Com 6.255 jogos atrás dela, o histórico é
muito mais preciso que os 32 jogos da temporada corrente, e a conta diz isso
sozinha.

Eu tinha arbitrado λ = 300 para os dois, e o descanso saía em **+0,62** contra
os +0,17 medidos. Corrigido, o modelo foi de 76% para **78%** do caminho até o
mercado.

E o prior de descanso era **chutado**: eu tinha escrito 0,10 no código, um
número que inventei e que ficou lá parecendo medição. O medido é 0,168.

## Empates: 11 vezes demais, até medir

Arredondar `(total ± margem)/2` produz placar igual sempre que a margem
contínua cai perto de zero. Resultado: **2,65% de empates na simulação, contra
0,24% na NFL de verdade**.

Empate na NFL exige a prorrogação inteira terminar nivelada, o que quase nunca
acontece — o normal é um field goal decidir. Então o empate de arredondamento é
resolvido como a prorrogação resolve, três pontos para o lado que a margem
contínua favorecia, e só a fração calibrada sobrevive.

Medido depois: **0,226% no Python e 0,243% no navegador**, contra 0,24% real.
