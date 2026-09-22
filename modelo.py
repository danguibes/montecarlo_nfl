"""Rating de força por time, e a margem esperada de cada jogo.

    margem = rating[casa] − rating[fora] + vantagem_casa + β·(diferença de descanso)

O problema central da NFL não é a fórmula, é a AMOSTRA: 17 jogos por time na
temporada inteira, e duas na semana 3. Ajustar 32 ratings livres em 32 jogos é
receita de ruído puro.

A saída é o prior: cada time começa a temporada com a força que terminou a
anterior, encolhida em direção à média, e os jogos novos puxam a estimativa a
partir daí. Formalmente é uma crista (ridge) centrada no prior em vez de em
zero:

    minimiza  Σ(margem − Xr)²  +  λ‖r − prior‖²

λ grande = o prior manda; λ pequeno = a temporada corrente manda. Os dois
números que governam isso — λ e o encolhimento do prior — NÃO são chutados:
saem de busca em grade sobre 24 temporadas, em regua.py.
"""
import numpy as np
import pandas as pd

# Vantagem de casa e descanso são efeitos de LIGA, e por isso têm prior próprio.
#
# Deixá-los flutuar livres parecia certo — são efeitos com milhares de jogos por
# trás — mas o ajuste só enxerga a temporada corrente. Na semana 3 de 2026, com
# 32 jogos, a vantagem de casa estimada saiu em **−0,77 ponto**: mando de campo
# negativo, que iria direto para a página sem nada quebrar.
#
# O prior vem da história da liga e é puxado com peso equivalente a algumas
# centenas de jogos, então o começo de temporada usa o histórico e o fim usa o
# que aconteceu.
N_EXTRA = 2
# Medidos por regressão sobre os 6.255 jogos de 2002 a 2026, controlando um
# pelo outro. O descanso estava CHUTADO em 0,10 — um número que eu inventei e
# que ficou no código parecendo medição. O medido é 0,168, com t = 2,2.
CASA_HIST = 2.237         # t = 12,0
DESCANSO_HIST = 0.1683    # ponto por dia de descanso a mais, t = 2,2
# O peso do prior nao e escolhido: numa crista, lambda = sigma^2 / tau^2, onde
# sigma e o desvio do erro por jogo (13,2 pontos) e tau a incerteza do proprio
# prior — o erro-padrao da estimativa historica. Com 6.255 jogos por tras, o
# historico e MUITO mais preciso que 32 jogos da temporada corrente, e a conta
# diz isso sozinha.
#
# LAM_LIGA = 300, que eu tinha arbitrado, deixava a temporada corrente com
# metade do peso e o descanso saia em +0,62 contra os +0,17 medidos.
SIGMA = 13.2
SE_CASA, SE_DESCANSO = 0.186, 0.0756
LAM_CASA = SIGMA ** 2 / SE_CASA ** 2          # ~5.000
LAM_DESCANSO = SIGMA ** 2 / SE_DESCANSO ** 2  # ~30.000
LAM_QB = 400.0

# Coluna opcional de QB: (reserva na casa − reserva fora). Serve para CORRIGIR
# O PASSADO — um time que perdeu dois jogos sem o titular não deve carregar
# isso no rating. Para a frente, a coluna é zero: assume-se que o titular joga,
# porque o QB de um jogo futuro simplesmente não existe na fonte (medido: 16 de
# 240 jogos de 2026 têm o nome preenchido).
COL_QB = "qb_dif"


def _matriz(jogos, times, com_qb=False):
    """Desenho: +1 para o mandante, −1 para o visitante, mais casa e descanso."""
    idx = {t: i for i, t in enumerate(times)}
    n, p = len(jogos), len(times)
    extra = N_EXTRA + (1 if com_qb else 0)
    X = np.zeros((n, p + extra))
    X[np.arange(n), jogos.home_team.map(idx).to_numpy()] = 1.0
    X[np.arange(n), jogos.away_team.map(idx).to_numpy()] = -1.0
    X[:, p] = 1.0 - jogos.neutro.to_numpy()          # campo neutro não tem mando
    X[:, p + 1] = jogos.dif_descanso.fillna(0).to_numpy()
    if com_qb:
        X[:, p + 2] = jogos[COL_QB].fillna(0).to_numpy()
    return X


def ajustar(jogos, times, prior=None, lam=8.0, com_qb=False):
    """Ratings a partir dos jogos dados, puxados para `prior`.

    `jogos` pode estar VAZIO — e é o caso da semana 1. Aí o resultado é o
    próprio prior, que é exatamente o comportamento desejado.
    """
    p = len(times)
    extra = N_EXTRA + (1 if com_qb else 0)
    pr = np.zeros(p + extra)
    if prior is not None:
        pr[:p] = pd.Series(prior).reindex(times).fillna(0.0).to_numpy()

    # Penalidade nos times (para o prior deles) e nos efeitos de liga (para o
    # histórico). Sem a segunda, começo de temporada devolve mando negativo.
    P = np.zeros(p + extra)
    P[:p] = lam
    P[p], P[p + 1] = LAM_CASA, LAM_DESCANSO
    pr[p], pr[p + 1] = CASA_HIST, DESCANSO_HIST
    if com_qb:
        P[p + 2] = LAM_QB
        pr[p + 2] = -4.4

    if len(jogos) == 0:
        # Sem jogo, a crista devolve o prior. Casa e descanso ficariam
        # indeterminados, então usam os valores de liga medidos.
        return _desempacotar(pr, times, com_qb)

    X = _matriz(jogos, times, com_qb)
    y = jogos.margem.to_numpy(float)
    A = X.T @ X + np.diag(P)
    b = X.T @ y + P * pr
    # Casa e descanso sem penalidade deixam A singular se a temporada tiver
    # pouquíssimo jogo; uma crista mínima os estabiliza sem mover o resultado.
    A[p:, p:] += np.eye(extra) * 1e-6
    sol = np.linalg.solve(A, b)
    return _desempacotar(sol, times, com_qb)


def _desempacotar(sol, times, com_qb=False):
    p = len(times)
    r = pd.Series(sol[:p], index=times)
    fit = {"rating": r - r.mean(), "casa": float(sol[p]),
           "descanso": float(sol[p + 1])}
    fit["qb"] = float(sol[p + 2]) if com_qb else 0.0
    return fit


def margem_esperada(fit, jogos):
    r = fit["rating"]
    return (r.reindex(jogos.home_team).to_numpy()
            - r.reindex(jogos.away_team).to_numpy()
            + fit["casa"] * (1.0 - jogos.neutro.to_numpy())
            + fit["descanso"] * jogos.dif_descanso.fillna(0).to_numpy()
            + fit.get("qb", 0.0) * (jogos[COL_QB].fillna(0).to_numpy()
                                    if COL_QB in jogos else 0.0))


def prior_da_temporada(jogos_temporada, times, encolhimento=0.7, lam=8.0):
    """Rating final de uma temporada, encolhido para virar prior da seguinte.

    Encolhimento 0 = a temporada anterior não vale nada; 1 = vale inteira.
    O valor certo é medido, não escolhido — ver regua.py.
    """
    fit = ajustar(jogos_temporada, times, prior=None, lam=lam)
    return fit["rating"] * encolhimento
