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

# Vantagem de casa e descanso entram sem penalidade: são efeitos de liga, com
# milhares de jogos por trás, e não precisam ser encolhidos como os times.
N_EXTRA = 2


def _matriz(jogos, times):
    """Desenho: +1 para o mandante, −1 para o visitante, mais casa e descanso."""
    idx = {t: i for i, t in enumerate(times)}
    n, p = len(jogos), len(times)
    X = np.zeros((n, p + N_EXTRA))
    X[np.arange(n), jogos.home_team.map(idx).to_numpy()] = 1.0
    X[np.arange(n), jogos.away_team.map(idx).to_numpy()] = -1.0
    X[:, p] = 1.0 - jogos.neutro.to_numpy()          # campo neutro não tem mando
    X[:, p + 1] = jogos.dif_descanso.fillna(0).to_numpy()
    return X


def ajustar(jogos, times, prior=None, lam=12.0):
    """Ratings a partir dos jogos dados, puxados para `prior`.

    `jogos` pode estar VAZIO — e é o caso da semana 1. Aí o resultado é o
    próprio prior, que é exatamente o comportamento desejado.
    """
    p = len(times)
    pr = np.zeros(p + N_EXTRA)
    if prior is not None:
        pr[:p] = pd.Series(prior).reindex(times).fillna(0.0).to_numpy()

    # Penalidade só nos times; casa e descanso ficam livres.
    P = np.zeros(p + N_EXTRA)
    P[:p] = lam

    if len(jogos) == 0:
        # Sem jogo, a crista devolve o prior. Casa e descanso ficariam
        # indeterminados, então usam os valores de liga medidos.
        pr[p], pr[p + 1] = 2.1, 0.0
        return _desempacotar(pr, times)

    X = _matriz(jogos, times)
    y = jogos.margem.to_numpy(float)
    A = X.T @ X + np.diag(P)
    b = X.T @ y + P * pr
    # Casa e descanso sem penalidade deixam A singular se a temporada tiver
    # pouquíssimo jogo; uma crista mínima os estabiliza sem mover o resultado.
    A[p:, p:] += np.eye(N_EXTRA) * 1e-6
    sol = np.linalg.solve(A, b)
    return _desempacotar(sol, times)


def _desempacotar(sol, times):
    p = len(times)
    r = pd.Series(sol[:p], index=times)
    return {"rating": r - r.mean(), "casa": float(sol[p]),
            "descanso": float(sol[p + 1])}


def margem_esperada(fit, jogos):
    r = fit["rating"]
    return (r.reindex(jogos.home_team).to_numpy()
            - r.reindex(jogos.away_team).to_numpy()
            + fit["casa"] * (1.0 - jogos.neutro.to_numpy())
            + fit["descanso"] * jogos.dif_descanso.fillna(0).to_numpy())


def prior_da_temporada(jogos_temporada, times, encolhimento=0.6, lam=12.0):
    """Rating final de uma temporada, encolhido para virar prior da seguinte.

    Encolhimento 0 = a temporada anterior não vale nada; 1 = vale inteira.
    O valor certo é medido, não escolhido — ver regua.py.
    """
    fit = ajustar(jogos_temporada, times, prior=None, lam=lam)
    return fit["rating"] * encolhimento
