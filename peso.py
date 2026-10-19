"""Quanto do rating de hoje ainda é 2025, e quanto já é 2026?

A crista centrada no prior não "troca" uma coisa pela outra num dia marcado:
ela mistura as duas continuamente, e o peso de cada lado muda a cada jogo. A
pergunta merece resposta exata, e tem uma, porque o ajuste é LINEAR:

    rating = A⁻¹ · (Xᵀy  +  P·prior)
             ╰──┬──╯      ╰────┬───╯
          jogos de 2026   priors

Os dois termos são vetores de pontos, somam exatamente o rating final, e podem
ser calculados separadamente. Não é aproximação nem regra de bolso — é a mesma
conta do `modelo.py`, aberta ao meio.

O prior se divide ainda em dois: o rating encolhido de 2025 (por time) e o
histórico de liga de mando e descanso, que tem peso de 6 mil jogos e por isso
quase não se move nunca.

    python peso.py
    python peso.py GB
"""
import sys

import numpy as np
import pandas as pd

from dados import carregar
from modelo import (CASA_HIST, DESCANSO_HIST, LAM_CASA, LAM_DESCANSO, N_EXTRA,
                    _matriz, prior_da_temporada)
from simular import LAM, ENC


def decompor(jogos_atual, times, prior, lam=LAM):
    """Separa o rating final nas parcelas que vêm de cada fonte."""
    p = len(times)
    pr = np.zeros(p + N_EXTRA)
    pr[:p] = pd.Series(prior).reindex(times).fillna(0.0).to_numpy()
    pr[p], pr[p + 1] = CASA_HIST, DESCANSO_HIST
    P = np.zeros(p + N_EXTRA)
    P[:p] = lam
    P[p], P[p + 1] = LAM_CASA, LAM_DESCANSO

    X = _matriz(jogos_atual, times)
    y = jogos_atual.margem.to_numpy(float)
    A = X.T @ X + np.diag(P)
    A[p:, p:] += np.eye(N_EXTRA) * 1e-6

    def parte(b):
        s = np.linalg.solve(A, b)
        r = pd.Series(s[:p], index=times)
        return r, float(s[p]), float(s[p + 1])

    de_2026 = parte(X.T @ y)
    pri = P * pr
    de_2025 = parte(np.where(np.arange(p + N_EXTRA) < p, pri, 0.0))
    de_liga = parte(np.where(np.arange(p + N_EXTRA) >= p, pri, 0.0))

    # A centragem (rating menos a média) é linear, então atravessa a soma.
    media = sum(x[0].mean() for x in (de_2026, de_2025, de_liga))
    return {
        "2026": (de_2026[0] - de_2026[0].mean(), de_2026[1], de_2026[2]),
        "2025": (de_2025[0] - de_2025[0].mean(), de_2025[1], de_2025[2]),
        "liga": (de_liga[0] - de_liga[0].mean(), de_liga[1], de_liga[2]),
        "_media": media,
    }


def sensibilidade(jogos_atual, times, prior, lam=LAM):
    """Se o prior de um time subisse 1 ponto, quanto subiria o rating dele?

    É a leitura mais direta de "quanto de 2025 ainda está aí": 1,00 seria o
    rating ainda ser o do ano passado; 0,00 seria 2025 ter sido esquecido.
    """
    p = len(times)
    P = np.zeros(p + N_EXTRA)
    P[:p] = lam
    P[p], P[p + 1] = LAM_CASA, LAM_DESCANSO
    X = _matriz(jogos_atual, times)
    A = X.T @ X + np.diag(P)
    A[p:, p:] += np.eye(N_EXTRA) * 1e-6
    J = np.linalg.solve(A, np.diag(P))[:p, :p]      # ∂rating / ∂prior
    return pd.Series(np.diag(J), index=times)


def main():
    df = carregar()
    temporada = int(df.season.max())
    times = sorted(set(df.home_team) | set(df.away_team))
    ant = df[(df.season == temporada - 1) & df.disputado]
    prior = prior_da_temporada(ant, times, ENC, LAM)
    atual = df[(df.season == temporada) & df.disputado]

    por_time = (atual.groupby("home_team").size()
                .add(atual.groupby("away_team").size(), fill_value=0))
    print(f"temporada {temporada}: {len(atual)} jogos disputados, "
          f"{int(por_time.min())} a {int(por_time.max())} por time\n")

    d = decompor(atual, times, prior)
    s = sensibilidade(atual, times, prior)

    print(f"PESO DE {temporada-1} NO RATING DE HOJE")
    print(f"  se o rating de {temporada-1} de um time subisse 1 ponto, o de hoje")
    print(f"  subiria {s.mean():.3f}  (de {s.min():.3f} a {s.max():.3f} entre os 32)")
    print(f"  equivale a {LAM * (1-s.mean()) / s.mean():.1f} jogos de informação nova,")
    print(f"  contra os {LAM:.0f} de peso que o prior tem por construcao (lambda)\n")

    peso25 = d["2025"][0].abs().sum()
    peso26 = d["2026"][0].abs().sum()
    print(f"  somando os 32 ratings em módulo:")
    print(f"    vem de {temporada-1}  {peso25:6.1f} pontos  "
          f"({peso25/(peso25+peso26)*100:.0f}%)")
    print(f"    vem de {temporada}  {peso26:6.1f} pontos  "
          f"({peso26/(peso25+peso26)*100:.0f}%)")

    print(f"\nEFEITOS DE LIGA — aqui o passado manda quase sozinho")
    for nome, i, hist in (("mando", 1, CASA_HIST), ("descanso", 2, DESCANSO_HIST)):
        de26, de25, deliga = (d[k][i] for k in ("2026", "2025", "liga"))
        tot = de26 + de25 + deliga
        print(f"  {nome:9s} {tot:+7.3f} = {deliga:+.3f} (histórico de liga) "
              f"{de26:+.3f} ({temporada}) {de25:+.3f} ({temporada-1})"
              f"   -> {abs(deliga)/abs(tot)*100:.0f}% histórico")

    alvo = [t for t in sys.argv[1:] if t in times]
    if not alvo:
        alvo = list(s.sort_values().index[:3]) + list(s.sort_values().index[-3:])
    print(f"\nPOR TIME")
    print(f"  {'':5s}{'rating':>8}{'  =  de 2026':>14}{'  +  de 2025':>14}"
          f"{'peso 2025':>11}{'jogos':>7}")
    for t in alvo:
        r26, r25 = d["2026"][0][t], d["2025"][0][t]
        print(f"  {t:5s}{r26+r25:+8.2f}{r26:+14.2f}{r25:+14.2f}"
              f"{s[t]*100:10.0f}%{int(por_time[t]):7d}")


if __name__ == "__main__":
    main()
