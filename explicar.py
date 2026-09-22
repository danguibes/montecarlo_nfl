"""Abre um jogo e mostra de onde vem cada número da probabilidade.

    python explicar.py GB ATL

Existe porque "M 69 / V 31" não é auditável. Este arquivo refaz a conta na
ordem em que ela acontece, com os valores de verdade em cada degrau.
"""
import sys

import numpy as np
import pandas as pd

from dados import carregar
from modelo import (CASA_HIST, DESCANSO_HIST, LAM_CASA, LAM_DESCANSO,
                    ajustar, margem_esperada, prior_da_temporada)
from simular import LAM, ENC, MANTER_EMPATE, residuos_historicos, resolver_empates


def main():
    casa, fora = sys.argv[1], sys.argv[2]
    df = carregar()
    temporada = int(df.season.max())
    times = sorted(set(df.home_team) | set(df.away_team))

    jogo = df[(df.season == temporada) & (df.home_team == casa)
              & (df.away_team == fora) & ~df.disputado]
    if not len(jogo):
        sys.exit(f"{casa} x {fora} nao esta entre os jogos restantes de {temporada}")
    jogo = jogo.iloc[[0]]

    print(f"{'='*66}\n{casa} (mandante) x {fora} (visitante)")
    print(f"semana {int(jogo.week.iloc[0])}, {jogo.gameday.iloc[0]} "
          f"{jogo.gametime.iloc[0]} ET\n{'='*66}")

    # ---------- 1. o prior ----------
    ant = df[(df.season == temporada - 1) & df.disputado]
    bruto = ajustar(ant, times, prior=None, lam=LAM)["rating"]
    prior = prior_da_temporada(ant, times, ENC, LAM)
    print(f"\n1. PRIOR — o que cada time era ao fim de {temporada-1}")
    print(f"   ajuste sobre {len(ant)} jogos, encolhido por {ENC}")
    for t in (casa, fora):
        print(f"   {t:4s} rating {temporada-1}: {bruto[t]:+6.2f}  "
              f"x {ENC} = prior {prior[t]:+6.2f}")
    print(f"   (encolher e o que impede 2 jogos de {temporada} de mandarem sozinhos)")

    # ---------- 2. a temporada corrente ----------
    atual = df[(df.season == temporada) & df.disputado]
    print(f"\n2. TEMPORADA CORRENTE — {len(atual)} jogos ate agora")
    for t in (casa, fora):
        j = atual[(atual.home_team == t) | (atual.away_team == t)]
        for r in j.itertuples():
            local = "casa " if r.home_team == t else "fora "
            m = r.margem if r.home_team == t else -r.margem
            adv = r.away_team if r.home_team == t else r.home_team
            print(f"   {t:4s} {local} vs {adv:4s}  margem {m:+3.0f}")

    # ---------- 3. o ajuste ----------
    fit = ajustar(atual, times, prior=prior, lam=LAM)
    r = fit["rating"]
    print(f"\n3. AJUSTE — crista centrada no prior, lambda = {LAM}")
    for t in (casa, fora):
        print(f"   {t:4s} prior {prior[t]:+6.2f}  ->  rating final {r[t]:+6.2f}"
              f"   (moveu {r[t]-prior[t]:+.2f})")
    print(f"   forca relativa: {casa} {r[casa]:+.2f} menos {fora} {r[fora]:+.2f}"
          f" = {r[casa]-r[fora]:+.2f} pontos")

    # ---------- 4. mando e descanso ----------
    dif_desc = float(jogo.dif_descanso.fillna(0).iloc[0])
    neutro = int(jogo.neutro.iloc[0])
    print(f"\n4. EFEITOS DE LIGA")
    print(f"   vantagem de casa   {fit['casa']:+6.3f}  "
          f"(prior historico {CASA_HIST:+.3f}, peso {LAM_CASA:.0f})")
    print(f"   descanso           {fit['descanso']:+6.3f} por dia  "
          f"(prior historico {DESCANSO_HIST:+.4f}, peso {LAM_DESCANSO:.0f})")
    print(f"   {casa} descansou {int(jogo.home_rest.iloc[0])} dias, "
          f"{fora} descansou {int(jogo.away_rest.iloc[0])} -> diferenca {dif_desc:+.0f}")
    print(f"   campo neutro? {'sim' if neutro else 'nao'}")

    # ---------- 5. a margem esperada ----------
    mu = float(margem_esperada(fit, jogo)[0])
    print(f"\n5. MARGEM ESPERADA")
    print(f"   {r[casa]-r[fora]:+.3f}  (forca)")
    print(f"   {fit['casa']*(1-neutro):+.3f}  (mando)")
    print(f"   {fit['descanso']*dif_desc:+.3f}  (descanso: {fit['descanso']:.3f} x {dif_desc:+.0f})")
    print(f"   {'-'*28}")
    print(f"   {mu:+.3f}  pontos a favor do mandante")

    # ---------- 6. o sorteio ----------
    res, tot = residuos_historicos(df)
    print(f"\n6. SORTEIO — nuvem empirica de {len(res)} residuos historicos")
    print(f"   desvio dos residuos: {res.std():.2f} pontos")
    print(f"   total de pontos: media {tot.mean():.1f}, desvio {tot.std():.1f}")
    print(f"   a margem de cada temporada simulada e {mu:+.2f} + um residuo sorteado")

    rng = np.random.default_rng(2026)
    N = 200000
    m = mu + rng.choice(res, size=N)
    t = rng.choice(tot, size=N)
    ph = np.maximum(0, np.rint((t + m) / 2))
    pa = np.maximum(0, np.rint((t - m) / 2))
    bruto_emp = float((ph == pa).mean() * 100)
    ph, pa = resolver_empates(ph, pa, m, rng)

    pv_casa = float((ph > pa).mean() * 100)
    pv_fora = float((ph < pa).mean() * 100)
    p_emp = float((ph == pa).mean() * 100)
    print(f"\n7. EMPATES")
    print(f"   arredondar deixaria {bruto_emp:.2f}% de placares iguais")
    print(f"   so {MANTER_EMPATE*100:.1f}% deles sobrevivem; o resto vira "
          f"decisao de 3 pontos")
    print(f"   empate final: {p_emp:.2f}%  (NFL real: 0,24%)")

    print(f"\n8. RESULTADO  ({N:,} sorteios)".replace(",", "."))
    print(f"   M  {casa} vence    {pv_casa:5.1f}%")
    print(f"   V  {fora} vence    {pv_fora:5.1f}%")
    print(f"   E  empate          {p_emp:5.2f}%")
    print(f"   soma               {pv_casa+pv_fora+p_emp:5.1f}%")

    print(f"\n9. O QUE NAO ENTROU")
    print("   EPA         medido, ganho de 0,0075 ponto com t=0,92 -> fora")
    print("   QB titular  medido, -4,37 pontos de efeito, mas so 37% previsivel -> fora")
    print("   spread      so 16 dos 240 jogos futuros tem linha -> impossivel")
    print("   horario, distancia de viagem, fuso, clima -> por decisao")


if __name__ == "__main__":
    main()
