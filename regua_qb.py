"""Corrigir o passado pelo QB melhora a previsão do futuro?

O efeito do titular está medido: −4,37 pontos (t = −13,07). Mas efeito grande
não é o mesmo que previsão melhor, e a diferença aqui é sutil.

O QB entra **corrigindo o passado**: um time que perdeu dois jogos com o
reserva não deve carregar esse buraco no rating. Para a frente a coluna é
ZERO — assume-se que o titular joga —, porque o QB de um jogo futuro não vem
da fonte: medido, 16 de 240 jogos de 2026 têm o nome preenchido.

Três cenários, e a diferença entre o segundo e o terceiro é o que separa
"limpar o passado" de "saber o futuro":

  sem QB          o modelo de hoje
  corrige passado usa o QB nos jogos já disputados, assume titular adiante
  sabe o futuro   usa o QB também no jogo previsto — IRREAL, é o teto

    python regua_qb.py
"""
import numpy as np
import pandas as pd

from dados import carregar
from modelo import ajustar, margem_esperada, prior_da_temporada, COL_QB
from qb import marcar_titular, marcar_titular_retrospectivo
from regua import resumo

LAM, ENC = 8.0, 0.7


def preparar(df, retro=True):
    """Coluna qb_dif = (reserva na casa) − (reserva fora), em {−1, 0, 1}."""
    c, f = ("tit_retro_casa", "tit_retro_fora") if retro else ("titular_casa", "titular_fora")
    df = df.copy()
    df[COL_QB] = (~df[c]).astype(float) - (~df[f]).astype(float)
    return df


def situacao_persistente(passado, alvo):
    """Assume que a situacao de QB do jogo mais recente de cada time CONTINUA.

    E a premissa honesta: lesao de quarterback nao evapora de uma semana para
    a outra. Assumir que o titular voltou — que e o que "corrige passado" faz —
    infla justamente o time que esta desfalcado.
    """
    ultimo = {}
    for r in passado.itertuples():
        ultimo[r.home_team] = not r.tit_retro_casa
        ultimo[r.away_team] = not r.tit_retro_fora
    casa = np.array([ultimo.get(t, False) for t in alvo.home_team], float)
    fora = np.array([ultimo.get(t, False) for t in alvo.away_team], float)
    return casa - fora


def avaliar(df, modo):
    """modo: 'sem', 'passado', 'persiste' ou 'futuro'."""
    com_qb = modo != "sem"
    times = sorted(set(df.home_team) | set(df.away_team))
    partes = []
    for temporada in sorted(df.season.unique()):
        ant = df[(df.season == temporada - 1) & df.disputado]
        if not len(ant):
            continue
        fit_ant = ajustar(ant, times, prior=None, lam=LAM, com_qb=com_qb)
        prior = fit_ant["rating"] * ENC
        atual = df[df.season == temporada]
        for semana in sorted(atual.week.unique()):
            passado = atual[(atual.week < semana) & atual.disputado]
            alvo = atual[(atual.week == semana) & atual.disputado].copy()
            if not len(alvo):
                continue
            fit = ajustar(passado, times, prior=prior, lam=LAM, com_qb=com_qb)
            if modo == "passado":
                alvo[COL_QB] = 0.0                       # assume titular de volta
            elif modo == "persiste":
                alvo[COL_QB] = situacao_persistente(passado, alvo)
            partes.append(pd.DataFrame({
                "season": temporada, "week": semana,
                "real": alvo.margem.to_numpy(float),
                "modelo": margem_esperada(fit, alvo),
                "spread": alvo.spread_line.to_numpy(float),
            }))
    return pd.concat(partes, ignore_index=True)


if __name__ == "__main__":
    base = marcar_titular_retrospectivo(marcar_titular(carregar()))
    df = preparar(base, retro=True)

    res = {}
    print(f"{'cenário':>18} {'RMSE':>9} {'log-loss':>10} {'acerto %':>9}")
    for modo, rot in (("sem", "sem QB"), ("passado", "corrige passado"),
                      ("persiste", "situação persiste"),
                      ("futuro", "sabe o futuro")):
        r = avaliar(df, modo)
        s = resumo(r)
        m = s.loc["modelo"]
        res[modo] = r
        print(f"{rot:>18} {m.RMSE:9.4f} {m.log_loss:10.5f} {m['acerto_%']:9.2f}")
    spread = resumo(res["sem"]).loc["spread"]
    print(f"{'mercado (spread)':>18} {spread.RMSE:9.4f} {spread.log_loss:10.5f} "
          f"{spread['acerto_%']:9.2f}")

    for alvo in ("passado", "persiste", "futuro"):
        a, b = res["sem"], res[alvo]
        e1 = (a.real - a.modelo) ** 2
        e2 = (b.real - b.modelo) ** 2
        d = e1 - e2
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f"  {alvo:16s} contra sem QB: "
              f"{np.sqrt(e1.mean()) - np.sqrt(e2.mean()):+.4f} ponto, t = {t:.2f}")

    a, b = res["sem"], res["passado"]
    e1 = (a.real - a.modelo) ** 2
    e2 = (b.real - b.modelo) ** 2
    d = e1 - e2
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\ncorrigir o passado contra não corrigir: "
          f"{np.sqrt(e1.mean()) - np.sqrt(e2.mean()):+.4f} ponto de RMSE, t = {t:.2f}")

    c = res["futuro"]
    e3 = (c.real - c.modelo) ** 2
    print(f"saber o futuro (irreal) daria mais: "
          f"{np.sqrt(e2.mean()) - np.sqrt(e3.mean()):+.4f} ponto")
    pd.DataFrame({k: [np.sqrt(((v.real - v.modelo) ** 2).mean())]
                  for k, v in res.items()}).to_csv("out/regua_qb.csv", index=False)
    print("\ngravado out/regua_qb.csv")
