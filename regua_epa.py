"""EPA melhora a previsão, ou só parece melhorar?

A afirmação corrente é que EPA estabiliza mais rápido que o placar. Ela é
plausível — 130 jogadas contra um número — mas plausível não é medido.

O teste é direto. O rating é ajustado sobre uma resposta misturada:

    y = α · margem_real  +  (1 − α) · margem_de_EPA

α = 1 é o modelo de hoje, que só vê o placar. α = 0 ignora o placar e olha só
EPA. O que se mede é sempre a previsão da margem REAL futura — o alvo não muda,
só a evidência usada para chegar nele.

Se a afirmação valer, o mínimo cai em algum α < 1. Se não valer, o mínimo fica
em 1 e o EPA sai do projeto sem cerimônia.

    python regua_epa.py
"""
import numpy as np
import pandas as pd

from dados import carregar
from epa import carregar as carregar_epa, margem_epa
from modelo import ajustar, margem_esperada
from regua import resumo

LAM, ENC = 8.0, 0.7
ALFAS = (1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0)


def prior_misturado(jogos, times, alfa, encolhimento=ENC, lam=LAM):
    if not len(jogos):
        return None
    j = jogos.copy()
    j["margem"] = _resposta(j, alfa)
    fit = ajustar(j, times, prior=None, lam=lam)
    return fit["rating"] * encolhimento


def _resposta(j, alfa):
    """Mistura placar e EPA. Onde faltar EPA, cai no placar — nunca em zero."""
    real = j.margem.to_numpy(float)
    epa = j.margem_epa.to_numpy(float)
    y = np.where(np.isfinite(epa), alfa * real + (1 - alfa) * epa, real)
    return y


def avaliar(df, alfa):
    times = sorted(set(df.home_team) | set(df.away_team))
    linhas = []
    for temporada in sorted(df.season.unique()):
        ant = df[(df.season == temporada - 1) & df.disputado]
        if not len(ant):
            continue
        prior = prior_misturado(ant, times, alfa)
        atual = df[df.season == temporada]
        for semana in sorted(atual.week.unique()):
            passado = atual[(atual.week < semana) & atual.disputado].copy()
            alvo = atual[(atual.week == semana) & atual.disputado]
            if not len(alvo):
                continue
            if len(passado):
                passado["margem"] = _resposta(passado, alfa)
            fit = ajustar(passado, times, prior=prior, lam=LAM)
            linhas.append(pd.DataFrame({
                "season": temporada, "week": semana,
                "real": alvo.margem.to_numpy(float),         # o alvo NUNCA muda
                "modelo": margem_esperada(fit, alvo),
                "spread": alvo.spread_line.to_numpy(float),
            }))
    return pd.concat(linhas, ignore_index=True)


if __name__ == "__main__":
    jogos = carregar()
    e = carregar_epa()
    df = margem_epa(e, jogos)
    com = df.disputado & df.margem_epa.notna()
    print(f"{int(com.sum())} jogos disputados com EPA nas duas pontas, "
          f"{int(df.disputado.sum())} disputados no total")
    print(f"temporadas com EPA: {int(df.loc[com, 'season'].min())}–"
          f"{int(df.loc[com, 'season'].max())}\n")

    print(f"{'α (peso do placar)':>20} {'RMSE':>8} {'log-loss':>9} {'acerto %':>9}")
    res = []
    for alfa in ALFAS:
        r = avaliar(df, alfa)
        s = resumo(r)
        m = s.loc["modelo"]
        res.append({"alfa": alfa, "RMSE": m.RMSE, "log_loss": m.log_loss,
                    "acerto": m["acerto_%"]})
        marca = "  <- só placar" if alfa == 1 else ("  <- só EPA" if alfa == 0 else "")
        print(f"{alfa:20.2f} {m.RMSE:8.4f} {m.log_loss:9.5f} {m['acerto_%']:9.2f}{marca}")

    t = pd.DataFrame(res)
    b = t.loc[t.RMSE.idxmin()]
    so_placar = t[t.alfa == 1].iloc[0]
    print(f"\nmelhor α = {b.alfa:.2f}  (RMSE {b.RMSE:.4f})")
    print(f"ganho sobre só placar: {so_placar.RMSE - b.RMSE:+.4f} ponto de RMSE")
    spread = resumo(avaliar(df, 1.0)).loc["spread", "RMSE"]
    print(f"\nmercado (spread) nos mesmos jogos: {spread:.4f}")
    print(f"  so placar fica a {so_placar.RMSE - spread:+.4f} do mercado")
    print(f"  melhor α  fica a {b.RMSE - spread:+.4f} do mercado")
    t.to_csv("out/regua_epa.csv", index=False)
    print("\ngravado out/regua_epa.csv")
