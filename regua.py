"""Caminhada para a frente: o modelo prevê semana a semana, sem ver o futuro.

E o mercado está no mesmo arquivo. Então, diferente do Brasileirão, a régua
existe desde a primeira linha de código: o `spread_line` do nflverse cobre 97%
dos jogos desde 1999 e é praticamente não-enviesado.

Duas medidas, nesta ordem de importância:

  RMSE da margem  — comparável diretamente ao spread, na mesma unidade (pontos)
  log-loss do vencedor — o que vale para simular playoffs

    python regua.py              # usa os hiperparâmetros já medidos
    python regua.py --buscar     # refaz a busca em grade de λ e encolhimento
"""
import sys

import numpy as np
import pandas as pd

from dados import carregar
from modelo import ajustar, margem_esperada, prior_da_temporada

TIMES = None       # preenchido em avaliar()
PRIMEIRA_AVALIADA = 2003   # 2002 não tem temporada anterior para o prior


def avaliar(df, lam, encolhimento, sd=13.2):
    """Devolve, por jogo previsto: erro do modelo, erro do spread, e log-loss."""
    global TIMES
    TIMES = sorted(set(df.home_team) | set(df.away_team))
    linhas = []
    for temporada in sorted(df.season.unique()):
        if temporada < PRIMEIRA_AVALIADA:
            continue
        ant = df[(df.season == temporada - 1) & df.disputado]
        prior = (prior_da_temporada(ant, TIMES, encolhimento, lam)
                 if len(ant) else None)
        atual = df[df.season == temporada]
        for semana in sorted(atual.week.unique()):
            passado = atual[(atual.week < semana) & atual.disputado]
            alvo = atual[(atual.week == semana) & atual.disputado]
            if not len(alvo):
                continue
            fit = ajustar(passado, TIMES, prior=prior, lam=lam)
            prev = margem_esperada(fit, alvo)
            linhas.append(pd.DataFrame({
                "season": temporada, "week": semana,
                "real": alvo.margem.to_numpy(float),
                "modelo": prev,
                "spread": alvo.spread_line.to_numpy(float),
            }))
    r = pd.concat(linhas, ignore_index=True)
    r["erro_modelo"] = r.real - r.modelo
    r["erro_spread"] = r.real - r.spread
    return r


def resumo(r, sd=13.2):
    from scipy.stats import norm
    # Baseline sem informacao nenhuma: todo jogo termina na vantagem de casa
    # media. E o "chutar 1/3" deste esporte, e sem ele os outros dois numeros
    # nao tem escala nenhuma.
    r = r.copy()
    r["so_casa"] = 2.1
    out = {}
    for nome, col in (("modelo", "modelo"), ("spread", "spread"),
                      ("so vantagem de casa", "so_casa")):
        ok = r[col].notna() & r.real.notna()
        e = (r.real - r[col])[ok]
        # probabilidade de vitoria do mandante; empate (0,2%) vai metade a metade
        pv = norm.cdf(r[col][ok] / sd)
        y = (r.real[ok] > 0).astype(float) + 0.5 * (r.real[ok] == 0)
        ll = -(y * np.log(np.clip(pv, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pv, 1e-9, 1)))
        out[nome] = {"n": int(ok.sum()), "RMSE": e.pow(2).mean() ** .5,
                     "MAE": e.abs().mean(), "vies": e.mean(),
                     "log_loss": ll.mean(),
                     "acerto_%": ((r[col][ok] > 0) == (r.real[ok] > 0)).mean() * 100}
    return pd.DataFrame(out).T


def buscar(df):
    print("busca em grade (RMSE fora da amostra, 2003+)\n")
    print(f"{'λ':>6} {'encolh.':>8} {'RMSE':>8} {'log-loss':>9}")
    melhor, res = None, []
    for lam in (4, 8, 12, 20, 30, 50):
        for enc in (0.0, 0.3, 0.5, 0.7, 0.9):
            r = avaliar(df, lam, enc)
            s = resumo(r)
            v = (lam, enc, s.loc["modelo", "RMSE"], s.loc["modelo", "log_loss"])
            res.append(v)
            print(f"{lam:6.0f} {enc:8.2f} {v[2]:8.4f} {v[3]:9.5f}")
            if melhor is None or v[2] < melhor[2]:
                melhor = v
    print(f"\nmelhor: λ={melhor[0]}, encolhimento={melhor[1]} "
          f"(RMSE {melhor[2]:.4f})")
    pd.DataFrame(res, columns=["lam", "encolhimento", "RMSE", "log_loss"]).to_csv(
        "out/busca_hiperparametros.csv", index=False)
    return melhor[0], melhor[1]


if __name__ == "__main__":
    df = carregar()
    lam, enc = (12.0, 0.6)
    if "--buscar" in sys.argv:
        lam, enc = buscar(df)
    r = avaliar(df, lam, enc)
    s = resumo(r)
    print(f"\ncaminhada para a frente, {len(r)} jogos previstos "
          f"(λ={lam}, encolhimento={enc})\n")
    print(s.round(4).to_string())
    m, e, b = s.loc["modelo"], s.loc["spread"], s.loc["so vantagem de casa"]
    print("\n  --- onde o modelo esta na regua (RMSE, em pontos) ---")
    print(f"    so vantagem de casa   {b.RMSE:.3f}")
    print(f"    ESTE MODELO           {m.RMSE:.3f}")
    print(f"    mercado (spread)      {e.RMSE:.3f}")
    frac = (b.RMSE - m.RMSE) / (b.RMSE - e.RMSE) * 100
    print(f"\n  do caminho entre o baseline e o mercado, o modelo andou {frac:.0f}%")
    print(f"  falta para o mercado: {m.RMSE - e.RMSE:+.3f} ponto de RMSE")
    fl = (b.log_loss - m.log_loss) / (b.log_loss - e.log_loss) * 100
    print(f"  em log-loss do vencedor, andou {fl:.0f}%")
    r.to_csv("out/regua.csv", index=False)
    print("\ngravado out/regua.csv")
