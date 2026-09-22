"""Monte Carlo da temporada: simula os jogos que faltam e semeia os playoffs.

O placar de cada jogo vem em duas partes independentes:

  margem ~ empírica, centrada na margem esperada do modelo
  total  ~ empírica, centrada no total médio da liga

e o placar sai de (total ± margem)/2. Margem e total são praticamente
independentes em futebol americano, e separá-los é o que permite usar a
distribuição empírica da margem — com os picos em 3 e 7 que nenhuma família
paramétrica reproduz.

Os pontos importam pouco para a classificação (só entram a partir do 7º
critério de desempate), mas importam para nada ser inventado: o motor de
playoffs recebe placares, como receberia da vida real.

    python simular.py --n 10000
"""
import argparse
import json

import numpy as np
import pandas as pd

from dados import carregar
from modelo import ajustar, margem_esperada, prior_da_temporada
from playoffs import (CONF_DE, DIV_DE, DIVISOES, TIMES, Temporada,
                      n_wildcards, semear)

LAM, ENC = 8.0, 0.7


def residuos_historicos(df, lam=LAM, enc=ENC):
    """Erros do modelo em temporadas passadas — a fonte do sorteio.

    Usar a distribuição empírica em vez de uma Normal preserva os picos de
    margem em 3 e 7. A Normal erra exatamente onde o futebol americano é mais
    concentrado.
    """
    times = sorted(set(df.home_team) | set(df.away_team))
    res, tot = [], []
    for temporada in sorted(df.season.unique()):
        ant = df[(df.season == temporada - 1) & df.disputado]
        if not len(ant):
            continue
        prior = prior_da_temporada(ant, times, enc, lam)
        atual = df[df.season == temporada]
        for semana in sorted(atual.week.unique()):
            passado = atual[(atual.week < semana) & atual.disputado]
            alvo = atual[(atual.week == semana) & atual.disputado]
            if not len(alvo):
                continue
            fit = ajustar(passado, times, prior=prior, lam=lam)
            res.append(alvo.margem.to_numpy(float) - margem_esperada(fit, alvo))
            tot.append(alvo.total_pts.to_numpy(float))
    return np.concatenate(res), np.concatenate(tot)


def ajustar_atual(df, temporada, lam=LAM, enc=ENC):
    times = sorted(set(df.home_team) | set(df.away_team))
    ant = df[(df.season == temporada - 1) & df.disputado]
    prior = prior_da_temporada(ant, times, enc, lam) if len(ant) else None
    passado = df[(df.season == temporada) & df.disputado]
    return ajustar(passado, times, prior=prior, lam=lam)


def simular(df, temporada, n=10000, seed=7):
    rng = np.random.default_rng(seed)
    hist_res, hist_tot = residuos_historicos(df)
    fit = ajustar_atual(df, temporada)

    atual = df[df.season == temporada]
    jogados = atual[atual.disputado]
    faltam = atual[~atual.disputado].reset_index(drop=True)
    mu = margem_esperada(fit, faltam)
    wc = n_wildcards(temporada)

    # base: o que já aconteceu
    base_casa = jogados.home_team.to_numpy()
    base_fora = jogados.away_team.to_numpy()
    base_ph = jogados.home_score.to_numpy(float)
    base_pa = jogados.away_score.to_numpy(float)
    f_casa = faltam.home_team.to_numpy()
    f_fora = faltam.away_team.to_numpy()

    # Vitorias REAIS ate agora. Ler isto do T da ultima simulacao — que foi o
    # primeiro erro aqui — mostra o registro de uma temporada inventada como se
    # fosse o de hoje, e ninguem estranha porque o numero parece plausivel.
    real = Temporada(base_casa, base_fora, base_ph, base_pa)
    v_hoje = {t: (real.v[t], real.d[t], real.e[t]) for t in TIMES}

    idx = {t: i for i, t in enumerate(TIMES)}
    cont_seed = np.zeros((len(TIMES), 8), int)     # 0 = fora; 1..7 = cabeça
    vitorias = np.zeros((len(TIMES), n), np.int16)

    for s in range(n):
        margem = mu + rng.choice(hist_res, size=len(faltam))
        total = rng.choice(hist_tot, size=len(faltam))
        ph = np.rint((total + margem) / 2).astype(int)
        pa = np.rint((total - margem) / 2).astype(int)
        ph = np.maximum(ph, 0)
        pa = np.maximum(pa, 0)

        T = Temporada(np.concatenate([base_casa, f_casa]),
                      np.concatenate([base_fora, f_fora]),
                      np.concatenate([base_ph, ph]),
                      np.concatenate([base_pa, pa]))
        for t in TIMES:
            vitorias[idx[t], s] = T.v[t]
        sorteio = {t: rng.random() for t in TIMES}
        for conf in ("AFC", "NFC"):
            for pos, t in enumerate(semear(T, conf, sorteio, wc), 1):
                cont_seed[idx[t], pos] += 1
        if (s + 1) % 500 == 0:
            print(f"  {s+1}/{n}", end="\r", flush=True)
    print(" " * 30, end="\r")

    linhas = []
    for t in TIMES:
        i = idx[t]
        c = cont_seed[i]
        v = vitorias[i]
        linhas.append({
            "time": t, "conf": CONF_DE[t], "divisao": DIV_DE[t],
            "v": v_hoje[t][0], "d": v_hoje[t][1], "e": v_hoje[t][2],
            "vitorias_medias": float(v.mean()),
            "v_p10": float(np.percentile(v, 10)), "v_p90": float(np.percentile(v, 90)),
            "playoffs_%": c[1:].sum() / n * 100,
            "divisao_%": c[1:5].sum() / n * 100,
            "cabeca1_%": c[1] / n * 100,
            **{f"seed{p}_%": c[p] / n * 100 for p in range(1, 8)},
        })
    return pd.DataFrame(linhas), {"n": n, "faltam": len(faltam),
                                  "jogados": len(jogados),
                                  "casa": fit["casa"], "descanso": fit["descanso"],
                                  "rating": fit["rating"].to_dict()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--temporada", type=int, default=None)
    args = ap.parse_args()

    df = carregar()
    temporada = args.temporada or int(df.season.max())
    print(f"simulando {args.n} temporadas de {temporada}…")
    t, meta = simular(df, temporada, args.n)
    t = t.sort_values(["conf", "playoffs_%"], ascending=[True, False])
    print(f"\n{meta['jogados']} jogos disputados, {meta['faltam']} restantes")
    print(f"vantagem de casa ajustada: {meta['casa']:+.2f} pontos\n")
    for conf in ("AFC", "NFC"):
        print(f"=== {conf} ===")
        c = t[t.conf == conf]
        print(c[["time", "divisao", "v", "d", "vitorias_medias",
                 "playoffs_%", "divisao_%", "cabeca1_%"]]
              .round(1).to_string(index=False))
        print()
    t.to_csv("out/simulacao.csv", index=False)
    json.dump(meta, open("out/simulacao_meta.json", "w"), indent=1)
    print("gravado out/simulacao.csv")
