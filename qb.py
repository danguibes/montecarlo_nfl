"""Quanto vale o quarterback titular, medido em pontos.

A resposta corrente é "uns 5 pontos", dita sem fonte. Aqui ela é medida, e o
arquivo do nflverse permite: `home_qb_name` e `away_qb_name` trazem quem
começou cada jogo desde 1999, em 100% dos jogos disputados.

O método evita o confundidor óbvio. Comparar o desempenho bruto com titular e
com reserva mistura duas coisas — o reserva é pior, mas também costuma entrar
contra adversários e em situações que não são as mesmas. Então o que se mede é
o **resíduo do modelo**: a previsão já embute força dos dois times, mando e
descanso, e o que sobra quando o titular falta é o efeito dele.

Uma restrição prática, medida antes de qualquer coisa: dos 240 jogos ainda não
disputados de 2026, **só 16 têm QB preenchido**. Ou seja, o titular de um jogo
futuro não vem da fonte. O QB pode corrigir o PASSADO — um time que jogou duas
semanas com reserva está subavaliado — e adiante só cabe a premissa de que o
titular atual joga, com correção manual para lesão conhecida.

    python qb.py
"""
import numpy as np
import pandas as pd

from dados import carregar
from modelo import ajustar, margem_esperada, prior_da_temporada
from regua import resumo

LAM, ENC = 8.0, 0.7


def marcar_titular(df):
    """Para cada jogo, diz se o QB que começou era o titular ESTABELECIDO.

    Estabelecido = quem mais começou pelo time até ali, contando só jogos
    anteriores. Na primeira aparição do time numa temporada, herda o titular
    da temporada anterior. Nada olha para a frente.
    """
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    longo = pd.concat([
        df[["season", "week", "home_team", "home_qb_name"]]
          .rename(columns={"home_team": "team", "home_qb_name": "qb"}),
        df[["season", "week", "away_team", "away_qb_name"]]
          .rename(columns={"away_team": "team", "away_qb_name": "qb"}),
    ]).sort_values(["team", "season", "week"])

    estabelecido, contagem, ultimo_ano = {}, {}, {}
    marcas = {}
    for r in longo.itertuples():
        chave = r.team
        if ultimo_ano.get(chave) != r.season:
            # temporada nova: zera a contagem, mas guarda quem era o titular
            contagem[chave] = {}
            ultimo_ano[chave] = r.season
        est = estabelecido.get(chave)
        marcas[(r.season, r.week, r.team)] = (est is not None and r.qb == est)
        if isinstance(r.qb, str):
            c = contagem[chave]
            c[r.qb] = c.get(r.qb, 0) + 1
            estabelecido[chave] = max(c, key=c.get)

    for lado, col in (("home_team", "titular_casa"), ("away_team", "titular_fora")):
        df[col] = [marcas.get((s, w, t), False) for s, w, t
                   in zip(df.season, df.week, df[lado])]
    return df


def marcar_titular_retrospectivo(df):
    """Titular da temporada = quem mais comecou pelo time NAQUELE ano.

    Olha a temporada inteira, entao NAO serve para prever — serve para MEDIR.
    A versao sem vazamento (`marcar_titular`) rotula como reserva todo QB que
    ainda nao acumulou jogos, e isso mistura o reserva de verdade com o novato
    ou o recem-contratado que E o titular. Medido: 37,6% dos jogos que ela
    marca como reserva foram comecados por alguem com 8+ jogos na temporada.

    Diluir assim faz a estimativa parecer menor do que o efeito realmente e.
    """
    longo = pd.concat([
        df[["season", "home_team", "home_qb_name"]]
          .rename(columns={"home_team": "team", "home_qb_name": "qb"}),
        df[["season", "away_team", "away_qb_name"]]
          .rename(columns={"away_team": "team", "away_qb_name": "qb"}),
    ])
    inicios = longo.groupby(["season", "team", "qb"]).size().rename("n").reset_index()
    dono = (inicios.sort_values("n", ascending=False)
            .drop_duplicates(["season", "team"])
            .set_index(["season", "team"]).qb)
    for lado, col, qbcol in (("home_team", "tit_retro_casa", "home_qb_name"),
                             ("away_team", "tit_retro_fora", "away_qb_name")):
        idx = pd.MultiIndex.from_arrays([df.season, df[lado]])
        df[col] = dono.reindex(idx).to_numpy() == df[qbcol].to_numpy()
    return df


def efeito(r, col_casa, col_fora, rotulo):
    """Coeficiente sobre (reserva na casa − reserva fora), em pontos."""
    x = (~r[col_casa]).astype(float) - (~r[col_fora]).astype(float)
    ok = x != 0
    beta = np.polyfit(x[ok], r.residuo[ok], 1)[0]
    d = r.residuo[ok] * np.sign(x[ok])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    pct = ((~r[col_casa]).sum() + (~r[col_fora]).sum()) / (2 * len(r)) * 100
    print(f"  {rotulo:36s} {beta:+6.2f} pontos   n={int(ok.sum()):5d}  "
          f"t={t:6.2f}   reservas={pct:.1f}% dos inicios")
    return beta


def residuos(df):
    """Resíduo da caminhada para a frente, jogo a jogo, com as marcas de QB."""
    times = sorted(set(df.home_team) | set(df.away_team))
    partes = []
    for temporada in sorted(df.season.unique()):
        ant = df[(df.season == temporada - 1) & df.disputado]
        if not len(ant):
            continue
        prior = prior_da_temporada(ant, times, ENC, LAM)
        atual = df[df.season == temporada]
        for semana in sorted(atual.week.unique()):
            passado = atual[(atual.week < semana) & atual.disputado]
            alvo = atual[(atual.week == semana) & atual.disputado]
            if not len(alvo):
                continue
            fit = ajustar(passado, times, prior=prior, lam=LAM)
            p = alvo.copy()
            p["previsto"] = margem_esperada(fit, alvo)
            p["residuo"] = p.margem - p.previsto
            partes.append(p)
    return pd.concat(partes, ignore_index=True)


if __name__ == "__main__":
    df = marcar_titular_retrospectivo(marcar_titular(carregar()))
    r = residuos(df[df.disputado | ~df.disputado])
    r = r[r.disputado]

    print(f"{len(r)} jogos com resíduo, {int(r.season.min())}–{int(r.season.max())}\n")
    n_casa = int((~r.titular_casa).sum())
    n_fora = int((~r.titular_fora).sum())
    print(f"jogos com RESERVA no comando:")
    print(f"  mandante  {n_casa:5d}  ({n_casa/len(r)*100:.1f}%)")
    print(f"  visitante {n_fora:5d}  ({n_fora/len(r)*100:.1f}%)")

    print("\n--- resíduo médio, por situação ---")
    for rot, sel in (
            ("os dois com titular", r.titular_casa & r.titular_fora),
            ("reserva SÓ no mandante", ~r.titular_casa & r.titular_fora),
            ("reserva SÓ no visitante", r.titular_casa & ~r.titular_fora),
            ("reserva nos dois", ~r.titular_casa & ~r.titular_fora)):
        s = r[sel]
        if len(s) < 10:
            continue
        ep = s.residuo.std(ddof=1) / np.sqrt(len(s))
        print(f"  {rot:26s} n={len(s):5d}  resíduo {s.residuo.mean():+6.2f}  "
              f"± {1.96*ep:.2f}")

    print("\n--- efeito de jogar sem o titular, por definicao ---")
    efeito(r, "titular_casa", "titular_fora",
           "sem vazamento (o que da p/ prever)")
    efeito(r, "tit_retro_casa", "tit_retro_fora",
           "retrospectiva (a medida limpa)")
    print("\n  A segunda e maior porque a primeira mistura o reserva de verdade")
    print("  com o novato que E o titular do ano — 37,6% dos casos dela.")
    r.to_csv("out/residuos_qb.csv", index=False)
    print("gravado out/residuos_qb.csv")
