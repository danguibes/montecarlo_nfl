"""Gera web/index.html: página única, autocontida, com o resultado da simulação.

A página carrega com os números do Python — 20 mil temporadas, autoritativo — e
o navegador refaz a simulação quando alguém fixa um resultado.

Isso exigiu portar o motor de desempate para JavaScript, o que eu tinha evitado
por medo de erro silencioso: uma cascata errada não quebra nada, só semeia
diferente. A defesa é submeter o porte ao MESMO teste de aceitação do Python,
dentro da própria página — as 24 temporadas históricas, com o resultado na tela
e não num log. Os dois motores concordam em 300 de 300 vagas.

    python exportar.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

import numpy as np

from dados import carregar
from modelo import margem_esperada
from playoffs import CONF_DE, DIVISOES, TIMES, n_wildcards
from simular import ajustar_atual, residuos_historicos

NOMES = {
    "ARI": "Arizona", "ATL": "Atlanta", "BAL": "Baltimore", "BUF": "Buffalo",
    "CAR": "Carolina", "CHI": "Chicago", "CIN": "Cincinnati", "CLE": "Cleveland",
    "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay",
    "HOU": "Houston", "IND": "Indianapolis", "JAX": "Jacksonville",
    "KC": "Kansas City", "LAC": "LA Chargers", "LA": "LA Rams", "LV": "Las Vegas",
    "MIA": "Miami", "MIN": "Minnesota", "NE": "New England", "NO": "New Orleans",
    "NYG": "NY Giants", "NYJ": "NY Jets", "PHI": "Philadelphia",
    "PIT": "Pittsburgh", "SEA": "Seattle", "SF": "San Francisco",
    "TB": "Tampa Bay", "TEN": "Tennessee", "WAS": "Washington",
}

def regua_medida():
    """Le a regua do arquivo que regua.py grava, em vez de repetir numeros.

    A versao anterior tinha os tres valores chumbados aqui. Duas horas depois,
    um conserto no peso do prior mudou o modelo de 13,553 para 13,528 e o
    rodape continuou anunciando 76% quando ja eram 78%. Numero copiado a mao
    envelhece em silencio e ninguem confere.
    """
    if not os.path.exists("out/regua.csv"):
        sys.exit("out/regua.csv ausente: rode regua.py antes de exportar")
    r = pd.read_csv("out/regua.csv")
    def rmse(col):
        e = (r.real - r[col]).dropna()
        return float((e ** 2).mean() ** 0.5)
    base = float(((r.real - 2.1) ** 2).mean() ** 0.5)
    return {"baseline": round(base, 3), "modelo": round(rmse("modelo"), 3),
            "mercado": round(rmse("spread"), 3), "n": int(len(r))}


def discordancia_medida():
    """Quando modelo e mercado discordam, quem erra menos — medido, nao dito.

    A secao de spreads mostra divergencia, e divergencia sem legenda vira
    recomendacao na cabeca de quem le. A legenda tem que sair do mesmo arquivo
    da regua, pelo mesmo motivo que a regua saiu: numero copiado a mao
    envelhece em silencio.
    """
    r = pd.read_csv("out/regua.csv").dropna(subset=["real", "modelo", "spread"])
    d = (r.modelo - r.spread).abs()
    g = r[d >= 3]
    lado = ((g.real - g.spread) * (g.modelo - g.spread) > 0)
    def rmse(sub, col):
        return float(((sub.real - sub[col]) ** 2).mean() ** 0.5)
    return {
        "n": int(len(g)),
        "corte": 3,
        "rmse_modelo": round(rmse(g, "modelo"), 2),
        "rmse_mercado": round(rmse(g, "spread"), 2),
        "lado": round(float(lado.mean()) * 100, 1),
    }


def main():
    if not os.path.exists("out/simulacao.csv"):
        sys.exit("rode simular.py antes de exportar")
    sim = pd.read_csv("out/simulacao.csv")
    meta = json.load(open("out/simulacao_meta.json", encoding="utf-8"))
    jogos = carregar()
    temporada = int(jogos.season.max())
    disputados = jogos[(jogos.season == temporada) & jogos.disputado]

    faltam_nomes = [t for t in sim.time if t not in NOMES]
    if faltam_nomes:
        sys.exit(f"time sem nome legivel: {faltam_nomes}")

    # Acesso por nome de coluna, e não por itertuples: as colunas terminam em
    # "%", que o itertuples renomeia para _10, _11 e afins. Um remendo ali
    # sobrevive até alguém acrescentar uma coluna no meio, e então a página
    # passa a mostrar a coluna errada sem erro nenhum.
    times = []
    for _, r in sim.iterrows():
        times.append({
            "sigla": r["time"], "nome": NOMES[r["time"]], "conf": r["conf"],
            "divisao": r["divisao"],
            "v": int(r["v"]), "d": int(r["d"]), "e": int(r["e"]),
            "vmed": round(float(r["vitorias_medias"]), 1),
            "vp10": int(r["v_p10"]), "vp90": int(r["v_p90"]),
            "playoffs": round(float(r["playoffs_%"]), 2),
            "divisao_pct": round(float(r["divisao_%"]), 2),
            "cabeca1": round(float(r["cabeca1_%"]), 2),
            "seeds": [round(float(r[f"seed{p}_%"]), 2) for p in range(1, 8)],
            "rating": round(float(meta["rating"][r["time"]]), 2),
        })

    payload = {
        "temporada": temporada,
        "n": int(meta["n"]),
        "jogados": int(meta["jogados"]),
        "faltam": int(meta["faltam"]),
        "casa": round(float(meta["casa"]), 2),
        "descanso": round(float(meta["descanso"]), 3),
        "rating": {t: round(float(v), 3) for t, v in meta["rating"].items()},
        "wildcards": n_wildcards(temporada),
        "regua": regua_medida(),
        "discordancia": discordancia_medida(),
        "ultimoJogo": str(disputados.gameday.max())[:10],
        "geradoEm": datetime.now(timezone.utc).astimezone(
            timezone(timedelta(hours=-3))).strftime("%d/%m %H:%M"),
        "times": times,
    }

    payload.update(motor(jogos, temporada))

    tpl = open("web/template.html", encoding="utf-8").read()
    tpl = tpl.replace("/*__MOTOR__*/",
                      open("web/motor.js.part", encoding="utf-8").read())
    if "/*__MOTOR__*/" in tpl:
        sys.exit("motor nao injetado na pagina")
    html = tpl.replace("/*__DADOS__*/null",
                       json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if "/*__DADOS__*/null" in html:
        sys.exit("placeholder nao substituido — a pagina sairia vazia")
    os.makedirs("web", exist_ok=True)
    open("web/index.html", "w", encoding="utf-8").write(html)
    print(f"web/index.html  {len(html)/1024:.0f} KB  "
          f"({len(times)} times, {payload['faltam']} jogos restantes)")


def motor(jogos, temporada):
    """Tudo que o navegador precisa para refazer a simulacao com condicional.

    Inclui o conjunto de VALIDACAO: temporadas historicas completas e os
    classificados reais de cada uma. O motor portado para JavaScript passa pelo
    mesmo teste de aceitacao do Python, dentro da pagina — se ele nao reproduzir
    as 336 vagas, o numero aparece na tela em vez de ficar escondido.
    """
    idx = {t: i for i, t in enumerate(TIMES)}
    atual = jogos[jogos.season == temporada]
    jogados = atual[atual.disputado]
    faltam = atual[~atual.disputado].reset_index(drop=True)
    fit = ajustar_atual(jogos, temporada)
    mu = margem_esperada(fit, faltam)

    # A nuvem INTEIRA, nao uma subamostra. A versao anterior mandava 4.000 dos
    # 5.999 residuos, e isso fazia o navegador sortear de uma nuvem levemente
    # diferente da do Python: 69,16% contra 69,40% no mesmo jogo, com desvio de
    # 13,566 contra 13,53. Diferenca pequena, mas SISTEMATICA — nao sai com
    # mais sorteios —, e uma pagina que exibe numero diferente do Python tira a
    # graca de ter as duas implementacoes conferindo uma a outra.
    #
    # Custa uns 12 KB.
    res, tot = residuos_historicos(jogos)

    # A pagina mostra a margem esperada DECOMPOSTA — forca, mando, descanso —
    # e a soma das parcelas tem que ser exatamente o mu que a simulacao usa.
    # Se algum dia o modelo ganhar um termo e a decomposicao nao souber dele,
    # o rodape do card explicaria um numero que nao e o da barra. Aqui isso
    # deixa de ser silencioso.
    parcelas = (fit["rating"].reindex(faltam.home_team).to_numpy()
                - fit["rating"].reindex(faltam.away_team).to_numpy()
                + fit["casa"] * (1 - faltam.neutro.to_numpy())
                + fit["descanso"] * faltam.dif_descanso.fillna(0).to_numpy())
    pior = float(np.abs(parcelas - mu).max()) if len(mu) else 0.0
    if pior > 1e-9:
        sys.exit(f"decomposicao nao fecha com a margem esperada: erro {pior:.2e}")

    todos = carregar(tipo=None)
    pos = todos[todos.game_type != "REG"]
    hist = []
    for ano, g in jogos[jogos.season < temporada].groupby("season"):
        g = g[g.disputado]
        p = pos[pos.season == ano]
        if not len(p) or len(g) < 200:
            continue
        reais = set(p.home_team) | set(p.away_team)
        hist.append({
            "ano": int(ano), "wc": n_wildcards(int(ano)),
            "j": [[idx[h], idx[a], int(ph), int(pa)] for h, a, ph, pa in
                  zip(g.home_team, g.away_team, g.home_score, g.away_score)],
            "reais": {c: sorted(idx[t] for t in reais if CONF_DE.get(t) == c)
                      for c in ("AFC", "NFC")},
        })

    return {
        "siglas": TIMES,
        "divisoes": {d: [idx[t] for t in ts] for d, ts in DIVISOES.items()},
        "base": [[idx[h], idx[a], int(ph), int(pa)] for h, a, ph, pa in
                 zip(jogados.home_team, jogados.away_team,
                     jogados.home_score, jogados.away_score)],
        "faltam_jogos": [
            {"h": idx[h], "a": idx[a], "mu": round(float(m), 3),
             "sem": int(w), "data": str(d)[:10],
             "dd": int(dd), "nt": int(nt),
             "sp": None if pd.isna(sp) else round(float(sp), 1)}
            for h, a, m, w, d, dd, nt, sp in zip(
                faltam.home_team, faltam.away_team, mu, faltam.week,
                faltam.gameday, faltam.dif_descanso.fillna(0), faltam.neutro,
                faltam.spread_line)],
        "resid": [round(float(x), 2) for x in res],
        "totais": [int(x) for x in tot],
        "hist": hist,
    }


if __name__ == "__main__":
    main()
