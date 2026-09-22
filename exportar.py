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

# Medidos em regua.py, caminhada para a frente sobre 5.999 jogos.
REGUA = {"baseline": 14.667, "modelo": 13.553, "mercado": 13.203}


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
        "wildcards": n_wildcards(temporada),
        "regua": REGUA,
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

    res, tot = residuos_historicos(jogos)
    rng = np.random.default_rng(3)
    amostra = rng.choice(len(res), size=min(4000, len(res)), replace=False)

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
             "sem": int(w), "data": str(d)[:10]}
            for h, a, m, w, d in zip(faltam.home_team, faltam.away_team, mu,
                                     faltam.week, faltam.gameday)],
        "resid": [round(float(x), 2) for x in res[amostra]],
        "totais": [int(x) for x in tot[amostra]],
        "hist": hist,
    }


if __name__ == "__main__":
    main()
