"""Gera web/index.html: página única, autocontida, com o resultado da simulação.

Diferente do projeto do Brasileirão, aqui o Monte Carlo **não** roda no
navegador. O motivo é o motor de desempate: ele custou quatro correções para
passar de 22,9% para 100% de acerto nas 336 vagas históricas, e portá-lo para
JavaScript duplicaria a superfície onde um erro silencioso cabe. A página
mostra o que o Python calculou e validou.

O preço é não ter condicional por jogo, que no Brasileirão era o melhor
brinquedo. Fica registrado como escolha, não como esquecimento.

    python exportar.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from dados import carregar
from playoffs import n_wildcards

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

    tpl = open("web/template.html", encoding="utf-8").read()
    html = tpl.replace("/*__DADOS__*/null",
                       json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if "/*__DADOS__*/null" in html:
        sys.exit("placeholder nao substituido — a pagina sairia vazia")
    os.makedirs("web", exist_ok=True)
    open("web/index.html", "w", encoding="utf-8").write(html)
    print(f"web/index.html  {len(html)/1024:.0f} KB  "
          f"({len(times)} times, {payload['faltam']} jogos restantes)")


if __name__ == "__main__":
    main()
