"""EPA por time em cada jogo, a partir do play-by-play do nflverse.

EPA (expected points added) já está em PONTOS — a soma do EPA das jogadas de um
time num jogo é, literalmente, quantos pontos esperados ele gerou. Então a
diferença de EPA entre os dois lados é uma "margem esperada", na mesma unidade
da margem real, e comparável a ela sem conversão nenhuma.

A hipótese a testar é a que você ouve em toda análise de NFL: **EPA estabiliza
mais rápido que o placar**, porque usa 130 jogadas por jogo em vez de um número.
Se for verdade aqui, o rating ajustado sobre margem de EPA deve prever a margem
REAL futura melhor que o rating ajustado sobre a margem real. É o que
`regua_epa.py` mede — nada entra no modelo por fé.

O play-by-play são ~20 MB por temporada, 450 MB no total. Nada disso fica em
disco: cada temporada é baixada para a memória, agregada em ~550 linhas, e
descartada.

    python epa.py              # usa o cache
    python epa.py --refresh    # refaz do zero (demora)
"""
import io
import os
import sys

import pandas as pd
import requests

BASE = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
        "play_by_play_{ano}.parquet")
UA = {"User-Agent": "Mozilla/5.0 (montecarlo-nfl; uso pessoal)"}
CACHE = "data/epa_por_jogo.csv"
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam",
        "epa", "play_type", "special"]


def _agregar_temporada(ano):
    r = requests.get(BASE.format(ano=ano), headers=UA, timeout=600)
    r.raise_for_status()
    d = pd.read_parquet(io.BytesIO(r.content), columns=COLS)
    d = d[d.season_type == "REG"]
    d = d[d.posteam.notna() & d.epa.notna()]

    # Duas contagens, de propósito: uma com tudo que o time gerou com a bola
    # (inclui retorno e jogada especial) e outra só de ataque. Qual serve é
    # pergunta de medição, não de gosto.
    d["scrimmage"] = d.play_type.isin(["pass", "run"])
    g = d.groupby(["game_id", "week", "posteam"], as_index=False).agg(
        epa_total=("epa", "sum"),
        jogadas=("epa", "size"),
        epa_scrimmage=("epa", lambda s: s[d.loc[s.index, "scrimmage"]].sum()),
        jogadas_scrimmage=("scrimmage", "sum"))
    g["season"] = ano
    return g.rename(columns={"posteam": "team"})


def carregar(refresh=False, de=2002, ate=2026):
    if os.path.exists(CACHE) and not refresh:
        return pd.read_csv(CACHE)
    partes = []
    for ano in range(de, ate + 1):
        try:
            p = _agregar_temporada(ano)
        except Exception as e:
            print(f"  {ano}: falhou ({type(e).__name__}) — pulado")
            continue
        partes.append(p)
        print(f"  {ano}: {len(p)} linhas de time-jogo, "
              f"{p.jogadas.sum()} jogadas", flush=True)
    df = pd.concat(partes, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def margem_epa(epa_df, jogos):
    """Junta o EPA por time na tabela de jogos e devolve a margem de EPA.

    Positivo = mandante gerou mais pontos esperados, na mesma escala da margem
    real. Um jogo sem as duas pontas vira NaN em vez de zero — zero seria
    afirmar equilíbrio onde não há dado.
    """
    e = epa_df.set_index(["game_id", "team"])
    out = jogos.copy()
    for lado, col in (("home_team", "casa"), ("away_team", "fora")):
        idx = pd.MultiIndex.from_arrays([jogos.game_id, jogos[lado]])
        out[f"epa_{col}"] = e.epa_total.reindex(idx).to_numpy()
        out[f"epas_{col}"] = e.epa_scrimmage.reindex(idx).to_numpy()
    out["margem_epa"] = out.epa_casa - out.epa_fora
    out["margem_epa_scrimmage"] = out.epas_casa - out.epas_fora
    return out


if __name__ == "__main__":
    df = carregar(refresh="--refresh" in sys.argv)
    print(f"\n{len(df)} linhas time-jogo, temporadas "
          f"{int(df.season.min())}–{int(df.season.max())}")
    print(f"jogadas totais: {int(df.jogadas.sum()):,}".replace(",", "."))

    from dados import carregar as carregar_jogos
    j = carregar_jogos()
    m = margem_epa(df, j[j.disputado])
    ok = m.margem_epa.notna()
    print(f"\njogos com EPA nas duas pontas: {int(ok.sum())} de {len(m)}")
    print(f"correlacao entre margem de EPA e margem real: "
          f"{m.loc[ok, 'margem_epa'].corr(m.loc[ok, 'margem']):.3f}")
    print(f"  (so ataque: "
          f"{m.loc[ok, 'margem_epa_scrimmage'].corr(m.loc[ok, 'margem']):.3f})")
    print(f"desvio da margem de EPA: {m.loc[ok, 'margem_epa'].std():.2f} pontos"
          f"   |   da margem real: {m.loc[ok, 'margem'].std():.2f}")
