"""Dados da NFL, do nflverse. Sem chave, atualizado diariamente.

Um arquivo resolve quase tudo: games.csv traz agenda, placar, descanso de cada
lado, spread e total do mercado, clima, e o QB titular de cada jogo desde 1999.

    python dados.py            # usa o cache
    python dados.py --refresh  # rebaixa
"""
import io
import os
import sys

import pandas as pd
import requests

URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
UA = {"User-Agent": "Mozilla/5.0 (montecarlo-nfl; uso pessoal)"}
CACHE = "data/games.csv"

# 2002 e o primeiro ano com 32 times e 8 divisoes — o formato de hoje. Antes
# disso a estrutura de conferencia e desempate era outra, e misturar as duas
# epocas poluiria tanto o ajuste quanto a validacao do motor de playoffs.
PRIMEIRA_TEMPORADA = 2002

# O games.csv usa o codigo da EPOCA; o play-by-play usa o da franquia ATUAL.
# Sem esta ponte, todo jogo de Raiders, Chargers e Rams antes da mudanca de
# cidade perde o EPA em silencio — eram 714 jogos, 11% da base, e o unico
# sintoma era um NaN que ninguem olha. A franquia e a mesma, entao o rating
# deve atravessar a mudanca: Raiders continuam Raiders.
FRANQUIA = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def baixar(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return pd.read_csv(CACHE, low_memory=False)
    r = requests.get(URL, headers=UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def carregar(refresh=False, tipo="REG"):
    """Jogos a partir de 2002, com as colunas que o modelo usa."""
    df = baixar(refresh)
    df = df[df.season >= PRIMEIRA_TEMPORADA].copy()
    if tipo:
        df = df[df.game_type == tipo]
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(FRANQUIA)
    df["data"] = pd.to_datetime(df.gameday, errors="coerce")
    df["margem"] = df.home_score - df.away_score          # positivo = casa venceu
    df["total_pts"] = df.home_score + df.away_score
    df["dif_descanso"] = df.home_rest - df.away_rest
    df["disputado"] = df.home_score.notna() & df.away_score.notna()
    # Jogo em campo neutro nao tem mandante de verdade (Londres, Super Bowl).
    # Tratar como se tivesse embutiria vantagem de casa que nao existe.
    df["neutro"] = (df.location.astype(str).str.lower() != "home").astype(int)
    return df.sort_values(["season", "week", "data"]).reset_index(drop=True)


if __name__ == "__main__":
    df = carregar(refresh="--refresh" in sys.argv)
    d = df[df.disputado]
    print(f"{len(df)} jogos de temporada regular, {int(df.season.min())}–{int(df.season.max())}")
    print(f"  disputados          : {len(d)}")
    print(f"  agendados sem placar: {int((~df.disputado).sum())}")
    atual = df[df.season == df.season.max()]
    print(f"\ntemporada {int(df.season.max())}: {len(atual)} jogos, "
          f"{int(atual.disputado.sum())} disputados, "
          f"semanas {sorted(atual[atual.disputado].week.unique().tolist())}")
    a = atual[atual.disputado]
    por_time = (a.groupby("home_team").size()
                .add(a.groupby("away_team").size(), fill_value=0))
    print(f"  jogos por time: min {int(por_time.min())}, max {int(por_time.max())}, "
          f"{len(por_time)} times")
    print(f"\ncampo neutro no historico: {int(df.neutro.sum())} jogos")
    print(f"cobertura de spread      : {df.spread_line.notna().mean()*100:.1f}%")
    print(f"cobertura de QB titular  : {df.home_qb_name.notna().mean()*100:.1f}%")
