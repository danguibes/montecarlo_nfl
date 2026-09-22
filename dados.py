"""Dados da NFL, do nflverse. Sem chave, atualizado diariamente.

Um arquivo resolve quase tudo: games.csv traz agenda, placar, descanso de cada
lado, spread e total do mercado, clima, e o QB titular de cada jogo desde 1999.

    python dados.py            # usa o cache
    python dados.py --refresh  # rebaixa
"""
import io
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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

# JOGO EM ANDAMENTO NAO CONTA.
#
# Nao consegui observar se o nflverse publica placar parcial — precisaria de um
# jogo rolando no instante da conferencia. Mas o projeto irmao do Brasileirao
# ensinou o custo de descobrir isso no dia da rodada: la as DUAS fontes
# publicavam parcial e CONCORDAVAM entre si, entao a conferencia cruzada — que
# era a defesa — passava com nota maxima enquanto o placar de um jogo no
# intervalo entrava no ajuste, na tabela e na projecao.
#
# A defesa que nao depende da fonte e o relogio: so vale como encerrado o jogo
# que comecou ha tempo suficiente. Jogo de NFL dura umas 3h10; seis horas e
# folgado, e errar para o lado de nao contar e barato — o jogo entra na proxima
# atualizacao.
#
# O horario do games.csv e de Nova York. Usar o relogio da maquina aqui seria
# repetir o bug que so aparecia no CI do Brasileirao, onde o runner roda em UTC
# e um jogo das 20:30 parecia ter comecado ha tres horas.
ET = ZoneInfo("America/New_York")
HORAS_ATE_ENCERRAR = 6


def agora_et():
    return datetime.now(timezone.utc).astimezone(ET)


def encerrado(gameday, gametime, agora=None):
    """O jogo ja terminou, pelo relogio de Nova York?"""
    if not isinstance(gameday, str):
        return False
    agora = agora or agora_et()
    hora = gametime if isinstance(gametime, str) and ":" in gametime else "13:00"
    try:
        ini = datetime.strptime(f"{gameday} {hora[:5]}", "%Y-%m-%d %H:%M")
    except ValueError:
        return False
    ini = ini.replace(tzinfo=ET)
    return agora >= ini + timedelta(hours=HORAS_ATE_ENCERRAR)


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
    tem_placar = df.home_score.notna() & df.away_score.notna()
    ja_acabou = [encerrado(g, t) for g, t in zip(df.gameday, df.gametime)]
    df["disputado"] = tem_placar & pd.Series(ja_acabou, index=df.index)
    # placar parcial nao pode vazar para o resto do projeto
    df.loc[tem_placar & ~df.disputado, ["home_score", "away_score"]] = pd.NA
    df["em_andamento"] = tem_placar & ~df.disputado
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
    if df.em_andamento.any():
        viv = df[df.em_andamento]
        print(f"\nEM ANDAMENTO agora ({len(viv)}), fora da conta:")
        for r in viv.itertuples():
            print(f"  {r.away_team} x {r.home_team}  ({r.gameday} {r.gametime} ET)")
    print(f"\ncampo neutro no historico: {int(df.neutro.sum())} jogos")
    print(f"cobertura de spread      : {df.spread_line.notna().mean()*100:.1f}%")
    print(f"cobertura de QB titular  : {df.home_qb_name.notna().mean()*100:.1f}%")
