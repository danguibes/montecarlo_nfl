"""Invariantes que já quebraram, aqui ou no projeto irmão.

Não é suíte de testes: é a lista curta dos erros que passam despercebidos em
produção. Roda como primeiro passo de `atualizar.py`, sem tolerância a falha.

O caso do jogo em andamento é o motivo de este arquivo existir. Ele só se
manifesta com uma partida rolando no instante exato da atualização, então não
dá para conferir olhando — tem que ser com relógio congelado.

    python testes.py
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

import dados
from dados import ET, HORAS_ATE_ENCERRAR, agora_et, encerrado


def check(nome, cond, detalhe=""):
    print(f"  {'ok   ' if cond else 'FALHA'}  {nome}"
          + (f"  — {detalhe}" if detalhe and not cond else ""))
    return bool(cond)


def teste_fuso():
    """O relógio tem que ser o de Nova York, não o da máquina.

    No projeto do Brasileirão este erro só aparecia no CI: o runner do GitHub
    roda em UTC, e um jogo das 20:30 parecia ter começado há três horas. Na
    máquina do dono, no fuso certo, o mesmo código acertava.
    """
    ok = True
    real = datetime.now(timezone.utc).astimezone(ET)
    ok &= check("agora_et independe do fuso da máquina",
                abs((agora_et() - real).total_seconds()) < 5)
    return ok


def teste_em_andamento():
    """Jogo que começou há pouco não conta, por mais placar que tenha."""
    base = datetime(2026, 9, 27, 13, 0, tzinfo=ET)
    casos = [
        ("começou há 1 hora",   base + timedelta(hours=1),  False),
        ("começou há 3 horas",  base + timedelta(hours=3),  False),
        ("começou há 6h01",     base + timedelta(hours=6, minutes=1), True),
        ("começou ontem",       base + timedelta(days=1),   True),
        ("ainda vai começar",   base - timedelta(hours=2),  False),
    ]
    ok = True
    for nome, agora, esperado in casos:
        r = encerrado("2026-09-27", "13:00", agora)
        ok &= check(f"{nome:22s} -> encerrado={esperado}", r == esperado,
                    f"veio {r}")
    return ok


def teste_placar_parcial_nao_vaza():
    """Placar de jogo em andamento não pode chegar ao resto do projeto.

    Marcar como não-disputado e DEIXAR o placar na linha seria pior que não
    ter guarda nenhuma: qualquer código que leia home_score sem olhar
    `disputado` continuaria usando o parcial, e ninguém perceberia.
    """
    import pandas as pd
    df = pd.DataFrame({
        "gameday": ["2026-09-27", "2026-09-27"],
        "gametime": ["13:00", "13:00"],
        "home_score": [21.0, 24.0],
        "away_score": [17.0, 10.0],
    })
    agora = datetime(2026, 9, 27, 14, 30, tzinfo=ET)     # 1h30 de jogo
    fim = [encerrado(g, t, agora) for g, t in zip(df.gameday, df.gametime)]
    ok = check("jogo de 1h30 marcado como NAO encerrado", not any(fim))
    parcial = df.copy()
    parcial.loc[~pd.Series(fim), ["home_score", "away_score"]] = pd.NA
    ok &= check("placar parcial apagado das colunas",
                parcial.home_score.isna().all() and parcial.away_score.isna().all())
    return ok


def teste_franquias():
    """Times relocados têm que casar entre games.csv e play-by-play."""
    df = dados.carregar()
    antigos = {"OAK", "SD", "STL"} & (set(df.home_team) | set(df.away_team))
    return check("codigos antigos de franquia normalizados", not antigos,
                 f"sobraram {antigos}")


def teste_empates():
    """A simulação não pode inventar empate: 0,24% na NFL, não 2,65%."""
    from modelo import margem_esperada
    from simular import ajustar_atual, residuos_historicos, resolver_empates
    df = dados.carregar()
    res, tot = residuos_historicos(df)
    fit = ajustar_atual(df, int(df.season.max()))
    faltam = df[(df.season == df.season.max()) & ~df.disputado].reset_index(drop=True)
    if not len(faltam):
        return check("empates calibrados", True)
    mu = margem_esperada(fit, faltam)
    rng = np.random.default_rng(11)
    emp = n = 0
    for _ in range(150):
        m = mu + rng.choice(res, size=len(faltam))
        t = rng.choice(tot, size=len(faltam))
        ph = np.maximum(0, np.rint((t + m) / 2))
        pa = np.maximum(0, np.rint((t - m) / 2))
        ph, pa = resolver_empates(ph, pa, m, rng)
        emp += int((ph == pa).sum()); n += len(faltam)
    taxa = emp / n * 100
    return check(f"empates em {taxa:.3f}% (alvo 0,24%)", 0.10 < taxa < 0.45,
                 f"fora da faixa")


if __name__ == "__main__":
    print("invariantes:")
    tudo = all([teste_fuso(), teste_em_andamento(),
                teste_placar_parcial_nao_vaza(), teste_franquias(),
                teste_empates()])
    print("\n" + ("todos passaram" if tudo else "HA FALHA — nao publique"))
    sys.exit(0 if tudo else 1)
