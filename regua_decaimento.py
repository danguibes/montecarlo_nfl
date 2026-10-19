"""E se o prior do ano anterior fosse desligado no meio da temporada?

O incômodo é legítimo: com 17 jogos na mão, 34% do rating ainda vem do ano
passado. A intuição diz que a essa altura o ano passado é resquício, e que o
modelo ganharia soltando-o.

A busca em grade do `regua.py` NÃO responde isso. Ela mediu λ **constante** e
achou 8. Um λ que decai no tempo é outro desenho, e nunca foi à régua.

Quatro famílias, todas caminhando para a frente, sem ver o futuro:

  constante     λ = λ0                      (o modelo de hoje)
  corte seco    λ = λ0 até W jogos, 0 depois (a proposta, na forma mais pura)
  linear        λ = λ0 · max(0, 1 − n/N)     (desliga suave, zera em N jogos)
  piso          λ = λ0 · max(f, 1 − n/N)     (decai, mas nunca abaixo de f)

λ0 entra em grade junto, porque um esquema que desliga cedo pode querer começar
mais pesado — fixar λ0 = 8 testaria a pergunta errada.

A medida principal é o RMSE da SEGUNDA METADE da temporada. O RMSE do ano
inteiro dilui a diferença nas semanas 1 a 8, onde todos os esquemas são
idênticos por construção: é a mesma armadilha de medir o remédio no corpo
inteiro quando a doença está num órgão.

    python regua_decaimento.py
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dados import carregar
from modelo import ajustar, margem_esperada, prior_da_temporada
from regua import PRIMEIRA_AVALIADA, resumo

ENC = 0.7
METADE = 9          # a partir desta semana, a "segunda metade"


def constante(lam0):
    return lambda n: lam0


def corte(lam0, w):
    return lambda n: lam0 if n < w else 0.0


def linear(lam0, N):
    return lambda n: lam0 * max(0.0, 1.0 - n / N)


def piso(lam0, N, f):
    return lambda n: lam0 * max(f, 1.0 - n / N)


def avaliar(df, flam, enc=ENC, lam_prior=8.0):
    """Igual ao regua.avaliar, mas com λ variando com os jogos já disputados.

    `lam_prior` é o λ usado para FORMAR o prior a partir da temporada anterior,
    e fica fixo: é outro ajuste, sobre uma temporada completa, e misturá-lo com
    o λ do ano corrente confundiria duas perguntas.
    """
    times = sorted(set(df.home_team) | set(df.away_team))
    linhas = []
    for temporada in sorted(df.season.unique()):
        if temporada < PRIMEIRA_AVALIADA:
            continue
        ant = df[(df.season == temporada - 1) & df.disputado]
        prior = prior_da_temporada(ant, times, enc, lam_prior) if len(ant) else None
        atual = df[df.season == temporada]
        for semana in sorted(atual.week.unique()):
            passado = atual[(atual.week < semana) & atual.disputado]
            alvo = atual[(atual.week == semana) & atual.disputado]
            if not len(alvo):
                continue
            n = 2 * len(passado) / len(times)        # jogos por time, em média
            # Piso de 1e-6: com lambda = 0 o bloco de times tem posto 31 de 32
            # — os ratings só ficam definidos a menos de uma constante, mesmo
            # com a temporada INTEIRA jogada. Sem o piso, desligar o prior não
            # mede "esquecer o ano passado": mede o solver devolvendo lixo, e o
            # RMSE sai em dezenas de milhares. O piso escolhe a solução de
            # norma mínima, que é o que a centragem já assume.
            fit = ajustar(passado, times, prior=prior, lam=max(flam(n), 1e-6))
            linhas.append(pd.DataFrame({
                "season": temporada, "week": semana,
                "real": alvo.margem.to_numpy(float),
                "modelo": margem_esperada(fit, alvo),
                "spread": alvo.spread_line.to_numpy(float),
            }))
    return pd.concat(linhas, ignore_index=True)


def rmse(r, col="modelo"):
    e = (r.real - r[col]).dropna()
    return float((e ** 2).mean() ** 0.5)


def main():
    df = carregar()
    base = avaliar(df, constante(8.0))
    b_tudo, b_fim = rmse(base), rmse(base[base.week >= METADE])
    ll_base = resumo(base).loc["modelo", "log_loss"]
    print(f"{len(base)} jogos previstos, 2003–2026")
    print(f"modelo de hoje (λ = 8 constante): RMSE {b_tudo:.4f} "
          f"(ano todo) · {b_fim:.4f} (semana {METADE}+)\n")

    esquemas = [("constante", f"λ0={l}", constante(l)) for l in (4, 8, 12, 20)]
    esquemas += [("corte seco", f"λ0={l}, zera em {w}j", corte(l, w))
                 for l in (8, 12, 20) for w in (6, 8, 10)]
    esquemas += [("linear", f"λ0={l}, zera em {N}j", linear(l, N))
                 for l in (8, 12, 20) for N in (8, 10, 12, 17)]
    esquemas += [("piso", f"λ0={l}, N={N}, piso {f}", piso(l, N, f))
                 for l in (8, 12) for N in (10, 17) for f in (0.3, 0.5)]

    print(f"{'esquema':>11} {'parâmetros':>24} {'RMSE ano':>9} {'RMSE sem9+':>11} "
          f"{'vs hoje':>8} {'log-loss':>9}")
    res = []
    for fam, nome, f in esquemas:
        r = avaliar(df, f)
        t, fim = rmse(r), rmse(r[r.week >= METADE])
        ll = resumo(r).loc["modelo", "log_loss"]
        res.append((fam, nome, t, fim, ll))
        marca = " <--" if fim < b_fim else ""
        print(f"{fam:>11} {nome:>24} {t:9.4f} {fim:11.4f} {fim-b_fim:+8.4f} "
              f"{ll:9.5f}{marca}")

    melhor = min(res, key=lambda x: x[3])
    print(f"\nmelhor na segunda metade: {melhor[0]} ({melhor[1]}) "
          f"RMSE {melhor[3]:.4f} ({melhor[3]-b_fim:+.4f})")
    print(f"melhor em log-loss:       {min(res, key=lambda x: x[4])[1]}"
          f"  ({min(r[4] for r in res):.5f} contra {ll_base:.5f} de hoje)")


if __name__ == "__main__":
    main()
