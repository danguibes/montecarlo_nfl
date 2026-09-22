"""Abre um empate específico e mostra o valor de cada critério, passo a passo.

Sem isto, "o motor errou 2004 NFC" não diz nada. Com isto, dá para ver em qual
degrau da cascata os dois times se separam, e comparar com a regra escrita.

    python diagnosticar.py 2004 NFC LA NO
"""
import sys

from dados import carregar
from playoffs import (CONF_DE, DIV_DE, DIVISOES, Temporada, _passos,
                      n_wildcards, semear)


def main():
    temporada = int(sys.argv[1])
    conf = sys.argv[2]
    times = sys.argv[3:]

    reg = carregar()
    g = reg[(reg.season == temporada) & reg.disputado]
    T = Temporada(g.home_team.to_numpy(), g.away_team.to_numpy(),
                  g.home_score.to_numpy(), g.away_score.to_numpy())

    print(f"{temporada} {conf} — minha semeadura:")
    for i, t in enumerate(semear(T, conf, None, n_wildcards(temporada)), 1):
        print(f"  {i}. {t:4s} {DIV_DE[t]:10s} "
              f"{T.v[t]}-{T.d[t]}-{T.e[t]}  ({T.pct(t):.3f})")

    if not times:
        return
    print(f"\ncomparando {times} pelo critério de WILD CARD:")
    for t in times:
        print(f"  {t}: {T.v[t]}-{T.d[t]}-{T.e[t]}  aproveitamento {T.pct(t):.4f}  "
              f"divisão {DIV_DE[t]}")

    h = T.h2h.get((times[0], times[1]))
    print(f"\n  confronto direto {times[0]} x {times[1]}: "
          f"{h if h else 'NAO SE ENFRENTARAM'}")

    for modo in ("wildcard", "divisao"):
        print(f"\n  --- cascata, modo {modo} ---")
        for nome, nota in _passos(T, times, modo):
            vals = {t: nota(t) for t in times}
            melhor = max(vals.values())
            sep = "SEPARA" if sum(v == melhor for v in vals.values()) < len(times) else "empata"
            txt = "  ".join(f"{t}={vals[t]:.4f}" for t in times)
            print(f"    {nome:26s} {txt}   -> {sep}")
            if sep == "SEPARA":
                break


if __name__ == "__main__":
    main()
