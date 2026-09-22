"""Dar a cada time o seu próprio mando melhora a previsão?

`casa.py` mede o efeito e a persistência. Aqui a pergunta é a única que decide:
levado à régua, caminhando para a frente, o RMSE cai?

O teste é barato porque o resíduo já traz tudo. Se o modelo de hoje erra por
`e` e eu acrescento à previsão o mando extra `h` do mandante, o erro novo é
`e − h`. Então basta estimar h com o passado — SÓ o passado — e medir.

Três eixos de grade, porque cada um poderia ser o motivo de não funcionar:

  janela      quantas temporadas anteriores entram na estimativa de h
  encolhimento  n/(n+k): com k alto, h fica perto de zero e o teste vira o
                modelo de hoje; é o controle que prova que a grade está viva
  sem nada    o modelo atual, mando único de liga

    python regua_casa.py
"""
import numpy as np
import pandas as pd

from dados import carregar
from casa import residuos_fora_da_amostra, por_time


def h_do_passado(r, ate, janela, k):
    """Mando extra por time, estimado só com temporadas < `ate`."""
    p = r[r.season < ate]
    if janela:
        p = p[p.season >= ate - janela]
    if len(p) < 100:
        return {}
    t = por_time(p)
    peso = t.n_casa / (t.n_casa + k)
    return (t.h * peso).to_dict()


def main():
    df = carregar()
    r = residuos_fora_da_amostra(df)

    # Só temporadas com história suficiente para estimar h antes de usar.
    PRIMEIRA = 2008
    alvo = r[(r.season >= PRIMEIRA) & (r.neutro == 0)].copy()
    base = float((alvo.resid ** 2).mean() ** 0.5)
    print(f"{len(alvo)} jogos avaliados, {PRIMEIRA}–{int(alvo.season.max())}")
    print(f"modelo de hoje (mando único de liga): RMSE {base:.4f}\n")

    print(f"{'janela':>8} {'k':>5} {'RMSE':>9} {'vs hoje':>9} {'|h| médio':>10}")
    melhor = None
    for janela in (3, 5, 10, None):
        for k in (0, 20, 50, 100, 200, 500):
            novo = np.empty(len(alvo)); mags = []
            i = 0
            for temporada, g in alvo.groupby("season"):
                h = h_do_passado(r, temporada, janela, k)
                aj = g.casa.map(h).fillna(0.0).to_numpy()
                mags.append(np.abs(aj).mean())
                novo[i:i + len(g)] = g.resid.to_numpy() - aj
                i += len(g)
            rmse = float((novo ** 2).mean() ** 0.5)
            nome = f"{janela}" if janela else "todas"
            print(f"{nome:>8} {k:5d} {rmse:9.4f} {rmse - base:+9.4f} "
                  f"{np.mean(mags):10.3f}")
            if melhor is None or rmse < melhor[0]:
                melhor = (rmse, nome, k)
    print(f"\nmelhor da grade: janela {melhor[1]}, k={melhor[2]}, "
          f"RMSE {melhor[0]:.4f} ({melhor[0] - base:+.4f} vs hoje)")


if __name__ == "__main__":
    main()
