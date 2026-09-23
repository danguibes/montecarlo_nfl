"""Quando o modelo e o mercado discordam, quem acerta mais?

A seção de spreads da página existe para mostrar divergência, e mostrar
divergência sem dizer o que ela significa é meio caminho para alguém ler como
recomendação. Esta é a medição que dá a legenda.

O material é o mesmo da régua: 24 temporadas de previsão fora da amostra, com a
linha do mercado ao lado. A pergunta é condicional — dentro de cada faixa de
discordância, qual dos dois erra menos.

    python discordancia.py
"""
import numpy as np
import pandas as pd

FAIXAS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]


def main():
    r = pd.read_csv("out/regua.csv").dropna(subset=["real", "modelo", "spread"])
    r["d"] = (r.modelo - r.spread).abs()
    print(f"{len(r)} jogos com previsão e linha, "
          f"{int(r.season.min())}–{int(r.season.max())}")
    print(f"discordância média {r.d.mean():.2f} pontos, "
          f"mediana {r.d.median():.2f}\n")

    print(f"{'discordância':>14} {'n':>6} {'RMSE modelo':>12} {'RMSE linha':>11} "
          f"{'dif':>7} {'modelo acerta o lado':>21}")
    for lo, hi in FAIXAS:
        g = r[(r.d >= lo) & (r.d < hi)]
        if not len(g):
            continue
        em = float(((g.real - g.modelo) ** 2).mean() ** 0.5)
        es = float(((g.real - g.spread) ** 2).mean() ** 0.5)
        # Quando discordam, um dos dois tem o lado certo contra a linha: o
        # placar real cai de um lado da linha, e o modelo aponta para um lado.
        lado = g[g.d >= 0.5]
        venceu = ((lado.real - lado.spread) * (lado.modelo - lado.spread) > 0)
        nome = f"{lo}–{hi} pts" if hi < 99 else f"{lo}+ pts"
        print(f"{nome:>14} {len(g):6d} {em:12.3f} {es:11.3f} {em-es:+7.3f} "
              f"{venceu.mean()*100:20.1f}%")

    print("\nO 'acerta o lado' é o teste direto: o placar real caiu do lado da")
    print("linha para onde o modelo apontava? 50% é o resultado de uma moeda.")
    g = r[r.d >= 3]
    v = ((g.real - g.spread) * (g.modelo - g.spread) > 0)
    se = (0.25 / len(v)) ** 0.5 * 100
    print(f"\nDiscordância de 3+ pontos: {len(v)} jogos, modelo acerta o lado em "
          f"{v.mean()*100:.1f}% (±{1.96*se:.1f})")
    print(f"t contra 50%: {(v.mean()-0.5)/ (0.25/len(v))**0.5:+.2f}")


if __name__ == "__main__":
    main()
