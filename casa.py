"""O mando de campo é o mesmo para os 32 times?

A intuição diz que não — Green Bay em dezembro, o barulho de Seattle, a altitude
de Denver. A pergunta desta página é se isso aparece no placar e, mais
importante, se PERSISTE: um efeito real mas que não se repete de um ano para o
outro é história, não previsão.

MÉTODO. Não dá para comparar a margem em casa de cada time direto: time bom
ganha em casa por ser bom. O que se usa é o RESÍDUO do modelo caminhando para a
frente — a previsão já embute força dos dois lados, mando de liga e descanso.

Mas o resíduo médio em casa ainda confunde duas coisas: o mando extra do time
(h) e o erro do próprio rating dele (b), que sobe quando o time está
subavaliado. A separação sai de graça, porque a margem é (casa − fora):

    resíduo médio em CASA  =  +h + b
    resíduo médio FORA     =       −b

Somar as duas cancela o rating e deixa h sozinho. Subtrair e dividir por dois
deixa b — que serve de CONTROLE: se o modelo é honesto fora da amostra, b tem
que ser ruído em volta de zero.

    python casa.py
"""
import numpy as np
import pandas as pd

from dados import carregar
from modelo import ajustar, margem_esperada, prior_da_temporada
from regua import PRIMEIRA_AVALIADA

LAM, ENC = 8.0, 0.7


def residuos_fora_da_amostra(df, lam=LAM, enc=ENC):
    """Um resíduo por jogo previsto, com os dois times identificados."""
    times = sorted(set(df.home_team) | set(df.away_team))
    linhas = []
    for temporada in sorted(df.season.unique()):
        if temporada < PRIMEIRA_AVALIADA:
            continue
        ant = df[(df.season == temporada - 1) & df.disputado]
        prior = prior_da_temporada(ant, times, enc, lam) if len(ant) else None
        atual = df[df.season == temporada]
        for semana in sorted(atual.week.unique()):
            passado = atual[(atual.week < semana) & atual.disputado]
            alvo = atual[(atual.week == semana) & atual.disputado]
            if not len(alvo):
                continue
            fit = ajustar(passado, times, prior=prior, lam=lam)
            linhas.append(pd.DataFrame({
                "season": temporada,
                "casa": alvo.home_team.to_numpy(),
                "fora": alvo.away_team.to_numpy(),
                "neutro": alvo.neutro.to_numpy(),
                "resid": alvo.margem.to_numpy(float) - margem_esperada(fit, alvo),
            }))
    return pd.concat(linhas, ignore_index=True)


def por_time(r):
    """h = mando extra do time; b = viés do rating (o controle)."""
    r = r[r.neutro == 0]
    em_casa = r.groupby("casa").resid.agg(["mean", "size"])
    fora = r.groupby("fora").resid.agg(["mean", "size"])
    j = em_casa.join(fora, lsuffix="_casa", rsuffix="_fora").dropna()
    out = pd.DataFrame({
        "n_casa": j.size_casa.astype(int),
        "h": j.mean_casa + j.mean_fora,
        "b": (j.mean_casa - j.mean_fora) / 2,
    })
    # Erro-padrão de h: soma de duas médias independentes.
    sd = r.resid.std()
    out["se_h"] = np.sqrt(sd ** 2 / j.size_casa + sd ** 2 / j.size_fora)
    return out.sort_values("h", ascending=False)


def main():
    df = carregar()
    r = residuos_fora_da_amostra(df)
    print(f"{len(r)} jogos previstos, {int(r.season.min())}–{int(r.season.max())}")
    print(f"desvio do resíduo por jogo: {r.resid.std():.2f} pontos\n")

    t = por_time(r)
    print("MANDO EXTRA POR TIME (h), em pontos além da vantagem de liga")
    print(f"{'':6s}{'h':>7}{'±se':>7}{'n casa':>8}   {'':6s}{'h':>7}{'±se':>7}{'n casa':>8}")
    ordem = list(t.index)
    meio = (len(ordem) + 1) // 2
    for i in range(meio):
        linha = ""
        for col in (i, i + meio):
            if col < len(ordem):
                s = ordem[col]; x = t.loc[s]
                linha += f"{s:6s}{x.h:+7.2f}{x.se_h:7.2f}{x.n_casa:8.0f}   "
        print(linha)

    print(f"\nDISPERSÃO")
    dp_obs = t.h.std()
    se_med = np.sqrt((t.se_h ** 2).mean())
    print(f"  desvio observado de h        {dp_obs:6.2f} pontos")
    print(f"  desvio esperado só por ruído {se_med:6.2f} pontos")
    verd = dp_obs ** 2 - se_med ** 2
    print(f"  sobra de dispersão VERDADEIRA {np.sqrt(verd) if verd > 0 else 0:6.2f} pontos"
          f"   ({'existe' if verd > 0 else 'nenhuma'})")
    print(f"  controle b (viés de rating)  {t.b.std():6.2f}  "
          f"— tem que ser pequeno, e é o teste de que o método separa as duas coisas")

    print(f"\nPERSISTÊNCIA — o h de um período prevê o do outro?")
    for nome, a, b in [
        ("anos pares x ímpares", r[r.season % 2 == 0], r[r.season % 2 == 1]),
        ("2003–2014 x 2015–2026", r[r.season <= 2014], r[r.season >= 2015]),
    ]:
        ta, tb = por_time(a), por_time(b)
        j = ta[["h"]].join(tb[["h"]], lsuffix="_a", rsuffix="_b").dropna()
        c = float(np.corrcoef(j.h_a, j.h_b)[0, 1])
        print(f"  {nome:24s} correlação {c:+.3f}  (n={len(j)} times)")
        print(f"  {'':24s} inclinação  {np.polyfit(j.h_a, j.h_b, 1)[0]:+.3f}"
              f"  — quanto do h de um lado se repete no outro")


if __name__ == "__main__":
    main()
