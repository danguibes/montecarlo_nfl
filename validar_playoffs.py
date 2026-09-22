"""O teste de aceitação do motor de playoffs.

Não é teste de unidade: é refazer a semeadura de todas as temporadas desde 2002
a partir dos jogos da temporada regular, e exigir que ela bata com o que
aconteceu de verdade. Ou passa em todas, ou o erro aparece e tem nome.

Duas conferências por ano, 24 anos, 7 vagas cada — 336 classificações para
errar. Um motor de desempate que acerta todas não está certo por sorte.

    python validar_playoffs.py
    python validar_playoffs.py --detalhe 2019
"""
import io
import sys

import pandas as pd
import requests

from dados import carregar
from playoffs import CONF_DE, n_wildcards, semear_temporada

UA = {"User-Agent": "Mozilla/5.0 (montecarlo-nfl; uso pessoal)"}
URL_STANDINGS = "https://github.com/nflverse/nfldata/raw/master/data/standings.csv"


def semeadura_real():
    """Cabeças de chave de verdade. Tenta o standings.csv do nflverse; se ele
    não trouxer a coluna, cai para a lista de participantes do pós-temporada,
    que é o que sempre dá para reconstruir dos jogos."""
    try:
        r = requests.get(URL_STANDINGS, headers=UA, timeout=60)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        col = next((c for c in d.columns if c.lower() in ("seed", "playoff_seed")), None)
        if col:
            d = d[d[col].notna()]
            return d, col
    except Exception:
        pass
    return None, None


def participantes_reais(todos):
    """Times que jogaram o pós-temporada, por ano e conferência."""
    pos = todos[todos.game_type != "REG"]
    fora = {}
    for temporada, g in pos.groupby("season"):
        times = set(g.home_team) | set(g.away_team)
        for conf in ("AFC", "NFC"):
            fora[(temporada, conf)] = {t for t in times if CONF_DE.get(t) == conf}
    return fora


def main():
    todos = carregar(tipo=None)
    reg = todos[(todos.game_type == "REG") & todos.home_score.notna()]
    reais = participantes_reais(todos)
    std, col = semeadura_real()

    detalhe = None
    if "--detalhe" in sys.argv:
        detalhe = int(sys.argv[sys.argv.index("--detalhe") + 1])

    linhas, erros = [], 0
    for temporada in sorted(reg.season.unique()):
        g = reg[reg.season == temporada]
        # a temporada tem que estar completa para a semeadura fazer sentido
        if (temporada, "AFC") not in reais:
            continue
        cab, _ = semear_temporada(g)
        for conf in ("AFC", "NFC"):
            meu = cab[conf]
            real = reais[(temporada, conf)]
            acertou_conjunto = set(meu) == real
            acertou_ordem = None
            if std is not None:
                s = std[(std.season == temporada)]
                s = s[s.team.isin(meu)]
                if len(s) == 7 and col in s:
                    ordem_real = list(s.sort_values(col).team)
                    acertou_ordem = (ordem_real == meu)
            linhas.append({"temporada": temporada, "conf": conf,
                           "conjunto": acertou_conjunto, "ordem": acertou_ordem})
            if not acertou_conjunto:
                erros += 1
                falta = sorted(real - set(meu))
                sobra = sorted(set(meu) - real)
                print(f"  {temporada} {conf}: faltou {falta}  sobrou {sobra}")
            if detalhe == temporada:
                print(f"  {temporada} {conf} minha semeadura: {meu}")
                print(f"  {temporada} {conf} classificados reais: {sorted(real)}")

    t = pd.DataFrame(linhas)
    n = len(t)
    ok = int(t.conjunto.sum())
    print(f"\nCONJUNTO DE CLASSIFICADOS: {ok} de {n} conferência-temporadas "
          f"({ok/n*100:.1f}%)")
    # Contar 7 vagas sempre estava errado: ate 2019 eram 6 por conferencia. O
    # porte para JavaScript foi quem denunciou, ao chegar a 300 onde o Python
    # dizia 336 — os dois acertavam as mesmas conferencias-temporadas, so que
    # um dos rotulos mentia.
    vagas = int((t.temporada.map(lambda a: n_wildcards(int(a)) + 4)).sum())
    ok_vagas = int((t[t.conjunto].temporada.map(
        lambda a: n_wildcards(int(a)) + 4)).sum())
    print(f"  ou seja {ok_vagas} de {vagas} vagas corretas")
    if t.ordem.notna().any():
        o = t[t.ordem.notna()]
        print(f"ORDEM DAS CABEÇAS: {int(o.ordem.sum())} de {len(o)} "
              f"({o.ordem.mean()*100:.1f}%)")
    else:
        print("ORDEM DAS CABEÇAS: não validada — a fonte não trouxe a coluna de seed")
    t.to_csv("out/validacao_playoffs.csv", index=False)
    print("\ngravado out/validacao_playoffs.csv")
    sys.exit(0 if erros == 0 else 1)


if __name__ == "__main__":
    main()
