"""Semeadura dos playoffs da NFL, com os critérios de desempate de verdade.

Os quatro campeões de divisão ocupam as cabeças 1 a 4, ordenados entre si; os
melhores restantes entram como wild card.

**Quantos wild cards depende do ano.** De 2002 a 2019 eram DOIS por
conferência — seis vagas —, e dois times pegavam semana de descanso. De 2020 em
diante são TRÊS, sete vagas, e só o primeiro descansa. Ignorar isso faz o motor
classificar um time a mais em dezoito temporadas seguidas, e foi exatamente o
que a validação apontou na primeira rodada.

O desempate é uma cascata, e tem três armadilhas que quebram implementações
ingênuas:

**1. A ordem é diferente para divisão e para wild card.** Dentro da divisão vem
confronto direto, depois registro na divisão, depois jogos comuns, depois
conferência. Entre wild cards, o registro na CONFERÊNCIA vem logo depois do
confronto direto, e registro de divisão não entra.

**2. Quando um time é eliminado de um empate múltiplo, o procedimento RECOMEÇA
do passo 1** com os que sobraram. Não se continua descendo a lista. Aqui isso
cai naturalmente numa recursão.

**3. Só um time por divisão disputa cada vaga de wild card.** Antes de comparar
wild cards, os times da mesma divisão são resolvidos entre si pelo critério de
DIVISÃO, e só o melhor segue na disputa. Esquecer isso troca a ordem sempre que
dois times de uma mesma divisão brigam pela mesma vaga.

Com três ou mais times, o confronto direto só vale como **varredura**: alguém
venceu todos os outros (vai na frente) ou perdeu para todos (vai atrás). Se não
houver varredura, o passo é pulado — não se soma o saldo dos confrontos.

O teste desta peça não é de unidade, é de aceitação: `validar_playoffs.py`
refaz a semeadura de todas as temporadas de 2002 em diante e exige que ela
bata com a real, time por time.
"""
import numpy as np
import pandas as pd

# Alinhamento em 8 divisões, estável desde 2002. Códigos do nflverse — LV, LAC
# e LA já são a franquia atual, porque dados.py normaliza OAK, SD e STL.
DIVISOES = {
    "AFC East":  ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West":  ["DEN", "KC", "LV", "LAC"],
    "NFC East":  ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West":  ["ARI", "LA", "SF", "SEA"],
}
# Primeira temporada com 7 vagas por conferência.
PRIMEIRO_ANO_7 = 2020


def n_wildcards(temporada):
    return 3 if temporada >= PRIMEIRO_ANO_7 else 2


DIV_DE = {t: d for d, ts in DIVISOES.items() for t in ts}
CONF_DE = {t: d.split()[0] for t, d in DIV_DE.items()}
TIMES = sorted(DIV_DE)


class Temporada:
    """Tudo que os critérios de desempate precisam, calculado uma vez."""

    def __init__(self, casa, fora, pts_casa, pts_fora):
        self.v = {t: 0 for t in TIMES}
        self.d = {t: 0 for t in TIMES}
        self.e = {t: 0 for t in TIMES}
        self.pf = {t: 0 for t in TIMES}
        self.pa = {t: 0 for t in TIMES}
        self.adv = {t: [] for t in TIMES}          # adversários, com repetição
        self.venceu = {t: [] for t in TIMES}       # adversários que derrotou
        self.h2h = {}                              # (a,b) -> [v, d, e] de a

        for h, a, ph, pa in zip(casa, fora, pts_casa, pts_fora):
            self.pf[h] += ph; self.pa[h] += pa
            self.pf[a] += pa; self.pa[a] += ph
            self.adv[h].append(a); self.adv[a].append(h)
            par_h = self.h2h.setdefault((h, a), [0, 0, 0])
            par_a = self.h2h.setdefault((a, h), [0, 0, 0])
            if ph > pa:
                self.v[h] += 1; self.d[a] += 1
                self.venceu[h].append(a)
                par_h[0] += 1; par_a[1] += 1
            elif ph < pa:
                self.v[a] += 1; self.d[h] += 1
                self.venceu[a].append(h)
                par_a[0] += 1; par_h[1] += 1
            else:
                self.e[h] += 1; self.e[a] += 1
                par_h[2] += 1; par_a[2] += 1

    # ---------- métricas ----------
    def pct(self, t):
        n = self.v[t] + self.d[t] + self.e[t]
        return (self.v[t] + 0.5 * self.e[t]) / n if n else 0.0

    def _pct_contra(self, t, alvos):
        v = d = e = 0
        for o in alvos:
            r = self.h2h.get((t, o))
            if r:
                v += r[0]; d += r[1]; e += r[2]
        n = v + d + e
        return (v + 0.5 * e) / n if n else None, n

    def pct_divisao(self, t):
        return self._pct_contra(t, [x for x in DIVISOES[DIV_DE[t]] if x != t])[0] or 0.0

    def pct_conferencia(self, t):
        conf = CONF_DE[t]
        return self._pct_contra(t, [x for x in TIMES
                                    if CONF_DE[x] == conf and x != t])[0] or 0.0

    def h2h_pct(self, t, outros):
        return self._pct_contra(t, outros)

    def comuns(self, times):
        """Adversários enfrentados por TODOS os times do conjunto."""
        conjuntos = [set(self.adv[t]) - set(times) for t in times]
        comum = set.intersection(*conjuntos) if conjuntos else set()
        return comum

    def _agregado(self, lista):
        """Aproveitamento COMBINADO dos adversários da lista.

        A regra soma vitórias e derrotas de todos e divide no fim; a média das
        porcentagens individuais é outra conta, e difere quando há empate ou
        número desigual de jogos.
        """
        v = d = e = 0
        for o in lista:
            v += self.v[o]; d += self.d[o]; e += self.e[o]
        n = v + d + e
        return (v + 0.5 * e) / n if n else 0.0

    def forca_vitorias(self, t):
        return self._agregado(self.venceu[t])

    def forca_calendario(self, t):
        return self._agregado(self.adv[t])

    def saldo(self, t):
        return self.pf[t] - self.pa[t]


# ---------- a cascata ----------
def _varredura(T, times):
    """Confronto direto com 3+ times só vale como VARREDURA.

    "Venceu cada um dos outros" exige ter **enfrentado** cada um dos outros.
    Quem não jogou contra todos não varreu ninguém, por melhor que tenha ido
    nos jogos que fez.

    Era aqui o erro que reprovava 2004 NFC: Minnesota venceu New Orleans e
    nunca enfrentou o Rams, e a versão antiga leu isso como aproveitamento
    1,000 — varredura. Com a varredura falsa, Minnesota pulava na frente e a
    ordem inteira saía errada. A regra certa pula o passo e manda para o
    registro na conferência, onde o Rams lidera.
    """
    notas, houve = {}, False
    for t in times:
        outros = [x for x in times if x != t]
        if not all(T.h2h.get((t, o)) for o in outros):
            notas[t] = 0.5                      # não enfrentou todos
            continue
        p, _ = T.h2h_pct(t, outros)
        if p == 1.0:
            notas[t], houve = 1.0, True         # venceu todos
        elif p == 0.0:
            notas[t], houve = 0.0, True         # perdeu para todos
        else:
            notas[t] = 0.5
    return notas if houve else None


def _passos(T, times, modo):
    """Lista de funções de nota, na ordem oficial. Maior é melhor."""
    n = len(times)
    passos = []
    if n == 2:
        a, b = times
        p, jogos = T.h2h_pct(a, [b])
        if jogos:
            passos.append(("confronto direto",
                           lambda t: T.h2h_pct(t, [x for x in times if x != t])[0] or 0.0))
    elif modo == "divisao":
        # DIVISAO com 3+: confronto direto COMBINADO entre os empatados. Nao e
        # varredura — basta ter o melhor aproveitamento contra os outros.
        # Era isto que reprovava 2025 NFC South: Carolina fez 3-1 contra
        # Atlanta e Tampa e ganharia logo no primeiro passo, mas a exigencia de
        # varredura (4-0) pulava o criterio e a divisao saia para outro.
        if all(T.h2h.get((a, b)) for a in times for b in times if a != b):
            passos.append(("confronto direto combinado",
                           lambda t: T.h2h_pct(t, [x for x in times if x != t])[0] or 0.0))
    else:
        # WILD CARD com 3+: so varredura vale.
        sw = _varredura(T, times)
        if sw is not None:
            passos.append(("varredura direta", lambda t: sw[t]))

    # A ORDEM DIFERE, e a diferenca decide vaga. Na divisao: registro na
    # divisao, depois jogos comuns, depois conferencia. No wild card: a
    # CONFERENCIA vem logo apos o confronto direto, e jogos comuns so depois —
    # com minimo de quatro adversarios em comum.
    #
    # Ter isso trocado reprovava 2009 AFC: Baltimore e Nova York lideravam na
    # conferencia (0,5833 contra 0,5000 do Houston), mas os jogos comuns
    # entravam primeiro e punham o Houston na frente.
    com = T.comuns(times)
    passo_comuns = ("jogos comuns", lambda t: T._pct_contra(t, com)[0] or 0.0)

    if modo == "divisao":
        passos.append(("registro na divisão", T.pct_divisao))
        if len(com) >= 1:
            passos.append(passo_comuns)
        passos.append(("registro na conferência", T.pct_conferencia))
    else:
        passos.append(("registro na conferência", T.pct_conferencia))
        if len(com) >= 4:
            passos.append(passo_comuns)

    passos.append(("força de vitórias", T.forca_vitorias))
    passos.append(("força de calendário", T.forca_calendario))
    passos.append(("saldo de pontos", T.saldo))
    return passos


def desempatar(T, times, modo, sorteio=None):
    """Ordena um grupo empatado, do melhor para o pior.

    A regra do recomeço está aqui: quando um passo separa o grupo, os dois
    lados voltam ao passo 1 em vez de continuar descendo a lista.
    """
    times = list(times)
    if len(times) <= 1:
        return times
    for nome, nota in _passos(T, times, modo):
        valores = {t: nota(t) for t in times}
        melhor = max(valores.values())
        lideres = [t for t in times if valores[t] == melhor]
        if len(lideres) < len(times):
            resto = [t for t in times if t not in lideres]
            return (desempatar(T, lideres, modo, sorteio)
                    + desempatar(T, resto, modo, sorteio))
    # esgotou os critérios implementados: decide por sorteio reproduzível
    if sorteio is not None:
        return [t for _, t in sorted((sorteio.get(t, 0.0), t) for t in times)]
    return sorted(times)


def ordenar(T, times, modo, sorteio=None):
    """Ordena um conjunto qualquer: primeiro por aproveitamento, e os empates
    pela cascata."""
    grupos = {}
    for t in times:
        grupos.setdefault(round(T.pct(t), 10), []).append(t)
    saida = []
    for p in sorted(grupos, reverse=True):
        saida += desempatar(T, grupos[p], modo, sorteio)
    return saida


def semear(T, conferencia, sorteio=None, wildcards=3):
    """Devolve as cabeças da conferência, da 1 em diante."""
    divisoes = [d for d in DIVISOES if d.startswith(conferencia)]

    campeoes = []
    for d in divisoes:
        campeoes.append(ordenar(T, DIVISOES[d], "divisao", sorteio)[0])
    cabecas = ordenar(T, campeoes, "wildcard", sorteio)

    restantes = [t for t in TIMES
                 if CONF_DE[t] == conferencia and t not in campeoes]
    for _ in range(wildcards):
        cabecas.append(_melhor_wildcard(T, restantes, sorteio))
        restantes.remove(cabecas[-1])
    return cabecas


def _melhor_wildcard(T, candidatos, sorteio=None):
    """O melhor entre os candidatos, respeitando "um por divisão".

    Antes de comparar, cada divisão é reduzida ao seu melhor representante pelo
    critério de DIVISÃO. Sem esse passo, dois times da mesma divisão brigando
    pela mesma vaga saem na ordem errada.
    """
    por_div = {}
    for t in candidatos:
        por_div.setdefault(DIV_DE[t], []).append(t)
    finalistas = [ordenar(T, ts, "divisao", sorteio)[0] for ts in por_div.values()]
    return ordenar(T, finalistas, "wildcard", sorteio)[0]


def semear_temporada(jogos, sorteio=None, temporada=None):
    """Atalho: recebe os jogos de uma temporada e devolve as cabeças."""
    if temporada is None:
        temporada = int(jogos.season.iloc[0])
    wc = n_wildcards(temporada)
    T = Temporada(jogos.home_team.to_numpy(), jogos.away_team.to_numpy(),
                  jogos.home_score.to_numpy(), jogos.away_score.to_numpy())
    return {"AFC": semear(T, "AFC", sorteio, wc),
            "NFC": semear(T, "NFC", sorteio, wc)}, T
