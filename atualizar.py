"""Atualiza tudo depois de uma rodada, numa tacada:

    python atualizar.py

Busca os jogos novos no nflverse, refaz a régua contra o mercado, simula a
temporada e regenera a página. Para no primeiro passo que falhar.

O motor de playoffs é revalidado a cada rodada, contra as 300 vagas históricas.
É barato e evita que uma mudança de código passe a semear errado em silêncio.

Antes de tudo, testes.py confere os invariantes que já quebraram — fuso, jogo
em andamento, placar parcial vazando, códigos de franquia e taxa de empate.
"""
import subprocess
import sys

PASSOS = [
    ("conferindo invariantes", [sys.executable, "testes.py"]),
    ("baixando jogos do nflverse", [sys.executable, "dados.py", "--refresh"]),
    ("revalidando o motor de playoffs", [sys.executable, "validar_playoffs.py"]),
    ("medindo contra o mercado", [sys.executable, "regua.py"]),
    ("simulando a temporada", [sys.executable, "simular.py", "--n", "20000"]),
    ("regerando a pagina", [sys.executable, "exportar.py"]),
]


def main():
    for i, (titulo, cmd) in enumerate(PASSOS, 1):
        print(f"\n=== {i}/{len(PASSOS)} {titulo} " + "=" * (44 - len(titulo)))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"\nParou no passo {i} ({titulo}). Nada depois dele foi refeito.")
            sys.exit(r.returncode)
    print("\n" + "=" * 60)
    print("pronto. abra web/index.html")


if __name__ == "__main__":
    main()
