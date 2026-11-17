"""Baixa os resultados das corridas de F1 e salva em data/raw/resultados.csv.

Uso:
    python src/coletar.py 2018 2026
"""
import sys
from pathlib import Path

import fastf1
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "data" / "cache"
SAIDA = RAIZ / "data" / "raw" / "resultados.csv"


def coletar_temporada(ano: int) -> list[dict]:
    """Retorna uma linha por piloto por corrida de uma temporada."""
    calendario = fastf1.get_event_schedule(ano, include_testing=False)
    linhas = []

    for _, evento in calendario.iterrows():
        data_evento = pd.Timestamp(evento["EventDate"]).tz_localize(None)
        if data_evento > pd.Timestamp.now():
            continue  # corrida ainda não aconteceu

        rodada = int(evento["RoundNumber"])
        try:
            sessao = fastf1.get_session(ano, rodada, "R")
            # só precisamos dos resultados, não da telemetria
            sessao.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as erro:
            print(f"  [aviso] {ano} rodada {rodada}: {erro}")
            continue

        for _, r in sessao.results.iterrows():
            linhas.append(
                {
                    "Ano": ano,
                    "Rodada": rodada,
                    "Circuito": evento["Location"],
                    "Piloto": r["Abbreviation"],
                    "Equipe": r["TeamName"],
                    "Largada": r["GridPosition"],
                    "Posicao": r["Position"],
                    "Status": r["Status"],
                    "Pontos": r["Points"],
                }
            )
        print(f"{ano} rodada {rodada} ok")

    return linhas


def main() -> None:
    ano_inicio = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
    ano_fim = int(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp.now().year

    CACHE.mkdir(parents=True, exist_ok=True)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    todas = []
    for ano in range(ano_inicio, ano_fim + 1):
        todas.extend(coletar_temporada(ano))

    pd.DataFrame(todas).to_csv(SAIDA, index=False)
    print(f"\nSalvo: {SAIDA} ({len(todas)} linhas)")


if __name__ == "__main__":
    main()