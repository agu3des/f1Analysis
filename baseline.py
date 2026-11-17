"""Baseline + primeiro modelo: prever se um piloto termina no pódio.

Uso:
    python src/baseline.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "raw" / "resultados.csv"

FEATURES = ["Largada", "forma_piloto", "forma_equipe"]


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Posicao", "Largada"]).copy()
    df = df.sort_values(["Ano", "Rodada"]).reset_index(drop=True)

    # Largada 0 = saída dos boxes; tratamos como último lugar
    df.loc[df["Largada"] == 0, "Largada"] = 20

    # Alvo: terminou no pódio?
    df["podio"] = (df["Posicao"] <= 3).astype(int)

    # Forma recente do piloto: média das 5 últimas posições (shift evita vazar a corrida atual)
    df["forma_piloto"] = df.groupby("Piloto")["Posicao"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Forma recente da equipe: média das posições dos carros nas 5 últimas corridas
    equipe = (
        df.groupby(["Equipe", "Ano", "Rodada"], as_index=False)["Posicao"]
        .mean()
        .sort_values(["Equipe", "Ano", "Rodada"])
    )
    equipe["forma_equipe"] = equipe.groupby("Equipe")["Posicao"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df = df.merge(
        equipe[["Equipe", "Ano", "Rodada", "forma_equipe"]],
        on=["Equipe", "Ano", "Rodada"],
        how="left",
    )

    # Primeiras corridas da história não têm passado: usa o meio do grid
    df[["forma_piloto", "forma_equipe"]] = df[["forma_piloto", "forma_equipe"]].fillna(10)
    return df


def main() -> None:
    df = preparar(pd.read_csv(ENTRADA))

    # Divisão temporal: treina no passado, testa na temporada mais recente
    ano_teste = df["Ano"].max()
    treino = df[df["Ano"] < ano_teste]
    teste = df[df["Ano"] == ano_teste]
    print(f"Treino: {len(treino)} linhas | Teste ({ano_teste}): {len(teste)} linhas\n")

    # Baseline: "quem larga no top 3 chega no pódio"
    pred_base = (teste["Largada"] <= 3).astype(int)
    print("== Baseline (largada no top 3) ==")
    print(f"Acurácia: {accuracy_score(teste['podio'], pred_base):.3f}")
    print(f"AUC:      {roc_auc_score(teste['podio'], -teste['Largada']):.3f}\n")

    # Modelo
    modelo = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    modelo.fit(treino[FEATURES], treino["podio"])
    prob = modelo.predict_proba(teste[FEATURES])[:, 1]

    print("== Random Forest ==")
    print(f"Acurácia: {accuracy_score(teste['podio'], prob > 0.5):.3f}")
    print(f"AUC:      {roc_auc_score(teste['podio'], prob):.3f}")
    print(f"Log loss: {log_loss(teste['podio'], prob):.3f}\n")

    print("Importância das features:")
    for nome, imp in sorted(zip(FEATURES, modelo.feature_importances_), key=lambda x: -x[1]):
        print(f"  {nome}: {imp:.3f}")


if __name__ == "__main__":
    main()