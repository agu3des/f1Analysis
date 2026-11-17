# f1Analysis
# Previsões de Fórmula 1

Projeto de estudo: prever quem termina no pódio de uma corrida.

## Estrutura

```
f1-previsoes/
├── requirements.txt
├── src/
│   ├── coletar.py    # baixa resultados via FastF1 -> data/raw/resultados.csv
│   └── baseline.py   # baseline + Random Forest com validação temporal
└── data/             # criada automaticamente (cache e CSVs)
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/coletar.py 2018 2026  # a primeira execução demora (cache é criado)
python src/baseline.py
```

## Próximos passos

1. Explorar os dados em um notebook (EDA)
2. Novas features: histórico no circuito, taxa de abandono, posição na classificação, clima
3. Testar XGBoost/LightGBM e regressão logística
4. Calibrar probabilidades e avaliar por corrida
5. Dashboard com Streamlit ou API com Django/FastAPI