# Dados do projeto

Os arquivos de dados não são versionados no Git porque podem ser reproduzidos pelas APIs.

## Reproduzir a Sprint 1

Na raiz do repositório:

```powershell
python src/coleta_integracao.py
```

O comando lê `config/params.yaml` e cria:

- `data/raw/air_quality_*.json` e `.csv`;
- `data/raw/weather_*.json` e `.csv`;
- `data/raw/integrado.csv`;
- `data/raw/coleta_integracao_log.json`.

O bruto não é sobrescrito por padrão.

## Reproduzir a Sprint 2

```powershell
python src/prepara_sprint2.py
```

O comando cria:

- dados limpos e splits em `data/interim/`;
- matrizes com features em `data/processed/`;
- estatísticas, logs e gráficos em `reports/sprint2/`.
