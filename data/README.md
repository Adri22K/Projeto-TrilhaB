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

## Reproduzir a Sprint 3

```powershell
python src/sprint3_baselines.py
```

O comando lê `data/processed/` e `data/interim/integrado_limpo.csv` e grava métricas, matrizes de confusão e erros em `reports/sprint3/`. Não altera `data/`.

## Reproduzir a Sprint 4

```powershell
python src/sprint4_features.py
```

O comando cria `data/processed/treino_features_s4.csv` e `teste_features_s4.csv` (features da Sprint 2 + novas), salva o pipeline congelado em `models/` e grava erros, seleção, métricas e lift em `reports/sprint4/`. Não altera os arquivos das Sprints anteriores.

## Reproduzir a Sprint 5

```powershell
python src/sprint5_modelos.py
```

O comando lê `data/processed/*_s4.csv` e `config/params.yaml`, compara 7 modelos no pipeline congelado, escolhe hiperparâmetros e limiar na validação temporal, mede o teste uma vez e grava `reports/sprint5/` e `models/modelo_final_s5.*`. Não altera `data/`.
