# Resultados de modelagem — Sprint 3 (calculados com CSV reais)

**Corte temporal:** 14/04/2026 00:00; **treino original:** 14016; **treino apos purga de 24h:** 13992; **teste:** 3480.

## Comparativo dos tres baselines

| Modelo | Acuracia | Precisao classe 1 | Recall classe 1 | F1 classe 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0.9141 | 0.0000 | 0.0000 | 0.0000 | 0 | 299 |
| Persistencia temporal | 0.9106 | 0.4801 | 0.4849 | 0.4825 | 157 | 154 |
| Gaussian Naive Bayes | 0.8626 | 0.2854 | 0.3980 | 0.3324 | 298 | 180 |

## Metricas por classe

| Modelo | Classe | Precisao | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 0 | 0.9141 | 1.0000 | 0.9551 | 3181 |
| DummyClassifier | 1 | 0.0000 | 0.0000 | 0.0000 | 299 |
| Persistencia temporal | 0 | 0.9515 | 0.9506 | 0.9511 | 3181 |
| Persistencia temporal | 1 | 0.4801 | 0.4849 | 0.4825 | 299 |
| Gaussian Naive Bayes | 0 | 0.9412 | 0.9063 | 0.9234 | 3181 |
| Gaussian Naive Bayes | 1 | 0.2854 | 0.3980 | 0.3324 | 299 |

## Interpretacao dos erros

O maior recall da classe 1 nesta comparacao foi o do modelo **Persistencia temporal** (0.4849), com **154 falsos negativos** e **157 falsos positivos**.
Falso negativo significa classificar como adequada (0) uma hora que estava inadequada (1), erro de maior custo para o projeto conforme o RFC da Sprint 1.
A acuracia do Dummy, que prediz a classe mais frequente, nao demonstra capacidade de detectar episodios inadequados.
O Naive Bayes e um baseline probabilistico e supoe independencia condicional dos atributos; PM2.5/PM10 e certas variaveis meteorologicas sao correlacionadas, portanto a hipotese e uma simplificacao.
Os tres modelos foram comparados no mesmo teste, sem SMOTE nem ajuste de limiar (Sprint 5). A analise dos FN pode orientar features da Sprint 4; nenhuma selecao de modelo final e feita aqui.

## Evidencias

- `reports/sprint3/comparativo_baselines.csv`
- `reports/sprint3/metricas_por_classe.csv`
- `reports/sprint3/erros_por_horario.csv`
- `reports/sprint3/matriz_dummy.png`
- `reports/sprint3/matriz_persistencia.png`
- `reports/sprint3/matriz_naive_bayes.png`
- `src/sprint3_baselines.py` e `notebooks/03_baseline.ipynb`
