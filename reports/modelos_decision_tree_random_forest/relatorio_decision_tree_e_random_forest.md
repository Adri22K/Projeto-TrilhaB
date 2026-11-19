# Comparacao reproduzivel: Decision Tree, Random Forest e XGBoost

> Relatorio de comparacao exploratoria. A selecao oficial de algoritmo ocorre na Sprint 5, por F2 na validacao temporal. O teste ja foi consultado em execucoes anteriores e nao constitui um novo conjunto cego para selecao.

- Treino purgado: 13992 observacoes; teste: 3480 horas, 299 positivas e 3181 negativas.
- Matriz X: 25 features congeladas em `config/params.yaml`, com dados ate o instante t; y = classificacao do ar em t+24h (CETESB N3+).
- Z: matriz numerica de X apos imputacao de medianas e transformacao Yeo-Johnson ajustadas somente no treino. Z nao e um alvo adicional.
- Os limiares foram escolhidos nas janelas de validacao (F2), sem escolha pelo teste.
- Fonte dos baselines: recalculo a partir de `teste_features_s4.csv`, conferido contra `reports/sprint4/comparativo_baselines_s4.csv`.

## Metricas no mesmo teste

| Modelo | Limiar | Acuracia | Precisao | Recall | F1 | F2 | ROC-AUC | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0.50 | 0.9141 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 299 | 0 |
| Persistencia Temporal | 0.50 | 0.9106 | 0.4801 | 0.4849 | 0.4825 | 0.4840 | 0.7178 | 154 | 157 |
| Decision Tree (Limiar Padrao 0.50) | 0.50 | 0.8517 | 0.3354 | 0.7391 | 0.4614 | 0.5957 | 0.8362 | 78 | 438 |
| Decision Tree (Limiar F2 0.35) | 0.35 | 0.8371 | 0.3144 | 0.7592 | 0.4447 | 0.5918 | 0.8362 | 72 | 495 |
| Random Forest (Limiar Padrao 0.50) | 0.50 | 0.8853 | 0.3967 | 0.6421 | 0.4904 | 0.5714 | 0.8986 | 107 | 292 |
| Random Forest (Limiar F2 0.30) | 0.30 | 0.8376 | 0.3168 | 0.7692 | 0.4488 | 0.5983 | 0.8986 | 69 | 496 |
| XGBoost (Limiar Padrao 0.50) | 0.50 | 0.8917 | 0.4089 | 0.5853 | 0.4814 | 0.5388 | 0.8812 | 124 | 253 |
| XGBoost (Limiar F2 0.20) | 0.20 | 0.7991 | 0.2674 | 0.7692 | 0.3969 | 0.5593 | 0.8812 | 69 | 630 |

## Limiar F2 escolhido exclusivamente na validacao

| Modelo | Limiar F2 | F2 na validacao |
|---|---:|---:|
| Decision Tree | 0.35 | 0.7233 | 
| Random Forest | 0.30 | 0.7373 | 
| XGBoost | 0.20 | 0.7277 | 

## Interpretacao e limites

- ROC-AUC mede ordenacao global, ao passo que recall, FN e F2 dependem do limiar.
- Horas consecutivas do mesmo episodio sao correlacionadas; nao equivalem a eventos independentes.
- O X historico e baseado em reanalise; a API futura usa previsoes meteorologicas: avaliar drift.
- Uma associacao do risco com horario de pico nao prova causalidade do trafego; necessita analise condicionada por poluentes, clima e origem dos dados.
- A selecao de 25 features na Sprint 4 usou F1, enquanto a selecao oficial da Sprint 5 usa F2; validar futura versao em teste cronologico independente.
- Os CSVs de treino/teste sao historicos. Somente a consulta da API ao vivo fornece previsoes atuais, com data da consulta e limite das fontes.

## Artefatos

- `models/pipeline_{decision_tree,random_forest,xgboost}.joblib`
- `models/limiares_modelos_arvore.json`
- `reports/modelos_decision_tree_random_forest/comparativo_completo_decision_tree_vs_random_forest.csv`
- `reports/sprint5/resultado_execucao.json` (selecao oficial independente)
