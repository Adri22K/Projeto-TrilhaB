# Avaliação e Comparativo: Decision Tree, Random Forest e XGBoost

## 1. Valores de Referência para Aprovação no Teste

Conforme estabelecido no RFC e no contrato de Engenharia de ML do projeto:
- **Dados Utilizados:** Dados tratados da **Sprint 5** (`treino_features_s4.csv` e `teste_features_s4.csv`), compostos pelas **25 features congeladas** preparadas pelo `ColumnTransformer` (mediana + Yeo-Johnson).
- **Custo Priorizado:** O **Falso Negativo (FN)** é o erro mais grave, pois classificar um horário como adequado quando está poluído (CETESB N3 - RUIM ou pior) induz a pessoa a praticar exercícios em condições prejudiciais à saúde respiratória e cardiovascular.
- **Piso 1 - Acurácia Ingênua (`DummyClassifier`):** 91,41%. Prediz apenas a classe majoritária (0), obtendo 0% de recall e **299 Falsos Negativos**. Não serve como modelo de previsão.
- **Piso 2 - Regra Operacional de Referência (`Persistência Temporal`):** É o baseline real da Sprint 3/4. Obtém **Recall de 48,49%**, **F1-Score de 0,4825** e **154 Falsos Negativos**.
- **Critério de Aprovação dos Modelos de ML:** Para justificar a complexidade de algoritmos de Machine Learning, o modelo deve obrigatoriamente superar a Persistência Temporal na classe positiva, alcançando **Recall > 48,49%** e reduzindo o teto de erro para **FN < 154**.

## 2. Quadro Comparativo Consolidado no Teste

| Modelo | Categoria | Limiar | Acurácia | Precisão (C1) | Recall (C1) | F1-Score (C1) | F2-Score (C1) | ROC-AUC | FN (Custo Alto) | FP | Status no Teste |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DummyClassifier | Piso de Acuracia | 0.50 | 91.41% | 0.0000 | 0.00% | 0.0000 | 0.0000 | 0.5000 | **299** | 0 | **Reprovado (Recall zero)** |
| Persistencia Temporal | Piso Real de Decisao | 0.50 | 91.06% | 0.4801 | 48.49% | 0.4825 | 0.4840 | 0.7178 | **154** | 157 | **Referencia a ser batida** |
| Decision Tree (Limiar Padrao 0.50) | Decision Tree | 0.50 | 85.17% | 0.3354 | 73.91% | 0.4614 | 0.5957 | 0.8362 | **78** | 438 | **APROVADO (Bateu Piso)** |
| Decision Tree (Limiar F2 0.35) | Decision Tree | 0.35 | 83.71% | 0.3144 | 75.92% | 0.4447 | 0.5918 | 0.8362 | **72** | 495 | **APROVADO (Bateu Piso)** |
| Random Forest (Limiar Padrao 0.50) | Random Forest | 0.50 | 89.14% | 0.4136 | 63.21% | 0.5000 | 0.5717 | 0.9006 | **110** | 268 | **APROVADO (Bateu Piso)** |
| Random Forest (Limiar F2 0.40) | Random Forest | 0.40 | 87.70% | 0.3776 | 66.56% | 0.4818 | 0.5775 | 0.9006 | **100** | 328 | **APROVADO (Bateu Piso)** |
| XGBoost (Limiar Padrao 0.50) | XGBoost | 0.50 | 89.20% | 0.4094 | 58.19% | 0.4807 | 0.5367 | 0.8869 | **125** | 251 | **APROVADO (Bateu Piso)** |
| XGBoost (Limiar F2 0.25) | XGBoost | 0.25 | 82.67% | 0.2968 | 74.25% | 0.4241 | 0.5710 | 0.8869 | **77** | 526 | **APROVADO (Bateu Piso)** |

## 3. Diferenciação Entre os Três Algoritmos de Árvores e Ensembles

### A. Decision Tree (Árvore de Decisão)
- **Conceito:** Modelo único e determinístico baseado em divisões ortogonais consecutivas do espaço de atributos.
- **Vantagens:** Alta interpretabilidade visual; permite extrair regras claras no formato `se [hora_cos >= -0.7 e ozone_media_8h > 115] então...`.
- **Desempenho no Teste (Sprint 5):** Com o limiar padrão (0,50), obteve **Recall de 73,91%** e **78 FNs** (eliminando 76 FNs em relação à persistência). Com limiar calibrado em $F_2 = 0,35$, alcançou **Recall de 75,92%** e **72 FNs**.

### B. Random Forest (Floresta Aleatória)
- **Conceito:** Ensemble de múltiplas Árvores de Decisão (200 árvores) combinadas por *bagging* (amostragem com reposição) e seleção aleatória de atributos a cada nó.
- **Vantagens:** Reduz a variância e a tendência de sobreajuste de árvores isoladas, produzindo estimativas de probabilidade suaves, menor taxa de alarmes falsos (apenas 268 FPs no limiar 0,50) e maior precisão.
- **Desempenho no Teste (Sprint 5):** Com limiar padrão (0,50), alcançou **Recall de 63,21%**, **F1-Score de 0,5000** e **110 FNs**. Com limiar calibrado em $F_2 = 0,40$, elevou o Recall para **66,56%** e reduziu os FNs para **100**, com **ROC-AUC de 0,9006**.

### C. XGBoost (Extreme Gradient Boosting)
- **Conceito:** Ensemble de *boosting* sequencial de árvores rasas, onde cada nova árvore corrige os erros residuais (gradientes) das árvores anteriores, com regularização L1/L2 e tratamento de desbalanceamento via `scale_pos_weight`.
- **Vantagens:** Alta precisão de ranking de probabilidade (**ROC-AUC de 0,8869** — a mais alta entre os modelos de árvore), excelente sensibilidade à calibração de ponto de operação e capacidade de capturar padrões sutis de interações não lineares entre clima e poluentes.
- **Desempenho no Teste (Sprint 5):**
  - **Limiar Padrão (0,50):** Acurácia de **89,20%**, **Recall de 58,19%**, **F1-Score de 0,4807**, reduzindo os FNs para **125** (eliminando 29 FNs em relação à persistência) com apenas 251 alarmes falsos.
  - **Limiar Calibrado em $F_2 = 0,25$ (Foco em Falso Negativo):** Alcança **Recall de 74,25%**, detectando **222 de 299 horas inadequadas** e derrubando os FNs para apenas **77** (redução de 50% dos erros graves da persistência).

## 4. Artefatos e Gráficos Gerados

- `models/pipeline_decision_tree.joblib`
- `models/pipeline_random_forest.joblib`
- `models/pipeline_xgboost.joblib`
- `reports/modelos_decision_tree_random_forest/decision_tree/`
- `reports/modelos_decision_tree_random_forest/random_forest/`
- `reports/modelos_decision_tree_random_forest/xgboost/`
- `reports/modelos_decision_tree_random_forest/comparativo_completo_decision_tree_vs_random_forest.csv`
- `reports/modelos_decision_tree_random_forest/metricas_por_classe.csv`
- `reports/modelos_decision_tree_random_forest/erros_detalhados_teste.csv`
