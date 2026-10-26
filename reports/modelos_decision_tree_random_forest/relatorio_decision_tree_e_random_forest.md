# Avaliação e Comparativo: Decision Tree vs. Random Forest

## 1. Valores de Referência para Aprovação no Teste

Conforme estabelecido no RFC e no contrato de Engenharia de ML do projeto:
- **Custo Priorizado:** O **Falso Negativo (FN)** é o erro mais grave, pois classificar um horário como adequado quando está poluído (CETESB N3 - RUIM ou pior) induz a pessoa a praticar exercícios em condições prejudiciais à saúde respiratória e cardiovascular.
- **Piso 1 - Acurácia Ingênua (`DummyClassifier`):** 91,41%. Prediz apenas a classe majoritária (0), obtendo 0% de recall e **299 Falsos Negativos**. Não serve como modelo de previsão.
- **Piso 2 - Regra Operacional de Referência (`Persistência Temporal`):** É o baseline real da Sprint 3. Obtém **Recall de 48,49%**, **F1-Score de 0,4825** e **154 Falsos Negativos**.
- **Critério de Aprovação dos Modelos de ML:** Para justificar a complexidade de algoritmos de Machine Learning, o modelo deve obrigatoriamente superar a Persistência Temporal na classe positiva, alcançando **Recall > 48,49%** e reduzindo o teto de erro para **FN < 154**.

## 2. Quadro Comparativo Consolidado no Teste

| Modelo | Categoria | Limiar | Acurácia | Precisão (C1) | Recall (C1) | F1-Score (C1) | FN (Custo Alto) | FP | Status no Teste |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DummyClassifier | Piso de Acuracia | 0.50 | 91.41% | 0.0000 | 0.00% | 0.0000 | **299** | 0 | **Reprovado (Recall zero)** |
| Persistencia Temporal | Piso Real de Decisao | 0.50 | 91.06% | 0.4801 | 48.49% | 0.4825 | **154** | 157 | **Referencia a ser batida** |
| Decision Tree (Limiar Padrao 0.50) | Decision Tree | 0.50 | 85.92% | 0.3310 | 62.54% | 0.4329 | **112** | 378 | **APROVADO (Bateu Piso)** |
| Decision Tree (Limiar F2 0.65) | Decision Tree | 0.65 | 86.06% | 0.3140 | 52.51% | 0.3930 | **142** | 343 | **APROVADO (Bateu Piso)** |
| Random Forest (Limiar Padrao 0.50) | Random Forest | 0.50 | 90.83% | 0.4621 | 40.80% | 0.4334 | **177** | 142 | **Abaixo do Piso em FN** |
| Random Forest (Limiar F1 0.57) | Random Forest | 0.57 | 91.41% | 0.5000 | 33.44% | 0.4008 | **199** | 100 | **Abaixo do Piso em FN** |
| Random Forest (Limiar F2 0.35) | Random Forest | 0.35 | 88.94% | 0.3966 | 55.18% | 0.4615 | **134** | 251 | **APROVADO (Bateu Piso)** |

## 3. Diferenciação Entre os Modelos

### A. Decision Tree (Árvore de Decisão)
- **Conceito:** Modelo único e determinístico baseado em divisões ortogonais consecutivas do espaço de atributos.
- **Vantagens:** Alta interpretabilidade visual; permite extrair regras claras no formato `se [hora >= 11 e ozone > 60] então...`.
- **Desempenho no Teste:** Com limiar padrão (0,50), alcançou o maior Recall do projeto (**62,54%**) e a menor contagem de erros graves (**112 FNs**, eliminando 42 FNs em relação à persistência). Em contrapartida, gerou mais falsos positivos (378 FPs).

### B. Random Forest (Floresta Aleatória)
- **Conceito:** Ensemble de múltiplas Árvores de Decisão (neste projeto, 150 árvores) combinadas por *bagging* (amostragem com reposição) e seleção aleatória de atributos a cada divisão.
- **Vantagens:** Reduz drasticamente a variância e a tendência de sobreajuste de árvores isoladas, produzindo estimativas de probabilidade suaves e maior capacidade de generalização.
- **Desempenho no Teste:** Com limiar calibrado em $F_2 = 0,35$ para priorizar a contenção de FN, obteve **Recall de 55,18%**, **F1-Score de 0,4615** e reduziu os FNs para **134**, mantendo uma acurácia elevada de **88,94%** e muito menos alarmes falsos (251 FPs contra 378 da árvore).

## 4. Artefatos e Gráficos Gerados

- `models/pipeline_decision_tree.joblib`
- `models/pipeline_random_forest.joblib`
- `reports/modelos_decision_tree_random_forest/decision_tree/`
- `reports/modelos_decision_tree_random_forest/random_forest/`
- `reports/modelos_decision_tree_random_forest/comparativo_completo_decision_tree_vs_random_forest.csv`
