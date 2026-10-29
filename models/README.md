# Modelos

| Arquivo | Sprint | Conteúdo |
|---|---|---|
| `column_transformer_s4.joblib` | 4 | `ColumnTransformer` **congelado** (mediana + Yeo-Johnson), ajustado só no treino. A Sprint 5 é obrigada a usá-lo. |
| `features_s4.json` | 4 | Lista congelada das 25 colunas e a normalização escolhida |
| `naive_bayes_s4.joblib` | 4 | `Pipeline` completo (transformer + `GaussianNB`) retreinado com as features da Sprint 4 |

| `modelo_final_s5.joblib` | 5 | **Modelo final**: `FixedThresholdClassifier` (limiar 0,20) envolvendo o Pipeline completo (transformer da Sprint 4 + `GaussianNB`). `predict` já aplica o limiar. |
| `modelo_final_s5.json` | 5 | Metadados: modelo, limiar e origem, features, `random_state`, versão do sklearn |
| `pipeline_decision_tree.joblib`, `pipeline_random_forest.joblib` | 5 (exploratório) | Experimento inicial com as features da Sprint 2. **Não** fazem parte do comparativo final, porque não usam o transformer congelado. |

Para recriar: `python src/sprint4_features.py` (Sprint 4) e `python src/sprint5_modelos.py` (Sprint 5).
