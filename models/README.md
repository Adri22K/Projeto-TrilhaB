# Modelos

| Arquivo | Sprint | Conteúdo |
|---|---|---|
| `column_transformer_s4.joblib` | 4 | `ColumnTransformer` **congelado** (mediana + Yeo-Johnson), ajustado só no treino. A Sprint 5 é obrigada a usá-lo. |
| `features_s4.json` | 4 | Lista congelada das 25 colunas e a normalização escolhida |
| `naive_bayes_s4.joblib` | 4 | `Pipeline` completo (transformer + `GaussianNB`) retreinado com as features da Sprint 4 |

Para recriar: `python src/sprint4_features.py`.
