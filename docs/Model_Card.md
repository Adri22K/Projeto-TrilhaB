# Model Card — Previsão de ar inadequado para treino ao ar livre (São Paulo, 24 h)

**Projeto:** Trilha B, Equipe 2 · **Versão:** 1.0 (Sprint 5) · **Artefato:** `models/modelo_final_s5.joblib`

## Uso pretendido

Avisar corredores e ciclistas, com 24 h de antecedência, se a qualidade do ar em São Paulo deve estar **Ruim ou pior** (CETESB N3+) no mesmo horário do dia seguinte, para que possam trocar o horário ou treinar em ambiente fechado.

**Não é para:** decisões médicas, alertas oficiais de saúde pública, previsão por bairro ou por rua, e pessoas com doença respiratória que precisem de alerta já na faixa Moderada.

## Dados

| Item | Descrição |
|---|---|
| Fontes | Open-Meteo Air Quality (modelo CAMS Global, ≈ 45 km; **modelado**) e Open-Meteo Historical Weather (reanálise ERA5/IFS) |
| Local e período | ponto central de São Paulo (−23,5489; −46,6388), 06/09/2024–06/09/2026, horário, `America/Sao_Paulo` |
| Unidade de análise | uma hora |
| Alvo | `qualidade_inadequada_24h = 1` se, em `t + 24 h`, algum poluente estiver em N3+ na média CETESB (PM10/PM2,5/SO₂ 24 h, O₃/CO 8 h, NO₂ 1 h) |
| Split | temporal; treino até 13/04/2026 (13.992 h após a purga de 24 h; 13,96 % positivas); teste de 14/04 a 05/09/2026 (3.480 h; 299 positivas, 8,6 %) |
| Features | 25, congeladas na Sprint 4 — ver [`Dicionario_de_Dados.md`](Dicionario_de_Dados.md) |

## Modelo

`FixedThresholdClassifier(Pipeline(ColumnTransformer[SimpleImputer(median) → PowerTransformer(yeo-johnson)] → GaussianNB()), threshold=0.20)`

- **Seleção:** sete modelos no mesmo pipeline e no mesmo teste (Dummy, persistência, Naive Bayes, Regressão Logística, Decision Tree, Random Forest, HistGradientBoosting). Os hiperparâmetros foram escolhidos por AP e o limiar por máximo F2, ambos nas 3 janelas de validação temporal dentro do treino. Regra registrada antes do teste: maior F2 na validação, com empate (≤ 0,005) decidido pelo modelo mais simples. NB (0,740) e Regressão Logística (0,744) empataram, com o Random Forest logo atrás (0,739), e ficou o NB, o mais simples.
- **Limiar 0,20:** máximo F2 na validação. O F2 foi escolhido porque o falso negativo, que expõe o atleta, custa mais que o falso positivo, que só muda o horário (RFC).
- **`random_state`:** não se aplica, porque o modelo é determinístico.

## Desempenho no teste (medido uma vez)

| Modelo | Recall c1 | Precisão c1 | F1 c1 | F2 c1 | FN | FP |
|---|---|---|---|---|---|---|
| **NB, limiar 0,20 (final)** | **0,682** | 0,310 | 0,427 | 0,550 | **95** | 453 |
| NB, `predict` padrão | 0,625 | 0,379 | 0,472 | 0,553 | 112 | 307 |
| Persistência (piso) | 0,485 | 0,480 | 0,482 | 0,484 | 154 | 157 |
| Dummy | 0,000 | 0,000 | 0,000 | 0,000 | 299 | 0 |

A classe 0 (ar adequado) tem recall de 0,858: 453 das 3.181 horas boas foram marcadas como ruins. A acurácia (84 %) é menor que a do Dummy (91 %), e isso é esperado: a acurácia não mede o custo deste problema.

No teste, outros modelos tiveram F2 um pouco maior (Decision Tree 0,592; Random Forest 0,578; Regressão Logística 0,577). O modelo não foi trocado, porque escolher pelo teste seria vazamento de seleção. Comparativo completo em `reports/sprint5/comparativo_teste.csv`.

## Limitações e riscos

1. **Episódios de partículas (MP2,5) no inverno:** 61 dos 95 FN, como em 14–15/05/2026 e 09–10/07/2026. O treino é dominado por episódios de ozônio, e a família `pm2_5` saiu na seleção da Sprint 4.
2. **Início de episódio:** 75 % dos FN acontecem quando o ar em `t` ainda está bom.
3. **Alarmes falsos:** 453 h no teste, em geral tardes de dias Moderados depois de um dia ruim.
4. **Dado modelado e um ponto só:** não reflete picos locais (corredores de tráfego) nem diferenças entre bairros.
5. **Operação real:** o modelo foi treinado com reanálise meteorológica. Em produção, precisaria de previsão do tempo, o que tende a piorar o desempenho.
6. **Probabilidades não calibradas:** o NB supõe independência entre features correlacionadas, então a probabilidade não deve ser mostrada como "chance real".
7. **Período curto:** dois anos de dados, com cada estação vista só duas vezes.

## Como reproduzir

```powershell
pip install -r requirements.txt
python src/sprint5_modelos.py        # ou notebooks/05_modelos.ipynb (Restart → Run All)
```

Como usar o artefato:

```python
import joblib, pandas as pd, json
modelo = joblib.load("models/modelo_final_s5.joblib")
features = json.load(open("models/modelo_final_s5.json", encoding="utf-8"))["features"]
X = pd.read_csv("data/processed/teste_features_s4.csv")[features]
alerta = modelo.predict(X)          # 1 = ar inadequado previsto para t + 24 h (limiar 0,20 embutido)
```
