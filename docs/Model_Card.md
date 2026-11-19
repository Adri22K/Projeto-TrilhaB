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

`FixedThresholdClassifier(Pipeline(ColumnTransformer[SimpleImputer(median) → PowerTransformer(yeo-johnson)] → XGBClassifier), threshold=0.20)`

- **Comparativo e seleção:** oito alternativas foram avaliadas: Dummy, persistência, Naive Bayes, Regressão Logística, Decision Tree, Random Forest, HistGradientBoosting e XGBoost. Dummy e os seis classificadores usam as features/pipeline congelados da Sprint 4; a persistência usa sua regra temporal própria. Hiperparâmetros por AP e limiar por máximo F2 nas três janelas de validação dentro do treino. Política revisada: maior recall da classe 1 entre alternativas com F2 ≥ persistência (0,6056), desempate por F2 e simplicidade. XGBoost venceu com recall 0,8974 (840/936) e F2 0,7316. A política foi revisada após consulta histórica ao teste, logo não constitui seleção validada prospectivamente.
- **Limiar 0,20:** máximo F2 na validação. O F2 foi escolhido porque o falso negativo, que expõe o atleta, custa mais que o falso positivo, que pode alterar o horário ou a rotina de treino (RFC).
- **Hiperparâmetros:** `learning_rate=0.03`, `max_depth=4`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=0.8`, peso positivo derivado do treino e `random_state=42`.

## Desempenho no teste (protocolo congelado; retreino de 18/11/2026)

| Modelo | Recall c1 | Precisão c1 | F1 c1 | F2 c1 | FN | FP |
|---|---|---|---|---|---|---|
| **XGBoost, limiar 0,20 (final)** | **0,729** | 0,276 | 0,400 | 0,549 | **81** | 573 |
| XGBoost, `predict` padrão | 0,562 | 0,414 | 0,477 | 0,524 | 131 | 238 |
| Decision Tree, limiar 0,35 | 0,759 | 0,314 | 0,445 | 0,592 | 72 | 495 |
| NB, limiar 0,20 (final anterior) | 0,682 | 0,311 | 0,427 | 0,550 | 95 | 453 |
| Persistência (piso) | 0,485 | 0,480 | 0,482 | 0,484 | 154 | 157 |
| Dummy | 0,000 | 0,000 | 0,000 | 0,000 | 299 | 0 |

A classe 0 (ar adequado) tem recall de 0,820: 573 das 3.181 horas adequadas foram marcadas como inadequadas. O modelo detectou 218 das 299 horas positivas. A acurácia de aproximadamente 81,2% é menor que a do Dummy (91,4%).

No teste, Decision Tree obteve maior recall e F2; Random Forest, maior ROC-AUC (0,8957) e AP (0,4920); Regressão Logística liderou em F2 na validação. A decisão operacional de priorizar recall na validação não demonstra superioridade universal do XGBoost. Comparativo completo em `reports/sprint5/comparativo_teste.csv`.

## Limitações e riscos

1. **Partículas (MP2,5):** 60 FN em 91 horas positivas associadas ao PM2,5; 21 FN em 221 horas associadas ao ozônio (poluentes podem coexistir). A família `pm2_5` não está nas features da Sprint 4.
2. **Episódios:** 5 de 46 episódios positivos contíguos sem qualquer alerta; 81 horas FN distribuídas em 17 grupos de erro.
3. **Alarmes falsos:** 573 h no teste, em 140 grupos de erro. Pico em dias úteis: recall 82,35% e taxa FP 27,71%; outras horas úteis: recall 66,41% e taxa FP 14,77%. Horário é proxy, não medição de tráfego.
4. **Dado modelado e um ponto só:** não reflete picos locais (corredores de tráfego) nem diferenças entre bairros.
5. **Operação real:** o modelo foi treinado com reanálise meteorológica. Em produção, precisaria de previsão do tempo, o que tende a piorar o desempenho.
6. **Probabilidades não calibradas:** o escore XGBoost não deve ser mostrado como "chance real".
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

## Revisão de 18/11/2026

O XGBoost oficial substitui o NB pela política revisada de recall na validação. O XGBoost exploratório é um ajuste distinto. Reproduzir o teste conhecido não fornece confirmação prospectiva.

[Relatório de revisão](../specs-dash/RELATORIO_REVISAO_CORRECOES_2026-11-18.md).
