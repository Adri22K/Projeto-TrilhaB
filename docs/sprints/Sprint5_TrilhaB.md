# Diário de Sprint 5 — Modelagem completa, limiar de decisão e seleção
**Período:** 12/10/2026 a 25/10/2026
**Trilha:** B — Qualidade do ar inadequada

**Equipe:** Equipe 2  
**Integrantes:** Adrielle Stollemberger RGM: 33948844 • Victor Almeida de Aquino RGM: 32901321 • Nicolas Santos Silva RGM: 32855885 • Samir Abdul Khalek RGM: 32657994 • João Pedro Garcia Almeida RGM: 32847629  
**Scrum Master do Sprint:** Nicolas Santos Silva  
**Repositório GitHub:** [Projeto-TrilhaB](https://github.com/Adri22K/Projeto-TrilhaB)  
**Board:** [Projeto Trilha B — Sprints](https://github.com/users/SamirAbdulKhalek/projects/2)

> Split da Sprint 2 e `ColumnTransformer` **congelado na Sprint 4**. No comparativo, Dummy e Naive Bayes reproduzem o pipeline da Sprint 4; a persistência reproduz o baseline temporal `estado_atual(t) → t+24h`. Todos entram na mesma regra de seleção pela validação e são medidos na mesma partição de teste. Pode **reproduzir** esses números no notebook desta sprint; **não** colar métricas da Sprint 3 (features antigas). Classificadores novos treinam no pipeline congelado.

### Contrato desta sprint

| | Artefato | Origem / destino |
|---|---|---|
| **Entra** | Split da Sprint 2 | Sprint 2 — **não mudar o teste** |
| **Entra** | Features finais + `ColumnTransformer` congelado + baselines retreinados (lift) | Sprint 4 |
| **Entra** | RFC (custo de FN) e dicionário v0.3 | Sprints 1 e 4 |
| **Sai** | Comparativo no pipeline final (baselines S4 + ≥2 modelos novos, mesma partição) | Entrega final |
| **Sai** | Limiar de decisão justificado pelo RFC | Entrega final |
| **Sai** | Modelo escolhido + caderno de experimentos + model card + dicionário v0.4 + `joblib` do Pipeline em `models/` | Entrega final / mostra |

**Não há sprint seguinte.** Dashboard, se houver, é extra e lê estes artefatos — não gera outro split.

- [x] Confirmei transformer da Sprint 4, split da Sprint 2 e que nenhuma métrica da Sprint 3 entrou no quadro

---

## 1. Revisão do split e da preparação

- [x] Mesmo split temporal da Sprint 2 (ou o recorte reaplicado e já usado nas Sprints 3–4)
- [x] `Pipeline`/`ColumnTransformer` da Sprint 4, `fit` só no treino
- [x] Vazamento reconfirmado com os atributos finais

**Corte, N treino/teste e resumo do transformer:**

- **Corte:** 14/04/2026 00:00 (Sprint 2). Treino: 14.016 linhas no CSV e 13.992 após a purga de 24 h (a mesma das Sprints 3 e 4). Teste: 3.480 horas (14/04 a 05/09/2026), com 299 positivas, **sem nenhuma alteração**.
- **Transformer:** o da Sprint 4, reconstruído com `sprint4_features.make_pipeline`: 25 colunas lidas de `config/params.yaml` (`sprint4.features_congeladas`), `SimpleImputer(median)` + `PowerTransformer(yeo-johnson)`, com `fit` só no treino. O script para com erro se a normalização ou a lista divergirem do que foi congelado.
- **Vazamento:** o teste automático da Sprint 4 foi reexecutado (6 instantes, features idênticas com e sem o futuro). A elevação solar é astronômica. O alvo usa `shift(-24)`, e as features usam só `t` e o passado.
- **Fonte das features:** `data/processed/treino_features_s4.csv` e `teste_features_s4.csv`, gerados na Sprint 4 e regenerados pelo pipeline completo, mantendo o contrato congelado.

## 2. Comparativo no pipeline final

- [x] Dummy, persistência e Naive Bayes **do pipeline da Sprint 4** no quadro (reproduzidos aqui; não usar números da Sprint 3)
- [x] No mínimo **dois** classificadores adicionais (ex.: regressão logística / Random Forest / XGBoost), com justificativa
- [x] Todos avaliados na **mesma** partição de teste
- [x] Hiperparâmetros documentados; se houver busca, ela usa só o treino (ou validação temporal dentro do treino) — nunca o teste
- [x] `class_weight="balanced"` (ou equivalente) é lícito; SMOTE / oversampling aleatório continua proibido
- [x] Independência do Naive Bayes continua tratada como hipótese pedagógica

**Modelos no comparativo (3 baselines da Sprint 4 + ≥2 novos):**
1. Dummy — `DummyClassifier(most_frequent)` no transformer da Sprint 4.
2. Persistência — regra `estado_atual(t) → t + 24 h`.
3. Naive Bayes — `GaussianNB` no transformer da Sprint 4. O script confere que ele reproduz o recall de 0,6254 e os 112 FN da Sprint 4.
4. **Regressão Logística** (`class_weight="balanced"`): modelo linear simples e interpretável. Diferente do NB, não supõe independência entre as features, o que importa porque elas são correlacionadas (lags e médias da mesma variável).
5. **Decision Tree** (`class_weight="balanced"`): gera regras legíveis. Partiu da configuração do experimento inicial do Nicolas (`min_samples_split=20`, `min_samples_leaf=15`).
6. **Random Forest** (`class_weight="balanced"`): captura interações não lineares (sol × calor × O₃) com menos variância que uma árvore. A grade inclui a configuração do Nicolas (`max_depth=12`, `min_samples_leaf=8`).
7. **HistGradientBoosting** (`class_weight="balanced"`): boosting de árvores, referência forte para dados tabulares.
8. **XGBoost**: boosting regularizado de árvores com ponderação da classe positiva e ajuste por AP temporal.

**Busca de hiperparâmetros (só no treino).** Grade pequena por modelo, avaliada pela **precisão média (AP)** nas 3 janelas de validação temporal da Sprint 4 (jul–set/2025, out–dez/2025 e jan–abr/2026), cada uma treinada só com o passado e com purga de 24 h. A AP não depende de limiar, então a busca não se mistura com a escolha do corte. Tabela completa em `reports/sprint5/busca_hiperparametros.csv`.

| Modelo | Grade | Escolhido (maior AP médio) | AP médio |
|---|---|---|---|
| Regressão Logística | C ∈ {0,01; 0,1; 1} | C = 0,01 | 0,622 |
| Decision Tree | max_depth ∈ {4, 6, 8} | max_depth = 4 | 0,553 |
| Random Forest (200 árvores) | (max_depth, min_samples_leaf) ∈ {(8, 20); (12, 8); (16, 20)} | (12, 8) | 0,6442 |
| HistGradientBoosting (300 iterações) | learning_rate ∈ {0,05; 0,1} × max_depth ∈ {3, 4} | 0,05 / 4 | 0,6148 |
| XGBoost (200 árvores) | learning_rate ∈ {0,03; 0,05} × max_depth ∈ {3, 4} | 0,03 / 4 | 0,6290 |
| Naive Bayes | sem hiperparâmetros | — | 0,607 |

O NB continua tratado como hipótese pedagógica: as features da mesma família são fortemente correlacionadas, o que viola a independência. É por isso que a Regressão Logística entrou como comparação direta.

## 3. Avaliação, limiar e custo do falso negativo

- [x] Recall, precisão, F1 e matriz de confusão por classe para todos, no `predict` padrão **e** no limiar escolhido
- [x] Recorte de **validação temporal no fim do treino** (ou `TimeSeriesSplit` só no treino)
- [x] Tabela ou curva de limiar (ex.: 0,3 / 0,5 / 0,7) na **validação**; o melhor ponto pelo custo de FN do RFC é congelado
- [x] Teste medido **uma vez** com esse limiar — não escolher o corte olhando o teste
- [x] Acurácia alta com recall baixo na classe positiva tratada como falha, se ocorrer
- [x] FN/FP também em **contagem**, não só em percentual

**Caderno de experimentos (validação e teste):**

| ID | Modelo | Features (S2/S4) | Limiar | Recall val. | Recall teste | Precisão teste | F1 teste | F2 teste | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| E01 | Dummy (most_frequent) | S4 (25, Yeo-Johnson) | 0,50 (predict) | — | 0,000 | 0,000 | 0,000 | 0,000 | 299 | 0 |
| E02 | Persistencia (regra estado(t) -> t+24h) | estado_atual | — | 0,604 | 0,485 | 0,480 | 0,482 | 0,484 | 154 | 157 |
| E03 | Naive Bayes (padrao) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,789 | 0,625 | 0,379 | 0,472 | 0,553 | 112 | 307 |
| E04 | Naive Bayes (padrao) | S4 (25, Yeo-Johnson) | 0,20 | 0,885 | 0,682 | 0,311 | 0,427 | 0,550 | 95 | 453 |
| E05 | Regressao Logistica (C=0.01) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,888 | 0,702 | 0,337 | 0,455 | 0,577 | 89 | 414 |
| E06 | Regressao Logistica (C=0.01) | S4 (25, Yeo-Johnson) | 0,50 | 0,888 | 0,702 | 0,337 | 0,455 | 0,577 | 89 | 414 |
| E07, rs=42 | Decision Tree (max_depth=4) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,816 | 0,739 | 0,335 | 0,461 | 0,596 | 78 | 438 |
| E08, rs=42 | Decision Tree (max_depth=4) | S4 (25, Yeo-Johnson) | 0,35 | 0,890 | 0,759 | 0,314 | 0,445 | 0,592 | 72 | 495 |
| E09, rs=42 | Random Forest (max_depth=12, min_samples_leaf=8) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,745 | 0,562 | 0,438 | 0,492 | 0,532 | 131 | 216 |
| E10, rs=42 | Random Forest (max_depth=12, min_samples_leaf=8) | S4 (25, Yeo-Johnson) | 0,35 | 0,863 | 0,662 | 0,375 | 0,479 | 0,574 | 101 | 330 |
| E11, rs=42 | HistGradientBoosting (learning_rate=0.05, max_depth=4) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,693 | 0,498 | 0,417 | 0,454 | 0,480 | 150 | 208 |
| E12, rs=42 | HistGradientBoosting (learning_rate=0.05, max_depth=4) | S4 (25, Yeo-Johnson) | 0,10 | 0,879 | 0,709 | 0,249 | 0,368 | 0,518 | 87 | 640 |
| E13, rs=42 | XGBoost (learning_rate=0.03, max_depth=4) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,753 | 0,562 | 0,414 | 0,477 | 0,524 | 131 | 238 |
| E14, rs=42 | XGBoost (learning_rate=0.03, max_depth=4) | S4 (25, Yeo-Johnson) | 0,20 | 0,897 | 0,729 | 0,276 | 0,400 | 0,549 | 81 | 573 |

Teste: 3.480 horas, 299 positivas. `rs=42` é o `random_state` dos modelos estocásticos; NB, Regressão Logística, Dummy e persistência são determinísticos. Random Forest e HistGradientBoosting podem variar na terceira casa decimal entre computadores (processamento paralelo e versões de bibliotecas); os números acima são os da execução versionada do notebook 05. NB, Regressão Logística e Decision Tree reproduzem exatamente. Recall val. = previsões fora da amostra nas 3 janelas (2.184 + 2.184 + 2.448 horas; 936 positivas). Tabelas completas em `reports/sprint5/caderno_experimentos.csv`, `comparativo_teste.csv` e `metricas_por_classe.csv`; matrizes em `reports/sprint5/matriz_*.png`.

**Acurácia × recall:** o Dummy tem 91,4 % de acurácia e recall zero (299 FN). Ele é tratado como falha e serve só de piso.

Anotar `random_state` na coluna ID ou numa nota abaixo. Persistência não tem limiar probabilístico — deixar “—” e avaliar só a classe persistida.

**Limiar escolhido (na validação) e justificativa:**

**Critério (RFC):** o falso negativo diz ao atleta que o ar estará bom quando estará ruim. Durante o esforço a ventilação pulmonar multiplica, e a dose inalada de poluente cresce junto. O falso positivo pode alterar a rotina de treino. Por isso o limiar de cada modelo é o que **maximiza o F2** da classe 1 (beta=2, favorecendo recall; beta²=4 na fórmula, sem equivalência automática a custo clínico) nas previsões fora da amostra das 3 janelas, numa grade de 0,05 a 0,95. Curva em `reports/sprint5/curva_limiar_validacao.png`.

Tabela do Naive Bayes na validação:

| Limiar | Recall | Precisão | F2 | FN | FP |
|---|---|---|---|---|---|
| 0,10 | 0,912 | 0,407 | 0,731 | 82 | 1.242 |
| **0,20** | **0,885** | **0,447** | **0,740** | **108** | **1.024** |
| 0,30 | 0,850 | 0,475 | 0,735 | 140 | 879 |
| 0,50 | 0,790 | 0,516 | 0,714 | 197 | 692 |
| 0,70 | 0,706 | 0,565 | 0,673 | 275 | 509 |

**Limiares congelados (retreino de 18/11/2026):** NB 0,20; Regressão Logística 0,50; Decision Tree 0,35; Random Forest 0,35; HistGradientBoosting 0,10; XGBoost 0,20.

**Regra de seleção revisada em 18/11/2026:** maior recall da classe 1 na validação temporal, considerando apenas alternativas com F2 ≥ F2 da persistência. Desempate por F2 e simplicidade. Cada limiar continua sendo escolhido pelo máximo F2 na validação. A revisão ocorre depois de conhecido o teste histórico: não constitui validação prospectiva independente.

| Modelo (limiar congelado) | Recall validação | F2 validação |
|---|---|---|
| **XGBoost (0,20)** | **89,74%** | 0,732 |
| Decision Tree (0,35) | 89,00% | 0,723 |
| Regressão Logística (0,50) | 88,78% | 0,744 |
| Naive Bayes (0,20) | 88,46% | 0,740 |
| HistGradientBoosting (0,10) | 87,93% | 0,718 |
| Random Forest (0,35) | 86,32% | 0,739 |
| Persistência | 60,36% | 0,606 |
| Dummy | 0,00% | 0,000 |

O XGBoost supera o piso F2 da persistência e tem o maior recall de validação: 840 das 936 horas positivas. Portanto é a escolha operacional segundo a regra revisada. Pela regra histórica de empate em F2, o NB era o vencedor.

## 4. Seleção do modelo, análise de erros e model card mínimo

- [x] Modelo final selecionado com justificativa (métrica da classe positiva + limiar + complexidade). **Não** escolher por acurácia nem pelo teste. Dummy e persistência participam da mesma seleção na validação e podem ser escolhidos se vencerem o critério pré-definido.
- [x] Análise qualitativa de erros concretos (datas/episódios, não só “erra mais na positiva”)
- [x] Model card mínimo preenchido abaixo
- [x] `joblib` (ou equivalente) do **Pipeline inteiro** em `models/` (pré-processamento + classificador + limiar documentado)
- [x] Dicionário conferido (features finais = as do modelo escolhido)

**Modelo final e justificativa:**

**XGBoost no pipeline da Sprint 4, com limiar 0,20.**

- **Métrica da classe positiva:** maior recall de validação (89,74%), F2 0,7316. No teste histórico: **recall 72,91%** (218 de 299 horas inadequadas), precisão 0,276, F1 0,400, F2 0,5486, **81 FN**, 573 FP e 2.608 TN.
- **Contra a persistência:** F2 de validação 0,7316 contra 0,6056. No teste, detecta 73 horas positivas a mais (81 FN contra 154), com F2 0,549 contra 0,484.
- **Limiar:** 0,20 reduz 50 FN frente ao `predict` padrão (131 → 81), ao custo de 335 FP a mais (238 → 573).
- **Complexidade:** boosting regularizado com 200 árvores, `learning_rate=0.03`, `max_depth=4`, `random_state=42`.
- **Transparência:** Decision Tree lidera no teste conhecido em recall (75,92%, 72 FN) e F2 (0,5918). Random Forest lidera em ROC-AUC (0,8957). Regressão Logística lidera em F2 de validação (0,7441). A decisão por recall de validação é operacional e necessita confirmação prospectiva.

**Principais erros analisados (FN/FP relevantes):**

Os erros foram agrupados em horas consecutivas (`reports/sprint5/episodios_erro_modelo_final.csv`): 81 FN em 17 grupos e 573 FP em 140 grupos. Dos 46 episódios positivos contíguos, cinco ficaram sem qualquer alerta.

**Padrões medidos:**
1. MP2,5: 60 FN entre 91 horas positivas associadas a esse poluente; ozônio: 21 FN entre 221 horas positivas. Há coocorrências. PM2,5 participa do alvo, mas não das 25 features finais.
2. Horários de pico em dias úteis: 84 TP, 18 FN e 143 FP (recall 82,35%; taxa FP 27,71%). Outras horas úteis: 85 TP, 43 FN e 255 FP (recall 66,41%; taxa FP 14,77%). Fim de semana: 49 TP, 20 FN e 175 FP (recall 71,01%). A diferença não prova causalidade de tráfego.
3. Os 573 FP mostram o custo operacional do limiar; alertas não devem ser tratados como garantia de qualidade do ar. A análise por data está nos CSVs regenerados.

### Model card mínimo

| Campo | Conteúdo |
|---|---|
| Problema / classe positiva | Prever se, **24 h depois**, o ar de São Paulo estará na classe **CETESB N3 (Ruim) ou pior** em pelo menos um poluente (PM10 e PM2,5 em 24 h, O₃ e CO em 8 h, NO₂ em 1 h, SO₂ em 24 h). Classe 1 = inadequado para corrida/ciclismo ao ar livre. |
| Unidade de análise e horizonte | Uma hora, no ponto central de São Paulo (−23,5489; −46,6388), hora local. Horizonte: 24 h (mesma hora do dia seguinte). |
| Split (treino / teste) | Temporal, corte em 14/04/2026. Treino: 06/09/2024–13/04/2026 (13.992 h após a purga de 24 h). Teste: 14/04–05/09/2026 (3.480 h; 299 positivas, 8,6 %). |
| Features (link do dicionário) | 25 features congeladas na Sprint 4: [`docs/Dicionario_de_Dados.md`](../Dicionario_de_Dados.md) (v0.4). |
| Algoritmo, hiperparâmetros, `random_state` e limiar (origem: validação) | `XGBClassifier` (`learning_rate=0.03`, `max_depth=4`, `random_state=42`) no pipeline congelado, envolvido em `FixedThresholdClassifier` com **limiar 0,20**, máximo F2 na validação temporal. Artefato: `models/modelo_final_s5.joblib`. |
| Métrica principal no teste (classe positiva) | **Recall 0,7291** (218/299); precisão 0,276; F1 0,400; F2 0,5486; **81 FN**, 573 FP. Persistência: recall 0,485, 154 FN. |
| O que o modelo **não** faz / limitações | Dados modelados (CAMS, ≈ 45 km) de um ponto da cidade; não mede tráfego nem a rua do atleta. 60 FN associados a PM2,5; 573 FP. Escores não calibrados, desvio entre reanálise e dados operacionais, ausência de validação prospectiva. Não substitui orientação médica nem alertas oficiais da CETESB. |
| Como reproduzir (`requirements.txt` + notebook/script) | `pip install -r requirements.txt` → `python src/sprint5_modelos.py` ou `notebooks/05_modelos.ipynb` (Restart → Run All). Usa `data/processed/*_s4.csv` e `config/params.yaml`. Model card completo: [`docs/Model_Card.md`](../Model_Card.md). |

## 5. Integração com Banco de Dados (opcional)

- [x] Necessidade de integração com Banco de Dados avaliada para poluentes, clima, previsões e apresentação do projeto
- [x] Situação atual registrada: não viabilizada nesta entrega por não ser necessária ao escopo

**Situação:** não viabilizada nesta entrega. O projeto é acadêmico, será apresentado por meio de um dashboard demonstrativo e não terá continuidade operacional depois da Mostra. O volume atual é atendido pelos CSVs versionados/gerados pelo pipeline, que também facilitam a reprodução dos experimentos. O dashboard deve consumir os CSVs e os artefatos finais (`joblib`/JSON) diretamente. Um banco de dados só acrescentaria complexidade de infraestrutura sem benefício para o objetivo desta entrega.

## 6. Scrum e diário de bordo

- [x] Board atualizado ([Projeto Trilha B — Sprints](https://github.com/users/SamirAbdulKhalek/projects/2))
- [x] Diário de bordo individual preenchido

| Integrante | O que fiz nesta Sprint | Dificuldades | O que pretendo manter/ajustar |
|---|---|---|---|
| Adrielle | Publiquei a página do projeto no GitHub Pages (`docs/index.html`, com identidade visual em `docs/assets/`) para apresentar o problema e os resultados na Mostra, e atualizei a estrutura do repositório no README | Ajustar os caminhos dos arquivos para o GitHub Pages funcionar a partir da pasta `docs/` | Atualizar a página com o modelo final e o model card desta sprint |
| Victor | Escrevi o artigo de divulgação sobre "inteligência atmosférica", que explica o problema da qualidade do ar para quem treina ao ar livre e o que o modelo entrega | Traduzir métricas como recall e falso negativo para uma linguagem que um atleta entenda | Revisar o artigo com os números finais (recall 0,68 e 95 FN) e os limites do modelo |
| Nicolas | Como Scrum Master, organizei o board. Implementei o primeiro experimento com Decision Tree e Random Forest (`class_weight="balanced"`, limiar por F2 na validação), cujos hiperparâmetros e a ideia do F2 foram reaproveitados no comparativo final | O experimento inicial usava as features da Sprint 2 e comparava vários limiares no teste. Foi preciso refazê-lo no transformer congelado e com o limiar fixado na validação | Trabalhar sempre a partir da `main` atualizada e em branch com PR, para não sobrescrever correções |
| Samir | Implementei o comparativo final (`src/sprint5_modelos.py` e notebook 05): 3 baselines reproduzidos da Sprint 4 e 4 modelos novos no pipeline congelado, busca de hiperparâmetros e limiar na validação temporal, regra de seleção fixada antes do teste, análise de erros por episódio, artefato `joblib` com o limiar e o model card | Resistir a trocar o modelo depois de ver que a Decision Tree foi melhor no teste; documentar por que isso seria vazamento | Na continuação, testar a família `pm2_5` e variáveis de previsão do tempo para os episódios de partículas |
| João Pedro | Fiz a revisão geral da Sprint 5, conferindo o comparativo dos modelos, os limiares definidos na validação temporal, as métricas da classe positiva e a justificativa para a escolha do Gaussian Naive Bayes como modelo final. Também revisei as evidências da própria Sprint 5, identifiquei divergências entre resultados exploratórios e resultados oficiais, corrigi descrições e números que não estavam alinhados com a entrega final e conferi o registro das atividades e pendências desta Sprint | A principal dificuldade foi separar os resultados exploratórios dos resultados oficiais da Sprint 5 e garantir que métricas, limiares, modelo final e justificativas estivessem coerentes nas evidências da própria Sprint | Manter a revisão cruzada das evidências da Sprint 5 antes do encerramento e deixar suas pendências registradas de forma clara |

---

## Rubrica de avaliação — Sprint 5 (nota de 0 a 4,0)

| Critério | Peso | O que caracteriza nota máxima | Nota atribuída | Observações |
|---|---|---|---|---|
| Revisão do split e da preparação | 0,5 | Split da Sprint 2 e transformer da Sprint 4, vazamento reconfirmado | | |
| Comparativo no pipeline final | 1,0 | Baselines retreinados + ≥2 modelos novos, mesma partição, hiperparâmetros documentados | | |
| Avaliação, limiar e custo de FN | 1,0 | Métricas por classe + limiar escolhido **na validação** ligado ao RFC; teste uma vez; FN/FP em contagem | | |
| Seleção, erros e model card | 1,0 | Justificativa clara (métrica da positiva, não acurácia), casos concretos, model card, `joblib` do Pipeline; baseline vencedor é aceito | | |
| Scrum + diário de bordo (e integração opcional com BD) | 0,5 | Board e diário refletindo o processo real da Sprint | | |
| **Nota final da Sprint 5** | **4,0** | | **___ / 4,0** | |

## Revisão de 18/11/2026

O retreino de 18/11/2026 inclui oito alternativas (dois controles e seis classificadores) e 14 experimentos. O XGBoost oficial tem limiar 0,20; a comparação exploratória usa ajuste próprio. A política revisada seleciona pelo maior recall de validação com piso F2 da persistência. Os registros individuais anteriores descrevem a política histórica. A reprodução em 18/11 não constitui nova avaliação independente. A unidade da matriz é hora positiva, não episódio contíguo. Recortes de pico são proxies de horário, não medição de tráfego.

Consulte a [spec de revisão](../../specs-dash/revisao_integral_sprints_modelos_dashboard.md) e o [relatório de correções](../../specs-dash/RELATORIO_REVISAO_CORRECOES_2026-11-18.md).
