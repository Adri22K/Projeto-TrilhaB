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
- **Fonte das features:** `data/processed/treino_features_s4.csv` e `teste_features_s4.csv`, gerados na Sprint 4 e não regenerados.

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

**Busca de hiperparâmetros (só no treino).** Grade pequena por modelo, avaliada pela **precisão média (AP)** nas 3 janelas de validação temporal da Sprint 4 (jul–set/2025, out–dez/2025 e jan–abr/2026), cada uma treinada só com o passado e com purga de 24 h. A AP não depende de limiar, então a busca não se mistura com a escolha do corte. Tabela completa em `reports/sprint5/busca_hiperparametros.csv`.

| Modelo | Grade | Escolhido (maior AP médio) | AP médio |
|---|---|---|---|
| Regressão Logística | C ∈ {0,01; 0,1; 1} | C = 0,01 | 0,622 |
| Decision Tree | max_depth ∈ {4, 6, 8} | max_depth = 4 | 0,553 |
| Random Forest (200 árvores) | (max_depth, min_samples_leaf) ∈ {(8, 20); (12, 8); (16, 20)} | (8, 20) | 0,643 |
| HistGradientBoosting (300 iterações) | learning_rate ∈ {0,05; 0,1} × max_depth ∈ {3, 4} | 0,05 / 3 | 0,610 |
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
| E09, rs=42 | Random Forest (max_depth=8, min_samples_leaf=20) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,805 | 0,632 | 0,414 | 0,500 | 0,572 | 110 | 268 |
| E10, rs=42 | Random Forest (max_depth=8, min_samples_leaf=20) | S4 (25, Yeo-Johnson) | 0,40 | 0,867 | 0,666 | 0,378 | 0,482 | 0,578 | 100 | 328 |
| E11, rs=42 | HistGradientBoosting (learning_rate=0.05, max_depth=3) | S4 (25, Yeo-Johnson) | 0,50 (predict) | 0,734 | 0,542 | 0,421 | 0,474 | 0,512 | 137 | 223 |
| E12, rs=42 | HistGradientBoosting (learning_rate=0.05, max_depth=3) | S4 (25, Yeo-Johnson) | 0,25 | 0,827 | 0,689 | 0,290 | 0,408 | 0,540 | 93 | 504 |

Teste: 3.480 horas, 299 positivas. `rs=42` é o `random_state` dos modelos estocásticos; NB, Regressão Logística, Dummy e persistência são determinísticos. Random Forest e HistGradientBoosting podem variar na terceira casa decimal entre computadores (processamento paralelo e versões de bibliotecas); os números acima são os da execução versionada do notebook 05. NB, Regressão Logística e Decision Tree reproduzem exatamente. Recall val. = previsões fora da amostra nas 3 janelas (2.184 + 2.184 + 2.448 horas; 936 positivas). Tabelas completas em `reports/sprint5/caderno_experimentos.csv`, `comparativo_teste.csv` e `metricas_por_classe.csv`; matrizes em `reports/sprint5/matriz_*.png`.

**Acurácia × recall:** o Dummy tem 91,4 % de acurácia e recall zero (299 FN). Ele é tratado como falha e serve só de piso.

Anotar `random_state` na coluna ID ou numa nota abaixo. Persistência não tem limiar probabilístico — deixar “—” e avaliar só a classe persistida.

**Limiar escolhido (na validação) e justificativa:**

**Critério (RFC):** o falso negativo diz ao atleta que o ar estará bom quando estará ruim. Durante o esforço a ventilação pulmonar multiplica, e a dose inalada de poluente cresce junto. O falso positivo só faz o treino mudar de horário. Por isso o limiar de cada modelo é o que **maximiza o F2** da classe 1 (recall com o dobro do peso da precisão) nas previsões fora da amostra das 3 janelas, numa grade de 0,05 a 0,95. Curva em `reports/sprint5/curva_limiar_validacao.png`.

Tabela do Naive Bayes na validação:

| Limiar | Recall | Precisão | F2 | FN | FP |
|---|---|---|---|---|---|
| 0,10 | 0,912 | 0,407 | 0,731 | 82 | 1.242 |
| **0,20** | **0,885** | **0,447** | **0,740** | **108** | **1.024** |
| 0,30 | 0,850 | 0,475 | 0,735 | 140 | 879 |
| 0,50 | 0,790 | 0,516 | 0,714 | 197 | 692 |
| 0,70 | 0,706 | 0,565 | 0,673 | 275 | 509 |

**Limiares congelados:** NB 0,20; Regressão Logística 0,50; Decision Tree 0,35; Random Forest 0,40; HistGradientBoosting 0,25.

**Regra de seleção, registrada antes do teste:** maior F2 na validação; empate (diferença ≤ 0,005) → modelo mais simples. A tolerância vem da estabilidade: o F2 do mesmo modelo varia de ~0,64 a ~0,80 entre as janelas (`reports/sprint5/validacao_por_janela.csv`), então milésimos no F2 médio não são sinal.

| Modelo (limiar congelado) | F2 validação |
|---|---|
| Regressão Logística (0,50) | 0,744 |
| **Naive Bayes (0,20)** | **0,740** |
| Random Forest (0,40) | 0,739 |
| Decision Tree (0,35) | 0,723 |
| HistGradientBoosting (0,25) | 0,707 |
| Persistência | 0,606 |
| Dummy | 0,000 |

NB e Regressão Logística empatam (diferença de 0,004, dentro da tolerância), e o Random Forest fica logo atrás (0,739). A regra escolhe o mais simples dos empatados: **Naive Bayes com limiar 0,20**.

## 4. Seleção do modelo, análise de erros e model card mínimo

- [x] Modelo final selecionado com justificativa (métrica da classe positiva + limiar + complexidade). **Não** escolher por acurácia nem pelo teste. Dummy e persistência participam da mesma seleção na validação e podem ser escolhidos se vencerem o critério pré-definido.
- [x] Análise qualitativa de erros concretos (datas/episódios, não só “erra mais na positiva”)
- [x] Model card mínimo preenchido abaixo
- [x] `joblib` (ou equivalente) do **Pipeline inteiro** em `models/` (pré-processamento + classificador + limiar documentado)
- [x] Dicionário conferido (features finais = as do modelo escolhido)

**Modelo final e justificativa:**

**Gaussian Naive Bayes no pipeline da Sprint 4, com limiar 0,20.**

- **Métrica da classe positiva:** ficou empatado em primeiro no F2 de validação (0,740 contra 0,744 da Regressão Logística) e foi escolhido pela regra de complexidade. No teste: **recall 0,682** (204 de 299 horas ruins detectadas), precisão 0,311, F1 0,427, F2 0,550, **95 FN** e 453 FP.
- **Contra a persistência (piso real):** na validação, a persistência teve F2 de 0,606, abaixo do NB (0,740), e por isso não entrou no grupo de empate que definiu o vencedor. No teste final, o NB detectou 59 horas ruins a mais (95 FN contra 154), com F2 de 0,550 contra 0,484; esses números servem apenas para medir o desempenho após a seleção.
- **Limiar:** 0,20 tira 17 FN em relação ao `predict` padrão (112 → 95), ao custo de 146 FP a mais. A troca é coerente com o custo do FN no RFC.
- **Complexidade:** não tem hiperparâmetros, treina em menos de 1 s e é fácil de explicar na Mostra.
- **Transparência sobre o teste:** no teste, a Decision Tree (F2 0,592; 72 FN), o Random Forest (0,578) e a Regressão Logística (0,577) ficaram acima do NB (0,550). **Não trocamos o modelo por isso:** escolher pelo teste seria usar o teste para seleção, o que o contrato proíbe. As diferenças (≈ 0,03–0,04 de F2) também estão dentro da variação entre janelas observada na validação (~0,15).

**Principais erros analisados (FN/FP relevantes):**

Os erros foram agrupados em episódios de horas consecutivas (`reports/sprint5/episodios_erro_modelo_final.csv`): 95 horas de FN em 20 episódios e 453 horas de FP em 111 episódios.

| Episódio (hora prevista) | Tipo | Horas | O que aconteceu | Por que o modelo errou |
|---|---|---|---|---|
| 09/07/2026 21 h → 10/07 13 h | FN | 17 | MP2,5 (24 h) chegou a 68,5 µg/m³; O₃ só 117 | episódio de **partículas no inverno**; probabilidade média 0,015 |
| 14/05/2026 21 h → 15/05 12 h | FN | 16 | MP2,5 (24 h) 69,4 µg/m³; O₃ baixo (78) | mesmo regime: inversão térmica de outono, sem ozônio |
| 14/05/2026 04 h → 12 h | FN | 9 | MP2,5 (24 h) 68,0 µg/m³ | início do mesmo episódio |
| 26/07/2026 15 h → 21 h | FN | 7 | O₃ (8 h) chegou a 175 µg/m³ | **início de episódio de ozônio**: no dia anterior o O₃ de 8 h estava em ~59, sem sinal de piora |
| 21/08/2026 11 h → 21 h | FP | 11 | O₃ (8 h) máximo de 106 (Moderado) | dia seguinte a um dia ruim (O₃ de 8 h ~137 em `t`), mas o episódio terminou |
| 29/08/2026 12 h → 21 h | FP | 10 | O₃ (8 h) máximo de 128,9 | "quase ruim": ficou 1 µg/m³ abaixo do limite |

**Padrões:**
1. **61 das 95 horas de FN (64 %)** são episódios de **MP2,5** em maio e julho. No treino quase todos os positivos são de ozônio, e a família `pm2_5` saiu na seleção da Sprint 4, ficando só o PM10. O modelo praticamente não "vê" episódios de partículas.
2. **75 % dos FN** acontecem quando o ar em `t` estava bom (início de episódio). É o limite de prever 24 h à frente só com o passado.
3. **Os FP** se concentram nas tardes depois de dias de ozônio alto. Em 39 % deles, o O₃ de 8 h previsto ficou acima de 100 (faixa Moderada). Para o atleta, é um alarme "barato".

### Model card mínimo

| Campo | Conteúdo |
|---|---|
| Problema / classe positiva | Prever se, **24 h depois**, o ar de São Paulo estará na classe **CETESB N3 (Ruim) ou pior** em pelo menos um poluente (PM10 e PM2,5 em 24 h, O₃ e CO em 8 h, NO₂ em 1 h, SO₂ em 24 h). Classe 1 = inadequado para corrida/ciclismo ao ar livre. |
| Unidade de análise e horizonte | Uma hora, no ponto central de São Paulo (−23,5489; −46,6388), hora local. Horizonte: 24 h (mesma hora do dia seguinte). |
| Split (treino / teste) | Temporal, corte em 14/04/2026. Treino: 06/09/2024–13/04/2026 (13.992 h após a purga de 24 h). Teste: 14/04–05/09/2026 (3.480 h; 299 positivas, 8,6 %). |
| Features (link do dicionário) | 25 features congeladas na Sprint 4: [`docs/Dicionario_de_Dados.md`](../Dicionario_de_Dados.md) (v0.4). |
| Algoritmo, hiperparâmetros, `random_state` e limiar (origem: validação) | `GaussianNB` (parâmetros padrão, determinístico) dentro de `Pipeline(ColumnTransformer[SimpleImputer(median) → PowerTransformer(yeo-johnson)] → GaussianNB)`, envolvido em `FixedThresholdClassifier` com **limiar 0,20**, escolhido por máximo F2 na validação temporal (3 janelas dentro do treino). Artefato: `models/modelo_final_s5.joblib`. |
| Métrica principal no teste (classe positiva) | **Recall 0,682** (204/299); precisão 0,311; F1 0,427; F2 0,550; **95 FN**, 453 FP. Persistência: recall 0,485, 154 FN. |
| O que o modelo **não** faz / limitações | Não mede a rua do atleta: os dados são **modelados** (CAMS, ≈ 45 km) e há um ponto para a cidade inteira. Erra a maioria dos **episódios de MP2,5 de inverno** (64 % dos FN). Erra inícios de episódio sem sinal no dia anterior. Gera muitos alarmes falsos (453 h no teste), em geral em dias Moderados. Suas probabilidades **não** são calibradas (o NB supõe independência). Na operação real, precisaria de previsão do tempo, e não de reanálise. Não substitui orientação médica nem os alertas oficiais da CETESB. |
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
