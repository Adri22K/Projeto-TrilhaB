# Diário de Sprint 4 — Iteração de atributos, retreino do baseline e pipeline congelado
**Período:** 05/10/2026 a 11/10/2026
**Trilha:** B — Qualidade do ar inadequada

**Equipe:** Equipe 2  
**Integrantes:** Adrielle Stollemberger RGM: 33948844 • Victor Almeida de Aquino RGM: 32901321 • Nicolas Santos Silva RGM: 32855885 • Samir Abdul Khalek RGM: 32657994 • João Pedro Garcia Almeida RGM: 32847629  
**Scrum Master do Sprint:** Nicolas Santos Silva  
**Repositório GitHub:** [Projeto-TrilhaB](https://github.com/Adri22K/Projeto-TrilhaB)

> Sprint 1: coleta bruta. Sprint 2: limpeza, split, alvo e features iniciais. Sprint 3: baselines. Esta sprint **evolui** as features a partir dos erros do baseline, **retreina Dummy + persistência + Naive Bayes no mesmo split** e congela o `ColumnTransformer` para a Sprint 5. Sem lift documentado, não há evolução — só colunas novas.

### Contrato desta sprint

| | Artefato | Origem / destino |
|---|---|---|
| **Entra** | Split da Sprint 2, features da Sprint 2, erros e métricas da Sprint 3 | Sprints 2 e 3 |
| **Entra** | `ColumnTransformer` inicial da Sprint 3 | Sprint 3 |
| **Sai** | Features **novas** (além das da Sprint 2), com justificativa pelos erros da Sprint 3 | Sprint 5 |
| **Sai** | `ColumnTransformer` **congelado** (`fit` só no treino) | Sprint 5 **é obrigada a usar este** |
| **Sai** | Dummy + persistência + NB **retreinados** + tabela de lift S3→S4 | Sprint 5 (piso do comparativo; **não colar métricas da Sprint 3**) |
| **Sai** | Dicionário v0.3 | Sprint 5 |

**Não sai daqui:** seleção do modelo final entre classificadores novos, limiar de decisão, model card.

- [x] Confirmei o mesmo split da Sprint 2 e que as features “novas” não são as da Sprint 2 renomeadas

---

## 1. Engenharia de atributos iterada

- [x] Atributos **novos** em relação à Sprint 2 (não são os mesmos com outro nome)
- [x] Motivação ligada aos erros da Sprint 3 (ex.: falsos negativos em determinado regime de poluente/clima)
- [x] Seleção/redução de atributos avaliada **só no treino** (não no teste nem no dataset inteiro) e justificada
- [x] Vazamento revisado em **todos** os atributos (antigos e novos); parâmetros ainda definidos só no treino
- [x] Cada atributo novo descrito e justificado individualmente
- [x] Dicionário (seção de derivados) atualizado

**Descrição dos atributos criados/selecionados e de quais erros da Sprint 3 eles atacam:**

**Onde os erros foram analisados.** Para que o teste não escolha features, os baselines da Sprint 3 foram retreinados com dados até 31/12/2025 e avaliados de 01/01/2026 até o corte (validação interna, dentro do treino; 2.448 horas, 330 positivas). Os padrões batem com os FN da Sprint 3 já publicados no teste.

| Erro observado (validação interna) | Evidência |
|---|---|
| (1) FN do Naive Bayes concentrados no fim da tarde e à noite | 123 FN, mediana às 19 h; nas horas perdidas o O₃ instantâneo já caiu (mediana 81 µg/m³), mas a **média de 8 h**, que define o alvo, segue perto do limite (127 µg/m³) |
| (2) Início de episódio | 98 das 330 horas positivas (30 %) começam um episódio; persistência acerta 0 % e o NB 30 % |
| (3) Hora tratada como número linear | o NB gaussiano supõe uma curva em sino por classe; 23 h e 0 h ficam distantes |
| (4) Redundância | PM10 × PM2,5 com correlação 0,97; cada variável aparece 4 vezes; o NB "conta duas vezes" a mesma evidência |

| Feature nova | Definição (só `t` e passado) | Erro que ataca |
|---|---|---|
| `ozone_media_8h` | média do O₃ nas 8 h até `t` | (1) métrica CETESB do alvo |
| `ozone_max_8h` | máximo do O₃ nas últimas 8 h | (1) pico da tarde após o pôr do sol |
| `ozone_8h_max_24h` | maior média de 8 h nas últimas 24 h | (2) intensidade do dia |
| `ozone_8h_max_24h_delta` | índice diário de hoje − de ontem | (2) episódio em formação |
| `horas_ruins_24h` | horas em N3+ nas últimas 24 h | (2) duração do episódio |
| `temperature_max_24h` | máxima de temperatura em 24 h | (2) calor acelera a formação de O₃ |
| `temperature_max_24h_delta` | máxima de hoje − de ontem | (2) aquecimento em curso |
| `hora_sin`, `hora_cos` | hora no círculo de 24 h | (3) ciclo contínuo |
| `mes_sin`, `mes_cos` | mês no círculo de 12 meses | (3) dez ↔ jan vizinhos |
| `elevacao_solar_alvo` | altura do sol na hora prevista (`t + 24 h`), fórmula astronômica da NOAA | (1)(2) O₃ é fotoquímico; a posição do sol é conhecida com antecedência |

**Vazamento:** todas as janelas terminam em `t`. A elevação solar depende só de data, hora e coordenada. Um teste automático recalcula as features apagando o futuro em 6 instantes e confirma valores idênticos. As 39 features da Sprint 2 já usavam `shift(1)` nas médias. A imputação e a normalização são ajustadas só no treino.

**Seleção (só no treino):** validação temporal com 3 janelas dentro do treino (jul–set/2025, out–dez/2025, jan–abr/2026), cada uma treinada só com o passado e com purga de 24 h. Partindo de 39 + 12 = 51 colunas, removeu-se a **família** cuja retirada mais aumentava o F1 médio da classe 1, até não haver ganho. O F1 foi usado, e não só o recall, porque sem ajuste de limiar (Sprint 5) um conjunto pode "ganhar recall" marcando tudo como ruim. Histórico completo em `reports/sprint4/selecao_validacao.csv`.

| Configuração (validação, média de 3 janelas) | Recall | F1 |
|---|---|---|
| 39 features S2 + StandardScaler (Sprint 3) | 0,463 | 0,403 |
| 39 features S2 + Yeo-Johnson | 0,644 | 0,458 |
| 21 features selecionadas + StandardScaler | 0,764 | 0,612 |
| **25 features selecionadas + Yeo-Johnson (congelado)** | **0,786** | **0,619** |

**Mantidas (25):** famílias de PM10, SO₂, precipitação e vento (valor, lag 1 h, lag 24 h, média 24 h); `ozone_media_8h`, `ozone_max_8h`, `temperature_max_24h`, `temperature_max_24h_delta`, `hora_sin/cos`, `mes_sin/cos`, `elevacao_solar_alvo`.
**Descartadas e por quê:**
- **O₃ bruto:** substituído pela janela de 8 h, na escala do alvo.
- **Temperatura bruta:** redundante com a máxima de 24 h e com a elevação solar.
- **PM2,5, NO₂ e CO:** correlacionados com o PM10.
- **Calendário bruto:** substituído pela versão cíclica; `dia_semana` não ajudou porque o alvo é dominado pelo O₃, que depende de sol e não do tráfego.
- **Pico diário e `horas_ruins_24h`:** informação já contida nas features de 8 h.

**Evidências:** [`src/sprint4_features.py`](../../src/sprint4_features.py), [`notebooks/04_features_iteracao.ipynb`](../../notebooks/04_features_iteracao.ipynb), [`reports/sprint4/`](../../reports/sprint4/), [`docs/Dicionario_de_Dados.md`](../Dicionario_de_Dados.md) (v0.3).

## 2. Pipeline de preparação congelado

- [x] Um `Pipeline` sklearn: `ColumnTransformer` **e** o classificador no mesmo objeto; lista de colunas **congelada** (a da seleção no treino)
- [x] Estratégias de imputação, encoding e normalização justificadas (não só o default do sklearn)
- [x] `fit` confirmado apenas no treino
- [x] Este Pipeline/transformer é o que a Sprint 5 vai usar. Não “descongelar” na Sprint 5; se o PO pedir mudança aqui, retreinar de novo os três baselines antes de entregar.

**Descrição da preparação final:**

| Etapa | Estratégia | Justificativa |
|---|---|---|
| Imputação | `SimpleImputer(strategy="median")` | só as primeiras ~48 h do histórico têm NaN por construção (janelas e deltas de 24 h); mediana é robusta às caudas longas dos poluentes |
| Encoding | hora e mês em seno/cosseno; nenhuma categórica | preserva o ciclo com 2 colunas, sem one-hot de 24 níveis |
| Normalização | `PowerTransformer(method="yeo-johnson", standardize=True)` | aproxima cada feature de uma normal (hipótese do `GaussianNB`); aceita zeros de chuva e vento; venceu o `StandardScaler` na validação temporal (F1 0,619 × 0,612; recall 0,786 × 0,764) |

O `fit` acontece só nas 13.992 linhas de treino (mesma purga de 24 h da Sprint 3). Artefatos: `models/column_transformer_s4.joblib` (transformer ajustado), `models/naive_bayes_s4.joblib` (Pipeline completo) e `models/features_s4.json`. A lista e a normalização também estão em `config/params.yaml` (`sprint4`). Se a seleção reexecutada divergir delas, o script para com erro. A persistência usa um Pipeline próprio com a coluna `estado_atual`, porque é uma regra e não consome as features.

## 3. Retreino do baseline e lift

- [x] Dummy, persistência e Naive Bayes **retreinados** com as features desta sprint, **no mesmo split** da Sprint 2/3
- [x] Métricas por classe (recall, precisão, F1, matriz) dos três, **depois** da iteração
- [x] Tabela de delta em relação à Sprint 3 (pelo menos recall e F1 da classe positiva)
- [x] Se não houve ganho: hipótese do porquê, e o que foi descartado

**Tabela de lift (classe positiva):**

| Modelo | Recall S3 | Recall S4 | F1 S3 | F1 S4 | Delta recall |
|---|---|---|---|---|---|
| Dummy | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |
| Persistência | 0,485 | 0,485 | 0,483 | 0,483 | 0,000 |
| Naive Bayes | 0,398 | **0,625** | 0,332 | **0,472** | **+0,227** |

| Modelo | FN S3 | FN S4 | FP S3 | FP S4 | Precisão S4 | Acurácia S4 |
|---|---|---|---|---|---|---|
| Dummy | 299 | 299 | 0 | 0 | 0,000 | 0,914 |
| Persistência | 154 | 154 | 157 | 157 | 0,480 | 0,911 |
| Naive Bayes | 180 | **112** | 298 | 307 | 0,379 | 0,880 |

Teste: 3.480 horas de 14/04 a 05/09/2026, 299 positivas, medido **uma vez**. Matrizes de confusão em `reports/sprint4/matriz_*_s4.png`; métricas por classe em `reports/sprint4/metricas_por_classe_s4.csv`.

**Interpretação do lift (ou da ausência de lift):**

- **Naive Bayes:** lift real. O recall subiu 0,23 e o F1 subiu 0,14, com 68 falsos negativos a menos e só 9 falsos positivos a mais. O ganho veio exatamente do regime apontado pela análise de erros: os FN das 17 h às 19 h caíram de 91 para 22 (`reports/sprint4/fn_teste_por_hora_s4.csv`). As features de 8 h deram ao modelo a informação de que o episódio de ozônio continua mesmo depois do pôr do sol.
- **Persistência:** delta zero, como esperado. É uma regra (classe de `t` copiada para `t + 24 h`) que não usa features; foi reexecutada no mesmo split e segue como piso.
- **Dummy:** delta zero, como esperado. Prevê sempre a classe majoritária. Continua mostrando que acurácia (91 %) não mede este problema.
- **NB × persistência:** o NB agora detecta mais horas ruins (112 FN contra 154), mas com quase o dobro de alarmes falsos (307 contra 157), e o F1 dos dois fica empatado (0,47 × 0,48). A escolha entre eles depende do limiar e do custo de FN do RFC, que é tarefa da Sprint 5.
- **Limitação:** o ganho na validação (F1 0,40 → 0,62) foi maior que no teste (0,33 → 0,47). O teste (abr–set) é outono/inverno, com menos episódios de ozônio, e a seleção gulosa pode ter se ajustado um pouco às janelas de validação. Os erros de início de episódio continuam sendo o ponto fraco. Candidatas para a próxima coleta versionada: radiação solar (`shortwave_radiation`) e nebulosidade (`cloud_cover`).

## 4. Sprint Review — checkpoint intermediário

**Incremento demonstrado ao PO (docente):** [DATA DA APRESENTAÇÃO]. Apresentamos o notebook `04_features_iteracao.ipynb` executado, em três pontos:

1. **O problema:** o gráfico de falsos negativos da Sprint 3 por hora mostrou que o Naive Bayes perdia os episódios de ozônio do fim da tarde, porque olhava o O₃ da hora e não a média de 8 h que define o alvo.
2. **A solução:** 12 features novas e uma seleção feita só com validação temporal dentro do treino, que resultou em 25 colunas e Yeo-Johnson no `ColumnTransformer` congelado.
3. **O resultado:** a tabela de lift S3 → S4 e o gráfico de FN por hora antes/depois. O Naive Bayes passou de 180 para 112 falsos negativos (recall 0,40 → 0,63), e os FN entre 17 h e 19 h caíram de 91 para 22. Persistência e Dummy não mudaram, como esperado.

Também deixamos explícito que o NB tem quase o dobro de falsos positivos da persistência e que essa escolha fica para a Sprint 5, com o limiar.

**Feedback recebido:** [REGISTRAR AQUI O QUE A PROFESSORA DISSER NA APRESENTAÇÃO]

## 5. Scrum

- [x] Papéis mantidos e registrados

Product Owner = Docente - Andrea Ono Sakai  
Scrum Master do Sprint = Nicolas Santos Silva  
Development Team = Adrielle, Victor, João Pedro e Samir

- [x] Atualizações assíncronas semanais registradas (comentários de andamento nas issues da Sprint 4 no GitHub)
- [x] Board refletindo o backlog, em andamento e concluído (uma issue por atividade abaixo, com responsável)

**Atividades da Sprint 4 para o board:**

| # | Atividade | Responsável | Evidência |
|---|---|---|---|
| 1 | Analisar os erros da Sprint 3 numa validação interna (dentro do treino) | Samir | `reports/sprint4/erros_s3_validacao_*` |
| 2 | Criar as 12 features novas e o teste de vazamento | Samir | `src/sprint4_features.py` |
| 3 | Selecionar features e normalização com validação temporal no treino | Samir e João Pedro | `reports/sprint4/selecao_validacao.csv` |
| 4 | Congelar o `ColumnTransformer` e salvar os artefatos em `models/` | João Pedro | `models/`, `config/params.yaml` |
| 5 | Retreinar Dummy, persistência e Naive Bayes no mesmo split e montar a tabela de lift | Victor | `reports/sprint4/lift_s3_s4.csv`, `README.md` |
| 6 | Atualizar o dicionário para a v0.3 | Adrielle | `docs/Dicionario_de_Dados.md` |
| 7 | Organizar o board, preparar a Sprint Review e consolidar este diário | Nicolas | este arquivo |

**Link do board:** [Projeto Trilha B — Sprints](https://github.com/users/SamirAbdulKhalek/projects/2)

## 6. Diário de bordo (retrospectiva individual)

| Integrante | O que fiz nesta Sprint | Dificuldades | O que pretendo manter/ajustar |
|---|---|---|---|
| Adrielle | Atualizei o dicionário de dados para a v0.3: documentei as 12 features derivadas (unidade, definição e se foram mantidas ou descartadas), a lista congelada de 25 colunas e a preparação do `ColumnTransformer`. Conferi se os nomes das colunas no dicionário batem com `config/params.yaml` e `models/features_s4.json` | Descrever features que não são medições diretas (deltas, codificação cíclica, elevação solar) de forma que alguém de fora do grupo entenda o que significam e por que não vazam informação futura | Manter o dicionário como contrato das colunas e atualizá-lo junto com o código, e não só no fim da sprint |
| Victor | Retreinei e avaliei os três baselines no mesmo split e montei a tabela de lift S3 → S4 com recall, F1, FN e FP. Interpretei por que Dummy e persistência não mudam e conferi no gráfico por hora que a queda de FN do Naive Bayes aconteceu no fim da tarde, como a análise de erros previa. Atualizei os resultados no README | Entender que "sem lift" na persistência e no Dummy não é falha, e sim consequência de serem regra fixa e classe majoritária. Explicar por que o NB ganhou recall mas empatou em F1 com a persistência | Sempre comparar em contagem (FN e FP), e não só em percentual, e deixar a escolha entre NB e persistência para o limiar da Sprint 5 |
| Nicolas | Como Scrum Master, criei uma issue para cada atividade no board, com responsável e evidência, acompanhei o andamento pelos comentários e conferi se cada entrega tinha evidência no repositório. Preparei a Sprint Review (problema → solução → resultado) e consolidei este diário | Coordenar cinco pessoas trabalhando em arquivos que dependem um do outro (o script gera os relatórios que entram no diário) sem que ninguém commitasse uma versão desatualizada | Manter a regra de `git pull` antes de cada commit e registrar as atualizações no board no mesmo dia em que a tarefa avança |
| Samir | Analisei os erros da Sprint 3 numa validação interna dentro do treino e identifiquei os dois regimes de falso negativo: fim de tarde (média de 8 h do O₃) e início de episódio. Implementei as 12 features novas, o teste automático de vazamento e a seleção por famílias com validação temporal de 3 janelas. Executei o notebook 04 e gerei os relatórios | Não usar o teste para escolher features (a tentação era olhar os erros do teste da Sprint 3). Montar a validação temporal com purga de 24 h. Garantir que a elevação solar fosse calculada para a hora prevista sem vazar informação | Para a Sprint 5, atacar os erros de início de episódio, que continuam sendo o ponto fraco; propor uma coleta versionada com radiação solar e nebulosidade |
| João Pedro | Revisei o vazamento temporal de todas as features (antigas e novas) e comparei `StandardScaler` × Yeo-Johnson na validação. Congelei o `ColumnTransformer` (mediana + Yeo-Johnson) e salvei os artefatos em `models/`. Travei a lista de 25 colunas no `params.yaml`, de forma que o script para com erro se a seleção mudar | Justificar a escolha da normalização além do "default do sklearn": entender por que o Naive Bayes gaussiano se beneficia de variáveis mais próximas da normal | Na Sprint 5, usar exatamente este transformer, sem "descongelar"; se algo precisar mudar, retreinar os três baselines antes |

---

## Rubrica de avaliação — Sprint 4 (nota de 0 a 4,0)

| Critério | Peso | O que caracteriza nota máxima | Nota atribuída | Observações |
|---|---|---|---|---|
| Engenharia de atributos iterada | 1,5 | Atributos novos, justificados pelos erros da Sprint 3, seleção criteriosa, sem vazamento | | |
| Pipeline congelado | 0,5 | ColumnTransformer atualizado, `fit` só no treino, estratégias justificadas | | |
| Retreino do baseline e lift | 1,0 | Dummy/persistência/NB retreinados no mesmo split; delta de recall/F1 interpretado | | |
| Sprint Review / incremento demonstrado | 0,5 | Incremento real apresentado ao PO, com feedback registrado | | |
| Scrum + diário de bordo | 0,5 | Board atualizado semanalmente, diário reflexivo de todos os integrantes | | |
| **Nota final da Sprint 4** | **4,0** | | **___ / 4,0** | |
