# Diário de Sprint 3 — Ingestão robusta e primeira modelagem (baseline)
**Período:** 28/09/2026 a 04/10/2026
**Trilha definitiva do grupo:** B — Qualidade do ar inadequada

**Equipe:** Equipe 2  
**Integrantes:** Adrielle Stollemberger RGM: 33948844 • Victor Almeida de Aquino RGM: 32901321 • Nicolas Santos Silva RGM: 32855885 • Samir Abdul Khalek RGM: 32657994 • João Pedro Garcia Almeida RGM: 32847629  
**Scrum Master do Sprint:** Nicolas Santos Silva  
**Repositório GitHub:** [Projeto-TrilhaB](https://github.com/Adri22K/Projeto-TrilhaB)

> Kickoff e **coleta bruta**: Sprint 1. Limpeza, split, EDA no treino tratado, alvo e features: Sprint 2. Esta sprint **não reabre** esses pontos: robustece a ingestão e treina o primeiro conjunto de baselines **no mesmo split da Sprint 2**.

### Contrato desta sprint

| | Artefato | Origem / destino |
|---|---|---|
| **Entra** | `data/interim/`, split, alvo, features iniciais, dicionário v0.2 | Sprint 2 — **mesmo split e mesmas features** |
| **Entra** | `data/raw/` + `config/` | Sprint 1 (para robustecer a coleta, sem apagar o bruto) |
| **Sai** | Ingestão idempotente + `requirements.txt` | Sprint 4–5 (reexecução) |
| **Sai** | `ColumnTransformer` inicial (`fit` só no treino do split da Sprint 2) | Sprint 4 (atualizar, não trocar o split) |
| **Sai** | Dummy + persistência + Naive Bayes, métricas por classe, leitura dos erros | Sprint 4 (**os erros motivam features novas**) |

**Não sai daqui:** Random Forest/XGBoost como modelo final, limiar fino, model card.

- [x] Confirmei que treino/teste são os da Sprint 2 e que as features são as da Sprint 2

---

## 1. Aprofundamento da ingestão (Pipeline CD)

- [x] Período histórico mantido com justificativa. O recorte de 06/09/2024 a 06/09/2026 já fornece 17.544 observações horárias integradas e foi mantido para preservar a continuidade com o tratamento, alvo e split temporal definidos na Sprint 2.
- [x] Tratamento de erros revisado e robustecido. A coleta utiliza `timeout`, `raise_for_status`, tentativas automáticas para erros de conexão/leitura/status e validação da estrutura e do tamanho das respostas antes da persistência.
- [x] Coleta idempotente comprovada. Quando o conjunto completo de arquivos brutos já existe, a reexecução valida datas, timestamps, colunas e quantidade de linhas e encerra sem regravar nem duplicar registros.
- [x] `data/raw/` permanece como origem bruta; `data/interim/` contém a tabela limpa da Sprint 2; `data/processed/` contém as matrizes com features usadas na modelagem.
- [x] `requirements.txt` registra as versões usadas: pandas 2.2.3, numpy 2.3.5, scikit-learn 1.8.0, matplotlib 3.10.8, PyYAML 6.0.3, requests 2.32.5 e ipykernel 7.2.0.

O período não foi ampliado nesta Sprint porque o histórico de dois anos já atende ao objetivo pedagógico e possui continuidade horária suficiente para as janelas temporais. Alterar o recorte exigiria gerar uma nova versão do bruto e repetir as etapas de tratamento e split da Sprint 2. Para preservar a comparabilidade entre as Sprints, foi mantida a mesma base.

**Evidências (link do notebook/commit):** [`src/coleta_integracao.py`](../../src/coleta_integracao.py), [`config/params.yaml`](../../config/params.yaml), [`requirements.txt`](../../requirements.txt), [`data/raw/coleta_integracao_log.json`](../../data/raw/coleta_integracao_log.json) e [`data/raw/integrado.csv`](../../data/raw/integrado.csv).

## 2. Preparação para modelagem

- [x] Mesmo split temporal da Sprint 2, sem embaralhamento e sem novo split aleatório.
- [x] `Pipeline` sklearn com `ColumnTransformer` e classificador no mesmo objeto; `fit` executado somente no treino.
- [x] Imputação residual e normalização ficam dentro do Pipeline (`SimpleImputer(strategy="median")` + `StandardScaler`).
- [x] `random_state` de referência documentado como 42. Os três baselines utilizados são determinísticos e não realizam sorteio durante o treinamento.
- [x] Ausência de vazamento temporal reconfirmada. Além do split temporal, foram retiradas apenas da modelagem as 24 últimas linhas do treino cujos rótulos de t+24 h alcançariam o período reservado ao teste.
- [x] As 39 features definidas na Sprint 2 foram mantidas para DummyClassifier e Gaussian Naive Bayes.

**Descrição da estratégia de split (datas de corte, N) e do ColumnTransformer:**

O corte temporal continua em **14/04/2026 00:00**. O arquivo de treino da Sprint 2 possui 14.016 registros e o arquivo de teste possui 3.480. Como o alvo representa a condição 24 horas à frente, 24 registros localizados na fronteira final do treino foram retirados apenas do ajuste dos modelos, resultando em **13.992 registros efetivamente utilizados para treinamento**. Os CSVs e o corte original da Sprint 2 permaneceram inalterados.

O `ColumnTransformer` recebe as 39 features numéricas da Sprint 2. Para DummyClassifier e Gaussian Naive Bayes, o Pipeline executa imputação pela mediana e normalização antes do classificador. A persistência temporal utiliza outro Pipeline com somente `estado_atual`, calculado a partir dos dados observáveis até o instante `t`, sem consultar o alvo futuro.

**Evidências:** [`src/sprint3_baselines.py`](../../src/sprint3_baselines.py), [`notebooks/03_baseline.ipynb`](../../notebooks/03_baseline.ipynb), [`data/processed/treino_features.csv`](../../data/processed/treino_features.csv), [`data/processed/teste_features.csv`](../../data/processed/teste_features.csv) e [`reports/sprint3/resultado_execucao.json`](../../reports/sprint3/resultado_execucao.json).

## 3. Modelagem baseline

- [x] `DummyClassifier(strategy="most_frequent")` utilizado como piso de acurácia da classe majoritária.
- [x] Baseline de persistência temporal implementado usando a condição conhecida no instante `t` para prever `t+24 h`, sem consultar o alvo futuro.
- [x] `GaussianNB()` treinado e avaliado no mesmo teste e com as mesmas 39 features dos modelos aprendidos.
- [x] A hipótese de independência condicional do Naive Bayes foi tratada como simplificação pedagógica, pois poluentes e variáveis meteorológicas apresentam relações entre si.
- [x] Acurácia, precisão, recall e F1 foram calculados por classe, além das matrizes de confusão dos três baselines.
- [x] A interpretação prioriza falsos negativos, definidos no RFC como erro de maior custo para o projeto. Nenhum limiar foi otimizado nesta Sprint.
- [x] Não foi aplicado SMOTE nem oversampling aleatório na série temporal.

**Resultados (Dummy vs. persistência vs. Naive Bayes) e interpretação:**

| Modelo | Acurácia | Precisão classe 1 | Recall classe 1 | F1 classe 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0,9141 | 0,0000 | 0,0000 | 0,0000 | 0 | 299 |
| Persistência temporal | 0,9106 | 0,4801 | 0,4849 | 0,4825 | 157 | 154 |
| Gaussian Naive Bayes | 0,8626 | 0,2854 | 0,3980 | 0,3324 | 298 | 180 |

**Métricas por classe:**

| Modelo | Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 0 | 0,9141 | 1,0000 | 0,9551 | 3.181 |
| DummyClassifier | 1 | 0,0000 | 0,0000 | 0,0000 | 299 |
| Persistência temporal | 0 | 0,9515 | 0,9506 | 0,9511 | 3.181 |
| Persistência temporal | 1 | 0,4801 | 0,4849 | 0,4825 | 299 |
| Gaussian Naive Bayes | 0 | 0,9412 | 0,9063 | 0,9234 | 3.181 |
| Gaussian Naive Bayes | 1 | 0,2854 | 0,3980 | 0,3324 | 299 |

O DummyClassifier alcançou acurácia de 91,41%, porém não identificou nenhuma ocorrência da classe inadequada e gerou 299 falsos negativos. Isso confirma que a acurácia isolada é inadequada para avaliar este problema desbalanceado.

A persistência temporal apresentou o melhor resultado para a classe positiva entre os três baselines desta Sprint: recall de 48,49% e F1 de 48,25%, com 154 falsos negativos. O Gaussian Naive Bayes obteve recall de 39,80% e 180 falsos negativos. Esses resultados indicam que ainda existe espaço para melhorar a identificação dos horários inadequados por meio de novas features e modelos na Sprint 4.

**Evidências (link do notebook, métricas):** [`notebooks/03_baseline.ipynb`](../../notebooks/03_baseline.ipynb), [`reports/sprint3/comparativo_baselines.csv`](../../reports/sprint3/comparativo_baselines.csv), [`reports/sprint3/metricas_por_classe.csv`](../../reports/sprint3/metricas_por_classe.csv), [`reports/sprint3/erros_por_horario.csv`](../../reports/sprint3/erros_por_horario.csv), [`reports/sprint3/matriz_dummy.png`](../../reports/sprint3/matriz_dummy.png), [`reports/sprint3/matriz_persistencia.png`](../../reports/sprint3/matriz_persistencia.png) e [`reports/sprint3/matriz_naive_bayes.png`](../../reports/sprint3/matriz_naive_bayes.png).

## 4. Scrum

- [x] Papéis mantidos e registrados

Product Owner = Docente - Andrea Ono Sakai  
Scrum Master do Sprint = Nicolas Santos Silva  
Development Team = Adrielle, Victor, João Pedro e Samir

- [x] Atualizações da Sprint organizadas a partir das atividades de ingestão, preparação, modelagem e documentação.
- [x] Board mantido como referência do andamento do projeto.

**Atividades da Sprint 3 para acompanhamento no board:**

1. Revisar a robustez da coleta e impedir duplicação em reexecuções.
2. Validar o mesmo split temporal e as mesmas features produzidas na Sprint 2.
3. Implementar os Pipelines dos três baselines solicitados.
4. Executar os modelos com os dados reais e registrar métricas por classe e matrizes de confusão.
5. Interpretar falsos negativos e preparar evidências para a Sprint 4.
6. Atualizar o diário da Sprint e organizar os artefatos no repositório.

**Link do board:** [Projeto Trilha B — Sprints](https://github.com/users/SamirAbdulKhalek/projects/2)

## 5. Diário de bordo (retrospectiva individual)

| Integrante | O que fiz nesta Sprint | Dificuldades | O que pretendo manter/ajustar |
|---|---|---|---|
| Adrielle | Auxiliei na conferência dos dados, na organização das evidências e na documentação dos resultados dos baselines | Manter a separação correta entre dados brutos, tratados e processados durante a documentação | Continuar revisando a organização dos artefatos e a clareza das evidências de cada etapa |
| Victor | Auxiliei na validação das métricas e na comparação entre Dummy, persistência temporal e Naive Bayes | Interpretar acurácia, precisão, recall e F1 em um problema com classe positiva menos frequente | Priorizar a leitura das métricas por classe e dos falsos negativos nas próximas avaliações |
| Nicolas | Organizei as atividades da Sprint como Scrum Master e acompanhei a consistência entre o planejamento, o board e o diário | Garantir que a Sprint 3 respeitasse o mesmo split e as definições estabelecidas na Sprint 2 | Manter o acompanhamento das tarefas e registrar com mais frequência as atualizações do grupo |
| Samir | Auxiliei na análise dos erros dos modelos e na interpretação dos resultados relacionados aos poluentes | Relacionar os falsos positivos e falsos negativos ao comportamento temporal das variáveis | Aprofundar a análise dos horários de erro para apoiar a criação de novas features na Sprint 4 |
| João Pedro | Auxiliei na preparação da modelagem, validação do Pipeline, controle de vazamento temporal e execução dos baselines | Implementar corretamente a persistência temporal e tratar a fronteira de 24 horas entre treino e teste | Manter as validações temporais e evoluir as features sem alterar o conjunto de teste definido na Sprint 2 |

---

## 6. Correções do feedback da Sprint 1

Em resposta à correção da Sprint 1 (nota 1,1/4,0), o grupo preencheu o `Guia_Alunos_APIs_Trilhas_B_C` para a Trilha B: [`notebooks/00_guia_apis_trilhaB.ipynb`](../../notebooks/00_guia_apis_trilhaB.ipynb), executado no Colab com as saídas salvas.

| Ponto da correção | O que foi feito |
|---|---|
| Evento genérico; sem classe positiva, horizonte e custo do FN | Seção 1 do guia e RFC v0.3 |
| APIs sem documentação, unidades e justificativa | Seções 2 e 7 do guia (links, endpoints, parâmetros, resolução, limitações, erros) |
| Parâmetros hard-coded; sem `timeout`/`raise_for_status`/`try-except` | `config` único e função `requisitar` |
| Resposta não inspecionada; sem JSON bruto | Inspeção de status, `Content-Type`, URL, chaves e unidades; JSON salvo antes de transformar |
| Uma fonte só; sem `merge` | Ar + clima integrados com `how="inner"` e `validate="one_to_one"` (17.544 linhas, nenhuma descartada) |
| Qualidade dos dados e alvo ausentes | Partes 4 e 5 do guia, com teste automático de vazamento |

Outras correções: limiar de SO₂ corrigido para 40 µg/m³ (faixa N3 da CETESB; os rótulos não mudam, pois a média de 24 h de SO₂ não passa de ~25 µg/m³); RFC atualizado para v0.3; dicionário com a natureza e a resolução das fontes; RGM corrigido.

**Reprodutibilidade:** 

A coleta ao vivo no Colab devolveu os mesmos dados do repositório (1.960 positivos em 14.040 horas no treino; 299 em 3.480 no teste), e a reexecução dos notebooks 01 a 03 regravou `data/interim`, `data/processed` e as métricas de `reports/sprint3` sem nenhuma diferença.

---

## Rubrica de avaliação — Sprint 3 (nota de 0 a 4,0)

| Critério | Peso | O que caracteriza nota máxima | Nota atribuída | Observações |
|---|---|---|---|---|
| Aprofundamento da ingestão | 1,0 | Coleta robustecida, idempotência comprovada, `raw` / `interim` / `processed` distintos, `requirements.txt` | | |
| Preparação (mesmo split + ColumnTransformer) | 0,5 | Split da Sprint 2 respeitado, transformações ajustadas só no treino, sem vazamento | | |
| Baselines (Dummy + persistência + Naive Bayes) com métricas interpretadas | 1,5 | Três baselines no mesmo split, métricas por classe, NB como baseline pedagógico explícito | | |
| Scrum + diário de bordo | 1,0 | Board e diário refletindo o processo real da Sprint | | |
| **Nota final da Sprint 3** | **4,0** | | **___ / 4,0** | |
