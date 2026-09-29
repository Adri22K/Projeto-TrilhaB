# Diário de Sprint 2 — Limpeza, tratamento, EDA e engenharia de atributos
**Período:** 18/09/2026 a 27/09/2026
**Trilha definitiva do grupo:** B — Qualidade do ar inadequada

**Equipe:** Equipe 2  
**Integrantes:** Adrielle Stollemberger RGM: 33948844 • Victor Almeida de Aquino RGM: 32901321 • Nicolas Santos Silva RGM: 3287380 • Samir Abdul Khalek RGM: 32657994 • João Pedro Garcia Almeida RGM: 32847629  
**Scrum Master do Sprint:** Nicolas Santos Silva  
**Repositório GitHub:** [Projeto-TrilhaB](https://github.com/Adri22K/Projeto-TrilhaB)

> A aula pode usar a **Trilha A (chuva intensa)** só como exemplo de método. **A entrega é nos dados brutos da Trilha B coletados na Sprint 1** (`data/raw/`).
>
> Ordem obrigatória: **inspeção → limpeza e tratamento → split → imputação estatística só no treino (se houver NA) → EDA no treino → alvo → features**. Não se faz EDA no bruto. Não há modelagem (baseline na Sprint 3).

### Contrato desta sprint

| | Artefato | Origem / destino |
|---|---|---|
| **Entra** | `data/raw/` + `config/` + N do merge | Sprint 1 — **não substituir por outra coleta sem versionar** |
| **Entra** | RFC v0.1 e dicionário v0.1 | Sprint 1 |
| **Sai** | `data/interim/` + log de limpeza (N antes/depois) | Sprint 3 em diante (tabela de trabalho) |
| **Sai** | Split (corte, N treino/teste) | Sprints 3, 4 e 5 — **mesmo corte** |
| **Sai** | Alvo formal (limiar, horizonte, desbalanceamento no treino) | Sprints 3–5 |
| **Sai** | Features iniciais justificadas + dicionário v0.2 | Sprint 3 (transformer) e 4 (base para iterar) |

**Não sai daqui:** Dummy, persistência, Naive Bayes, F1, modelo escolhido.

- [x] Confirmei que o processo desta sprint lê `data/raw/integrado.csv` gerado na Sprint 1

---

## 1. Limpeza e tratamento (antes da EDA)

`data/raw/` permanece intocado. A tabela tratada vai para `data/interim/`.

### 1.1 Inspeção (contar o sujo, ainda sem corrigir)

- [x] Estatística descritiva das colunas numéricas
- [x] Ausentes quantificados por coluna
- [x] Duplicatas (linhas e por chave de unidade de análise)
- [x] Valores inválidos de domínio (ex.: poluente negativo) e sentinelas da API

**Diagnóstico (números):**

Foram inspecionadas 17.544 linhas e 10 colunas. Todas as nove variáveis numéricas têm 17.544 valores válidos. Foram encontrados 0 ausentes, 0 timestamps inválidos, 0 linhas duplicadas, 0 timestamps duplicados, 0 sentinelas (`-999`, `-9999`, `9999`) e 0 valores negativos inválidos. As estatísticas completas estão em [`reports/sprint2/relatorio_execucao.json`](../../reports/sprint2/relatorio_execucao.json) e [`reports/sprint2/estatisticas_treino.csv`](../../reports/sprint2/estatisticas_treino.csv).

### 1.2 Regras aplicadas

Não imputar com média/mediana/moda do dataset **inteiro**. Isso vaza o teste. Imputação estatística, se precisar, é a seção 3 (depois do split).

- [x] Tipos e unidades padronizados
- [x] Duplicatas tratadas com contagem
- [x] Sentinelas → `NaN` explícito
- [x] Inválidos de domínio tratados com regra escrita (remover / `NaN` / correção só se a fonte tiver erro conhecido)
- [x] Log: o que foi feito, em quantas linhas/células, N depois
- [x] Tabela salva em `data/interim/`
- [x] Dicionário (log de limpeza) atualizado

| Problema encontrado | Regra aplicada | Linhas/células afetadas | N depois |
|---|---|---|---|
| Tipos | `timestamp` convertido para data/hora; demais colunas convertidas para numérico | 0 coerções inválidas | 17.544 |
| Linhas/timestamps duplicados | Remover duplicatas, mantendo a primeira ocorrência por timestamp | 0 | 17.544 |
| Sentinelas `-999`, `-9999`, `9999` | Converter para `NaN` | 0 | 17.544 |
| Valores negativos em poluentes, precipitação ou vento | Converter para `NaN` | 0 | 17.544 |

**N após a limpeza:** 17.544 linhas.  
**Evidências (link do notebook/commit):** [`notebooks/02_limpeza_eda_features.ipynb`](../../notebooks/02_limpeza_eda_features.ipynb), [`src/prepara_sprint2.py`](../../src/prepara_sprint2.py), [`data/interim/integrado_limpo.csv`](../../data/interim/integrado_limpo.csv) e [`reports/sprint2/relatorio_execucao.json`](../../reports/sprint2/relatorio_execucao.json).

## 2. Split temporal

Sobre a tabela **já limpa** (`data/interim/`).

- [x] Treino = período mais antigo; teste = mais recente (sem embaralhar)
- [x] Data de corte explícita; N treino e N teste
- [x] Teste isolado: não usa para limiar, janelas, lags nem parâmetros de imputação

**Corte:** 14/04/2026 às 00:00.  
**N treino / N teste:** 14.040 / 3.504.  
**Justificativa:** Divisão temporal de aproximadamente 80% para treino e 20% para teste, arredondada para o início do dia. O treino cobre 06/09/2024 00:00 a 13/04/2026 23:00; o teste cobre 14/04/2026 00:00 a 06/09/2026 23:00. Não houve embaralhamento.

## 3. Tratamento estatístico residual (depois do split, antes da EDA)

Só NA que a limpeza de domínio não resolveu. Parâmetros saem **somente do treino**.

- [x] Estratégia justificada (imputar agora / deixar NA para o `ColumnTransformer` da Sprint 3)
- [x] Se imputar: regra ajustada no treino e aplicada ao teste — não se aplica, pois há 0 NA após a limpeza

**Regra residual (ou “não há NA restantes”):** Não há NA restantes após a limpeza; portanto, nenhuma imputação estatística foi realizada. O segundo item não se aplica.

## 4. Análise exploratória (EDA) — treino já tratado

- [x] Estatística descritiva da variável de origem do alvo **no treino**
- [x] Série temporal, histogramas, boxplots
- [x] Correlação no treino (pode ser espúria em série temporal)
- [x] Outliers discutidos (não só plotados)

> Desbalanceamento de **classes** só depois da seção 5 (quando o alvo existir). Nesta seção explora-se a variável de origem (contínua ou categórica bruta).

**Principais achados da EDA (treino):** PM10 e PM2,5 apresentaram a maior correlação entre poluentes (`r = 0,9658`). O ozônio apresentou correlação positiva com temperatura (`r = 0,6960`) e negativa com NO₂ (`r = -0,5581`). As medianas no treino foram 17,7 µg/m³ para PM10, 16,5 µg/m³ para PM2,5 e 64,0 µg/m³ para O₃. Pelo critério do intervalo interquartil foram sinalizados, entre outros, 687 valores de PM10, 641 de PM2,5 e 135 de O₃. Eles foram mantidos porque não violam o domínio e podem representar episódios reais. Na precipitação, como Q1, mediana e Q3 são zero, o critério IQR marca qualquer chuva positiva; esses valores também foram mantidos.

**Evidências (gráficos, link do notebook):** [`notebooks/02_limpeza_eda_features.ipynb`](../../notebooks/02_limpeza_eda_features.ipynb), [série temporal](../../reports/sprint2/serie_temporal_treino.png), [histogramas](../../reports/sprint2/histogramas_treino.png), [boxplots](../../reports/sprint2/boxplots_treino.png), [correlação](../../reports/sprint2/correlacao_treino.png) e [`estatisticas_treino.csv`](../../reports/sprint2/estatisticas_treino.csv).

## 5. Definição da variável-alvo

**Classe positiva:** `qualidade_inadequada_24h = 1` quando, 24 horas após o instante das features, pelo menos um poluente atinge a classe CETESB N3 — RUIM ou pior.  
**Limiar e justificativa (treino, alinhada ao RFC):** PM10 (média de 24 h) > 100 µg/m³; PM2,5 (média de 24 h) > 50 µg/m³; SO₂ (média de 24 h) > 50 µg/m³; O₃ (média de 8 h) > 130 µg/m³; NO₂ (1 h) > 240 µg/m³; CO (média de 8 h) > 11 ppm. A regra usa a categoria RUIM, na qual toda a população pode apresentar sintomas, e prioriza evitar falsos negativos conforme o RFC. Fonte: [estrutura do índice da CETESB](https://sistemasinter.cetesb.sp.gov.br/Ar/php/boletim_por_poluente.php).  
**Horizonte / deslocamento:** 24 horas à frente, implementado com `.shift(-24)` somente na construção do alvo.  
**Desbalanceamento no treino:** 1.960 positivos (13,96%) e 12.080 negativos (86,04%), em 14.040 linhas. As últimas 24 horas da base não possuem futuro observável e foram excluídas apenas do conjunto de modelagem do teste.

- [x] Classe positiva sem ambiguidade, ligada ao custo de FN
- [x] Desbalanceamento das classes quantificado **no treino** (ex.: “12% positivos vs. 88% negativos”)
- [x] Seção do alvo no dicionário preenchida
- [x] Colunas de construção do alvo na lista de exclusão (vazamento)

## 6. Engenharia de atributos (Feature Engineering)

- [x] Médias móveis, lags e agregações só com informação anterior ao ponto de previsão
- [x] Cada feature seria calculável **no momento real da previsão** (mesmo horizonte do RFC)
- [x] Timestamp e IDs **não** entram como número; calendário (hora, dia da semana, estação) vale
- [x] Variáveis de calendário quando pertinente (hora, dia da semana, estação)
- [x] Parâmetros de janela definidos **no treino**, depois aplicados ao teste
- [x] Vazamento ausente (texto + código: `.shift()` / `.rolling()` sem o instante-alvo nem o futuro)
- [x] Cada atributo justificado um a um; dicionário atualizado

**Descrição e justificativa dos atributos:** Foram mantidas as nove variáveis brutas disponíveis no instante da previsão. Para cada uma foram criados `lag_1h` (persistência recente), `lag_24h` (mesma hora do dia anterior) e `media_24h` (histórico recente). A média móvel usa `.shift(1).rolling(24)`, excluindo o instante atual e o futuro. Também foram criadas `hora`, `dia_semana` e `mes` para representar ciclos diários, semanais e sazonais. O timestamp permanece apenas como chave, não como número. As seis flags auxiliares usadas para construir o alvo não são incluídas nas features. As primeiras 24 linhas do treino, sem histórico completo para os lags, foram excluídas do arquivo de features em vez de imputadas; assim, `treino_features.csv` possui 14.016 linhas e `teste_features.csv` possui 3.480 linhas completas.

**Evidências (trecho de código, link do notebook):** [`notebooks/02_limpeza_eda_features.ipynb`](../../notebooks/02_limpeza_eda_features.ipynb), [`src/prepara_sprint2.py`](../../src/prepara_sprint2.py), [`docs/Dicionario_de_Dados.md`](../Dicionario_de_Dados.md), [`data/processed/treino_features.csv`](../../data/processed/treino_features.csv) e [`data/processed/teste_features.csv`](../../data/processed/teste_features.csv).

## 7. Scrum

- [ ] Atualizações semanais no board
- [ ] Board refletindo o estado real

**Link do board:** [BOARD](https://github.com/Adri22K/Projeto-TrilhaB/issues/2#issue-5481547970)

O link informado aponta para uma issue aberta da Sprint 1 e não apresenta histórico da Sprint 2; por isso, os dois itens acima permanecem desmarcados.

## 8. Diário de bordo (retrospectiva individual)

| Integrante | O que fiz nesta Sprint | Dificuldades | O que pretendo manter/ajustar |
|---|---|---|---|
Adrielle | Auxiliei na análise e organização dos dados utilizados na Sprint 2 | Compreender o tratamento necessário antes da análise exploratória | Manter a organização e documentação das etapas do projeto

Victor | Auxiliei no processo de limpeza e análise exploratória dos dados | Identificar corretamente valores inválidos e possíveis outliers |Melhorar a análise dos dados nas próximas etapas

Nicolas | Auxiliei na organização das atividades da Sprint e acompanhamento do board | Garantir que as etapas fossem realizadas na ordem definida | Manter o acompanhamento das tarefas e a organização do grupo

Samir | Auxiliei na análise das variáveis de qualidade do ar | Interpretar as relações entre os diferentes poluentes | Aprofundar a análise das variáveis utilizadas no projeto

João Pedro | Auxiliei na definição da variável-alvo e na engenharia inicial de atributos | Evitar vazamento de dados durante a criação de variáveis temporais | Manter o cuidado com dados temporais e aprimorar as features utilizadas pelo projetoos para caracterizar a qualidade do ar.

## Rubrica de avaliação — Sprint 2 (nota de 0 a 4,0)

| Critério | Peso | O que caracteriza nota máxima | Nota atribuída | Observações |
|---|---|---|---|---|
| Limpeza e tratamento | 1,0 | Inspeção numérica, regras explícitas, log com N, `interim` gerado, raw intocado, sem estatística global para imputar | | |
| Split temporal | 0,5 | Corte explícito na tabela limpa; teste isolado | | |
| EDA e variável-alvo | 1,0 | EDA no treino tratado; alvo formalizado no treino; dicionário atualizado | | |
| Engenharia de atributos | 1,0 | Janelas/lags coerentes, sem vazamento, parâmetros no treino, atributos justificados um a um | | |
| Scrum + diário de bordo | 0,5 | Board com histórico; diário reflexivo de todos | | |
| **Nota final da Sprint 2** | **4,0** | | **___ / 4,0** | |
