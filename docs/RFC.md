# RFC: Proposta de Projeto

| Campo | Valor |
|---|---|
| **Título** |  Predição da Qualidade do Ar com Foco nos Melhores Horários para a Prática de Exercícios ao Ar Livre em São Paulo |
| **Trilha** | B — Qualidade do ar inadequada
| **Equipe** | Equipe 2 |
| **Autores** | • Adrielle Stollemberger RGM: 33948844 • Victor Almeida de Aquino RGM: 32901321 • Nicolas Santos Silva RGM: 32855885 • Samir Abdul Khalek RGM: 32657994 • João Pedro Garcia Almeida RGM: 32847629 |
| **Status** | Revisado (v0.3) |
| **Data** | 14/09/2026 |
| **Sprint de referência** | 1 |

> Um RFC ("Request for Comments") é um documento curto que formaliza uma proposta antes de ela ser executada, para que o time (e quem revisa) concorde com o problema e o escopo antes de investir tempo em código. Aqui, ele reúne o canvas de kickoff numa proposta legível por alguém de fora do grupo. Algoritmo (Random Forest, XGBoost, etc.) **não** se escolhe neste documento.

---

## 1. Resumo (TL;DR)

O projeto propõe prever as condições da qualidade do ar em São Paulo para identificar os horários mais adequados para a prática de exercícios ao ar livre. A previsão será baseada principalmente em dados de poluentes atmosféricos e condições meteorológicas. O objetivo é auxiliar a população na escolha de períodos com condições mais favoráveis para atividades físicas externas.

---

## 2. Contexto e motivação

A qualidade do ar é um fator relevante para a prática de atividades físicas ao ar livre, pois durante o exercício ocorre aumento da frequência respiratória e, consequentemente, da quantidade de ar inalado. Em períodos de maior concentração de poluentes atmosféricos, como **PM2.5, PM10, ozônio (O₃) e dióxido de nitrogênio (NO₂),** a exposição pode ser menos favorável à realização de exercícios externos.

Diante disso, o projeto busca utilizar dados históricos de qualidade do ar e condições meteorológicas para identificar padrões e prever períodos com condições mais ou menos adequadas para atividades físicas ao ar livre em São Paulo. A proposta possui aplicação principalmente relacionada à saúde pública e qualidade de vida, oferecendo informações que possam apoiar a escolha de horários para a realização dessas atividades.

---

## 3. Problema e evento a ser previsto

| Pergunta | Resposta |
|---|---|
| **Qual evento será previsto** | A ocorrência de um período com **qualidade do ar inadequada para a prática de exercícios ao ar livre em São Paulo**. | 
| **Como será definida a classe positiva?** | `qualidade_inadequada_24h = 1` quando, **24 horas após o instante da previsão**, pelo menos um poluente estiver na faixa **CETESB N3 — RUIM ou pior**, cada um na média de referência da CETESB: MP10 (24 h) > 100 µg/m³; MP2,5 (24 h) > 50 µg/m³; O₃ (8 h) > 130 µg/m³; NO₂ (1 h) > 240 µg/m³; SO₂ (24 h) > 40 µg/m³; CO (8 h) > 11 ppm. Classe `0` = Boa ou Moderada. Definido formalmente na Sprint 2; no treino, 13,96 % das horas são classe 1 (1.960 de 14.040), quase todas por ozônio. |
| **Qual é o horizonte da previsão?** | **24 horas à frente**: na hora `t` prevê-se a condição do ar na mesma hora do dia seguinte, permitindo ao atleta planejar o treino de amanhã. |
| **Qual é a unidade de análise (o que representa cada linha do dataset)?** | Cada linha representa **uma hora** (`timestamp` em hora local `America/Sao_Paulo`) no ponto central de São Paulo (lat −23,5489, lon −46,6388), com 6 poluentes e 3 variáveis meteorológicas. São 17.544 linhas (06/09/2024 00h a 06/09/2026 23h). |

---

## 4. Escopo

| Pergunta | Resposta |
|---|---|
| Período histórico considerado | 06/09/2024 a 06/09/2026 (dois anos completos, horários). |
| O que está dentro do escopo deste projeto | Coleta e integração de dados de qualidade do ar e meteorológicos; análise dos poluentes PM2.5, PM10, CO, O₃, NO₂ e SO₂; utilização de variáveis meteorológicas relevantes; tratamento e análise dos dados; desenvolvimento de um modelo para predizer períodos com qualidade do ar adequada ou inadequada; e apresentação dos resultados em um dashboard com indicação dos horários mais favoráveis para exercícios ao ar livre em São Paulo.| 
| O que está fora de escopo (explicitamente não será feito) | Diagnóstico ou recomendação médica individual; avaliação das condições de saúde de cada usuário; medição própria da qualidade do ar por sensores; previsão para localidades fora da área definida do projeto; identificação da causa específica da poluição; análise de trânsito como relação causal; e garantia de que determinado horário seja seguro para todas as pessoas. | 

---

## 5. Usuários e decisão apoiada

Os principais usuários do projeto serão pessoas que praticam exercícios físicos ao ar livre em São Paulo, incluindo praticantes amadores e atletas profissionais, especialmente em atividades como caminhada, corrida e ciclismo. Com base na previsão da qualidade do ar, o sistema poderá indicar os períodos com condições mais ou menos favoráveis, auxiliando esses usuários a escolher o melhor horário para realizar atividades físicas, treinos ou competições e evitar períodos com qualidade do ar inadequada.

O alerta terá caráter informativo e preventivo, não substituindo orientações médicas ou avaliações individuais de saúde.

---

## 6. Dados e fontes

Resumo de alto nível. O detalhe (unidade, resolução, medido vs. modelado, códigos IBGE) vai no `Dicionário de Dados`.

| Fonte | O que fornece | Papel no projeto (feature / alvo / ambos) |
|---|---|---|
| Open-Meteo Air Quality API | Dados horários de qualidade do ar, incluindo PM2.5, PM10, CO, NO₂, SO₂ e O₃ para a localização analisada em São Paulo. | Ambos — os poluentes serão utilizados como dados de entrada e também servirão de base para a definição da classe-alvo de qualidade do ar adequada/inadequada.|
| Open-Meteo Weather / Historical Weather API | Dados meteorológicos horários de reanálise: temperatura a 2 m (°C), precipitação (mm) e velocidade do vento a 10 m (km/h), no mesmo período, local e fuso. | Feature — variáveis utilizadas para auxiliar o modelo na predição da condição futura da qualidade do ar. |

**https://github.com/Adri22K/Projeto-TrilhaB/blob/main/docs/Dicionario_de_Dados.md**

---

## 7. Custo dos erros

**O falso negativo** é considerado mais grave, pois o sistema deixaria de alertar sobre uma condição inadequada da qualidade do ar e poderia indicar como favorável um horário que deveria ser evitado. Já o falso positivo gera principalmente um inconveniente, como a mudança desnecessária do horário de uma atividade.

**Tipo de erro**	
O projeto dará maior atenção ao recall da classe positiva (qualidade do ar inadequada), buscando reduzir principalmente os falsos negativos. O limiar de decisão será definido posteriormente, durante a validação do modelo, considerando o equilíbrio entre a identificação dos períodos inadequados e a quantidade de alertas incorretos.

**Custo/consequência**
Falso negativo	O modelo prevê que determinado horário apresenta condições adequadas para exercícios ao ar livre, quando na realidade a qualidade do ar está inadequada.	O usuário, incluindo atletas amadores ou profissionais, pode realizar exercícios em um período com maior concentração de poluentes, aumentando desnecessariamente sua exposição à poluição atmosférica.

**Falso positivo**	
- O modelo prevê que determinado horário apresenta qualidade do ar inadequada, quando na realidade as condições estão adequadas para exercícios ao ar livre.	
- O usuário pode evitar ou alterar um treino desnecessariamente, escolhendo outro horário mesmo quando as condições ambientais seriam favoráveis.

---

## 8. Abordagem proposta (visão de alto nível)

- A ingestão (`src/coleta_integracao.py`) lê `config/params.yaml` e consulta duas APIs: **Open-Meteo Air Quality** (PM10, PM2.5, CO, NO₂, SO₂ e O₃, horários, em µg/m³) e **Open-Meteo Historical Weather** (temperatura, precipitação e vento, horários), para São Paulo, de 06/09/2024 a 06/09/2026, no fuso `America/Sao_Paulo`. O CO₂, presente na primeira versão, foi retirado por não fazer parte do índice CETESB.

- As requisições usam `requests` com `timeout`, `raise_for_status()`, novas tentativas automáticas para erros temporários (429/5xx) e validação da estrutura do JSON. A resposta de cada API é inspecionada (status, `Content-Type`, URL, chaves, unidades, tamanho) e **salva como JSON bruto** em `data/raw/` antes de qualquer transformação. A coleta é idempotente: reexecutar não duplica nem sobrescreve o bruto.

- A granularidade é **horária** em todas as etapas (a agregação diária da primeira versão foi abandonada). As duas fontes são integradas pelo `timestamp` com `how="inner"` e `validate="one_to_one"`, resultando em 17.544 linhas e nenhuma hora descartada.

- Em seguida: limpeza com log (Sprint 2) → split temporal (corte em 14/04/2026) → EDA só no treino → alvo `qualidade_inadequada_24h` → features com lags e médias móveis apenas do passado → Pipeline do scikit-learn (`ColumnTransformer` + classificador) com `fit` só no treino → baselines (Dummy, persistência, Naive Bayes) → iteração de features (Sprint 4) → limiar escolhido em validação temporal dentro do treino → teste medido uma única vez → Model Card e `joblib` do Pipeline.

- Um painel para o atleta (horários favoráveis / atenção / evitar) poderá ser apresentado na Mostra como recurso adicional.

**O fluxo fica assim:**
API Open-Meteo (ar + clima) → requests com timeout/retry → inspeção + JSON bruto em `data/raw` → merge horário → limpeza (`data/interim`) → split temporal → EDA no treino → alvo → features (`data/processed`) → Pipeline scikit-learn → baselines → iteração → limiar na validação → teste final → Model Card + joblib

---

## 9. Riscos e limitações conhecidas

O projeto apresenta alguns riscos e limitações que deverão ser considerados durante seu desenvolvimento:

1. Dependência de APIs externas, que podem apresentar indisponibilidade, alterações ou ausência temporária de dados.

2. Possibilidade de dados ausentes ou inconsistentes nas séries históricas de qualidade do ar e meteorologia.

3. Os dados utilizados podem ser modelados ou estimados, não representando necessariamente medições realizadas diretamente no local onde o usuário fará o exercício.

4. A cobertura espacial pode não representar igualmente todas as regiões de São Paulo.

5. Possibilidade de desbalanceamento das classes, principalmente se houver poucos registros classificados como qualidade do ar inadequada.

6. O período histórico de 2024 a 2026 é relativamente curto, sendo que os dados de 2026 ainda correspondem a um período parcial.

7. A qualidade do ar pode ser influenciada por fatores que não estarão presentes no dataset, limitando a capacidade preditiva do modelo.

8. A recomendação considera principalmente a qualidade do ar, não avaliando condições individuais de saúde ou todos os demais fatores ambientais relevantes para a prática de exercícios.

---

## 10. Critérios de sucesso

A solução será considerada útil se conseguir identificar antecipadamente períodos com qualidade do ar inadequada para a prática de exercícios ao ar livre, apresentando desempenho superior ao baseline definido pelo grupo.

Como o falso negativo foi considerado o erro de maior impacto, será priorizado o recall da classe positiva no conjunto de teste, utilizando o limiar de decisão escolhido nas etapas posteriores do projeto. O valor mínimo aceitável será definido após a análise exploratória, avaliação do balanceamento das classes e construção do baseline.

Além do desempenho preditivo, serão considerados critérios de sucesso a integração consistente das fontes de dados, tratamento adequado de valores ausentes, dicionário de dados alinhado às variáveis efetivamente utilizadas pelo modelo e documentação das características, limitações e resultados do modelo no Model Card.

Referência mínima já medida (Sprint 3): o baseline de persistência teve recall de 0,48 na classe 1 (145 de 299 horas inadequadas detectadas, 154 FN). Um modelo só será adotado se superar esse recall no limiar congelado na validação.

---

## 11. Alternativas consideradas 

Durante a definição do projeto, foram consideradas outras abordagens relacionadas à qualidade do ar, como comparação da poluição entre regiões de São Paulo, análise da relação entre poluição e trânsito e avaliação do risco da poluição para grupos específicos, como idosos. Essas alternativas foram descartadas para manter um escopo mais objetivo e viável.

O grupo optou pela predição da qualidade do ar com foco nos melhores horários para a prática de exercícios ao ar livre, por permitir a utilização de dados históricos de poluentes e condições meteorológicas em um problema preditivo com aplicação prática para praticantes de atividades físicas e atletas profissionais.

---

## 12. Perguntas em aberto

Algumas definições serão realizadas nas próximas etapas do projeto a partir da Análise Exploratória dos Dados (EDA) e do desempenho do modelo, entre elas:

1. Qual será o limiar definitivo para classificar um horário como adequado ou inadequado para exercícios ao ar livre?

2. Quais poluentes e variáveis meteorológicas apresentam maior relevância para a predição?

3. Qual será a janela temporal mais adequada para utilizar dados anteriores na previsão dos períodos seguintes?

4. O horizonte de 24 horas apresenta desempenho satisfatório ou deverá ser ajustado?

5. Como será definido o limiar de probabilidade do modelo, considerando que o falso negativo possui maior impacto para o projeto?

6. O modelo apresentará lift e desempenho suficientes em relação a uma estratégia de referência (baseline) para justificar seu uso?

7. Será necessário criar níveis intermediários de recomendação, como Favorável, Atenção e Evitar, além da classificação binária utilizada pelo modelo?

---

## 13. Cronograma e contrato entre sprints

| Sprint | Período | Produz (sai) | A próxima sprint é obrigada a usar |
|---|---|---|---|
| 1 | 17/08/2026 – 17/09/2026 | RFC v0.1, dicionário v0.1, `data/raw`, `config`, N do merge | O bruto desta coleta |
| 2 | 18/09/2026 – 27/09/2026 | `data/interim`, split, alvo, features iniciais, dicionário v0.2 | Split, alvo e features daqui |
| 3 | 28/09/2026 – 04/10/2026 | Transformer inicial, Dummy + persistência + NB, erros | Os erros (para features novas) e o mesmo split |
| 4 | 05/10/2026 – 11/10/2026 | Features novas, transformer congelado, lift S3→S4, dicionário v0.3 | Este transformer e estes baselines retreinados |
| 5 | 12/10/2026 – 25/10/2026 | Comparativo final, limiar **na validação**, caderno, model card, dicionário v0.4, `joblib` do Pipeline | — (entrega final) |

---

## 14. Histórico de revisões

| Versão | Data | Autor | O que mudou |
|---|---|---|---|
| v0.1 | 14/09/2026 | Professora | Arquivo base do RFC (Sprint 1) |
| v0.2 | 14/09/2026 | Todos os integrantes | Primeira versão do RFC (Sprint 1) |
| v0.3 | 19/10/2026 | Todos os integrantes | Correções pós-feedback da Sprint 1: classe positiva formal, horizonte, unidade de análise e período alinhados ao código; CO₂ e agregação diária removidos; abordagem atualizada (duas APIs, JSON bruto, merge `one_to_one`); limiar de SO₂ corrigido para 40 µg/m³ |
| | | | |

