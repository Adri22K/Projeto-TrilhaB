# Dicionário de Dados

**Projeto: Predição da Qualidade do Ar com Foco nos Melhores Horários para a Prática de Exercícios ao Ar Livre em São Paulo**
<br>
**Trilha: B — Qualidade do ar inadequada**
<br>
**Equipe:Equipe 2**
<br>
**Última atualização: 17/09/2026** 
<br>
**Versão: v0.1 (Sprint 1)**

> Contrato das colunas entre sprints. Sprint 1: fontes e variáveis **brutas**. Sprint 2: log de limpeza, alvo formal, derivados iniciais e exclusões por vazamento. Sprint 4: features iteradas. Sprint 5: conferência com o model card — as colunas do modelo escolhido são estas.

**Unidade de análise (o que é uma linha):**
**N após o merge (Sprint 1, bruto):**
**N após a limpeza (Sprint 2, `data/interim`):**
**Split (Sprint 2):** treino = ________ / teste = ________ (N treino = ____, N teste = ____)

---

## 1. Fontes de dados

| Fonte | API / Endpoint | Cobertura temporal disponível | Resolução temporal | Medido ou modelado? | Limitações conhecidas |
|---|---|---|---|---|---|
| Open-Meteo Air Quality API | /v1/air-quality  | 06/09/2024 a 06/09/2026 | Dados retornados em resolução horária; para São Paulo, a fonte Open-Meteo possui resolução nativa de 1 hora. | Modelado | Não representa medição direta de uma estação local. Para São Paulo, os dados são provenientes do modelo global CAMS, com resolução espacial de aproximadamente 45 km, podendo não representar exatamente as condições de um ponto específico da cidade. |
| Open-Meteo Historical Weather API | | | | | | **--> em construção**

> **Trilha B:** poluentes Open-Meteo em geral são produto **modelado**, não medição de estação local — declarar na coluna acima.

---

## 2. Variáveis brutas (coletadas na Sprint 1)

| Nome da coluna | Fonte | Tipo | Unidade | Descrição | Observações |
|---|---|---|---|---|---|
| `date` | Ambas | data-hora | hr | Data e hora referentes ao registro. | Utilizada para alinhar e integrar as duas fontes. |
| `pm2_5` | Open-Meteo Air Quality API | numérica contínua | µg/m³ | Concentração de material particulado fino com diâmetro inferior a 2,5 µm. | Dado modelado pelo CAMS. |
| `pm10` | Open-Meteo Air Quality API | numérica contínua | µg/m³ | Concentração de material particulado com diâmetro inferior a 10 µm. | Dado modelado pelo CAMS. |
| `carbon_monoxide` | Open-Meteo Air Quality API | numérica contínua | µg/m³ | Concentração de monóxido de carbono (CO) no ar. | Dado modelado pelo CAMS. |
| `nitrogen_dioxide` | Open-Meteo Air Quality API | numérica contínua | µg/m³ | Concentração de dióxido de nitrogênio (NO₂) no ar. | Dado modelado pelo CAMS. |
| `sulphur_dioxide` | Open-Meteo Air Quality API | numérica contínua | µg/m³ | Concentração de dióxido de enxofre (SO₂) no ar. | Dado modelado pelo CAMS. |
| `ozone` | Open-Meteo Air Quality API | numérica contínua | µg/m³ | Concentração de ozônio (O₃) no ar. | Dado modelado pelo CAMS. |

*Tipo: numérica contínua / numérica discreta / categórica / data-hora / identificador.*

---

## 2.1 Log de limpeza e tratamento (Sprint 2, antes da EDA)

`data/raw/` permanece intocado. A tabela da EDA é `data/interim/`.

Não imputar com média/mediana/moda do dataset inteiro nesta etapa.

| Problema | Regra aplicada | Linhas/células afetadas | N depois | Observação |
|---|---|---|---|---|
| Registros duplicados | Remoção de registros com timestamp duplicado | 5 linhas | 17.515 | Mantido apenas um registro por timestamp |
| Valores inválidos | Remoção de registros que não atendiam à regra definida | 2 linhas | 17.513 | Regra documentada no notebook |

**Ausentes restantes após a limpeza de domínio (se houver, tratar no treino na Sprint 2):**

---

## 3. Variável-alvo *(formalizar na Sprint 2; limiar escolhido no treino)*

| Campo | Descrição |
|---|---|
| Nome da coluna | `qualidade_ar_inadequada` |
| Definição da classe positiva | Classe **1**: horário considerado com qualidade do ar inadequada para a prática de exercícios ao ar livre. Classe **0**: horário considerado adequado. A regra definitiva será formalizada na Sprint 2. |
| Limiar adotado e justificativa (evidência do **treino**) | **A definir na Sprint 2**, utilizando exclusivamente os dados de treino e a análise dos poluentes selecionados. |
| Fonte da variável de origem | Variáveis de poluentes coletadas pela **Open-Meteo Air Quality API**, como PM2.5, PM10, O₃, NO₂, CO e SO₂. |
| Horizonte de previsão (deslocamento aplicado) | Horizonte de até **24 horas à frente**. O deslocamento temporal será formalizado na Sprint 2 de acordo com a definição final do alvo. |
| Nível de desbalanceamento no treino (% positivos / negativos) | **A calcular na Sprint 2**, após a definição formal da classe positiva e a separação dos dados de treino. |
---

## 4. Atributos derivados (Sprints 2 e 4)

| Nome do atributo | Variável(is) de origem | Tipo de transformação | Janela/parâmetro (definido no treino) | Calculável no instante da previsão? | Justificativa | Sprint (2 ou 4) | Feature ou alvo? |
|---|---|---|---|---|---|---|---|
| A definir na Sprint 2 | — | — | — | — | — | 2 | — |

*Tipo de transformação: média móvel / valor defasado (lag) / agregação (soma, contagem) / variável de calendário / outro (especificar).*

> Cada linha precisa de justificativa — não copiar só o nome da coluna. Parâmetros de janela **não** se reajustam no teste. A Sprint 4 acrescenta linhas novas; não apaga as da Sprint 2 se ainda estiverem no modelo (pode marcar “descartada na seleção”). Timestamp, código IBGE e nome de município **não** entram como número; calendário (mês, safra) vale. “Calculável no instante da previsão” = sim só se os dados de origem já existiriam na hora da decisão (mesmo horizonte do RFC).

---

## 5. Variáveis excluídas das features (risco de vazamento)

| Nome da coluna | Motivo da exclusão |
|---|---|
| **A definir na Sprint 2** | A exclusão será avaliada após a construção da variável-alvo e identificação das informações que não estariam disponíveis no instante real da previsão. |

> Incluir colunas usadas para construir o alvo e qualquer informação que só existiria depois do evento ou depois do instante de previsão. Elas **não** entram no `ColumnTransformer`.

---

## 6. Observações gerais e limitações do dataset

- Os dados de qualidade do ar obtidos pela **Open-Meteo Air Quality API são modelados**, provenientes do CAMS (Copernicus Atmosphere Monitoring Service). Para São Paulo, entra a cobertura do CAMS Global Atmospheric Composition Forecasts, e não correspondem a medições diretas realizadas por uma estação local.

- Os dados representam uma determinada área geográfica e, portanto, podem não refletir exatamente as condições de qualidade do ar observadas em todos os pontos da cidade de São Paulo.

- O projeto utiliza o período de **06/09/2024 a 06/09/2026**, o que limita a análise a aproximadamente dois anos de dados.

- Os dados são trabalhados em **resolução horária**. No caso da qualidade do ar, a fonte CAMS Global possui resolução temporal nativa de 1 hora, sendo os valores horários disponibilizados pela Open-Meteo por meio de interpolação temporal.

- Os dados meteorológicos históricos também são provenientes de modelos/reanálises e podem apresentar diferenças em relação às condições observadas por estações meteorológicas locais.

- A integração entre as fontes depende da compatibilidade dos timestamps, resolução temporal e fuso horário dos dados de qualidade do ar e clima.

- Possíveis valores ausentes, duplicados ou inconsistentes serão identificados e documentados na **Sprint 2**, sem alteração dos arquivos originais armazenados em `data/raw/`.

- As recomendações futuras do projeto considerarão principalmente a **qualidade do ar e as variáveis disponíveis no dataset**, não representando uma avaliação individual de saúde ou condição física do usuário.
---

## 7. Histórico de alterações

| Versão | Sprint | Data | O que mudou |
|---|---|---|---|
| v0.1 | 1 | | Fontes, variáveis brutas, N após o merge (bruto) |
| v0.2 | 2 | | Log de limpeza; N em `interim`; split; alvo; derivados iniciais; exclusões |
| v0.3 | 4 | | Features iteradas; seleção |
| v0.4 | 5 | | Conferência com o modelo final / model card |
