# Diário de Sprint 1 — Kickoff e coleta bruta

**Período:** 17/08/2026 a 17/09/2026
**Trilha definitiva do grupo:** B — Qualidade do ar inadequada
**Predição da Qualidade do Ar com Foco nos Melhores Horários para a Prática de Exercícios ao Ar Livre em São Paulo**
**Equipe:*Equipe 2*
**Integrantes:*• Adrielle Stollemberger RGM: 33948844 • Victor Almeida de Aquino RGM: 32901321 • Nicolas Santos Silva RGM: 3287380 • Samir Abdul Khalek RGM: 32657994 • João Pedro Garcia Almeida RGM: 32847629*
**Scrum Master do Sprint:*Nicolas Santos Silva*
**Repositório GitHub:** (link)

> Ingestão (Pipeline CD) = coleta e integração do **bruto** — não confundir com Continuous Delivery.
>
> Esta sprint **só entrega dado cru**: problema, RFC, dicionário das variáveis brutas, coleta das duas APIs, merge e `data/raw/` intocável. **Não há** limpeza, tratamento, split, EDA, engenharia de atributos nem modelo. Isso é a Sprint 2.

### Contrato desta sprint

| | Artefato | Quem usa depois |
|---|---|---|
| **Entra** | Enunciado da Trilha B (nada de sprint anterior) | — |
| **Sai** | RFC v0.1 (problema, horizonte, custo FN/FP, fora de escopo) | Sprint 2 (alvo formal) e 5 (limiar) |
| **Sai** | Dicionário v0.1 (fontes e variáveis brutas) | Sprint 2 |
| **Sai** | `data/raw/` + `config/` + `requirements.txt` (APIs, merge, N após o merge) | Sprint 2 **é obrigada a usar este bruto** |
| **Sai** | Repositório, board, este diário | Sprints seguintes |

**Não sai daqui:** `data/interim`, split, limiar do alvo, features, modelo.

---

## 1. Definição do problema (Canvas de Kickoff)

| Pergunta | Resposta |
|---|---|
| Qual evento será previsto? |A predição de horários com qualidade do ar inadequada e com horários adequeados para a prática de exercícios ao ar livre em São Paulo.|
| Para qual cidade/região e período? |São Paulo – SP, utilizando dados históricos de 06/09/2024 a 06/09/2026.|
| Qual é a unidade de análise (o que representa cada linha após o merge)? |Cada linha representará uma hora em São Paulo, reunindo os dados de qualidade do ar e os dados meteorológicos correspondentes ao mesmo período após a integração das fontes.|
| Qual é o horizonte da previsão? |A previsão terá horizonte de até 24 horas à frente, com o objetivo de identificar antecipadamente quais horários apresentam condições mais ou menos favoráveis para exercícios ao ar livre.|
| Quem usaria o alerta e qual decisão ele apoiaria? |O alerta poderá ser utilizado por praticantes de atividades físicas ao ar livre, tanto amadores quanto atletas profissionais. A informação apoiará a decisão de escolher, alterar ou evitar determinado horário de treino ou atividade física, considerando a qualidade do ar prevista.|
| Como será definida a classe positiva (provisoriamente)? | Provisoriamente, a classe positiva (1) representará um horário com qualidade do ar inadequada para exercícios ao ar livre, considerando os níveis dos poluentes atmosféricos disponíveis. A definição e o limiar formal da classe serão estabelecidos na Sprint 2, após a análise exploratória dos dados de treino.|
| Quais dados estarão disponíveis no momento real da previsão? |Dados históricos de qualidade do ar disponíveis até o momento da previsão, incluindo PM2.5, PM10, CO, NO₂, SO₂ e O₃, além das informações de data e horário dos registros. Outras variáveis somente serão consideradas caso sejam incorporadas e validadas nas próximas Sprints do projeto.|
| Qual é o custo de um falso negativo e de um falso positivo? | Falso negativo: o sistema considera um horário adequado quando, na realidade, a qualidade do ar está inadequada, podendo aumentar a exposição do praticante ou atleta aos poluentes durante o exercício. Falso positivo: o sistema considera um horário inadequado quando as condições estão adequadas, podendo fazer com que o usuário altere ou evite um treino desnecessariamente. O falso negativo é considerado mais grave para o projeto.|
| Justificativa da escolha da Trilha B (relevância, disponibilidade de dados, viabilidade) | A Trilha B — Qualidade do ar inadequada foi escolhida pela relevância da qualidade do ar para pessoas que realizam exercícios ao ar livre e pela possibilidade de apoiar a escolha de horários mais favoráveis para essas atividades. O projeto apresenta disponibilidade de dados históricos de qualidade do ar por meio de APIs, além de viabilidade técnica para coleta, integração, análise e posterior desenvolvimento de um modelo preditivo.|

<img src="https://img.shields.io/badge/feito-008000?style=flat-square" />  RFC da Sprint 1 preenchido a partir deste canvas 
<br>
<img src="https://img.shields.io/badge/feito-008000?style=flat-square" />  Dicionário v0.1 só com fontes e variáveis brutas 

## 2. Documentação das APIs e parâmetros

**Tabela de variáveis escolhidas:**

| Variável | API | Unidade | Justificativa | Feature ou alvo? |
|---|---|---|---|---|
|Pm2_5 | Open-Meteo Air Quality API| µg/m³ | Partículas relevantes para avaliar a condição da qualidade do ar. | Ambos |
|Pm10 | Open-Meteo Air Quality API| µg/m³ | Representa partículas inaláveis e complementa a avaliação da concentração de material particulado.| Ambos |
|O₃ | Open-Meteo Air Quality API| µg/m³ | Poluente relevante para avaliação da qualidade do ar e exposição ao ar livre. | Ambos |
|NO₂ | Open-Meteo Air Quality API|µg/m³ | Poluente atmosférico relacionado principalmente a processos de combustão e importante para caracterizar a qualidade do ar. | Ambos |
|CO | Open-Meteo Air Quality API| µg/m³| Complementa a caracterização da presença de poluentes provenientes de processos de combustão. | Ambos |
|SO₂ | Open-Meteo Air Quality API| µg/m³ | Complementa a análise da presença de gases poluentes na atmosfera. | Ambos |


<img src="https://img.shields.io/badge/feito-008000?style=flat-square" /> Para cada API (Open-Meteo Air Quality API): endpoint, parâmetros obrigatórios, resolução temporal, período histórico disponível e limitações registrados

• Endpoint: /v1/air-quality

• Parâmetros utilizados: latitude, longitude, start_date, end_date e hourly, com as variáveis de qualidade do ar selecionadas.
Latitude e longitude identificam a localização; hourly determina quais variáveis serão retornadas.

• Resolução temporal: os dados utilizados são horários. Para São Paulo, a documentação indica que a fonte global CAMS tem resolução nativa de 3 horas; a API apresenta dados horários.

• Período utilizado pelo projeto: 06/09/2024 a 06/09/2026.

• Limitações: por serem provenientes de modelos atmosféricos, os valores representam uma estimativa para uma célula espacial e não uma medição realizada exatamente no ponto onde uma pessoa realizará o exercício. A fonte global CAMS também possui resolução espacial relativamente ampla, de aproximadamente 45 km.

<img src="https://img.shields.io/badge/feito-008000?style=flat-square" /> Dados de poluentes declarados como **medidos ou modelados**

 <img src="https://img.shields.io/badge/feito-008000?style=flat-square" /> Seleção criteriosa de variáveis (não todas as disponíveis)

## 3. Coleta bruta

- [❌] Parâmetros em configuração externa (não hard-coded)
Hoje latitude, longitude, datas e variáveis estão dentro do código em PARAMS. Precisam ir para um arquivo de configuração, por exemplo config/config.json.

- [⚠️] Requisição HTTP com `timeout`, `raise_for_status` e tratamento de exceções
Possuimos o retry, mas o código atual não mostra explicitamente timeout, raise_for_status nem try/except. Precisamos ajustar ainda.

- [❌] Resposta inspecionada antes de transformar (status, headers, estrutura do JSON) — só para confirmar que a coleta funcionou
O código vai direto de responses para transformação. Precisamos registrar/verificar status, headers e estrutura da resposta antes do processamento.

- [❌] As duas fontes (qualidade do ar + clima) coletadas
O código possui coleta apenas Open-Meteo Air Quality. Ainda precisamos implementar a coleta meteorológica.

- [⚠️] `requirements.txt` (ou equivalente) com as bibliotecas da coleta
Em processo de criação.

- [❌] Arquivos da API gravados em `data/raw/` (JSON/CSV da resposta) e **não sobrescritos** depois
Atualmente salva dados_saida/qualidade_do_ar_por_dia.csv. Precisamos salvar os dados brutos em data/raw/, preservando-os.

```
 Legendas
 
 ❌ - Não atende
 ⚠️ - Atende parcial
 ✅ - Atende
```


**Evidências (prints, trechos de código, link do notebook/commit):**
<hr> 
<img width="739" height="485" alt="print git" src="https://github.com/user-attachments/assets/f0afac57-be23-44fb-84fa-1503062ae7a8" />
<hr>

## 4. Integração do bruto (ainda sem limpar)

Juntar as fontes numa tabela de trabalho. Duplicatas, ausentes e inválidos **não se corrigem** aqui — apenas se registra o N do merge.

- [ ] Colunas de tempo convertidas corretamente em cada fonte
- [ ] Junção por timestamp com `how` e `validate` justificados
- [ ] Cardinalidade e fuso/resolução verificados
- [ ] N após o merge registrado
- [ ] Tabela integrada salva em `data/raw/integrado.csv` (ou equivalente), **sem** substituir os arquivos da API

**N após o merge (bruto):**
**Evidências (link do notebook/commit):**

## 5. Organização Scrum

- [✅] Papéis definidos
Product Owner = Docente - Andrea Ono Sakai 
Scrum Master do sprint = Nicolas Santos Silva 
Development Team = Adrielle, Victor, João e Samir 

- [✅] Board criado 

- [✅] Backlog inicial com pelo menos 3 user stories

**User stories do backlog inicial:**
1. Como praticante de exercícios ao ar livre, quero saber se a qualidade do ar estará adequada nas próximas horas, para escolher um horário mais favorável para realizar minha atividade.
2. Como praticante de exercícios ao ar livre, quero visualizar quais horários apresentam melhores condições de qualidade do ar, para planejar meu treino ao longo do dia.
3. Como usuário, quero receber um alerta quando houver previsão de qualidade do ar inadequada, para considerar alterar o horário da minha atividade ao ar livre.

**Link do board:[BOARD](https://github.com/Adri22K/Projeto-TrilhaB/issues/2#issue-5481547970)**


## 6. Diário de bordo (retrospectiva individual)

| Integrante | O que fiz nesta Sprint | Dificuldades | O que pretendo manter/ajustar |
|---|---|---|---|
| Adrielle | Preenchi a Sprint 1 com base no ipynb criado na aula do dia 31/08/2026 | A tabela de clima ainda não havia sido criada | Pretendo criar o merge com a tabela de clima que ainda está em construção |
| Victor |Preenchi a Sprint 1 com base no ipynb criado na aula do dia 31/08/2026 | A tabela de clima ainda não havia sido criada | Pretendo criar o merge com a tabela de clima que ainda está em construção |
| Nicolas |Preenchi a Sprint 1 com base no ipynb criado na aula do dia 31/08/2026 | A tabela de clima ainda não havia sido criada | Pretendo criar o merge com a tabela de clima que ainda está em construção |


## 7. Evidências gerais

- Link do RFC: [RFC](https://github.com/Adri22K/Projeto-TrilhaB/blob/6007a278c88fa1a6ae0ff768201da7b66456c039/docs/RFC.md)

- Link do dicionário v0.1:

- Link de commits desta Sprint: [SPRINT #1](https://github.com/Adri22K/Projeto-TrilhaB/commits/main/)

- Link do board atualizado: [BOARD](https://github.com/Adri22K/Projeto-TrilhaB/issues/2#issue-54815479701)

- Link do notebook / `config` / `data/raw`: [NOTBOOK](https://github.com/Adri22K/Projeto-TrilhaB/tree/6007a278c88fa1a6ae0ff768201da7b66456c039/config)

---

## Rubrica de avaliação — Sprint 1 (nota de 0 a 4,0)

| Critério | Peso | O que caracteriza nota máxima | Nota atribuída | Observações |
|---|---|---|---|---|
| Definição do problema (canvas + RFC) | 0,5 | Perguntas respondidas; unidade de análise e horizonte explícitos; RFC legível | | |
| Documentação das APIs e parâmetros | 0,5 | Duas APIs documentadas, seleção justificada, medido vs. modelado declarado | | |
| Coleta bruta | 1,0 | Config externa, erros tratados, duas fontes em `data/raw` intocável | | |
| Integração do bruto | 1,0 | Datas corretas, `how`/`validate` justificados, N do merge registrado, **sem limpar** | | |
| Scrum + diário de bordo | 1,0 | Papéis, board, backlog e diário reflexivo de todos | | |
| **Nota final da Sprint 1** | **4,0** | | **___ / 4,0** | |
