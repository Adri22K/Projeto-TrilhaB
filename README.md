# 📍  Ciência de Dados e Aprendizado de Máquina

Esse repositorio será sobre o desenvolvimento de um Data Frame para apresentação da Mostra de Tecnologia:


| Trilha | Problema | Fontes mínimas |
|---|---|---|
| B | Qualidade do ar inadequada | Open-Meteo Air Quality + Open-Meteo Historical Weather |

Neste material, **Pipeline CD** = coleta e integração do bruto (Sprint 1). Não é Continuous Delivery e **não** inclui limpeza — limpeza é Sprint 2.

---

## 1. O que o grupo preenche

| Artefato | Quando | O que é |
|---|---|---|
| `RFC_Proposta_de_Projeto_Template.md` | Sprint 1 (revisar se o alvo mudar) | Proposta do problema, escopo, usuários, custo dos erros |
| `Dicionario_de_Dados_Template.md` | Sprint 1 → 5 | Contrato das variáveis |
| `SprintN_TrilhaB.md` ou `SprintN_TrilhaC.md` | Cada sprint | Diário, evidências, retrospectiva individual |

`SprintN_Template_Correcao.md` é do corretor.

Não há Sprint 6 neste projeto. Dashboard, se existir na mostra, é extra.

---

## 2. Ordem das sprints (única sequência válida)

```text
Sprint 1  17/08–17/09   RFC, coleta bruta, merge, `data/raw`
Sprint 2  18/09–27/09   Limpeza e tratamento → split → EDA → alvo → features
Sprint 3  28/09–04/10   Ingestão robusta + Dummy + persistência + Naive Bayes
Sprint 4  05/10–11/10   Features novas a partir dos erros; retreino; pipeline congelado
Sprint 5  12/10–25/10   Todos os modelos no pipeline final; limiar; model card
```

Regras que atravessam as sprints:

1. A aula da Trilha A (chuva intensa) é **exemplo**. A entrega das Sprints 1–5 é sempre B ou C.
2. Sprint 1 entrega **só o bruto**. Limpeza, tratamento, EDA e features são a Sprint 2.
3. **Não se faz EDA em dado sujo.** Inspeção → limpeza → split → EDA no treino.
4. O teste não escolhe limiar, janelas, hiperparâmetros, imputação **nem** lista de features. Limiar se escolhe numa validação temporal **dentro do treino**.
5. Features novas na Sprint 4 obrigam **retreino** dos baselines.
6. Trilha C: N ≈ municípios × safras **depois da limpeza**.

---

## 2.1 Contrato entre sprints

Cada sprint **só consome** o que a anterior declarou como saída. Recoletar, relimpar ou mudar o split sem versionar quebra o contrato.

| De → para | Obrigatório usar | Obrigatório produzir | Proibido nesta etapa |
|---|---|---|---|
| Início → **1** | Enunciado da trilha | RFC v0.1, dicionário v0.1, `data/raw` + `config`, N do merge | Limpeza, EDA, features, modelo |
| **1 → 2** | `data/raw` da Sprint 1 | `data/interim`, log de limpeza, split, alvo formal, features iniciais, dicionário v0.2 | Dummy, F1, modelo |
| **2 → 3** | `interim`, split, alvo, features da Sprint 2 | Ingestão robusta, `ColumnTransformer` inicial, Dummy + persistência + NB, erros interpretados | RF/XGB como modelo final, limiar fino |
| **3 → 4** | Split S2, features S2, erros S3 | Features novas, transformer **congelado**, baselines retreinados, lift S3→S4, dicionário v0.3 | Seleção do modelo final, limiar |
| **4 → 5** | Split S2, transformer S4, lift S4 | Comparativo no pipeline final, limiar na validação, caderno, modelo + model card, dicionário v0.4, `joblib` em `models/` | Colar métricas da Sprint 3; novo split; otimizar limiar no teste |

O RFC amarra o problema (S1) ao limiar (S5). O dicionário é o contrato das colunas (v0.1 bruto → v0.2 limpo/alvo/features → v0.3 iteração → v0.4 modelo final).

---

## 2.2 Orientações de engenharia de ML

Regras que um engenheiro de ML exigiria neste projeto. Não são MLOps: são o mínimo para o experimento ser válido.

1. **Um `Pipeline` do sklearn, não um scaler solto.** `ColumnTransformer` + classificador no mesmo objeto. `fit` só no treino; teste só `predict` / `predict_proba`. O artefato em `models/` é esse objeto inteiro (`joblib`), não só o classificador.
2. **Semente e dependências.** `random_state` documentado onde o algoritmo for estocástico. `requirements.txt` (ou `environment.yml`) desde a Sprint 1, atualizado se a stack mudar.
3. **Feature tem de existir na hora de prever.** Cada coluna usada no modelo precisa ser calculável com dados disponíveis no instante da previsão (o RFC já pergunta isso). Janela de 24 h com horizonte de 24 h usa só o passado; clima do fim da safra não entra se a decisão é no plantio.
4. **Seleção de atributos só no treino.** Correlação, importância ou filtro calculados no teste (ou no dataset inteiro) vazam. A Sprint 4 escolhe features no treino e aplica a lista ao teste.
5. **Limiar não se otimiza no teste.** Recortar o **final do treino** como validação temporal (ou `TimeSeriesSplit` só dentro do treino), escolher o ponto de operação pelo custo de FN **nessa validação**, congelar o limiar e **medir o teste uma vez**. Olhar 0,3 / 0,5 / 0,7 no teste e ficar com o melhor é vazar o teste.
6. **Desbalanceamento: medir e tratar com método adequado.** Dummy `most_frequent` é piso de **acurácia**, não de recall na classe rara. `class_weight="balanced"` (ou equivalente) é lícito. SMOTE / oversampling aleatório em série temporal **não** — inventa instantes que não existiram.
7. **Se o baseline ganhar, o baseline é o resultado.** Persistência ou Dummy no pipeline final com recall/F1 iguais ou melhores que RF/XGBoost não é falha de entrega: é o modelo a ser escolhido. Complexidade extra precisa de ganho no teste, no limiar congelado.
8. **Métrica com N pequeno se reporta em contagem.** Trilha C (e testes curtos na B): escrever “6 FN em 11 eventos” além de “recall 0,55”. Percentual sozinho mente quando o denominador é dezena.
9. **Caderno de experimentos.** Uma tabela: modelo, conjunto de features (S2 ou S4), `random_state`, limiar, recall/precisão/F1 na **validação** e no **teste**. Sem isso não dá para saber o que foi tentado.
10. **Mesmas regras na coleta nova.** Ampliar período na Sprint 3 implica reaplicar o log de limpeza da Sprint 2 e o critério de split — não um tratamento “de emergência” diferente.
11. **Identificador não é feature numérica.** Timestamp, código IBGE e nome de município não entram como `float`. Calendário (mês, estação, safra) vale. Município como categoria só se **todos** os municípios do teste existirem no treino; mesmo assim o modelo pode só memorizar o lugar — justificar ou deixar de fora.
12. **Acurácia não escolhe o modelo.** O critério de seleção é a métrica da classe positiva do RFC (em geral recall/F1), no limiar congelado na validação e então medido no teste.

---

## 3. Arquitetura mínima

```text
                    RFC (problema, usuário, custo FN/FP)
                                    │
                                    ▼
         APIs ──► coleta ──► data/raw/  (resposta da API + integrado.csv)
                                    │
                                    ▼
              inspeção → limpeza e tratamento  (Sprint 2)
                                    │
                                    ▼
                         data/interim/  (limpo, sem features de modelo)
                                    │
                                    ▼
                         split temporal / por safra
                           treino │ teste (intocado)
                                  ▼
              imputação residual só com estatística do treino
                                  ▼
              EDA da variável de origem → alvo → desbalanceamento
                                  ▼
              engenharia de atributos sem dado futuro
                                  ▼
                    data/processed/  (matriz para o modelo)
                                  ▼
         sklearn Pipeline (ColumnTransformer + classificador)
           fit exclusivo no treino; teste só em predict
                                  ▼
           baseline → iteração medida (lift) → modelos + limiar
                                  ▼
                    artefato + dicionário + model card
```

Pastas: `raw` = bruto da API/merge; `interim` = limpo (Sprint 2); `processed` = com features, pronto para treinar (Sprints 3–5).

---

## 4. Estrutura do repositório

```text
projeto-trilha-b-ou-c/
├── README.md
├── requirements.txt
├── config/
│   └── params.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_ingestao.ipynb
│   ├── 02_limpeza_eda_features.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_features_iteracao.ipynb
│   └── 05_modelos.ipynb
├── src/                         # opcional
├── models/
├── docs/
│   ├── RFC.md
│   ├── Dicionario_de_Dados.md
│   └── sprints/
└── reports/
```

`data/` grande não sobe para o Git; sobe um `data/README.md` com o comando de recoleta.

---

## 5. README do repositório do grupo (copiar e preencher)

```markdown
# 🌱 Predição da Qualidade do Ar com Pandas 💨

**Disciplina:** Ciência de Dados e Aprendizado de Máquina
**Trilha:** B 
** Equipe:**
- Adrielle Stollemberger RGM: 33948844
- Victor Almeida de Aquino RGM: 32901321
- Nicolas Santos Silva RGM: 3287380
- Samir Abdul Khalek RGM: 32657994
- João Pedro Garcia Almeida RGM: 32847629

**Repositório / board:**

## Problema

Evento, usuário da decisão e horizonte (3 frases).

Classe positiva:
Custo priorizado (FN ou FP):

## Como reproduzir

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Configuração: `config/params.yaml` (nada de município/datas no código).
Notebooks: 01 → 05.

## Dados

| Fonte | Papel | Resolução | Medido ou modelado | Período |
|---|---|---|---|---|
| | | | | |

Unidade de análise:
N após o merge:
Split:
Dicionário: `docs/Dicionario_de_Dados.md`

## Modelo

Baseline:
Modelo final:
Limiar (escolhido na validação) e por quê:
Métrica principal na classe positiva (teste, uma vez):

## Documentação

- RFC: `docs/RFC.md`
- Sprints: `docs/sprints/`
- Model card: no diário da Sprint 5
```

---

## 6. Checklist antes de chamar o modelo de pronto

- [ ] RFC e dicionário batem com o notebook (mesmo alvo, mesmo horizonte).
- [ ] Duas fontes integradas com `how`/`validate` conscientes.
- [ ] Inspeção de qualidade **antes** da correção (Sprint 2); limpeza com log e `data/interim`; `data/raw` da Sprint 1 intocado.
- [ ] EDA feita na tabela tratada, não no bruto.
- [ ] Desbalanceamento com porcentagem real **do treino**.
- [ ] Nenhuma feature usa informação posterior ao instante de previsão; IDs/timestamps não vão como número.
- [ ] Colunas do alvo estão na seção de exclusão do dicionário.
- [ ] Split temporal (B) ou por safra (C); teste = período mais recente.
- [ ] Um `Pipeline` sklearn (`ColumnTransformer` + classificador); `fit` só no treino; artefato = `joblib` desse objeto.
- [ ] Dummy e persistência como piso; se ganharem no teste, **eles** são o modelo final.
- [ ] Limiar escolhido na **validação** temporal (fim do treino); teste medido **uma vez**.
- [ ] Caderno de experimentos (modelo, features, seed, limiar, val. e teste).
- [ ] Matriz de confusão com FN/FP em **contagem**; erros concretos (datas ou município/safra).
- [ ] Sem SMOTE em série/safra; `class_weight` ok; acurácia não escolhe o modelo.
- [ ] Trilha C: N declarado; complexidade compatível.
- [ ] `requirements.txt` e `config/` permitem reexecutar a ingestão.
- [ ] Model card mínimo da Sprint 5 preenchido.

---

## 7. Mapa dos templates

| Arquivo | Público |
|---|---|
| `RFC_Proposta_de_Projeto_Template.md` | Grupo |
| `Dicionario_de_Dados_Template.md` | Grupo |
| `Sprint{1–5}_TrilhaB.md` | Grupo na Trilha B |
| `Sprint{1–5}_TrilhaC.md` | Grupo na Trilha C |
