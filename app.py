use_container_width=True
        )

    col_melhor1, col_melhor2 = st.columns(2)
    with col_melhor1:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #059669;">
            <h4 style="color: #059669; margin-top: 0;">🥇 Vencedor Geral para Produção: Random Forest (Limiar 0,40)</h4>
            <p><strong>Por que foi o melhor modelo global:</strong></p>
            <ul>
                <li><strong>Maior ROC-AUC do projeto (0,9006):</strong> Liderança absoluta na capacidade de discriminar ar puro de ar poluído em qualquer ponto de corte.</li>
                <li><strong>Menor taxa de alarmes falsos (Falsos Positivos):</strong> Apenas <strong>328 FPs</strong> contra 495 da Árvore e 526 do XGBoost, evitando cancelamentos desnecessários de treinos.</li>
                <li><strong>Alta Acurácia (87,70%):</strong> Excelente equilíbrio probabilístico e robustez de ensemble (200 árvores com bagging e seleção aleatória de atributos que eliminam ruídos atmosféricos).</li>
                <li><strong>Redução de Falsos Negativos:</strong> Baixou os 154 erros da persistência para <strong>100</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_melhor2:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #2563eb;">
            <h4 style="color: #2563eb; margin-top: 0;">🛡️ Vencedor em Proteção Máxima ao Atleta: Decision Tree (Limiar 0,35)</h4>
            <p><strong>Por que se destaca em segurança crítica:</strong></p>
            <ul>
                <li><strong>Maior Recall de todo o projeto (75,92%):</strong> Detectou mais de 3 em cada 4 episódios críticos de poluição com 24h de antecedência.</li>
                <li><strong>Menor número absoluto de Falsos Negativos (72):</strong> Uma <strong>redução drástica de 53,2%</strong> nos erros fatais frente à Persistência Temporal (154 erros).</li>
                <li><strong>Maior Score F₂ (0,5918):</strong> Líder na métrica prioritária de saúde respiratória.</li>
                <li><strong>Transparência Total (White-box):</strong> Regras explicáveis em texto e visualizáveis em árvore gráfica.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📐 Quais Foram os Critérios Formais para Aprovação do Modelo e o Porquê?")

    c_crit1, c_crit2 = st.columns(2)
    with c_crit1:
        st.markdown(r"""
        ##### 1️⃣ Critério 1: Superar o Piso Real da Persistência Temporal
        * **Definição:** Em meteorologia, a *Persistência Temporal* (chutar que amanhã o ar estará igual ao de hoje) é uma baseline forte devido à inércia do ar, atingindo $F_2 = 0,4840$ e Recall de 48,49%.
        * **Critério de Aprovação:** O modelo de ML só tem viabilidade se bater a persistência em Recall e $F_2$.
        * **Resultado:** Random Forest ($F_2=0,5775$), Decision Tree ($F_2=0,5918$) e XGBoost ($F_2=0,5710$) **superaram amplamente o piso**.
        <br>
        ##### 2️⃣ Critério 2: Minimização Rigorosa de Falsos Negativos (Métrica $F_2$)
        * **Definição:** A métrica oficial é o $F_2$ ($\beta=2$):
          $$F_2 = \frac{5 \cdot \text{Precisão} \cdot \text{Recall}}{4 \cdot \text{Precisão} + \text{Recall}}$$
        * **Por que adotar $\beta=2$?** Em atividades físicas ao ar livre, o volume respiratório de um atleta sobe de $6\,\text{L/min}$ para até $150\,\text{L/min}$.
        * **Falso Negativo (Grave):** O sistema não avisa, e o atleta treina sob ozônio tóxico ou $\text{PM}_{2,5}$, sofrendo inflamação alveolar e estresse oxidativo.
        * **Falso Positivo (Tolerável):** O sistema avisa ar ruim, mas o ar estava razoável; o atleta apenas transfere o treino para a esteira por precaução.
        * **Conclusão:** O Recall tem peso **4 vezes maior** que a Precisão!
        """, unsafe_allow_html=True)

    with c_crit2:
        st.markdown(r"""
        ##### 3️⃣ Critério 3: Reprovação Sumária da Armadilha do DummyClassifier
        * **Definição:** O ar em São Paulo é inadequado ($\text{N3+}$) em apenas ~8,5% das horas.
        * **O Perigo da Acurácia Pura:** Um classificador ingênuo (*Dummy*) que sempre responde "ar adequado" atinge impressionantes **91,41% de acurácia**, mas seu **Recall é 0,0%** (deixa passar todos os 299 episódios críticos).
        * **Critério:** Modelos com Recall nulo ou sem discriminação são imediatamente **reprovados**.
        <br>
        ##### 4️⃣ Critério 4: Calibração de Limiar sem Vazamento Temporal
        * **Definição:** Em vez do limiar ingênuo de 0,50, os limiares ótimos ($0,40$ para RF, $0,35$ para DT e $0,25$ para XGB) foram calibrados exclusivamente nas janelas de validação do passado (*Out-of-Fold Temporal* com purga de 24h).
        * **Garantia:** Zero vazamento de dados do futuro para o passado (*data leakage*).
        """, unsafe_allow_html=True)



# VISÃO 6: EXECUTAR PIPELINE COMPLETO
elif menu == "⚡ 6. Executar Pipeline Completo":
    render_hero_header(
        titulo="⚡ Orquestrador do Pipeline de Ponta a Ponta",
        subtitulo="Dispare a execução completa e automatizada: coleta, tratamento, modelagem e comparação contra baselines",
        badges=[
            "🔄 Pipeline Automatizado",
            "🧼 Limpeza e Purga 24h",
            "🧠 Treinamento e Calibração",
            "📊 Geração de Relatórios"
        ]
    )

    st.markdown("""
    <div class="metric-card" style="border-left: 4px solid #3b82f6; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #1e3a8a;">Fluxo de Execução Sequencial</h4>
        <ol style="margin-bottom: 0; padding-left: 20px; line-height: 1.8;">
            <li><strong>Coleta e Integração:</strong> Consulta ou verifica o repositório bruto das APIs públicas da Open-Meteo.</li>
            <li><strong>Limpeza e Split Temporal:</strong> Trata sentinelas, padroniza unidades e aplica purga temporal estrita de 24 horas.</li>
            <li><strong>Modelagem e Calibração:</strong> Treina Decision Tree, Random Forest e XGBoost com ColumnTransformer e calibra limiares por F2.</li>
            <li><strong>Comparação e Auditoria:</strong> Avalia contra a Persistência Temporal e gera relatórios e métricas.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "🚀 Executar Pipeline Completo Agora",
        type="primary",
        use_container_width=True
    ):
        import pipeline_completo

        with st.spinner("Executando pipeline completo... Isso pode levar de 30 a 60 segundos."):
            res = pipeline_completo.executar_pipeline_completo(
                forcar_recoleta=False
            )

            st.success(
                f"Pipeline executado com sucesso em {res['tempo_execucao_s']:.2f} segundos!"
            )

            st.dataframe(
                pd.DataFrame(res["comparativo"]),
                use_container_width=True
            )
