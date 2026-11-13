"""CADENC.IA — Dashboard Interativo de Inteligência Atmosférica e Previsão da Qualidade do Ar.

Trilha B: Qualidade do ar inadequada para exercícios ao ar livre em São Paulo (Horizonte 24h).
Apresenta o pipeline completo de ponta a ponta:
- Coleta de dados abertos (Open-Meteo)
- Análise exploratória e tratamento
- Modelagem e calibração (Decision Tree, Random Forest, XGBoost)
- Comparação direta contra a Baseline (Persistência Temporal e DummyClassifier)
- Simulador de inferência e recomendações para atletas

Como executar:
    streamlit run app.py
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="CADENC.IA | Previsão de Qualidade do Ar",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.0,
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "text.color": "#1e293b",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
})

def estilizar_eixo(ax):
    """Aplica acabamento minimalista e moderno a qualquer eixo Matplotlib."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner do Dashboard */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
        color: white;
        padding: 26px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #e2e8f0;
        margin: 0 0 14px 0;
        line-height: 1.5;
        font-weight: 400;
    }
    .hero-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Cards e Containers Modernos */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.06);
    }
    .highlight-card {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
        color: #1e3a8a;
        line-height: 1.6;
    }
    .alert-card-danger {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
        color: #991b1b;
        line-height: 1.6;
    }
    .alert-card-success {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
        color: #166534;
        line-height: 1.6;
    }

    /* Estilização nativa dos componentes st.metric */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* Badges de Status */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-ok {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-alert {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
    }

    /* Barra Lateral e Navegação */
    /* Barra lateral CADENC.IA — alto contraste e identidade visual */
    [data-testid="stSidebar"] {
        background: #f4f0df !important;
        border-right: 1px solid #d5dfd7 !important;
    }

    /* Textos, títulos e descrições da barra lateral */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] small {
        color: #0f3b4c !important;
    }

    /* Opções do menu de navegação */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #0f3b4c !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* Marcador das opções */
    [data-testid="stSidebar"] [role="radio"] {
        accent-color: #1f7a7a;
    }

    /* Cartão de status do pipeline */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
        color: #0f3b4c !important;
    }

    /* Divisórias e legenda */
    [data-testid="stSidebar"] hr {
        border-color: #cbd8d1 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #526b72 !important;
    }
</style>
""", unsafe_allow_html=True)

def render_hero_header(titulo: str, subtitulo: str, badges: list[str]):
    """Gera o cabeçalho estilizado padrão para todas as visões."""
    badges_html = "".join([f'<span class="hero-badge">{b}</span>' for b in badges])
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">{titulo}</div>
        <div class="hero-subtitle">{subtitulo}</div>
        <div class="hero-badge-container">{badges_html}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def carregar_dados():
    """Carrega dados brutos, tratados, features e relatórios com cache."""
    caminho_interim = ROOT / "data/interim/integrado_limpo.csv"
    caminho_teste = ROOT / "data/processed/teste_features_s4.csv"
    caminho_treino = ROOT / "data/processed/treino_features_s4.csv"
    caminho_cmp = ROOT / "reports/modelos_decision_tree_random_forest/comparativo_completo_decision_tree_vs_random_forest.csv"
    caminho_erros = ROOT / "reports/modelos_decision_tree_random_forest/erros_detalhados_teste.csv"
    caminho_params = ROOT / "config/params.yaml"

    interim = pd.read_csv(caminho_interim, parse_dates=["timestamp"]) if caminho_interim.is_file() else None
    teste = pd.read_csv(caminho_teste, parse_dates=["timestamp"]) if caminho_teste.is_file() else None
    treino = pd.read_csv(caminho_treino, parse_dates=["timestamp"]) if caminho_treino.is_file() else None
    comparativo = pd.read_csv(caminho_cmp) if caminho_cmp.is_file() else None
    erros = pd.read_csv(caminho_erros, parse_dates=["timestamp"]) if caminho_erros.is_file() else None

    params = yaml.safe_load(caminho_params.read_text(encoding="utf-8")) if caminho_params.is_file() else {}
    features = list(params.get("sprint4", {}).get("features_congeladas", []))

    return interim, treino, teste, comparativo, erros, features


@st.cache_resource
def carregar_pipelines():
    """Carrega os 3 modelos de machine learning treinados em models/."""
    modelos = {}
    for nome, arq in [
        ("Decision Tree", "pipeline_decision_tree.joblib"),
        ("Random Forest", "pipeline_random_forest.joblib"),
        ("XGBoost", "pipeline_xgboost.joblib"),
    ]:
        caminho = ROOT / "models" / arq
        if caminho.is_file():
            modelos[nome] = joblib.load(caminho)
    return modelos


interim, treino, teste, comparativo, erros, features = carregar_dados()
modelos = carregar_pipelines()

st.sidebar.image("docs/assets/cadencia-logo-clara.png" if (ROOT / "docs/assets/cadencia-logo-clara.png").is_file() else None, width=220)
st.sidebar.markdown("### Navegação")
menu = st.sidebar.radio(
    "Selecione uma visão:",
    [
        "📊 1. Visão Geral & Dados Abertos",
        "📈 2. Análise Exploratória (EDA)",
        "⚖️ 3. Comparativo: Baseline vs. Modelos",
        "🔬 4. Diagnóstico Detalhado dos Modelos",
        "📅 5. Testar Nova Data & Previsão 24h (API ao Vivo)",
        "⚡ 6. Executar Pipeline Completo",
    ]
)

st.sidebar.markdown("""
<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-top: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
    <div style="font-size: 0.78rem; font-weight: 800; color: #1e3a8a; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Status do Pipeline</div>
    <div style="font-size: 0.83rem; color: #334155; margin-bottom: 6px;">🟢 <strong>Coleta:</strong> 17.544 h (Open-Meteo)</div>
    <div style="font-size: 0.83rem; color: #334155; margin-bottom: 6px;">🟢 <strong>Features:</strong> 25 (Yeo-Johnson)</div>
    <div style="font-size: 0.83rem; color: #334155;">🟢 <strong>Modelos:</strong> 3 Homologados + Baseline</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption("Projeto Trilha B — Equipe 2 · Mostra Tecnológica")


# VISÃO 1: VISÃO GERAL & COLETA DE DADOS ABERTOS

if menu == "📊 1. Visão Geral & Dados Abertos":
    render_hero_header(
        titulo="🌱 CADENC.IA — O Ritmo do Ar",
        subtitulo="Previsão de Qualidade do Ar Inadequada para Atividades Físicas em São Paulo com 24h de Antecedência",
        badges=["🛰️ Open-Meteo APIs", "📍 São Paulo (-23.5489, -46.6388)", "⏱️ Horizonte 24h", "🛡️ Norma CETESB N3+", "⚡ 25 Features Congeladas"]
    )

    st.markdown("""
    <div class="highlight-card">
        <strong>🎯 Objetivo do Projeto:</strong> Auxiliar corredores, ciclistas e praticantes de exercícios ao ar livre a planejar treinos com <strong>24 horas de antecedência</strong>, 
        identificando previamente quando a concentração de poluentes atingirá níveis prejudiciais à saúde respiratória (classificação <strong>CETESB N3 — RUIM ou pior</strong>).
        <br><strong>⚠️ Custo Crítico Priorizado (RFC):</strong> <strong>Falso Negativo (FN)</strong> — deixar de alertar sobre o ar poluído expõe o atleta a respirar altas doses de ozônio e material particulado durante esforço cardiovascular intenso.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros Horários", "17.544 h", "2 anos completos")
    with col2:
        st.metric("Fontes de Dados Abertos", "2 APIs", "Open-Meteo Ar + Clima")
    with col3:
        st.metric("Features no Pipeline", f"{len(features)}", "Lags, Médias e Ciclos")
    with col4:
        taxa_pos = (treino['qualidade_inadequada_24h'].mean() * 100) if treino is not None else 13.91
        st.metric("Desbalanceamento Treino", f"{taxa_pos:.1f}%", "Classe Ruim (CETESB N3+)")

    st.markdown("### 🌐 Dados Abertos Coletados e Integrados")
    st.markdown("""
    Os dados são obtidos via requisições idempotentes e automatizadas a duas APIs abertas públicas da **Open-Meteo**, integradas horariamente pela coordenada central de São Paulo (`-23.5489, -46.6388`):
    - **Open-Meteo Air Quality API:** Poluentes atmosféricos horários (`PM10`, `PM2.5`, `O₃`, `NO₂`, `SO₂`, `CO`).
    - **Open-Meteo Historical Weather API:** Variáveis meteorológicas horárias de reanálise (`temperatura`, `velocidade do vento`, `precipitação`).
    """)

    if interim is not None:
        st.dataframe(
            interim.head(10)[["timestamp", "pm10", "pm2_5", "ozone", "nitrogen_dioxide", "temperature_2m", "wind_speed_10m", "precipitation"]],
            use_container_width=True
        )
        st.caption(f"Amostra dos dados limpos integrados. Total de observações: {len(interim):,} linhas horárias.")


#  VISÃO 2: ANÁLISE EXPLORATÓRIA (EDA)
elif menu == "📈 2. Análise Exploratória (EDA)":
    render_hero_header(
        titulo="📈 Análise Exploratória dos Dados Abertos (EDA)",
        subtitulo="Comportamento temporal de poluentes e variáveis climáticas em São Paulo frente aos limites regulatórios",
        badges=["📊 17.544 Observações", "💨 Ozônio (O₃)", "🌫️ Partículas (PM10 e PM2.5)", "🌡️ Temperatura & Vento"]
    )

    if interim is not None:
        col_sel, col_range = st.columns([1, 2])
        with col_sel:
            variavel = st.selectbox(
                "Selecione a variável para visualizar:",
                ["ozone", "pm10", "pm2_5", "temperature_2m", "wind_speed_10m", "precipitation", "nitrogen_dioxide"],
                format_func=lambda x: {
                    "ozone": "Ozônio (O₃) [µg/m³]",
                    "pm10": "Material Particulado (PM10) [µg/m³]",
                    "pm2_5": "Partículas Finas (PM2.5) [µg/m³]",
                    "temperature_2m": "Temperatura do Ar [°C]",
                    "wind_speed_10m": "Velocidade do Vento [km/h]",
                    "precipitation": "Precipitação [mm]",
                    "nitrogen_dioxide": "Dióxido de Nitrogênio (NO₂) [µg/m³]"
                }.get(x, x)
            )

        min_date = interim["timestamp"].min().date()
        max_date = interim["timestamp"].max().date()
        with col_range:
            datas = st.date_input("Intervalo de datas:", value=(min_date, min_date + pd.Timedelta(days=90)), min_value=min_date, max_value=max_date)

        if isinstance(datas, tuple) and len(datas) == 2:
            df_filtro = interim[(interim["timestamp"].dt.date >= datas[0]) & (interim["timestamp"].dt.date <= datas[1])]
        else:
            df_filtro = interim.head(1000)

        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(df_filtro["timestamp"], df_filtro[variavel], color="#0284c7", lw=1.5, label=variavel)
        
        if variavel == "ozone":
            ax.axhline(130, color="#ef4444", ls="--", lw=1.8, label="Limiar CETESB Ruim (130 µg/m³ - média 8h)")
        elif variavel == "pm10":
            ax.axhline(100, color="#ef4444", ls="--", lw=1.8, label="Limiar CETESB Ruim (100 µg/m³ - média 24h)")
        elif variavel == "pm2_5":
            ax.axhline(50, color="#ef4444", ls="--", lw=1.8, label="Limiar CETESB Ruim (50 µg/m³ - média 24h)")

        ax.set_ylabel("Concentração / Valor", fontweight="600")
        ax.set_xlabel("Data e Hora", fontweight="600")
        estilizar_eixo(ax)
        ax.legend(loc="upper right", framealpha=0.9)
        st.pyplot(fig)
        plt.close(fig)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Média", f"{df_filtro[variavel].mean():.2f}")
        col_s2.metric("Mediana", f"{df_filtro[variavel].median():.2f}")
        col_s3.metric("Máximo", f"{df_filtro[variavel].max():.2f}")
        col_s4.metric("Desvio Padrão", f"{df_filtro[variavel].std():.2f}")


# VISÃO 3: COMPARATIVO: BASELINE VS. MODELOS
    elif menu == "⚖️ 3. Comparativo: Baseline vs. Modelos":
        render_hero_header(
            titulo="⚖️ Comparativo Geral: Baseline vs. Modelos de Machine Learning",
            subtitulo="Avaliação empírica no conjunto de teste independente (3.480 horas, 299 casos positivos observados)",
            badges=["🏆 Random Forest Líder ROC-AUC", "🛡️ Decision Tree Líder Recall F₂", "🚀 XGBoost Gradient Boosting", "📉 -53% Falsos Negativos"]
    )    

    st.markdown("""
    <div class="highlight-card">
        <strong>Piso Operacional a Ser Batido:</strong> A <strong>Persistência Temporal</strong> (regra operacional que assume que o ar de amanhã será igual ao de hoje) 
        alcança <strong>48,49% de Recall</strong> e gera <strong>154 Falsos Negativos</strong>. Para justificar o uso de Machine Learning, qualquer modelo deve bater esse piso com folga.
    </div>
    """, unsafe_allow_html=True)

    if comparativo is not None:
        st.dataframe(
            comparativo.style.format({
                "limiar": "{:.2f}",
                "acuracia": "{:.2%}",
                "precisao_c1": "{:.4f}",
                "recall_c1": "{:.2%}",
                "f1_c1": "{:.4f}",
                "f2_c1": "{:.4f}",
                "roc_auc": "{:.4f}",
            }),
            use_container_width=True
        )

        st.markdown("### 📉 Redução do Erro Crítico (Falsos Negativos)")
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        
        df_plot = comparativo[comparativo["modelo"] != "DummyClassifier"].copy()
        
        cores_map = {
            "Persistencia": "#ef4444",
            "Decision Tree": "#2563eb",
            "Random Forest": "#059669",
            "XGBoost": "#7c3aed"
        }
        cores = []
        for m in df_plot["modelo"]:
            c = "#94a3b8"
            for k, val in cores_map.items():
                if k in m:
                    c = val
                    break
            cores.append(c)
        
        barras = ax.barh(df_plot["modelo"], df_plot["fn"], color=cores, height=0.6)
        ax.axvline(154, color="#ef4444", ls="--", lw=1.8, label="Teto Máximo da Persistência (154 FNs)")
        ax.set_xlabel("Quantidade de Falsos Negativos (Menor é Melhor)", fontweight="bold")
        ax.set_title("Comparação de Falsos Negativos no Teste (Total de 299 Horas Críticas)", pad=12, fontweight="bold", fontsize=12)
        estilizar_eixo(ax)
        ax.legend(framealpha=0.9)
        
        for bar in barras:
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{int(width)}", va="center", fontweight="bold", color="#1e293b")

        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #2563eb;">
                <h4 style="color: #2563eb; margin-top: 0;">🌲 Decision Tree</h4>
                <ul style="padding-left: 18px; margin-bottom: 0;">
                    <li><strong>Recall:</strong> 73,91% (Padrão) / <strong>75,92% ($F_2$)</strong></li>
                    <li><strong>Falsos Negativos:</strong> Reduzidos para <strong>72</strong> (corte de 53% nos erros graves)</li>
                    <li><strong>Vantagem:</strong> Regras explícitas e transparentes.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #059669;">
                <h4 style="color: #059669; margin-top: 0;">🌳 Random Forest</h4>
                <ul style="padding-left: 18px; margin-bottom: 0;">
                    <li><strong>Recall:</strong> 63,21% (Padrão) / <strong>66,56% ($F_2$)</strong></li>
                    <li><strong>Falsos Negativos:</strong> Reduzidos para <strong>100</strong></li>
                    <li><strong>Vantagem:</strong> Liderança em <strong>ROC-AUC (0,9006)</strong> e menor taxa de alarmes falsos.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #7c3aed;">
                <h4 style="color: #7c3aed; margin-top: 0;">🚀 XGBoost</h4>
                <ul style="padding-left: 18px; margin-bottom: 0;">
                    <li><strong>Recall:</strong> 58,19% (Padrão) / <strong>74,25% ($F_2$)</strong></li>
                    <li><strong>Falsos Negativos:</strong> Reduzidos para <strong>77</strong></li>
                    <li><strong>Vantagem:</strong> Alta flexibilidade e excelente ranking probabilístico (<strong>ROC-AUC 0,8869</strong>).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# VISÃO 4: DIAGNÓSTICO DETALHADO DOS MODELOS
    elif menu == "🔬 4. Diagnóstico Detalhado dos Modelos":
    render_hero_header(
        titulo="🔬 Diagnóstico Detalhado e Interpretação dos Modelos",
        subtitulo="Matrizes de confusão com limiar padrão vs. calibrado por F₂, importância de variáveis e regras extraídas",
        badges=["🔍 Matrizes de Confusão", "📊 Importância de Features", "🌳 Visualização de Árvore", "📜 Regras Transparentes"]
    )

    modelo_escolhido = st.selectbox("Selecione o modelo para inspecionar:", ["Decision Tree", "Random Forest", "XGBoost"])
    subpasta = {
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }[modelo_escolhido]

    dir_rel = ROOT / f"reports/modelos_decision_tree_random_forest/{subpasta}"

    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.markdown("#### Matriz de Confusão — Limiar Padrão (0,50)")
        p_img_padrao = dir_rel / f"matriz_{subpasta}_padrao.png"
        if p_img_padrao.is_file():
            st.image(str(p_img_padrao), use_container_width=True)

    with col_mat2:
        st.markdown("#### Matriz de Confusão — Limiar Calibrado por $F_2$")
        p_img_f2 = dir_rel / f"matriz_{subpasta}_calibrado_f2.png"
        if p_img_f2.is_file():
            st.image(str(p_img_f2), use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### 🏆 Atributos Mais Determinantes (Importância de Features) — {modelo_escolhido}")

    col_imp1, col_imp2 = st.columns([1.2, 1])
    with col_imp1:
        p_img_imp = dir_rel / f"importancia_features_{subpasta}.png"
        if p_img_imp.is_file():
            st.image(str(p_img_imp), use_container_width=True)

    with col_imp2:
        p_csv_imp = dir_rel / f"importancia_features_{subpasta}.csv"
        if p_csv_imp.is_file():
            df_imp = pd.read_csv(p_csv_imp)
            st.dataframe(df_imp.head(10).style.format({"importancia": "{:.4f}"}), use_container_width=True)

    if modelo_escolhido == "Decision Tree":
        with st.expander("📜 Ver Regras Textuais da Árvore de Decisão"):
            p_rules = dir_rel / "regras_arvore_decisao.txt"
            if p_rules.is_file():
                st.code(p_rules.read_text(encoding="utf-8")[:2500], language="text")


# VISÃO 5: TESTAR NOVA DATA & PREVISÃO 24H (API AO VIVO)
    elif menu == "📅 5. Testar Nova Data & Previsão 24h (API ao Vivo)":
    render_hero_header(
        titulo="📅 Consulta ao Vivo, Previsão 24h e Auditoria de Modelos",
        subtitulo="Escolha qualquer data e horário para São Paulo. O sistema consulta a API Open-Meteo em tempo real, monitora latência e consumo de RAM, projeta as condições para as 24 horas seguintes e audita as previsões frente ao padrão oficial CETESB N3+.",
        badges=["🌐 Open-Meteo API ao Vivo", "⚡ Telemetria de RAM e Latência", "🎯 Previsão t₀ ➔ t₀+24h", "📈 Trajetória 24h Seguintes", "⚖️ Critérios Formais de Aprovação"]
    )

    import importlib
    import auditoria_api
    importlib.reload(auditoria_api)

    st.markdown("#### 1. Selecionar Ponto de Início da Pesquisa (Data e Horário)")

    data_padrao = st.session_state.get("data_alvo_input", datetime.date(2024, 10, 15))
    hora_padrao = st.session_state.get("hora_alvo_input", 14)

    col_d1, col_d2, col_d3 = st.columns([1.4, 1.2, 2.4])
    with col_d1:
        data_selecionada = st.date_input(
            "📅 Data de Início:",
            value=data_padrao,
            min_value=datetime.date(2022, 1, 1),
            max_value=datetime.date.today() + datetime.timedelta(days=2),
            help="Data inicial para a previsão. Pode ser data histórica ou o dia de hoje."
        )

    with col_d2:
        horas_opcoes = [f"{h:02d}:00" for h in range(24)]
        hora_idx = hora_padrao if 0 <= hora_padrao < 24 else 14
        hora_str = st.selectbox(
            "⏰ Horário de Início (t₀):",
            options=horas_opcoes,
            index=hora_idx,
            help="Momento exato em que a previsão para as próximas 24 horas será disparada."
        )
        hora_selecionada = int(hora_str.split(":")[0])

    with col_d3:
        st.markdown("**Atalhos de cenários para teste:**")
        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("🚨 15/10/24 14h (Pico O₃)", use_container_width=True):
            st.session_state["data_alvo_input"] = datetime.date(2024, 10, 15)
            st.session_state["hora_alvo_input"] = 14
            st.rerun()
        if col_b2.button("🟢 15/07/25 08h (Limpo)", use_container_width=True):
            st.session_state["data_alvo_input"] = datetime.date(2025, 7, 15)
            st.session_state["hora_alvo_input"] = 8
            st.rerun()
        if col_b3.button("📅 Hoje / Recente", use_container_width=True):
            st.session_state["data_alvo_input"] = datetime.date.today()
            st.session_state["hora_alvo_input"] = 12
            st.rerun()

    btn_consultar = st.button("🚀 Consultar API e Gerar Previsão para as Próximas 24 Horas", type="primary", use_container_width=True)

    if btn_consultar:
        ts_inicio_str = f"{data_selecionada} às {hora_selecionada:02d}:00"
        with st.spinner(f"Consultando APIs Open-Meteo (Qualidade do Ar + Meteorologia) para São Paulo iniciando em {ts_inicio_str}..."):
            try:
                importlib.reload(auditoria_api)
                df_bruto, meta = auditoria_api.consultar_api_open_meteo(data_selecionada, hora_selecionada)
                sanidade = auditoria_api.validar_sanidade_dados(df_bruto)
                res_features = auditoria_api.extrair_features_e_alvo(df_bruto, data_selecionada, hora_selecionada)
                prev_t0 = auditoria_api.gerar_previsao_t0(res_features["linha_t0"], modelos, res_features["tem_gt_t0_24h"])
                audit_janela, resumo_j = auditoria_api.auditar_janela_24h_seguinte(res_features["df_janela_24h"], modelos, res_features["tem_gt_janela"])

                st.session_state["resultado_v5"] = {
                    "data_selecionada": data_selecionada,
                    "hora_selecionada": hora_selecionada,
                    "meta": meta,
                    "sanidade": sanidade,
                    "res_features": res_features,
                    "prev_t0": prev_t0,
                    "audit_janela": audit_janela,
                    "resumo_j": resumo_j,
                }
                st.success(f"Consulta finalizada! Latência total: {meta['latencia_total_ms']} ms | RAM do processo: {meta['recursos']['memoria_processo_mb']} MB")
            except Exception as e:
                st.error(f"Erro na consulta à API ou processamento: {e}")

    if "resultado_v5" in st.session_state:
        res = st.session_state["resultado_v5"]
        meta = res["meta"]
        sanidade = res["sanidade"]
        res_features = res["res_features"]
        prev_t0 = res["prev_t0"]
        audit_janela = res["audit_janela"]
        resumo_j = res["resumo_j"]
        recursos = meta["recursos"]

        st.markdown("---")
        st.markdown("### ⚡ 2. Monitoramento de Recursos e Performance Técnica")

        c_rec1, c_rec2, c_rec3, c_rec4 = st.columns(4)
        with c_rec1:
            st.metric(
                label="⏱️ Latência da API",
                value=f"{meta['latencia_total_ms']} ms",
                delta=f"Qualidade: {meta['latencia_aq_ms']}ms | Clima: {meta['latencia_weather_ms']}ms",
                delta_color="off"
            )
        with c_rec2:
            st.metric(
                label="🧠 RAM do Processo Python",
                value=f"{recursos['memoria_processo_mb']} MB",
                delta="Consumo da aplicação",
                delta_color="off"
            )
        with c_rec3:
            st.metric(
                label="💻 Memória RAM do Sistema",
                value=f"{recursos['memoria_sistema_pct']}%",
                delta=f"{recursos['memoria_sistema_usada_gb']} GB de {recursos['memoria_sistema_total_gb']} GB",
                delta_color="off"
            )
        with c_rec4:
            st.metric(
                label="🛡️ Integridade dos Dados",
                value="100% Válidos" if sanidade["integro"] else "Com Alertas",
                delta=f"{sanidade['total_linhas']}h analisadas | 0 sentinelas",
                delta_color="off"
            )

        with st.expander("🔍 Inspecionar Diagnóstico Completo de Sanidade e Headers da API"):
            st.json({
                "latencia_detalhada": {
                    "qualidade_do_ar_ms": meta["latencia_aq_ms"],
                    "meteorologia_ms": meta["latencia_weather_ms"],
                    "total_ms": meta["latencia_total_ms"],
                },
                "memoria_sistema": recursos,
                "auditoria_dados_brutos": sanidade,
            })

        st.markdown("---")
        st.markdown("### 🎯 3. Previsão Emitida no Ponto de Partida (Horizonte de 24h)")

        if prev_t0:
            t0_fmt = prev_t0["t0"].strftime("%d/%m/%Y às %H:00")
            t24_fmt = prev_t0["t_alvo_24h"].strftime("%d/%m/%Y às %H:00")

            st.info(f"📍 **Momento da Previsão (t₀):** `{t0_fmt}`  ➔  **Previsão para 24h à Frente (t₀ + 24h):** `{t24_fmt}`")

            # Condição observada em t0
            c_cond1, c_cond2, c_cond3, c_cond4 = st.columns(4)
            c_cond1.metric("Temperatura em t₀", f"{prev_t0['temp_t0']:.1f} °C")
            c_cond2.metric("Vento em t₀", f"{prev_t0['vento_t0']:.1f} m/s")
            c_cond3.metric("Ozônio (O₃) em t₀", f"{prev_t0['o3_t0']:.1f} µg/m³")
            c_cond4.metric("Qualidade Atual do Ar (t₀)", prev_t0["condicao_atual_t0"], prev_t0["poluente_agora"])

            st.markdown(f"#### 🔮 Projeção dos Modelos para exatamente 24 Horas Depois (`{t24_fmt}`):")

            cols_prev = st.columns(4)
            modelos_nomes = ["XGBoost", "Random Forest", "Decision Tree", "Persistência Temporal"]
            for col_p, nome_m in zip(cols_prev, modelos_nomes):
                info_p = prev_t0["previsoes"].get(nome_m, {})
                with col_p:
                    st.markdown(f"**{nome_m}**")
                    st.metric(
                        label="Risco Previsto P(N3+)",
                        value=f"{info_p.get('prob', 0)*100:.1f}%",
                        delta=info_p.get("rotulo", "N/D"),
                        delta_color="inverse" if "Inadequado" in info_p.get("rotulo", "") else "normal"
                    )
                    st.caption(f"Limiar de decisão: {info_p.get('limiar', 0.5):.2f}")

            if prev_t0["tem_ground_truth"] and prev_t0["real_rotulo"]:
                st.markdown("##### ⚖️ Verificação com a Realidade Observada 24 Horas Depois:")
                st.markdown(f"""
                <div class="highlight-card">
                    <strong>Condição Oficial Observada pela CETESB em {t24_fmt}:</strong> {prev_t0['real_rotulo']}<br>
                    <strong>Poluente Crítico Registrado:</strong> {prev_t0['poluente_critico_real']}
                </div>
                """, unsafe_allow_html=True)

                col_aud1, col_aud2, col_aud3, col_aud4 = st.columns(4)
                for col_a, nome_m in zip([col_aud1, col_aud2, col_aud3, col_aud4], modelos_nomes):
                    info_p = prev_t0["previsoes"].get(nome_m, {})
                    pred = info_p.get("pred", 0)
                    real = prev_t0["real_valor"]
                    with col_a:
                        if pred == real:
                            st.success(f"✅ **{nome_m} ACERTOU!**")
                        elif pred == 0 and real == 1:
                            st.error(f"❌ **{nome_m}: Falso Negativo**")
                        else:
                            st.warning(f"⚠️ **{nome_m}: Alarme Falso**")
            else:
                st.info("ℹ️ **Data Recente / Hoje:** O resultado real em 24h ainda não foi concluído. Os modelos acima representam o alerta preventivo para o seu treino amanhã!")

        st.markdown("---")
        st.markdown(f"### 📈 4. Trajetória Horária das Próximas 24 Horas (t₀ até t₀ + 24h)")

        if len(audit_janela) > 0:
            fig, ax = plt.subplots(figsize=(12, 4.5))
            horas_janela = audit_janela["Data/Hora"].tolist()

            probs_xgb = [float(p.replace('%', '')) / 100 for p in audit_janela["XGBoost (Risco %)"]] if "XGBoost (Risco %)" in audit_janela else []
            probs_rf = [float(p.replace('%', '')) / 100 for p in audit_janela["Random Forest (Risco %)"]] if "Random Forest (Risco %)" in audit_janela else []
            probs_dt = [float(p.replace('%', '')) / 100 for p in audit_janela["Decision Tree (Risco %)"]] if "Decision Tree (Risco %)" in audit_janela else []

            if probs_xgb:
                ax.plot(horas_janela, probs_xgb, marker="o", linewidth=2.5, color="#2563eb", label="XGBoost (Probabilidade)")
            if probs_rf:
                ax.plot(horas_janela, probs_rf, marker="s", linewidth=2.0, color="#059669", label="Random Forest (Probabilidade)")
            if probs_dt:
                ax.plot(horas_janela, probs_dt, marker="^", linewidth=1.8, color="#d97706", linestyle="--", label="Decision Tree (Probabilidade)")

           
            if "Realidade CETESB" in audit_janela.columns:
                ja_anotou = False
                for idx_h, status_real in enumerate(audit_janela["Realidade CETESB"]):
                    if "Inadequado" in status_real:
                        lbl = "Ar Realmente Inadequado (CETESB N3+)" if not ja_anotou else ""
                        ax.axvspan(idx_h - 0.4, idx_h + 0.4, color="#ef4444", alpha=0.25, label=lbl)
                        ja_anotou = True

            ax.axhline(0.25, color="#2563eb", linestyle=":", alpha=0.7, label="Limiar Decisão XGBoost (0,25)")
            ax.axhline(0.40, color="#059669", linestyle=":", alpha=0.7, label="Limiar Decisão Random Forest (0,40)")

            ax.set_title(f"Evolução do Risco nas 24 Horas Seguintes a {res_features['t0'].strftime('%d/%m %H:00')}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Hora Futura")
            ax.set_ylabel("Probabilidade de Risco P(N3+)")
            ax.set_ylim(-0.05, 1.05)
            plt.xticks(rotation=45, ha="right")
            estilizar_eixo(ax)
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9, framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### 📋 Tabela Detalhada das 24 Horas Seguintes:")
            st.dataframe(audit_janela.drop(columns=["timestamp"], errors="ignore"), use_container_width=True, height=360)

    
    # SEÇÃO OBRIGATÓRIA: COMPROVAÇÃO DE QUAL MODELO FOI O MELHOR E CRITÉRIOS
    st.markdown("---")
    st.markdown('<div class="main-title" style="font-size: 1.8rem;">🏆 Comprovação: Qual Modelo Foi o Melhor e Quais os Critérios de Aprovação?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evidências empíricas auditadas no conjunto de teste independente (3.480 horas / 299 eventos críticos).</div>', unsafe_allow_html=True)

    if comparativo is not None:
        st.markdown("#### 📊 Tabela Oficial de Desempenho no Conjunto de Teste")
        st.dataframe(
            comparativo.style.format({
                "limiar": "{:.2f}",
                "acuracia": "{:.2%}",
                "precisao_c1": "{:.4f}",
                "recall_c1": "{:.2%}",
                "f1_c1": "{:.4f}",
                "f2_c1": "{:.4f}",
                "roc_auc": "{:.4f}",
            }),
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
        badges=["🔄 Pipeline Automatizado", "🧼 Limpeza e Purga 24h", "🧠 Treinamento e Calibração", "📊 Geração de Relatórios"]
    )

    st.markdown("""
    <div class="metric-card" style="border-left: 4px solid #3b82f6; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #1e3a8a;">Fluxo de Execução Sequencial</h4>
        <ol style="margin-bottom: 0; padding-left: 20px; line-height: 1.8;">
            <li><strong>Coleta e Integração:</strong> Consulta ou verifica o repositório bruto das APIs públicas da Open-Meteo.</li>
            <li><strong>Limpeza e Split Temporal:</strong> Trata sentinelas, padroniza unidades e aplica purga temporal estrita de 24 horas.</li>
            <li><strong>Modelagem e Calibração:</strong> Treina Decision Tree, Random Forest e XGBoost com <code>ColumnTransformer</code> (Yeo-Johnson) e calibra limiares ótimos por $F_2$.</li>
            <li><strong>Comparação e Auditoria:</strong> Avalia contra a Persistência Temporal e regera todos os relatórios e métricas.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Executar Pipeline Completo Agora", type="primary", use_container_width=True):
        import pipeline_completo
        with st.spinner("Executando pipeline completo... Isso pode levar de 30 a 60 segundos."):
            res = pipeline_completo.executar_pipeline_completo(forcar_recoleta=False)
            st.success(f"Pipeline executado com sucesso em {res['tempo_execucao_s']:.2f} segundos!")
            st.dataframe(pd.DataFrame(res["comparativo"]), use_container_width=True)



