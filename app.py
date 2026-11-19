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
import html
import io
import json
import logging
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use("Agg")  # Streamlit renderiza imagens no servidor, sem janela Tk.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from limiares_modelos import limiar_para_modelo
from metricas_dashboard import (
    carregar_resultado_oficial, carregar_comparativo_oficial, destaques_exploratorios,
    linha_persistencia, selecionar_linha_calibrada,
    carregar_resumo_sprint5, comparacao_exploratoria_com_nb,
)

# Configuração da página Streamlit
st.set_page_config(
    page_title="CADENC.IA | Previsão de Qualidade do Ar",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

# Configuração global de estilo para gráficos Matplotlib
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

# A camada de aparência fica isolada do conteúdo acadêmico e das regras do app.
st.html("<style>" + (ROOT / "app.css").read_text(encoding="utf-8") + "</style>")

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


def assinatura_arquivo(caminho: Path):
    try:
        info = caminho.stat()
        return info.st_mtime_ns, info.st_size
    except FileNotFoundError:
        return None


ARQUIVOS_DADOS = {
    "interim": "data/interim/integrado_limpo.csv",
    "treino": "data/processed/treino_features_s4.csv",
    "teste": "data/processed/teste_features_s4.csv",
    "comparativo": "reports/modelos_decision_tree_random_forest/comparativo_completo_decision_tree_vs_random_forest.csv",
    "erros": "reports/modelos_decision_tree_random_forest/erros_detalhados_teste.csv",
    "features": "config/params.yaml",
}


@st.cache_data(show_spinner=False, max_entries=16)
def carregar_arquivo(nome, assinatura):
    if assinatura is None:
        return [] if nome == "features" else None
    caminho = ROOT / ARQUIVOS_DADOS[nome]
    if nome == "features":
        params = yaml.safe_load(caminho.read_text(encoding="utf-8"))
        return list(params.get("sprint4", {}).get("features_congeladas", []))
    if nome == "treino":
        # A visao geral usa apenas a proporcao do alvo.
        return pd.read_csv(caminho, usecols=["qualidade_inadequada_24h"])
    return pd.read_csv(caminho, parse_dates=["timestamp"] if nome in ("interim", "teste", "erros") else None)


def carregar_dados(necessarios: tuple[str, ...]):
    return tuple(
        carregar_arquivo(nome, assinatura_arquivo(ROOT / ARQUIVOS_DADOS[nome]))
        if nome in necessarios else ([] if nome == "features" else None)
        for nome in ARQUIVOS_DADOS
    )


@st.cache_data(show_spinner=False, max_entries=4)
def resumo_arquivos(assinatura_csv, assinatura_config) -> tuple[int | None, int]:
    caminho = ROOT / ARQUIVOS_DADOS["interim"]
    quantidade = None
    if assinatura_csv is not None:
        with caminho.open(encoding="utf-8") as arquivo:
            quantidade = sum(1 for _ in arquivo) - 1
    return quantidade, len(carregar_arquivo("features", assinatura_config))


ARQUIVOS_MODELOS = {
    "Decision Tree": "pipeline_decision_tree.joblib",
    "Random Forest": "pipeline_random_forest.joblib",
    "XGBoost": "pipeline_xgboost.joblib",
    "Naive Bayes": "naive_bayes_s4.joblib",
    "Modelo oficial (Sprint 5)": "modelo_final_s5.joblib",
}


def assinatura_modelo(nome):
    return assinatura_arquivo(ROOT / "models" / ARQUIVOS_MODELOS[nome])


@st.cache_resource(show_spinner=False, max_entries=10)
def _carregar_pipeline(nome, assinatura, assinatura_validacao):
    if assinatura is None:
        return None
    if nome == "Naive Bayes":
        limiar_para_modelo(nome)
    return joblib.load(ROOT / "models" / ARQUIVOS_MODELOS[nome])


def carregar_pipeline(nome):
    try:
        validacao = assinatura_arquivo(ROOT / "reports/sprint5/comparativo_teste.csv") if nome == "Naive Bayes" else None
        return _carregar_pipeline(nome, assinatura_modelo(nome), validacao)
    except Exception:
        logging.getLogger(__name__).exception("Falha ao carregar o modelo %s", nome)
        return None


def carregar_pipelines():
    return {nome: pipe for nome in ARQUIVOS_MODELOS if (pipe := carregar_pipeline(nome)) is not None}


@st.cache_data(show_spinner=False, max_entries=16)
def probabilidades_diagnostico(nome, versao_modelo, versao_teste, features, _pipe, _teste):
    return _pipe.predict_proba(_teste[list(features)])[:, 1]


@st.cache_data(show_spinner=False, max_entries=8)
def importancia_permutacao_nb(versao_modelo, versao_teste, features, limiar, _pipe, _teste):
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import fbeta_score

    def f2_no_limiar(estimator, x, y):
        return fbeta_score(y, estimator.predict_proba(x)[:, 1] >= limiar, beta=2, zero_division=0)

    return permutation_importance(
        _pipe, _teste[list(features)], _teste["qualidade_inadequada_24h"],
        scoring=f2_no_limiar, n_repeats=5, random_state=42,
    )


def plotar_matriz_confusao_dinamica(y_true, y_pred, titulo: str):
    """Gera a matriz de confusão dinamicamente com Matplotlib."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Adequada (0)", "Inadequada (1)"], fontweight="bold", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Adequada (0)", "Inadequada (1)"], fontweight="bold", fontsize=10)
    ax.set_xlabel("Classe Prevista", fontweight="bold", fontsize=10)
    ax.set_ylabel("Classe Real", fontweight="bold", fontsize=10)
    ax.set_title(f"Matriz de Confusão — {titulo}", pad=12, fontweight="bold", fontsize=11)
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            pct = (val / total) * 100 if total > 0 else 0
            cor = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=cor, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plotar_importancia_dinamica(importancias, nomes_features, titulo: str, eixo="Importância Relativa (Gini / Ganho)"):
    """Gera o gráfico de barras horizontais de importância de atributos dinamicamente."""
    df_imp = pd.DataFrame({"feature": nomes_features, "importancia": importancias})
    df_imp = df_imp.sort_values("importancia", ascending=False).reset_index(drop=True)
    df_top = df_imp.head(12).sort_values("importancia", ascending=True)
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(df_top["feature"], df_top["importancia"], color="#2563eb", edgecolor="#1d4ed8")
    ax.set_xlabel(eixo, fontweight="bold", fontsize=10)
    ax.set_title(f"Top 12 Atributos Mais Determinantes — {titulo}", pad=12, fontweight="bold", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig, df_imp


@st.cache_data(show_spinner=False, max_entries=24)
def imagem_matriz(y_true, y_pred, titulo):
    return imagem_figura(plotar_matriz_confusao_dinamica(y_true, y_pred, titulo))


@st.cache_data(show_spinner=False, max_entries=12)
def imagem_importancia(importancias, nomes_features, titulo, eixo=None):
    if eixo is None:
        fig, _ = plotar_importancia_dinamica(importancias, nomes_features, titulo)
    else:
        fig, _ = plotar_importancia_dinamica(importancias, nomes_features, titulo, eixo)
    return imagem_figura(fig)


def imagem_figura(fig):
    try:
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
            return buffer.getvalue()
    finally:
        plt.close(fig)


@st.cache_data(show_spinner=False, max_entries=8)
def grafico_reducao_erro(quadro, oficial):
    fig, ax = plt.subplots(figsize=(10, 5.3))
    nomes = quadro["modelo"].tolist()
    barras = ax.barh(nomes, quadro["fn"], color=["#2563eb" if n == oficial else "#94a3b8" for n in nomes], height=.6)
    ax.invert_yaxis()
    ax.bar_label(barras, padding=4, fmt="%d")
    ax.set_xlim(0, max(1, quadro["fn"].max()) * 1.16)
    ax.set_xlabel("Quantidade de falsos negativos (menor é melhor)")
    piso = quadro.loc[quadro["modelo"] == "Persistencia", "fn"]
    if not piso.empty:
        ax.axvline(piso.iloc[0], color="#ef4444", ls="--", label="Persistência")
        ax.legend()
    estilizar_eixo(ax)
    fig.tight_layout()
    return imagem_figura(fig)


def render_reducao_erro():
    quadro = carregar_resumo_sprint5()
    if quadro is None or quadro.empty:
        st.info("Comparativo da Sprint 5 indisponível. Execute o pipeline.")
        return
    oficial = carregar_resultado_oficial().get("modelo_final")
    st.markdown("### Redução do Erro Crítico (Falsos Negativos)")
    st.caption("Sprint 5: mesmo teste cronológico, uma linha por modelo, limiares escolhidos na validação temporal. Menor FN é melhor.")
    st.image(grafico_reducao_erro(quadro, oficial), width="stretch")
    for inicio in range(0, len(quadro), 3):
        for coluna, (_, linha) in zip(st.columns(3), quadro.iloc[inicio:inicio+3].iterrows()):
            with coluna:
                st.markdown(f"**{linha['modelo']}{' — oficial' if linha['modelo'] == oficial else ''}**")
                st.metric("Recall no teste", f"{linha['recall_c1']:.2%}")
                st.metric("Falsos negativos", str(int(linha['fn'])))
                st.caption(f"F₂: {linha['f2_c1']:.4f} · FP: {int(linha['fp'])}")
    dt = quadro.loc[quadro.modelo == "Decision Tree"]
    rf = quadro.loc[quadro.modelo == "Random Forest"]
    if not dt.empty and not rf.empty:
        d, r = dt.iloc[0], rf.iloc[0]
        st.info(f"Decision Tree: recall {d.recall_c1:.2%}, {int(d.fn)} FN, {int(d.fp)} FP e F₂ {d.f2_c1:.4f}. "
                f"Random Forest: recall {r.recall_c1:.2%}, {int(r.fn)} FN, {int(r.fp)} FP e F₂ {r.f2_c1:.4f}. "
                f"ROC-AUC: {d.roc_auc:.4f} vs. {r.roc_auc:.4f}. A prioridade do projeto é detectar horas inadequadas, "
                "considerando também os falsos alarmes. A seleção oficial usa recall na validação com piso F₂ da persistência; "
                "o teste histórico descreve o desempenho e não redefine o vencedor. A comparação exploratória usa ajustes próprios.")


@st.cache_data(show_spinner=False, max_entries=12)
def grafico_eda(df_filtro, variavel):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(df_filtro["timestamp"], df_filtro[variavel], color="#0284c7", lw=1.5, label=variavel)

    # Limiar CETESB para Ozônio se aplicável
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
    return imagem_figura(fig)


@st.cache_data(show_spinner=False, max_entries=12)
def grafico_trajetoria(audit_janela, limiares, nome_oficial):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    horas_janela = audit_janela["Alvo (+24h)"].tolist()

    def probabilidades(nome: str):
        coluna = f"{nome} (Risco %)"
        if coluna not in audit_janela:
            return []
        return (pd.to_numeric(audit_janela[coluna].str.rstrip("%"), errors="coerce") / 100).tolist()

    probs_xgb = probabilidades("XGBoost")
    probs_rf = probabilidades("Random Forest")
    probs_dt = probabilidades("Decision Tree")
    probs_nb = probabilidades("Naive Bayes")
    probs_oficial = probabilidades("Modelo oficial (Sprint 5)")

    if probs_oficial and any(pd.notna(x) for x in probs_oficial):
        ax.plot(horas_janela, probs_oficial, marker="d", linewidth=2.8, color="#475569", label=f"{nome_oficial} oficial (Sprint 5)")

    if probs_xgb and any(pd.notna(x) for x in probs_xgb):
        ax.plot(horas_janela, probs_xgb, marker="o", linewidth=2.5, color="#2563eb", label="XGBoost (exploratório)")
    if probs_rf and any(pd.notna(x) for x in probs_rf):
        ax.plot(horas_janela, probs_rf, marker="s", linewidth=2.0, color="#059669", label="Random Forest (exploratório)")
    if probs_dt and any(pd.notna(x) for x in probs_dt):
        ax.plot(horas_janela, probs_dt, marker="^", linewidth=1.8, color="#d97706", linestyle="--", label="Decision Tree (exploratório)")
    if probs_nb and any(pd.notna(x) for x in probs_nb):
        ax.plot(horas_janela, probs_nb, marker="x", linewidth=2, color="#9333ea", label="Naive Bayes (Sprint 5)")

    # Destacar horas com ar inadequado se houver dados reais
    if "Referência retrospectiva Open-Meteo" in audit_janela.columns:
        ja_anotou = False
        for idx_h, status_real in enumerate(audit_janela["Referência retrospectiva Open-Meteo"]):
            if "Inadequado" in status_real:
                lbl = "Ar classificado como inadequado (referência Open-Meteo)" if not ja_anotou else ""
                ax.axvspan(idx_h - 0.4, idx_h + 0.4, color="#ef4444", alpha=0.25, label=lbl)
                ja_anotou = True

    for modelo_linha, cor in (("Modelo oficial (Sprint 5)", "#475569"), ("XGBoost", "#2563eb"), ("Random Forest", "#059669"), ("Decision Tree", "#d97706"), ("Naive Bayes", "#9333ea")):
        if modelo_linha not in limiares:
            continue
        valor_limiar = limiares[modelo_linha]
        rotulo = f"{nome_oficial} oficial" if modelo_linha == "Modelo oficial (Sprint 5)" else "Naive Bayes (Sprint 5)" if modelo_linha == "Naive Bayes" else f"{modelo_linha} exploratório"
        ax.axhline(valor_limiar, color=cor, linestyle=":", alpha=0.7,
                   label=f"Limiar {rotulo} ({valor_limiar:.2f})")

    ax.set_title("Projeções para t + 24h, emitidas hora a hora", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hora do alvo (+24h)")
    ax.set_ylabel("Escore do modelo para N3+ (não calibrado)")
    ax.set_ylim(-0.05, 1.05)
    plt.xticks(rotation=45, ha="right")
    estilizar_eixo(ax)
    tem_series = bool(ax.get_legend_handles_labels()[0])
    if tem_series:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return imagem_figura(fig), tem_series


# Barra lateral (Sidebar)
logo_sidebar = ROOT / "docs/assets/cadencia-logo-clara.png"
if logo_sidebar.is_file():
    st.sidebar.image(str(logo_sidebar), width=220)
else:
    st.sidebar.markdown("## CADENC.IA")
st.sidebar.markdown("### Navegação")
menu = st.sidebar.radio(
    "Selecione uma visão:",
    [
        "1. Visão Geral & Dados Abertos",
        "2. Análise Exploratória (EDA)",
        "3. Comparativo: Baseline vs. Modelos",
        "4. Diagnóstico Detalhado dos Modelos",
        "5. Testar Nova Data & Previsão 24h (API ao Vivo)",
        "6. Executar Pipeline Completo",
    ],
    index=4,
)

artefatos_presentes = sum(
    (ROOT / "models" / nome).is_file()
    for nome in (
        "pipeline_decision_tree.joblib",
        "pipeline_random_forest.joblib",
        "pipeline_xgboost.joblib",
        "naive_bayes_s4.joblib",
        "modelo_final_s5.joblib",
    )
)
total_horas, numero_features = resumo_arquivos(
    assinatura_arquivo(ROOT / ARQUIVOS_DADOS["interim"]),
    assinatura_arquivo(ROOT / ARQUIVOS_DADOS["features"]),
)
coleta_status = f"{total_horas:,} h".replace(",", ".") if total_horas is not None else "Indisponível"
st.sidebar.markdown(f"""
<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-top: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
    <div style="font-size: 0.78rem; font-weight: 800; color: #1e3a8a; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Status do Pipeline</div>
    <div style="font-size: 0.83rem; color: #334155; margin-bottom: 6px;"><strong>Coleta histórica:</strong> {coleta_status} (Open-Meteo)</div>
    <div style="font-size: 0.83rem; color: #334155; margin-bottom: 6px;"><strong>Features:</strong> {numero_features} (Yeo-Johnson)</div>
    <div style="font-size: 0.83rem; color: #334155;"><strong>Artefatos de modelos locais:</strong> {artefatos_presentes}/5 + Persistência</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption("Projeto Trilha B — Equipe 2 · Mostra Tecnológica")

# Dados acadêmicos e pipelines são acessados sob demanda, conforme a visão.
arquivos_por_visao = {
    "1": ("interim", "treino", "features"),
    "2": ("interim",),
    "3": ("comparativo",),
    "4": ("teste", "features"),
    "5": ("comparativo",),
    "6": (),
}
visao = menu[:1]
interim, treino, teste, comparativo, erros, features = carregar_dados(arquivos_por_visao[visao])


@st.cache_data(ttl=600, max_entries=12, show_spinner=False)
def consultar_api_cache(data: datetime.date, hora: int):
    import auditoria_api
    return auditoria_api.consultar_api_open_meteo(data, hora)


# ==============================================================================
# VISÃO 1: VISÃO GERAL & COLETA DE DADOS ABERTOS
# ==============================================================================
if menu == "1. Visão Geral & Dados Abertos":
    render_hero_header(
        titulo=" CADENC.IA — O Ritmo do Ar",
        subtitulo="Previsão de Qualidade do Ar Inadequada para Atividades Físicas em São Paulo com 24h de Antecedência",
        badges=[" Open-Meteo APIs", " São Paulo (-23.5489, -46.6388)", " Horizonte 24h", " Norma CETESB N3+", " 25 Features Congeladas"]
    )

    st.markdown("""
    <div class="highlight-card">
        <strong> Objetivo do Projeto:</strong> Auxiliar corredores, ciclistas e praticantes de exercícios ao ar livre a planejar treinos com <strong>24 horas de antecedência</strong>, 
        identificando previamente quando a concentração de poluentes atingirá níveis prejudiciais à saúde respiratória (classificação <strong>CETESB N3 — RUIM ou pior</strong>).
        <br><strong> Custo Crítico Priorizado (RFC):</strong> <strong>Falso Negativo (FN)</strong> — deixar de alertar sobre o ar poluído expõe o atleta a respirar altas doses de ozônio e material particulado durante esforço cardiovascular intenso.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros Horários", coleta_status, "Amostra histórica do arquivo local")
    with col2:
        st.metric("Fontes de Dados Abertos", "2 APIs", "Open-Meteo Ar + Clima")
    with col3:
        st.metric("Features no Pipeline", f"{len(features)}", "Lags, Médias e Ciclos")
    with col4:
        taxa_pos = (treino['qualidade_inadequada_24h'].mean() * 100) if treino is not None else None
        st.metric("Desbalanceamento Treino", f"{taxa_pos:.1f}%" if taxa_pos is not None else "N/D", "Classe Ruim (CETESB N3+)")

    st.markdown("###  Dados Abertos Coletados e Integrados")
    st.markdown("""
    Os dados são obtidos via requisições idempotentes e automatizadas a duas APIs abertas públicas da **Open-Meteo**, integradas horariamente pela coordenada central de São Paulo (`-23.5489, -46.6388`):
    - **Open-Meteo Air Quality API:** Poluentes atmosféricos horários (`PM10`, `PM2.5`, `O₃`, `NO₂`, `SO₂`, `CO`).
    - **Open-Meteo Historical Weather API:** Variáveis meteorológicas horárias de reanálise (`temperatura`, `velocidade do vento`, `precipitação`).
    """)

    if interim is not None:
        st.dataframe(
            interim.head(10)[["timestamp", "pm10", "pm2_5", "ozone", "nitrogen_dioxide", "temperature_2m", "wind_speed_10m", "precipitation"]],
            width="stretch"
        )
        st.caption(f"Amostra dos dados limpos integrados. Total de observações: {len(interim):,} linhas horárias.")


# ==============================================================================
# VISÃO 2: ANÁLISE EXPLORATÓRIA (EDA)
# ==============================================================================
elif menu == "2. Análise Exploratória (EDA)":
    render_hero_header(
        titulo=" Análise Exploratória dos Dados Abertos (EDA)",
        subtitulo="Comportamento temporal de poluentes e variáveis climáticas em São Paulo frente aos limites regulatórios",
        badges=[f" {coleta_status} Observações", " Ozônio (O₃)", " Partículas (PM10 e PM2.5)", " Temperatura & Vento"]
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

        # Filtro de intervalo de datas
        min_date = interim["timestamp"].min().date()
        max_date = interim["timestamp"].max().date()
        with col_range:
            datas = st.date_input("Intervalo de datas:", value=(min_date, min(min_date + datetime.timedelta(days=90), max_date)), min_value=min_date, max_value=max_date)

        if isinstance(datas, tuple) and len(datas) == 2:
            df_filtro = interim[(interim["timestamp"] >= pd.Timestamp(datas[0])) & (interim["timestamp"] < pd.Timestamp(datetime.datetime.combine(datas[1] + datetime.timedelta(days=1), datetime.time.min)))]
        else:
            df_filtro = interim.head(1000)

        # Gráfico interativo
        st.image(grafico_eda(df_filtro[["timestamp", variavel]], variavel), width="stretch")

        # Estatísticas resumidas da variável
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Média", f"{df_filtro[variavel].mean():.2f}")
        col_s2.metric("Mediana", f"{df_filtro[variavel].median():.2f}")
        col_s3.metric("Máximo", f"{df_filtro[variavel].max():.2f}")
        col_s4.metric("Desvio Padrão", f"{df_filtro[variavel].std():.2f}")


# ==============================================================================
# VISÃO 3: COMPARATIVO: BASELINE VS. MODELOS
# ==============================================================================
elif menu == "3. Comparativo: Baseline vs. Modelos":
    oficial = carregar_resultado_oficial()
    persistencia = linha_persistencia(comparativo)
    total_positivos = int(oficial.get("positivos_teste", 0))
    total_teste = int(oficial.get("n_teste", 0))
    render_hero_header(
        titulo=" Comparativo Geral: Baseline vs. Modelos de Machine Learning",
        subtitulo=f"Avaliação cronológica em {total_teste:,} horas de teste e {total_positivos} horas positivas; decisão oficial baseada na validação temporal".replace(",", "."),
        badges=[" Seleção oficial na Sprint 5", " Comparativo exploratório separado", " Foco no risco para atletas"]
    )
    if oficial:
        selecionado = oficial.get("selecionado_na_validacao", "não registrado")
        f2 = oficial.get("f2_validacao", {}).get(selecionado)
        recall = oficial.get("recall_validacao", {}).get(selecionado)
        st.success(f"Modelo oficial da Sprint 5: **{selecionado}** (Recall de validação: {recall:.2%}; F₂: {f2:.4f}). "
                   "Prioridade: detectar horas inadequadas, com limiar de máximo F₂ e piso da persistência." if f2 is not None and recall is not None
                   else f"Modelo oficial: {selecionado}.")
        calibrados = carregar_comparativo_oficial()
        if calibrados is not None:
            st.markdown("#### Comparação oficial dos seis candidatos — Sprint 5")
            colunas = ["modelo", "limiar", "recall_val", "f2_val", "recall_c1", "f2_c1", "tp", "fn", "fp", "tn"]
            colunas += [c for c in ("roc_auc", "average_precision") if c in calibrados]
            st.dataframe(calibrados[colunas].style.format({
                "recall_val": "{:.2%}", "f2_val": "{:.4f}", "recall_c1": "{:.2%}", "f2_c1": "{:.4f}",
                "roc_auc": "{:.4f}", "average_precision": "{:.4f}"
            }), width="stretch")
            st.caption("Recall mede a proporção das horas realmente inadequadas detectadas (TP / (TP + FN)). O teste histórico já consultado é uma avaliação retrospectiva da política revisada.")
    if persistencia is not None:
        st.markdown(f"**Piso operacional — Persistência Temporal:** recall {persistencia['recall_c1']:.2%}; "
                    f"{int(persistencia['fn'])} falsos negativos no teste cronológico.")

    if comparativo is not None:
        st.caption("Resultados exploratórios dos três algoritmos de árvores, com Naive Bayes da Sprint 5 identificado à parte. "
                   "O teste histórico já foi consultado; estas métricas não redefinem o vencedor oficial.")
        st.dataframe(
            comparacao_exploratoria_com_nb(comparativo).style.format({
                "limiar": "{:.2f}",
                "acuracia": "{:.2%}",
                "precisao_c1": "{:.4f}",
                "recall_c1": "{:.2%}",
                "f1_c1": "{:.4f}",
                "f2_c1": "{:.4f}",
                "roc_auc": "{:.4f}",
            }),
            width="stretch"
        )

    else:
        st.warning(
            " **Quadro comparativo ainda não gerado neste ambiente.**\n\n"
            "Para computar as métricas de acurácia, recall, F₂, ROC-AUC e redução de falsos negativos:\n"
            " Acesse a aba **' 6. Executar Pipeline Completo'** e clique em **' Executar Pipeline Completo Agora'**."
        )
    render_reducao_erro()


# ==============================================================================
# VISÃO 4: DIAGNÓSTICO DETALHADO DOS MODELOS
# ==============================================================================
elif menu == "4. Diagnóstico Detalhado dos Modelos":
    render_hero_header(
        titulo=" Diagnóstico Detalhado e Interpretação dos Modelos",
        subtitulo="Matrizes de confusão com limiar padrão vs. calibrado por F₂, importância de variáveis e regras extraídas",
        badges=[" Matrizes de Confusão", " Importância de Features", " Visualização de Árvore", " Regras Transparentes"]
    )

    chave_oficial = "Modelo oficial (Sprint 5)"
    nome_final = carregar_resultado_oficial().get("modelo_final", "indisponível")
    modelo_escolhido = st.selectbox("Selecione o modelo para inspecionar:",
        [chave_oficial, "Decision Tree", "Random Forest", "XGBoost", "Naive Bayes"],
        format_func=lambda nome: f"{nome} — {nome_final}" if nome == chave_oficial else f"{nome} (Sprint 5)" if nome == "Naive Bayes" else f"{nome} (exploratório)")
    subpasta = {
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost", "Naive Bayes": "naive_bayes", chave_oficial: "oficial"
    }[modelo_escolhido]

    dir_rel = ROOT / f"reports/modelos_decision_tree_random_forest/{subpasta}"

    p_img_padrao = dir_rel / f"matriz_{subpasta}_padrao.png"
    p_img_f2 = dir_rel / f"matriz_{subpasta}_calibrado_f2.png"
    p_img_imp = dir_rel / f"importancia_features_{subpasta}.png"
    p_csv_imp = dir_rel / f"importancia_features_{subpasta}.csv"
    p_rules = dir_rel / "regras_arvore_decisao.txt"
    if modelo_escolhido == chave_oficial:
        dir_rel = ROOT / "reports/sprint5"
        p_img_padrao = dir_rel / "matriz_final_padrao.png"
        p_img_f2 = dir_rel / "matriz_final_limiar.png"

    precisa_pipeline = (
        modelo_escolhido == "Naive Bayes"
        or not all(p.is_file() for p in (p_img_padrao, p_img_f2, p_img_imp, p_csv_imp))
        or (modelo_escolhido == "Decision Tree" and not p_rules.is_file())
    )
    pipe = carregar_pipeline(modelo_escolhido) if precisa_pipeline else None
    if modelo_escolhido == chave_oficial and pipe is not None:
        pipe = getattr(pipe, "estimator_", pipe)
    classificador = getattr(pipe, "named_steps", {}).get("classificador")
    alvo_col = "qualidade_inadequada_24h"
    tem_teste = (teste is not None) and (alvo_col in teste.columns) and all(f in teste.columns for f in features)

    try:
        limiar_f2_val = limiar_para_modelo(modelo_escolhido)
    except ValueError as erro:
        limiar_f2_val = None
        st.warning(f"Diagnóstico com limiar validado indisponível: {erro}. Execute o pipeline completo.")

    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.markdown("#### Matriz de Confusão — Limiar Padrão (0,50)")
        if p_img_padrao.is_file():
            st.image(str(p_img_padrao), width="stretch")
        elif pipe is not None and tem_teste:
            probs = probabilidades_diagnostico(
                modelo_escolhido, assinatura_modelo(modelo_escolhido),
                assinatura_arquivo(ROOT / ARQUIVOS_DADOS["teste"]), tuple(features), pipe, teste,
            )
            y_pred_padrao = (probs >= 0.50).astype(int)
            st.image(imagem_matriz(teste[alvo_col].values, y_pred_padrao, f"{modelo_escolhido} (0,50)"), width="stretch")
        else:
            st.info("ℹ Matriz de confusão indisponível localmente. Treine os modelos na aba ' 6. Executar Pipeline Completo'.")

    with col_mat2:
        rotulo_limiar = f"{limiar_f2_val:.2f}" if limiar_f2_val is not None else "indisponível"
        st.markdown(f"#### Matriz de Confusão — Limiar Calibrado por $F_2$ ({rotulo_limiar})")
        if limiar_f2_val is None:
            st.info("Matriz com limiar validado indisponível.")
        elif p_img_f2.is_file():
            st.image(str(p_img_f2), width="stretch")
        elif pipe is not None and tem_teste:
            probs = probabilidades_diagnostico(
                modelo_escolhido, assinatura_modelo(modelo_escolhido),
                assinatura_arquivo(ROOT / ARQUIVOS_DADOS["teste"]), tuple(features), pipe, teste,
            )
            y_pred_f2 = (probs >= limiar_f2_val).astype(int)
            st.image(imagem_matriz(teste[alvo_col].values, y_pred_f2, f"{modelo_escolhido} (F₂={limiar_f2_val:.2f})"), width="stretch")
        else:
            st.info(f"ℹ Matriz calibrada por F₂ indisponível. Treine os modelos na aba ' 6. Executar Pipeline Completo'.")

    st.markdown("---")
    st.markdown(f"####  Atributos Mais Determinantes (Importância de Features) — {modelo_escolhido}")

    col_imp1, col_imp2 = st.columns([1.2, 1])
    importancia_nb = None
    if modelo_escolhido == "Naive Bayes" and pipe is not None and tem_teste and limiar_f2_val is not None:
        resultado_imp = importancia_permutacao_nb(
            assinatura_modelo(modelo_escolhido), assinatura_arquivo(ROOT / ARQUIVOS_DADOS["teste"]),
            tuple(features), limiar_f2_val, pipe, teste,
        )
        importancia_nb = resultado_imp.importances_mean
        st.caption("Naive Bayes: importância por permutação (queda média de F₂ no teste, 5 repetições). Valores negativos são possíveis; análise descritiva, sem causalidade.")
    with col_imp1:
        if importancia_nb is not None:
            st.image(imagem_importancia(importancia_nb, features, modelo_escolhido, "Queda média de F₂ por permutação"), width="stretch")
        elif p_img_imp.is_file():
            st.image(str(p_img_imp), width="stretch")
        elif hasattr(classificador, "feature_importances_"):
            imps = classificador.feature_importances_
            st.image(imagem_importancia(imps, features, modelo_escolhido), width="stretch")
        else:
            st.info("ℹ Gráfico de importância indisponível neste ambiente.")

    with col_imp2:
        if importancia_nb is not None:
            df_imp = pd.DataFrame({"feature": features, "importancia": importancia_nb, "desvio_padrao": resultado_imp.importances_std}).sort_values("importancia", ascending=False)
            st.dataframe(df_imp.style.format({"importancia": "{:.4f}", "desvio_padrao": "{:.4f}"}), width="stretch")
        elif p_csv_imp.is_file():
            df_imp = pd.read_csv(p_csv_imp)
            st.dataframe(df_imp.head(10).style.format({"importancia": "{:.4f}"}), width="stretch")
        elif hasattr(classificador, "feature_importances_"):
            imps = classificador.feature_importances_
            df_imp_din = pd.DataFrame({"feature": features, "importancia": imps}).sort_values("importancia", ascending=False)
            st.dataframe(df_imp_din.head(10).style.format({"importancia": "{:.4f}"}), width="stretch")
        else:
            st.info("ℹ Tabela de atributos indisponível neste ambiente.")

    if modelo_escolhido == "Decision Tree":
        with st.expander(" Ver Regras Textuais da Árvore de Decisão"):
            if p_rules.is_file():
                st.code(p_rules.read_text(encoding="utf-8")[:2500], language="text")
            elif pipe is not None:
                from sklearn.tree import export_text
                clf_dt = pipe.named_steps.get("classificador")
                if clf_dt is not None:
                    regras_txt = export_text(clf_dt, feature_names=features, max_depth=4)
                    st.code(regras_txt[:2500], language="text")
                else:
                    st.info("Classificador da árvore não disponível no pipeline.")
            else:
                st.info("Regras textuais indisponíveis. Execute o treinamento na aba ' 6. Executar Pipeline Completo'.")


# ==============================================================================
# VISÃO 5: TESTAR NOVA DATA & PREVISÃO 24H (API AO VIVO)
# ==============================================================================
elif menu == "5. Testar Nova Data & Previsão 24h (API ao Vivo)":
    render_hero_header(
        titulo=" Consulta ao Vivo, Previsão 24h e Auditoria de Modelos",
        subtitulo="Escolha qualquer data e horário para São Paulo. O sistema consulta a API Open-Meteo em tempo real, monitora latência e consumo de RAM, projeta as condições para as 24 horas seguintes e audita as previsões frente ao padrão oficial CETESB N3+.",
        badges=[" Open-Meteo API ao Vivo", " Telemetria de RAM e Latência", " Previsão t₀  t₀+24h", " Trajetória 24h Seguintes", " Critérios Formais de Aprovação"]
    )

    import auditoria_api
    from forecast_ui import render_previsao_horaria

    agora_sp = datetime.datetime.now(ZoneInfo("America/Sao_Paulo"))

    st.markdown("#### 1. Selecionar Ponto de Início da Pesquisa (Data e Horário)")

    # Widgets vinculados ao session_state: atalhos atualizam os valores exibidos.
    if "data_alvo_input" not in st.session_state:
        st.session_state["data_alvo_input"] = agora_sp.date()
    if "hora_alvo_input" not in st.session_state:
        st.session_state["hora_alvo_input"] = f"{agora_sp.hour:02d}:00"

    def selecionar_cenario(data: datetime.date, hora: str) -> None:
        """Callback executado antes dos widgets, permitindo trocar os atalhos."""
        st.session_state["data_alvo_input"] = data
        st.session_state["hora_alvo_input"] = hora

    col_d1, col_d2, col_d3 = st.columns([1.4, 1.2, 2.4])
    with col_d1:
        data_selecionada = st.date_input(
            " Data de Início:",
            key="data_alvo_input",
            min_value=datetime.date(2022, 1, 1),
            max_value=agora_sp.date() + datetime.timedelta(days=2),
            help="Data inicial para a previsão. Pode ser data histórica ou o dia de hoje."
        )

    with col_d2:
        horas_opcoes = [f"{h:02d}:00" for h in range(24)]
        hora_str = st.selectbox(
            " Horário de Início (t₀):",
            options=horas_opcoes,
            key="hora_alvo_input",
            help="Momento exato em que a previsão para as próximas 24 horas será disparada."
        )
        hora_selecionada = int(hora_str.split(":")[0])

    with col_d3:
        st.markdown("**Atalhos de cenários para teste:**")
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.button(
            " 15/10/24 14h (Pico O₃)", width="stretch",
            on_click=selecionar_cenario, args=(datetime.date(2024, 10, 15), "14:00"),
        )
        col_b2.button(
            " 15/07/25 08h (Limpo)", width="stretch",
            on_click=selecionar_cenario, args=(datetime.date(2025, 7, 15), "08:00"),
        )
        col_b3.button(
            " Hoje / Recente", width="stretch",
            on_click=selecionar_cenario, args=(agora_sp.date(), f"{agora_sp.hour:02d}:00"),
        )

    col_consultar, col_atualizar = st.columns([3, 1])
    with col_consultar:
        btn_consultar = st.button(" Consultar API e Gerar Previsão para as Próximas 24 Horas", type="primary", width="stretch")
    with col_atualizar:
        btn_atualizar = st.button("Atualizar dados", width="stretch")

    # Execução da consulta
    chave_atual = (data_selecionada, hora_selecionada)
    resultado_anterior = st.session_state.get("resultado_v5")
    resultado_correto = (
        resultado_anterior is not None
        and (resultado_anterior["data_selecionada"], resultado_anterior["hora_selecionada"]) == chave_atual
    )
    # Não repetir automaticamente chamadas que falharam em cada rerun da interface.
    tentativa = st.session_state.get("tentativa_v5")
    consulta_expirada = bool(
        resultado_correto
        and (agora_sp - resultado_anterior.get("consultado_em", agora_sp)).total_seconds() >= 600
    )
    consulta_inicial = (
        data_selecionada == agora_sp.date()
        and (not resultado_correto or consulta_expirada)
        and (
            tentativa is None or tentativa[0] != chave_atual
            or (consulta_expirada and tentativa[1] <= resultado_anterior["consultado_em"])
        )
    )
    if btn_consultar or btn_atualizar or consulta_inicial:
        st.session_state["tentativa_v5"] = (chave_atual, agora_sp)
        ts_inicio_str = f"{data_selecionada} às {hora_selecionada:02d}:00"
        with st.spinner(f"Consultando APIs Open-Meteo (Qualidade do Ar + Meteorologia) para São Paulo iniciando em {ts_inicio_str}..."):
            try:
                if btn_atualizar:
                    consultar_api_cache.clear()
                df_bruto, meta = consultar_api_cache(data_selecionada, hora_selecionada)
                sanidade = auditoria_api.validar_sanidade_dados(df_bruto)
                res_features = auditoria_api.extrair_features_e_alvo(
                    df_bruto, data_selecionada, hora_selecionada,
                    referencia_historica=meta["referencia_historica"],
                )
                modelos = carregar_pipelines()
                prev_t0 = auditoria_api.gerar_previsao_t0(res_features["linha_t0"], modelos, res_features["tem_gt_t0_24h"])
                audit_janela, resumo_j = auditoria_api.auditar_janela_24h_seguinte(res_features["df_janela_24h"], modelos, res_features["tem_gt_janela"])

                st.session_state["resultado_v5"] = {
                    "data_selecionada": data_selecionada,
                    "hora_selecionada": hora_selecionada,
                    "meta": meta,
                    "sanidade": sanidade,
                    "res_features": res_features,
                    "prev_t0": prev_t0,
                    "modelos_consultados": tuple(modelos),
                    "audit_janela": audit_janela,
                    "resumo_j": resumo_j,
                    "consultado_em": agora_sp,
                }
                st.success(f"Consulta finalizada! Latência total: {meta['latencia_total_ms']} ms | RAM do processo: {meta['recursos']['memoria_processo_mb']} MB")
            except Exception as e:
                st.session_state.pop("resultado_v5", None)
                st.error(f"Erro na consulta à API ou processamento: {e}")

    # Exibição dos resultados
    if "resultado_v5" in st.session_state and (
        st.session_state["resultado_v5"]["data_selecionada"],
        st.session_state["resultado_v5"]["hora_selecionada"],
    ) == chave_atual:
        res = st.session_state["resultado_v5"]
        if (agora_sp - res.get("consultado_em", agora_sp)).total_seconds() >= 600:
            st.info("Esta consulta tem mais de 10 minutos. Clique em Atualizar dados para obter os valores mais recentes.")
        meta = res["meta"]
        sanidade = res["sanidade"]
        res_features = res["res_features"]
        prev_t0 = res["prev_t0"]
        audit_janela = res["audit_janela"]
        resumo_j = res["resumo_j"]
        recursos = meta["recursos"]

        render_previsao_horaria(res)
        st.caption("Previsão meteorológica e da qualidade do ar: Open-Meteo (dados modelados). "
                   "A inferência de Machine Learning abaixo estima o risco para t₀ + 24h; "
                   "nenhuma das duas constitui medição oficial da CETESB.")

        st.markdown("---")
        st.markdown("###  2. Monitoramento de Recursos e Performance Técnica")

        c_rec1, c_rec2, c_rec3, c_rec4 = st.columns(4)
        with c_rec1:
            st.metric(
                label=" Latência da API",
                value=f"{meta['latencia_total_ms']} ms",
                delta=f"Qualidade: {meta['latencia_aq_ms']}ms | Clima: {meta['latencia_weather_ms']}ms",
                delta_color="off"
            )
        with c_rec2:
            st.metric(
                label=" RAM do Processo Python",
                value=f"{recursos['memoria_processo_mb']} MB",
                delta="Consumo da aplicação",
                delta_color="off"
            )
        with c_rec3:
            st.metric(
                label=" Memória RAM do Sistema",
                value=f"{recursos['memoria_sistema_pct']}%",
                delta=f"{recursos['memoria_sistema_usada_gb']} GB de {recursos['memoria_sistema_total_gb']} GB",
                delta_color="off"
            )
        with c_rec4:
            st.metric(
                label=" Integridade dos Dados",
                value="100% Válidos" if sanidade["integro"] else "Com Alertas",
                delta=f"{sanidade['total_linhas']}h analisadas | {sanidade['sentinelas_detectadas']} sentinelas",
                delta_color="off"
            )

        with st.expander(" Inspecionar Diagnóstico Completo de Sanidade e Headers da API"):
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
        st.markdown("###  3. Previsão Emitida no Ponto de Partida (Horizonte de 24h)")

        if prev_t0:
            t0_fmt = prev_t0["t0"].strftime("%d/%m/%Y às %H:00")
            t24_fmt = prev_t0["t_alvo_24h"].strftime("%d/%m/%Y às %H:00")

            st.info(f" **Momento da Previsão (t₀):** `{t0_fmt}`    **Previsão para 24h à Frente (t₀ + 24h):** `{t24_fmt}`")

            # Condição observada em t0
            c_cond1, c_cond2, c_cond3, c_cond4 = st.columns(4)
            c_cond1.metric("Temperatura em t₀", f"{prev_t0['temp_t0']:.1f} °C")
            c_cond2.metric("Vento em t₀", f"{prev_t0['vento_t0']:.1f} km/h")
            c_cond3.metric("Ozônio (O₃) em t₀", f"{prev_t0['o3_t0']:.1f} µg/m³")
            c_cond4.metric("Qualidade Atual do Ar (t₀)", prev_t0["condicao_atual_t0"], prev_t0["poluente_agora"])

            st.markdown(f"####  Projeção dos Modelos para exatamente 24 Horas Depois (`{t24_fmt}`):")

            info_oficial = prev_t0["previsoes"].get("Modelo oficial (Sprint 5)", {})
            nome_oficial = carregar_resultado_oficial().get("modelo_final", "não registrado")
            st.metric(f"Modelo oficial — {nome_oficial}",
                      f"{info_oficial['prob'] * 100:.1f}%" if "prob" in info_oficial else "Indisponível",
                      info_oficial.get("rotulo", "Modelo indisponível"))
            st.caption(f"Limiar oficial: {limiar_para_modelo('Modelo oficial (Sprint 5)'):.2f}. Escore do modelo para t₀ + 24h. As probabilidades não passaram por calibração probabilística; não equivalem a chances reais nem à classificação horária direta da API.")

            st.caption("Comparação complementar: árvores exploratórias com ajustes próprios, Naive Bayes da Sprint 5 e baseline de persistência.")
            cols_prev = st.columns(5)
            modelos_nomes = ["XGBoost", "Random Forest", "Decision Tree", "Naive Bayes", "Persistência Temporal"]
            for col_p, nome_m in zip(cols_prev, modelos_nomes):
                info_p = prev_t0["previsoes"].get(nome_m, {})
                with col_p:
                    st.markdown(f"**{nome_m}{' (Sprint 5)' if nome_m == 'Naive Bayes' else ' (exploratório)' if nome_m != 'Persistência Temporal' else ''}**")
                    st.metric(
                         label="Escore para N3+",
                        value=f"{info_p['prob']*100:.1f}%" if "prob" in info_p else "Indisponível",
                        delta=info_p.get("rotulo", "Modelo indisponível ou dados insuficientes"),
                        delta_color="inverse" if "Inadequado" in info_p.get("rotulo", "") else "normal"
                    )
                    st.caption(f"Limiar de decisão: {info_p.get('limiar', 0.5):.2f}")

            # Auditoria de Acerto (se data passada)
            if prev_t0["tem_ground_truth"] and prev_t0["real_rotulo"]:
                st.markdown("##### Verificação retrospectiva com dados modelados da Open-Meteo, 24 horas depois:")
                st.markdown(f"""
                <div class="highlight-card">
                    <strong>Classificação retrospectiva com dados Open-Meteo em {t24_fmt}:</strong> {html.escape(str(prev_t0['real_rotulo']))}<br>
                    <strong>Poluente crítico estimado:</strong> {html.escape(str(prev_t0['poluente_critico_real']))}
                </div>
                """, unsafe_allow_html=True)

                nomes_auditoria = ["Modelo oficial (Sprint 5)", *modelos_nomes]
                for col_a, nome_m in zip(st.columns(len(nomes_auditoria)), nomes_auditoria):
                    info_p = prev_t0["previsoes"].get(nome_m, {})
                    pred = info_p.get("pred")
                    real = prev_t0["real_valor"]
                    with col_a:
                        if pred is None:
                            st.info(f"{nome_m}: modelo indisponível.")
                        elif pred == real:
                            st.success(f" **{nome_m} ACERTOU!**")
                        elif pred == 0 and real == 1:
                            st.error(f" **{nome_m}: Falso Negativo**")
                        else:
                            st.warning(f" **{nome_m}: Alarme Falso**")
            else:
                st.info("**Data Recente / Hoje:** O resultado real em 24h ainda não foi concluído. Os modelos acima representam o alerta preventivo para o seu treino amanhã!")

        st.markdown("---")
        st.markdown("###  4. Trajetória Horária das Emissões (cada alvo 24h depois)")

        if len(audit_janela) > 0:
            limiares = {
                nome: limiar_para_modelo(nome) for nome in ARQUIVOS_MODELOS
                if nome in res.get("modelos_consultados", ()) or f"{nome} (Risco %)" in audit_janela
            }
            imagem, tem_series = grafico_trajetoria(
                audit_janela, limiares, carregar_resultado_oficial().get("modelo_final", "Modelo"),
            )
            if not tem_series:
                st.info("Modelos ainda sem escores calculáveis para este intervalo. Os horários da API continuam disponíveis acima.")
            st.image(imagem, width="stretch")

            st.markdown("#### Tabela Detalhada das 24 Horas Seguintes: emissão e alvo (+24h)")
            st.caption("Cada linha usa os dados da hora de emissão para prever 24 horas depois. Emissões futuras são simulações condicionadas à previsão da API.")
            st.dataframe(audit_janela.drop(columns=["timestamp"], errors="ignore"), width="stretch", height=360)

    # ==========================================================================
    # SEÇÃO OBRIGATÓRIA: COMPROVAÇÃO DE QUAL MODELO FOI O MELHOR E CRITÉRIOS
    # ==========================================================================
    st.markdown("---")
    st.markdown('<div class="main-title" style="font-size: 1.8rem;"> Comprovação: Qual Modelo Foi o Melhor e Quais os Critérios de Aprovação?</div>', unsafe_allow_html=True)
    oficial = carregar_resultado_oficial()
    destaque_expl = destaques_exploratorios(comparativo)
    st.markdown('<div class="sub-title">Resultados do conjunto de teste cronológico já consultado; '
                'a política revisada usa a validação temporal; este teste histórico já era conhecido.</div>', unsafe_allow_html=True)

    if oficial:
        nome_final = oficial.get("selecionado_na_validacao", "Indisponível")
        limiar_oficial = oficial.get("limiar_final")
        f2_oficial = oficial.get("f2_validacao", {}).get(nome_final)
        recall_oficial = oficial.get("recall_validacao", {}).get(nome_final)
        st.success(f"**Modelo oficial selecionado na Sprint 5:** {nome_final} "
                   f"(limiar {limiar_oficial:.2f}, recall na validação {recall_oficial:.2%}, F₂ {f2_oficial:.4f}). "
                   "Maior recall nos limiares de máximo F₂, com piso da persistência. O teste conhecido não é validação independente." if limiar_oficial is not None and f2_oficial is not None and recall_oficial is not None
                   else f"Modelo oficial da Sprint 5: {nome_final}.")

    if comparativo is not None:
        st.markdown("#### Comparação exploratória de árvores no teste cronológico e Naive Bayes (Sprint 5)")
        st.caption("O Naive Bayes usa o pipeline da Sprint 4 reproduzido na Sprint 5 e seu limiar validado. Os destaques exploratórios abaixo consideram apenas as árvores.")
        st.dataframe(
            comparacao_exploratoria_com_nb(comparativo).style.format({
                "limiar": "{:.2f}",
                "acuracia": "{:.2%}",
                "precisao_c1": "{:.4f}",
                "recall_c1": "{:.2%}",
                "f1_c1": "{:.4f}",
                "f2_c1": "{:.4f}",
                "roc_auc": "{:.4f}",
            }),
            width="stretch"
        )

    col_melhor1, col_melhor2 = st.columns(2)
    with col_melhor1:
        nome_auc = destaque_expl["maior_auc"]
        if nome_auc:
            linha = destaque_expl["linhas"][nome_auc]
            st.markdown(f"#### Maior ROC-AUC exploratório: {nome_auc}")
            st.metric("ROC-AUC no teste", f"{linha['roc_auc']:.4f}")
            st.caption(f"F₂: {linha['f2_c1']:.4f} · Recall: {linha['recall_c1']:.2%} · "
                       f"FN: {int(linha['fn'])} · FP: {int(linha['fp'])}. "
                       "AUC mede capacidade de ordenação e não define sozinho o modelo selecionado.")
    with col_melhor2:
        nome_fn = destaque_expl["menor_fn"]
        if nome_fn:
            linha = destaque_expl["linhas"][nome_fn]
            empatados = destaque_expl["empates_fn"]
            st.markdown(f"#### Menos falsos negativos entre as árvores: {' e '.join(empatados)}")
            st.metric("Falsos negativos no teste", f"{int(linha['fn'])}")
            for nome in empatados:
                item = destaque_expl["linhas"][nome]
                st.caption(f"{nome}: recall {item['recall_c1']:.2%} · F₂ {item['f2_c1']:.4f} · "
                           f"limiar {item['limiar']:.2f} · FP {int(item['fp'])}.")
            st.caption("Estatística descritiva, sem substituir a decisão oficial.")

    st.markdown("---")
    st.markdown("####  Quais Foram os Critérios Formais para Aprovação do Modelo e o Porquê?")

    c_crit1, c_crit2 = st.columns(2)
    with c_crit1:
        st.markdown(r"""
        #####  Critério 1: Superar o Piso Real da Persistência Temporal
        * **Definição:** Em meteorologia, a *Persistência Temporal* (chutar que amanhã o ar estará igual ao de hoje) é uma baseline forte devido à inércia do ar. Seus resultados estão nas tabelas de avaliação.
        * **Critério de Aprovação:** Entre as alternativas com $F_2$ de validação igual ou superior ao da persistência, seleciona-se o maior Recall da classe inadequada, com desempate por $F_2$ e simplicidade.
        * **Resultado:** A tabela exploratória acima apresenta os resultados efetivos de cada execução. A Sprint 5 registra separadamente o modelo oficial.
        <br>
        #####  Critério 2: Minimização Rigorosa de Falsos Negativos (Métrica $F_2$)
        * **Definição:** A métrica oficial é o $F_2$ ($\beta=2$):
          $$F_2 = \frac{5 \cdot \text{Precisão} \cdot \text{Recall}}{4 \cdot \text{Precisão} + \text{Recall}}$$
        * **Por que adotar $\beta=2$?** Em atividades físicas ao ar livre, o volume respiratório de um atleta sobe de $6\,\text{L/min}$ para até $150\,\text{L/min}$.
        * **Falso Negativo (Grave):** O sistema não avisa, e o atleta treina sob ozônio tóxico ou $\text{PM}_{2,5}$, sofrendo inflamação alveolar e estresse oxidativo.
        * **Falso Positivo (Tolerável):** O sistema avisa ar ruim, mas o ar estava razoável; o atleta apenas transfere o treino para a esteira por precaução.
        * **Conclusão:** O fator $\beta^2=4$ da fórmula favorece o Recall; isso não representa uma razão de custo clínico entre erros.
        """, unsafe_allow_html=True)

    with c_crit2:
        st.markdown(r"""
        #####  Critério 3: Reprovação Sumária da Armadilha do DummyClassifier
        * **Definição:** A classe inadequada ($\text{N3+}$) é minoritária no conjunto de teste histórico; sua frequência não é uma estimativa universal para São Paulo.
        * **O Perigo da Acurácia Pura:** Um classificador ingênuo (*Dummy*) que sempre responde "ar adequado" pode atingir acurácia elevada, mas seu **Recall é 0,0%** (deixa passar todas as horas positivas). Os valores reproduzidos constam da tabela de comparação.
        * **Critério:** Modelos com Recall nulo ou sem discriminação são imediatamente **reprovados**.
        <br>
        #####  Critério 4: Calibração de Limiar sem Vazamento Temporal
        * **Definição:** Em vez do limiar fixo de 0,50, os limiares operacionais são lidos do JSON de calibração das três janelas temporais (*Out-of-Fold* com purga de 24h).
        * **Controle:** O pré-processamento e a calibração são ajustados somente no treino. A consulta ao teste não autoriza reajustar o limiar nem redefinir o vencedor.
        """, unsafe_allow_html=True)



# ==============================================================================
# VISÃO 6: EXECUTAR PIPELINE COMPLETO
# ==============================================================================
elif menu == "6. Executar Pipeline Completo":
    render_hero_header(
        titulo=" Orquestrador do Pipeline de Ponta a Ponta",
        subtitulo="Dispare a execução completa e automatizada: coleta, tratamento, modelagem e comparação contra baselines",
        badges=[" Pipeline Automatizado", " Limpeza e Purga 24h", " Treinamento e Calibração", " Geração de Relatórios"]
    )

    st.markdown("""
    <div class="metric-card" style="border-left: 4px solid #3b82f6; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #1e3a8a;">Fluxo de Execução Sequencial</h4>
        <ol style="margin-bottom: 0; padding-left: 20px; line-height: 1.8;">
            <li><strong>Coleta e Integração:</strong> Consulta ou verifica o repositório bruto das APIs públicas da Open-Meteo.</li>
            <li><strong>Limpeza e Split Temporal:</strong> Trata sentinelas, padroniza unidades e aplica purga temporal estrita de 24 horas.</li>
            <li><strong>Modelagem e Calibração:</strong> Reexecuta as baselines, a engenharia de atributos, os seis candidatos da Sprint 5 e a comparação exploratória de árvores com <code>ColumnTransformer</code> (Yeo-Johnson) e limiares por F₂.</li>
            <li><strong>Comparação e Auditoria:</strong> Avalia contra a Persistência Temporal e regera todos os relatórios e métricas.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        " **Nota de Desempenho:** Em servidores com recursos limitados (como o plano gratuito do Streamlit Cloud), "
        "o retreinamento completo pode demorar conforme os recursos disponíveis. Os modelos já treinados e relatórios "
        "incluídos no repositório dispensam reexecução imediata."
    )

    forcar_recoleta = st.checkbox(
        "Atualizar dados de treinamento a partir das APIs antes de retreinar",
        value=False,
        help="A coleta segue as datas configuradas no projeto. O retreinamento ocorre mesmo com esta opção desmarcada.",
    )
    if st.button(" Executar Pipeline Completo Agora", type="primary", width="stretch"):
        import contextlib
        import io
        import traceback

        log_buffer = io.StringIO()
        try:
            import pipeline_completo
            with st.spinner("Executando todas as etapas de preparação, treinamento e validação..."):
                with contextlib.redirect_stdout(log_buffer):
                    res = pipeline_completo.executar_pipeline_completo(forcar_recoleta=forcar_recoleta)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state.pop("resultado_v5", None)
                st.session_state.pop("tentativa_v5", None)
                st.success(f" Pipeline executado com sucesso em {res['tempo_execucao_s']:.2f} segundos! Todas as imagens e modelos foram atualizados.")
                st.dataframe(comparacao_exploratoria_com_nb(pd.DataFrame(res["comparativo"])), width="stretch")
                with st.expander(" Ver Log de Execução"):
                    st.code(log_buffer.getvalue(), language="text")
                render_reducao_erro()
        except ImportError as e:
            st.error(
                f" **Dependência ausente no ambiente Python:** {e}\n\n"
                f" **Como resolver:** Instale os pacotes necessários executando no seu terminal:\n"
                f"```bash\npip install -r requirements.txt\n```"
            )
        except Exception as e:
            st.error(
                f" **Erro durante a execução do pipeline:** {e}\n\n"
                " *Dica para Windows/OneDrive:* Se o projeto estiver dentro de uma pasta sincronizada pelo OneDrive ou com arquivos abertos, "
                "pause a sincronização ou mova a pasta do projeto para `C:\\Projetos\\` e tente novamente."
            )
            with st.expander(" Ver detalhes técnicos do erro (Traceback)"):
                if log_buffer.getvalue():
                    st.code(log_buffer.getvalue(), language="text")
                st.code(traceback.format_exc(), language="python")
