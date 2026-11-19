"""Boletim horário da Open-Meteo para atletas, preservando a auditoria ML da tela."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st


def _numero(valor: object, unidade: str = "", casas: int = 0) -> str:
    if pd.isna(valor):
        return "N/D"
    try:
        return f"{float(valor):.{casas}f}{unidade}"
    except (ValueError, TypeError):
        return "N/D"


def render_previsao_horaria(resultado: dict) -> None:
    """Exibe horas devolvidas pela API. ML 24h permanece na seção original."""
    dados = resultado["res_features"]
    meta = resultado["meta"]
    atual = dados["linha_t0"]
    horas = pd.concat([atual, dados["df_janela_24h"]], ignore_index=True)
    t0 = pd.Timestamp(dados["t0"])
    horas = horas[horas["timestamp"] >= t0]
    if horas.empty:
        st.warning("Nenhum horário retornado pelas APIs para o período selecionado.")
        return

    horas = horas.sort_values("timestamp").drop_duplicates("timestamp").head(25)
    primeira = horas.iloc[0]
    inicio = pd.Timestamp(primeira["timestamp"])
    if inicio > t0:
        st.warning(f"A API não retornou o horário inicial {t0:%d/%m %H:00}. Exibindo somente as horas efetivamente disponíveis.")
    hoje = pd.Timestamp.now(tz="America/Sao_Paulo").date()
    data_ref = inicio.date()
    titulo = "Hoje" if data_ref == hoje else inicio.strftime("%d/%m/%Y")

    estados_validos = horas[horas["dados_validos"]]
    horas_alerta = int(estados_validos["estado_atual"].sum())
    quantidade_validas = len(estados_validos)
    if quantidade_validas == 0:
        mensagem_periodo = "Dados insuficientes"
    elif horas_alerta:
        mensagem_periodo = f"{horas_alerta}h com indicação de ar inadequado"
    else:
        mensagem_periodo = "Sem indicação de ar inadequado"

    if quantidade_validas == 0:
        orientacao_treino = "Verifique dados locais antes de planejar o treino."
    elif horas_alerta:
        orientacao_treino = "Para os horários com alerta, considere reduzir o esforço ou treinar em ambiente interno."
    else:
        orientacao_treino = "Consulte as condições locais antes da atividade ao ar livre."

    if not bool(primeira["dados_validos"]):
        estado = "Dados insuficientes"
    elif int(primeira["estado_atual"]) == 1:
        estado = "Ar inadequado (N3+)"
    else:
        estado = "Ar adequado (N1–N2)"

    atualizado = escape(meta.get("consulta_realizada_em", "horário não informado"))
    fonte = escape(meta.get("fonte_meteorologia", "Open-Meteo"))
    cartoes = []
    for _, linha in horas.iterrows():
        instante = pd.Timestamp(linha["timestamp"])
        dia = "Hoje" if instante.date() == hoje else "Amanhã" if instante.date() == hoje + pd.Timedelta(days=1) else instante.strftime("%d/%m")
        disponivel = bool(linha["dados_validos"])
        alerta = disponivel and int(linha["estado_atual"]) == 1
        classe = "alert" if alerta else "" if disponivel else "unknown"
        descricao = "Inadequado" if alerta else "Adequado" if disponivel else "Sem dados"
        cartoes.append(
            f'<div class="air-hour {classe}">'
            f'<div class="air-hour-time">{escape(dia)} · {instante:%H}:00</div>'
            f'<div class="air-hour-temp">{_numero(linha["temperature_2m"], "°")}</div>'
            f'<div class="air-hour-state">{descricao}</div>'
            f'<div class="air-hour-meta">PM2,5: {_numero(linha["pm2_5"], " µg/m³", 1)}<br>'
            f'Vento: {_numero(linha["wind_speed_10m"], " km/h", 1)}</div>'
            '</div>'
        )

    nota_fonte = (
        "Histórico modelado pelas APIs Open-Meteo"
        if data_ref < hoje else "Previsão horária fornecida pelas APIs Open-Meteo"
    )
    st.markdown(
        f'<section class="air-forecast" aria-label="Boletim da qualidade do ar">'
        f'<div class="air-forecast-head"><div class="air-forecast-title">Previsão do ar · {escape(titulo)}</div>'
        f'<div class="air-forecast-source">São Paulo · Consulta: {atualizado}</div></div>'
        f'<div class="air-today"><div class="air-now">'
        f'<div class="air-kicker">{inicio:%H}:00 · Qualidade do ar</div>'
        f'<div class="air-current"><div><div class="air-current-temp">{_numero(primeira["temperature_2m"], "°")}</div>'
        f'<div class="air-current-status">{escape(estado)}</div>'
        f'<div class="air-current-description">PM2,5: {_numero(primeira["pm2_5"], " µg/m³", 1)} · '
        f'Ozônio: {_numero(primeira["ozone"], " µg/m³", 1)}</div>'
        f'</div></div></div><div class="air-outlook">'
        f'<div class="air-outlook-title">Panorama das próximas horas</div>'
        f'<div class="air-outlook-value">{escape(mensagem_periodo)}</div>'
        f'<div class="air-outlook-caption">{quantidade_validas}h com dados válidos de {len(horas)}h disponíveis</div>'
        f'<div class="air-outlook-caption" style="margin-top:10px">{escape(orientacao_treino)}</div>'
        f'</div></div><div class="air-grid">{"".join(cartoes)}</div>'
        f'<div class="air-forecast-note">{escape(nota_fonte)} · Clima: {fonte}. '
        f'Classificação indicativa segundo os limiares CETESB usados no projeto; '
        f'este painel não representa medições oficiais da CETESB. '
        f'As probabilidades do modelo de ML para o horizonte de 24h aparecem abaixo.</div></section>',
        unsafe_allow_html=True,
    )
