"""Modulo de integracao e auditoria de novas datas via API Open-Meteo.

Permite consultar qualquer data e horario para a regiao metropolitana de Sao Paulo,
medir o uso de memoria RAM e latencia da API em tempo real,
validar a sanidade dos dados coletados, extrair as 25 features congeladas da Sprint 4/5,
executar os modelos de Machine Learning (XGBoost, Random Forest, Decision Tree e Persistencia),
obter a previsao para as 24 horas seguintes e auditar detalhadamente se a previsao esta correta.
"""

from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import psutil
import requests
import yaml
from limiares_modelos import limiar_para_modelo

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config/params.yaml"

LAT = -23.5489
LON = -46.6388
TIMEZONE = "America/Sao_Paulo"
UTC_OFFSET_H = -3
HORIZON_H = 24
LOGGER = logging.getLogger(__name__)

FROZEN_FEATURES = [
    "pm10",
    "pm10_lag_1h",
    "pm10_lag_24h",
    "pm10_media_24h",
    "sulphur_dioxide",
    "sulphur_dioxide_lag_1h",
    "sulphur_dioxide_lag_24h",
    "sulphur_dioxide_media_24h",
    "precipitation",
    "precipitation_lag_1h",
    "precipitation_lag_24h",
    "precipitation_media_24h",
    "wind_speed_10m",
    "wind_speed_10m_lag_1h",
    "wind_speed_10m_lag_24h",
    "wind_speed_10m_media_24h",
    "ozone_media_8h",
    "ozone_max_8h",
    "temperature_max_24h",
    "temperature_max_24h_delta",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    "elevacao_solar_alvo",
]

CETESB_THRESHOLDS = {
    "pm10_24h": 100.0,
    "pm2_5_24h": 50.0,
    "so2_24h": 40.0,
    "ozone_8h": 130.0,
    "no2_1h": 240.0,
    "co_8h_ppm": 11.0,
    "co_factor": 0.0008727,
}


def medir_recursos_sistema() -> dict[str, Any]:
    """Mede o consumo atual de memoria RAM do processo Python e do sistema operacional."""
    proc = psutil.Process(os.getpid())
    ram_proc_mb = round(proc.memory_info().rss / (1024 * 1024), 1)

    vm = psutil.virtual_memory()
    ram_sys_total_gb = round(vm.total / (1024**3), 2)
    ram_sys_used_gb = round(vm.used / (1024**3), 2)
    ram_sys_free_gb = round(vm.available / (1024**3), 2)
    ram_sys_pct = vm.percent

    return {
        "memoria_processo_mb": ram_proc_mb,
        "memoria_sistema_total_gb": ram_sys_total_gb,
        "memoria_sistema_usada_gb": ram_sys_used_gb,
        "memoria_sistema_livre_gb": ram_sys_free_gb,
        "memoria_sistema_pct": ram_sys_pct,
    }


def calcular_elevacao_solar(ts: pd.Series) -> np.ndarray:
    """Calcula elevacao solar NOAA sem vazamento."""
    doy = ts.dt.dayofyear.to_numpy()
    hour = ts.dt.hour.to_numpy() + 0.5
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (
        0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    )
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )
    true_solar_min = hour * 60 + eqtime + 4 * LON - 60 * UTC_OFFSET_H
    hour_angle = np.radians(true_solar_min / 4 - 180)
    lat_rad = np.radians(LAT)
    cos_zen = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(cos_zen, -1, 1)))


def consultar_api_open_meteo(data_alvo: str | datetime.date, hora_inicio: int = 0, *args, **kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Consulta as APIs Open-Meteo de Qualidade do Ar e Clima com buffer de 2 dias antes e 2 dias depois.
    
    Mede a latencia individual de cada chamada e o consumo de memoria RAM.
    """
    if isinstance(data_alvo, str):
        dt_alvo = datetime.datetime.strptime(data_alvo, "%Y-%m-%d").date()
    else:
        dt_alvo = data_alvo

    dt_inicio = dt_alvo - datetime.timedelta(days=2)
    dt_fim = dt_alvo + datetime.timedelta(days=2)
    agora_local = datetime.datetime.now(ZoneInfo(TIMEZONE))
    # O arquivo histórico pode ter atraso de dias; a previsão cobre o período recente.
    usar_previsao_clima = dt_alvo >= agora_local.date() - datetime.timedelta(days=5)

    str_inicio = dt_inicio.strftime("%Y-%m-%d")
    str_fim = dt_fim.strftime("%Y-%m-%d")

    # As APIs são independentes: requisitá-las em paralelo reduz a espera da tela.
    url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_aq = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str_inicio,
        "end_date": str_fim,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": TIMEZONE,
    }
    url_weather = (
        "https://api.open-meteo.com/v1/forecast" if usar_previsao_clima
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    params_w = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str_inicio,
        "end_date": str_fim,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": TIMEZONE,
    }
    def obter(nome: str, url: str, params: dict) -> tuple[pd.DataFrame, int, int]:
        inicio = time.monotonic()
        resposta = requests.get(url, params=params, timeout=25)
        latencia = int((time.monotonic() - inicio) * 1000)
        if resposta.status_code != 200:
            raise RuntimeError(f"Erro na API de {nome} ({resposta.status_code}): {resposta.text[:350]}")
        horario = resposta.json().get("hourly")
        if not horario or "time" not in horario:
            raise ValueError(f"API de {nome} não retornou dados horários válidos.")
        df = pd.DataFrame(horario)
        df["timestamp"] = pd.to_datetime(df.pop("time"))
        return df, latencia, resposta.status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        aq = executor.submit(obter, "Qualidade do Ar", url_aq, params_aq)
        clima = executor.submit(obter, "Meteorologia", url_weather, params_w)
        df_aq, latencia_aq_ms, status_aq = aq.result()
        df_w, latencia_weather_ms, status_clima = clima.result()

    # Merge
    df_bruto = pd.merge(df_aq, df_w, on="timestamp", how="inner", validate="one_to_one")
    if df_bruto.empty:
        raise ValueError("As APIs não retornaram horas coincidentes no período solicitado.")
    latencia_total_ms = max(latencia_aq_ms, latencia_weather_ms)

    # Recursos de memoria
    recursos = medir_recursos_sistema()

    meta = {
        "data_alvo": str(dt_alvo),
        "hora_inicio": hora_inicio,
        "timestamp_inicio": f"{str(dt_alvo)} {hora_inicio:02d}:00:00",
        "periodo_coletado": f"{str_inicio} a {str_fim}",
        "total_horas_buffer": len(df_bruto),
        "latencia_aq_ms": latencia_aq_ms,
        "latencia_weather_ms": latencia_weather_ms,
        "latencia_total_ms": latencia_total_ms,
        "status_code_aq": status_aq,
        "status_code_weather": status_clima,
        "consulta_realizada_em": agora_local.strftime("%d/%m/%Y às %H:%M"),
        "fonte_meteorologia": "Open-Meteo Forecast" if usar_previsao_clima else "Open-Meteo Archive",
        "referencia_historica": not usar_previsao_clima,
        "ultima_hora_api": df_bruto["timestamp"].max().strftime("%d/%m/%Y às %H:%M"),
        "recursos": recursos,
    }

    return df_bruto, meta


def validar_sanidade_dados(df: pd.DataFrame) -> dict[str, Any]:
    """Audita a integridade, presenca de sentinelas, nulos e limites fisicos dos dados recebidos da API."""
    colunas_poluentes = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    colunas_clima = ["temperature_2m", "precipitation", "wind_speed_10m"]
    colunas_analise = colunas_poluentes + colunas_clima

    total_linhas = len(df)
    nulos = df[colunas_analise].isna().sum().to_dict()
    total_nulos = sum(nulos.values())

    # Sentinelas
    sentinelas = [-999, -9999, 9999]
    sentinela_count = int(df[colunas_analise].isin(sentinelas).sum().sum())

    # Nao-negativos obrigatorios
    negativos = {}
    for col in ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "precipitation", "wind_speed_10m"]:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0:
            negativos[col] = neg_count

    # Limites fisicos plausiveis
    anomalias_fisicas = []
    if (df["pm10"] > 800).any():
        anomalias_fisicas.append("PM10 > 800 ug/m³ detectado")
    if (df["pm2_5"] > 500).any():
        anomalias_fisicas.append("PM2.5 > 500 ug/m³ detectado")
    if (df["temperature_2m"] < -10).any() or (df["temperature_2m"] > 50).any():
        anomalias_fisicas.append("Temperatura fora da faixa esperada (-10°C a 50°C)")
    if (df["wind_speed_10m"] > 216).any():
        anomalias_fisicas.append("Vento > 216 km/h detectado")

    dados_integros = (total_nulos == 0) and (sentinela_count == 0) and (len(negativos) == 0) and (len(anomalias_fisicas) == 0)

    return {
        "integro": dados_integros,
        "total_linhas": total_linhas,
        "total_nulos": total_nulos,
        "nulos_por_coluna": nulos,
        "sentinelas_detectadas": sentinela_count,
        "negativos_invalidos": negativos,
        "anomalias_fisicas": anomalias_fisicas,
        "status_texto": "Dados 100% Íntegros e Validados" if dados_integros else "Dados com Alertas de Sanidade",
    }


def extrair_features_e_alvo(
    df_bruto: pd.DataFrame,
    data_alvo: str | datetime.date,
    hora_inicio: int = 0,
    referencia_historica: bool = False,
    *args,
    **kwargs,
) -> dict[str, Any]:
    """Calcula as 25 features congeladas, o alvo CETESB N3+ e separa o instante t0 e as 24h seguintes."""
    if isinstance(data_alvo, str):
        str_alvo = data_alvo
    else:
        str_alvo = data_alvo.strftime("%Y-%m-%d")

    df = df_bruto.sort_values("timestamp").reset_index(drop=True).copy()
    if df.empty or df["timestamp"].duplicated().any():
        raise ValueError("Série horária vazia ou com timestamps duplicados")
    # Uma hora ausente não pode deslocar lags ou o horizonte real de 24h.
    grade = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", name="timestamp")
    df = df.set_index("timestamp").reindex(grade).reset_index()

    # Tratamento de sentinelas e negativos
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    for sent in [-999, -9999, 9999]:
        df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] == sent)

    for c in ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "precipitation", "wind_speed_10m"]:
        df[c] = df[c].mask(df[c] < 0)

    validos_brutos = df[[
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "temperature_2m", "wind_speed_10m", "precipitation",
    ]].notna().all(axis=1)
    # Sem uma janela de 24h completa, não é seguro classificar o ar como adequado.
    df["dados_validos"] = validos_brutos.rolling(24, min_periods=24).sum().eq(24)
    df["features_validas"] = validos_brutos.rolling(48, min_periods=48).sum().eq(48)
    # Preenchimento apenas com horas anteriores; não usar horas futuras como passado.
    df[numeric_cols] = df[numeric_cols].ffill()

    # 1. Regras CETESB N3 para cada poluente
    o3_8h = df["ozone"].rolling(8, min_periods=8).mean()
    pm10_24h = df["pm10"].rolling(24, min_periods=24).mean()
    pm25_24h = df["pm2_5"].rolling(24, min_periods=24).mean()
    so2_24h = df["sulphur_dioxide"].rolling(24, min_periods=24).mean()
    co_8h = df["carbon_monoxide"].rolling(8, min_periods=8).mean() * CETESB_THRESHOLDS["co_factor"]
    no2_1h = df["nitrogen_dioxide"]

    flag_o3 = o3_8h > CETESB_THRESHOLDS["ozone_8h"]
    flag_pm10 = pm10_24h > CETESB_THRESHOLDS["pm10_24h"]
    flag_pm25 = pm25_24h > CETESB_THRESHOLDS["pm2_5_24h"]
    flag_so2 = so2_24h > CETESB_THRESHOLDS["so2_24h"]
    flag_co = co_8h > CETESB_THRESHOLDS["co_8h_ppm"]
    flag_no2 = no2_1h > CETESB_THRESHOLDS["no2_1h"]

    estado_atual = (flag_o3 | flag_pm10 | flag_pm25 | flag_so2 | flag_co | flag_no2).astype(int)
    df["estado_atual"] = estado_atual

    def definir_responsavel(row_idx):
        criticos = []
        if flag_o3.iloc[row_idx]:
            criticos.append(f"O₃ ({o3_8h.iloc[row_idx]:.1f} > 130)")
        if flag_pm25.iloc[row_idx]:
            criticos.append(f"PM2.5 ({pm25_24h.iloc[row_idx]:.1f} > 50)")
        if flag_pm10.iloc[row_idx]:
            criticos.append(f"PM10 ({pm10_24h.iloc[row_idx]:.1f} > 100)")
        if flag_so2.iloc[row_idx]:
            criticos.append(f"SO₂ ({so2_24h.iloc[row_idx]:.1f} > 40)")
        if flag_no2.iloc[row_idx]:
            criticos.append(f"NO₂ ({no2_1h.iloc[row_idx]:.1f} > 240)")
        if flag_co.iloc[row_idx]:
            criticos.append(f"CO ({co_8h.iloc[row_idx]:.1f} > 11)")
        return ", ".join(criticos) if criticos else "Normal (N1-N2)"

    df["poluente_critico_agora"] = [definir_responsavel(i) for i in range(len(df))]

    # Ground truth no horizonte t + 24 h
    futuro_valido = df["dados_validos"].shift(-HORIZON_H, fill_value=False)
    df["qualidade_inadequada_24h"] = df["estado_atual"].shift(-HORIZON_H).where(futuro_valido)
    df["poluente_critico_24h"] = df["poluente_critico_agora"].shift(-HORIZON_H).where(futuro_valido)

    # 2. Construcao das 25 features congeladas
    for col in ["pm10", "sulphur_dioxide", "precipitation", "wind_speed_10m"]:
        df[f"{col}_lag_1h"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(24)
        df[f"{col}_media_24h"] = df[col].shift(1).rolling(24, min_periods=24).mean()

    df["ozone_media_8h"] = o3_8h
    df["ozone_max_8h"] = df["ozone"].rolling(8, min_periods=8).max()
    df["temperature_max_24h"] = df["temperature_2m"].rolling(24, min_periods=24).max()
    df["temperature_max_24h_delta"] = df["temperature_max_24h"] - df["temperature_max_24h"].shift(24)

    hora = df["timestamp"].dt.hour
    mes = df["timestamp"].dt.month
    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["mes_sin"] = np.sin(2 * np.pi * (mes - 1) / 12)
    df["mes_cos"] = np.cos(2 * np.pi * (mes - 1) / 12)
    df["elevacao_solar_alvo"] = calcular_elevacao_solar(df["timestamp"] + pd.Timedelta(hours=HORIZON_H))

    # Ponto de partida exato t0 selecionado pelo usuario
    t0 = pd.Timestamp(f"{str_alvo} {hora_inicio:02d}:00:00")
    t_fim_24h = t0 + pd.Timedelta(hours=24)

    # Linha t0
    linha_t0 = df[df["timestamp"] == t0].copy()

    # Janela das 24 horas seguintes a t0 (t0+1h ate t0+24h)
    df_janela_24h = df[(df["timestamp"] > t0) & (df["timestamp"] <= t_fim_24h)].copy()

    # Dia completo de 00h a 23h da data alvo
    df_dia_completo = df[df["timestamp"].dt.strftime("%Y-%m-%d") == str_alvo].copy()

    # Verifica disponibilidade de ground truth observado
    agora = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    tem_gt_t0_24h = bool(
        referencia_historica and len(linha_t0) > 0 and t_fim_24h < agora
        and pd.notna(linha_t0["qualidade_inadequada_24h"].iloc[0])
        and bool(linha_t0["dados_validos"].iloc[0])
        and bool(df.loc[df["timestamp"] == t_fim_24h, "dados_validos"].any())
    )
    tem_gt_janela = bool(
        referencia_historica and len(df_janela_24h) == 24
        and (df_janela_24h["timestamp"] + pd.Timedelta(hours=HORIZON_H) < agora).all()
        and df_janela_24h["dados_validos"].all()
        and df_janela_24h["qualidade_inadequada_24h"].notna().all()
    )

    return {
        "df_completo": df,
        "t0": t0,
        "t_fim_24h": t_fim_24h,
        "linha_t0": linha_t0,
        "df_janela_24h": df_janela_24h,
        "df_dia_completo": df_dia_completo,
        "tem_gt_t0_24h": tem_gt_t0_24h,
        "tem_gt_janela": tem_gt_janela,
    }


def _entrada_modelo(df: pd.DataFrame, pipeline: Any) -> tuple[pd.DataFrame, pd.Series]:
    """Respeita o contrato do artefato; a persistência exige apenas o estado atual."""
    features = list(getattr(pipeline, "feature_names_in_", FROZEN_FEATURES))
    if features == ["estado_atual"]:
        validade = df["dados_validos"]
    elif features == FROZEN_FEATURES:
        validade = df.get("features_validas", df["dados_validos"])
    else:
        raise ValueError("Contrato de features do modelo incompatível com a consulta")
    x = df.reindex(columns=features)
    finitos = np.isfinite(x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all(axis=1)
    return x, pd.Series(finitos, index=x.index) & validade.astype(bool)


def gerar_previsao_t0(
    linha_t0: pd.DataFrame,
    modelos: dict[str, Any],
    tem_gt: bool,
) -> dict[str, Any]:
    """Gera a previsao especifica emitida no instante t0 para exatamente t0 + 24h."""
    if len(linha_t0) == 0:
        return {}

    row = linha_t0.iloc[0]
    t0 = row["timestamp"]
    t_alvo = t0 + pd.Timedelta(hours=24)

    prevs = {}
    for nome_modelo, pipeline in modelos.items():
        limiar = limiar_para_modelo(nome_modelo)
        try:
            X, validas = _entrada_modelo(linha_t0, pipeline)
            if not bool(validas.iloc[0]):
                prevs[nome_modelo] = {"limiar": limiar, "rotulo": "Dados insuficientes"}
                continue
            prob = float(pipeline.predict_proba(X)[0, 1])
            if not np.isfinite(prob):
                raise ValueError("Probabilidade não finita")
            pred = int(prob >= limiar)
            prevs[nome_modelo] = {
                "prob": prob,
                "pred": pred,
                "limiar": limiar,
                "rotulo": "Inadequado (N3+)" if pred == 1 else "Adequado (N1-N2)",
            }
        except (ValueError, TypeError, AttributeError) as exc:
            LOGGER.warning("Modelo %s indisponível: %s", nome_modelo, exc)
            prevs[nome_modelo] = {"limiar": limiar, "rotulo": "Modelo indisponível"}

    # Persistencia
    if bool(row["dados_validos"]):
        pred_pers = int(row["estado_atual"])
        prevs["Persistência Temporal"] = {
            "prob": float(pred_pers),
            "pred": pred_pers,
            "limiar": 0.50,
            "rotulo": "Inadequado (N3+)" if pred_pers == 1 else "Adequado (N1-N2)",
        }
    else:
        prevs["Persistência Temporal"] = {"limiar": 0.50, "rotulo": "Dados insuficientes"}

    real_valor = None
    real_rotulo = None
    poluente_critico = None
    if tem_gt and pd.notna(row["qualidade_inadequada_24h"]):
        real_valor = int(row["qualidade_inadequada_24h"])
        real_rotulo = "Inadequado (N3+)" if real_valor == 1 else "Adequado (N1-N2)"
        poluente_critico = row["poluente_critico_24h"]

    return {
        "t0": t0,
        "t_alvo_24h": t_alvo,
        "condicao_atual_t0": ("Inadequado (N3+)" if row["estado_atual"] == 1 else "Adequado (N1-N2)") if bool(row["dados_validos"]) else "Dados insuficientes",
        "poluente_agora": row["poluente_critico_agora"],
        "temp_t0": row["temperature_2m"],
        "vento_t0": row["wind_speed_10m"],
        "o3_t0": row["ozone"],
        "pm25_t0": row["pm2_5"],
        "previsoes": prevs,
        "tem_ground_truth": tem_gt,
        "real_valor": real_valor,
        "real_rotulo": real_rotulo,
        "poluente_critico_real": poluente_critico,
    }


def auditar_janela_24h_seguinte(
    df_janela: pd.DataFrame,
    modelos: dict[str, Any],
    tem_ground_truth: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gera tabela hora a hora para as 24 horas seguintes a t0 com predicoes e auditoria."""
    if df_janela.empty:
        return pd.DataFrame(), {}
    audit_df = pd.DataFrame(index=df_janela.index)
    audit_df["Data/Hora"] = df_janela["timestamp"].dt.strftime("%d/%m %H:00")
    audit_df["timestamp"] = df_janela["timestamp"]
    audit_df["Alvo (+24h)"] = (df_janela["timestamp"] + pd.Timedelta(hours=HORIZON_H)).dt.strftime("%d/%m %H:00")
    audit_df["Temp (°C)"] = df_janela["temperature_2m"].round(1)
    audit_df["Vento (km/h)"] = df_janela["wind_speed_10m"].round(1)
    audit_df["O₃ (ug/m³)"] = df_janela["ozone"].round(1)
    audit_df["PM2.5 (ug/m³)"] = df_janela["pm2_5"].round(1)

    # Previsoes dos modelos geradas em cada hora
    probs_por_modelo = {}
    for nome_modelo, pipeline in modelos.items():
        limiar = limiar_para_modelo(nome_modelo)
        probs = pd.Series(np.nan, index=df_janela.index, dtype=float)
        try:
            X, mask_validas = _entrada_modelo(df_janela, pipeline)
            if mask_validas.any():
                calculadas = np.asarray(pipeline.predict_proba(X.loc[mask_validas])[:, 1], dtype=float)
                if not np.isfinite(calculadas).all():
                    raise ValueError("Probabilidade não finita")
                probs.loc[mask_validas] = calculadas
        except (ValueError, TypeError, AttributeError) as exc:
            LOGGER.warning("Falha na projeção horária de %s: %s", nome_modelo, exc)
        probs_por_modelo[nome_modelo] = probs
        audit_df[f"{nome_modelo} (Risco %)"] = probs.map(
            lambda valor: f"{valor * 100:.1f}%" if pd.notna(valor) else "N/D"
        )
        audit_df[f"{nome_modelo} (Previsão)"] = probs.map(
            lambda valor: "Indisponível" if pd.isna(valor) else "Inadequado" if valor >= limiar else "Adequado"
        )

    resumo = {}
    if tem_ground_truth:
        y_real = df_janela["qualidade_inadequada_24h"].astype(int).values
        audit_df["Referência retrospectiva Open-Meteo"] = np.where(y_real == 1, "Inadequado (N3+)", "Adequado (N1-N2)")
        audit_df["Poluente Crítico"] = df_janela["poluente_critico_24h"].values

        for nome_modelo, probs in probs_por_modelo.items():
            limiar = limiar_para_modelo(nome_modelo)
            validas = probs.notna()
            if not validas.any():
                continue
            preds = (probs.loc[validas].to_numpy() >= limiar).astype(int)
            reais = df_janela.loc[validas, "qualidade_inadequada_24h"].astype(int).to_numpy()

            status_col = pd.Series("Indisponível", index=X.index)
            for idx, y_t, y_p in zip(X.loc[validas].index, reais, preds):
                if y_t == 1 and y_p == 1:
                    status_col.loc[idx] = "Acerto (Verdadeiro Positivo)"
                elif y_t == 0 and y_p == 0:
                    status_col.loc[idx] = "Acerto (Verdadeiro Negativo)"
                elif y_t == 1 and y_p == 0:
                    status_col.loc[idx] = "Falso Negativo (Crítico)"
                else:
                    status_col.loc[idx] = "Falso Positivo (Conservador)"

            audit_df[f"Auditoria {nome_modelo}"] = status_col

            acertos = int((preds == reais).sum())
            total = len(reais)
            resumo[nome_modelo] = {
                "acertos": acertos,
                "total": total,
                "acuracia_pct": round(acertos / total * 100, 1) if total > 0 else 0,
                "tp": int(((reais == 1) & (preds == 1)).sum()),
                "tn": int(((reais == 0) & (preds == 0)).sum()),
                "fp": int(((reais == 0) & (preds == 1)).sum()),
                "fn": int(((reais == 1) & (preds == 0)).sum()),
            }

    return audit_df, resumo
