"""Modulo de integracao e auditoria de novas datas via API Open-Meteo.

Permite consultar qualquer data e horario para a regiao metropolitana de Sao Paulo,
medir o uso de memoria RAM e latencia da API em tempo real,
validar a sanidade dos dados coletados, extrair as 25 features congeladas da Sprint 4/5,
executar os modelos de Machine Learning (XGBoost, Random Forest, Decision Tree e Persistencia),
obter a previsao para as 24 horas seguintes e auditar detalhadamente se a previsao esta correta.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config/params.yaml"

LAT = -23.5489
LON = -46.6388
TIMEZONE = "America/Sao_Paulo"
UTC_OFFSET_H = -3
HORIZON_H = 24

FROZEN_FEATURES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "hora",
    "dia_semana",
    "mes",
    "pm10_lag_1h",
    "pm10_lag_24h",
    "pm10_media_24h",
    "pm2_5_lag_1h",
    "pm2_5_lag_24h",
    "pm2_5_media_24h",
    "carbon_monoxide_lag_1h",
    "carbon_monoxide_lag_24h",
    "carbon_monoxide_media_24h",
    "nitrogen_dioxide_lag_1h",
    "nitrogen_dioxide_lag_24h",
    "nitrogen_dioxide_media_24h",
    "sulphur_dioxide_lag_1h",
    "sulphur_dioxide_lag_24h",
    "sulphur_dioxide_media_24h",
    "ozone_lag_1h",
    "ozone_lag_24h",
    "ozone_media_24h",
    "temperature_2m_lag_1h",
    "temperature_2m_lag_24h",
    "temperature_2m_media_24h",
    "precipitation_lag_1h",
    "precipitation_lag_24h",
    "precipitation_media_24h",
    "wind_speed_10m_lag_1h",
    "wind_speed_10m_lag_24h",
    "wind_speed_10m_media_24h",
]

THRESHOLDS_MODELOS = {
    "Decision Tree": 0.35,
    "Random Forest": 0.40,
    "XGBoost": 0.25,
    "Persistência Temporal": 0.50,
}

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
