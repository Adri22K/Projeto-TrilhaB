"""Coleta e integra as fontes brutas da Sprint 1 sem limpar os dados."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "params.yaml"
RAW_DIR = ROOT / "data" / "raw"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_session(retries: int, backoff_factor: float) -> requests.Session:
    retry_policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_api(
    session: requests.Session,
    name: str,
    api_config: dict[str, Any],
    common_params: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {
        **common_params,
        "hourly": ",".join(api_config["hourly"]),
    }
    try:
        response = session.get(api_config["url"], params=params, timeout=timeout)
        status_code = response.status_code
        content_type = response.headers.get("content-type", "")
        response.raise_for_status()
        if "json" not in content_type.lower():
            raise ValueError(f"{name}: resposta não é JSON ({content_type!r})")
        payload = response.json()
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Falha na coleta de {name}: {exc}") from exc

    required_keys = {"hourly", "hourly_units", "timezone", "utc_offset_seconds"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(f"{name}: chaves ausentes no JSON: {sorted(missing_keys)}")
    if "time" not in payload["hourly"]:
        raise ValueError(f"{name}: hourly.time ausente")

    row_count = len(payload["hourly"]["time"])
    for variable in api_config["hourly"]:
        if variable not in payload["hourly"]:
            raise ValueError(f"{name}: variável {variable!r} ausente")
        if len(payload["hourly"][variable]) != row_count:
            raise ValueError(f"{name}: tamanho inconsistente em {variable!r}")

    inspection = {
        "status_code": status_code,
        "content_type": content_type,
        "top_level_keys": sorted(payload.keys()),
        "hourly_keys": sorted(payload["hourly"].keys()),
        "hourly_units": payload["hourly_units"],
        "row_count": row_count,
        "request_url": response.url,
    }
    return payload, inspection


def payload_to_frame(payload: dict[str, Any], variables: list[str]) -> pd.DataFrame:
    data = {"timestamp": payload["hourly"]["time"]}
    data.update({variable: payload["hourly"][variable] for variable in variables})
    return pd.DataFrame(data)


def validate_existing_outputs(paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Valida a coleta já persistida antes de tratar uma reexecução como concluída."""
    air = pd.read_csv(paths["air_csv"], parse_dates=["timestamp"])
    weather = pd.read_csv(paths["weather_csv"], parse_dates=["timestamp"])
    merged = pd.read_csv(paths["merged_csv"], parse_dates=["timestamp"])

    expected_air = {"timestamp", *config["apis"]["air_quality"]["hourly"]}
    expected_weather = {"timestamp", *config["apis"]["weather"]["hourly"]}
    if not expected_air.issubset(air.columns):
        raise ValueError("Coleta existente de qualidade do ar está incompleta")
    if not expected_weather.issubset(weather.columns):
        raise ValueError("Coleta existente de clima está incompleta")

    for name, frame in (("air_quality", air), ("weather", weather), ("integrado", merged)):
        if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
            raise ValueError(f"Coleta existente inválida em {name}: timestamp ausente ou duplicado")

    expected_start = pd.Timestamp(config["period"]["start_date"])
    expected_end = pd.Timestamp(config["period"]["end_date"])
    if merged["timestamp"].min().normalize() != expected_start:
        raise ValueError("Coleta existente não corresponde à data inicial configurada")
    if merged["timestamp"].max().normalize() != expected_end:
        raise ValueError("Coleta existente não corresponde à data final configurada")

    with paths["log_json"].open(encoding="utf-8") as file:
        log = json.load(file)
    logged_rows = int(log["integration"]["merged_rows"])
    if logged_rows != len(merged):
        raise ValueError("Log da coleta existente diverge do número de linhas integradas")

    return {
        "air_rows": len(air),
        "weather_rows": len(weather),
        "merged_rows": len(merged),
        "merged_duplicate_timestamps": int(merged["timestamp"].duplicated().sum()),
    }


def existing_outputs_complete(paths: dict[str, Path], overwrite: bool) -> bool:
    """Retorna True quando uma reexecução pode reutilizar a coleta completa existente."""
    if overwrite:
        return False
    existing = [key for key, path in paths.items() if path.exists()]
    if not existing:
        return False
    if len(existing) == len(paths):
        return True
    missing = [key for key, path in paths.items() if not path.exists()]
    raise FileExistsError(
        "Conjunto parcial de saídas brutas detectado. "
        f"Existentes: {existing}; ausentes: {missing}. "
        "Os arquivos existentes foram preservados para evitar sobrescrita acidental."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    period_tag = (
        config["period"]["start_date"].replace("-", "")
        + "_"
        + config["period"]["end_date"].replace("-", "")
    )
    paths = {
        "air_json": RAW_DIR / f"air_quality_{period_tag}.json",
        "air_csv": RAW_DIR / f"air_quality_{period_tag}.csv",
        "weather_json": RAW_DIR / f"weather_{period_tag}.json",
        "weather_csv": RAW_DIR / f"weather_{period_tag}.csv",
        "merged_csv": RAW_DIR / "integrado.csv",
        "log_json": RAW_DIR / "coleta_integracao_log.json",
    }
    if existing_outputs_complete(paths, args.overwrite):
        summary = validate_existing_outputs(paths, config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("Reexecução idempotente: coleta completa já existe e nenhuma linha foi duplicada.")
        return

    location = config["location"]
    common_params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": config["period"]["start_date"],
        "end_date": config["period"]["end_date"],
        "timezone": location["timezone"],
    }
    http = config["http"]
    session = build_session(http["retries"], http["backoff_factor"])

    payloads: dict[str, dict[str, Any]] = {}
    inspections: dict[str, dict[str, Any]] = {}
    for name in ("air_quality", "weather"):
        payloads[name], inspections[name] = fetch_api(
            session,
            name,
            config["apis"][name],
            common_params,
            http["timeout_seconds"],
        )

    air = payload_to_frame(payloads["air_quality"], config["apis"]["air_quality"]["hourly"])
    weather = payload_to_frame(payloads["weather"], config["apis"]["weather"]["hourly"])
    air["timestamp"] = pd.to_datetime(air["timestamp"], errors="raise")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="raise")

    merge_config = config["merge"]
    merged = air.merge(
        weather,
        on="timestamp",
        how=merge_config["how"],
        validate=merge_config["validate"],
        sort=True,
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, json_key in (("air_quality", "air_json"), ("weather", "weather_json")):
        with paths[json_key].open("w", encoding="utf-8") as file:
            json.dump(payloads[name], file, ensure_ascii=False, indent=2)
    air.to_csv(paths["air_csv"], index=False, encoding="utf-8")
    weather.to_csv(paths["weather_csv"], index=False, encoding="utf-8")
    merged.to_csv(paths["merged_csv"], index=False, encoding="utf-8")

    log = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(args.config.resolve()),
        "inspection": inspections,
        "integration": {
            "join_key": "timestamp",
            "how": merge_config["how"],
            "validate": merge_config["validate"],
            "air_rows": len(air),
            "weather_rows": len(weather),
            "air_duplicate_timestamps": int(air["timestamp"].duplicated().sum()),
            "weather_duplicate_timestamps": int(weather["timestamp"].duplicated().sum()),
            "air_unmatched_rows": int((~air["timestamp"].isin(weather["timestamp"])).sum()),
            "weather_unmatched_rows": int((~weather["timestamp"].isin(air["timestamp"])).sum()),
            "merged_rows": len(merged),
            "merged_columns": list(merged.columns),
            "missing_by_column": {key: int(value) for key, value in merged.isna().sum().items()},
        },
    }
    with paths["log_json"].open("w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=2)

    print(json.dumps(log["integration"], ensure_ascii=False, indent=2))
    print(f"Dados brutos preservados em: {RAW_DIR}")


if __name__ == "__main__":
    main()
