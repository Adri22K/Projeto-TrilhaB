"""Limpeza, split, EDA, alvo e atributos iniciais da Sprint 2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "params.yaml"
RAW_PATH = ROOT / "data" / "raw" / "integrado.csv"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "sprint2"

POLLUTANTS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]
WEATHER = ["temperature_2m", "precipitation", "wind_speed_10m"]
NON_NEGATIVE = POLLUTANTS + ["precipitation", "wind_speed_10m"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def inspect_raw(frame: pd.DataFrame, sentinels: list[int]) -> dict[str, Any]:
    numeric = [column for column in frame.columns if column != "timestamp"]
    numeric_coerced = frame[numeric].apply(pd.to_numeric, errors="coerce")
    sentinel_counts = {
        column: int(numeric_coerced[column].isin(sentinels).sum()) for column in numeric
    }
    invalid_counts = {
        column: int((numeric_coerced[column] < 0).sum()) for column in NON_NEGATIVE
    }
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
        "invalid_timestamps": int(timestamps.isna().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "sentinels_by_column": sentinel_counts,
        "negative_invalid_by_column": invalid_counts,
        "numeric_describe": json_ready(numeric_coerced.describe().round(4).to_dict()),
    }


def clean(frame: pd.DataFrame, sentinels: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = frame.copy()
    rows_before = len(cleaned)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    invalid_timestamp_rows = int(cleaned["timestamp"].isna().sum())
    cleaned = cleaned.dropna(subset=["timestamp"])

    numeric = [column for column in cleaned.columns if column != "timestamp"]
    coercion_counts: dict[str, int] = {}
    for column in numeric:
        missing_before = int(cleaned[column].isna().sum())
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        coercion_counts[column] = int(cleaned[column].isna().sum()) - missing_before

    sentinel_cells = 0
    for sentinel in sentinels:
        matches = cleaned[numeric].eq(sentinel)
        sentinel_cells += int(matches.sum().sum())
        cleaned[numeric] = cleaned[numeric].mask(matches)

    invalid_cells: dict[str, int] = {}
    for column in NON_NEGATIVE:
        invalid = cleaned[column] < 0
        invalid_cells[column] = int(invalid.sum())
        cleaned.loc[invalid, column] = pd.NA

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()
    duplicate_timestamps = int(cleaned["timestamp"].duplicated().sum())
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="first")
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    log = {
        "rows_before": rows_before,
        "invalid_timestamp_rows_removed": invalid_timestamp_rows,
        "values_coerced_to_nan_by_column": coercion_counts,
        "sentinel_cells_changed_to_nan": sentinel_cells,
        "negative_cells_changed_to_nan_by_column": invalid_cells,
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_timestamps_removed": duplicate_timestamps,
        "rows_after": len(cleaned),
        "missing_after_by_column": {
            key: int(value) for key, value in cleaned.isna().sum().items()
        },
    }
    return cleaned, log


def temporal_split(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    candidate_index = math.floor(len(frame) * (1 - test_fraction))
    cutoff = frame.loc[candidate_index, "timestamp"].ceil("D")
    train = frame.loc[frame["timestamp"] < cutoff].copy()
    test = frame.loc[frame["timestamp"] >= cutoff].copy()
    return train, test, cutoff


def target_sources(frame: pd.DataFrame, target_config: dict[str, Any]) -> pd.DataFrame:
    limits = target_config["thresholds"]
    sources = pd.DataFrame(index=frame.index)
    sources["pm10_ruim"] = (
        frame["pm10"].rolling(24, min_periods=24).mean() > limits["pm10_24h_ug_m3"]
    )
    sources["pm2_5_ruim"] = (
        frame["pm2_5"].rolling(24, min_periods=24).mean()
        > limits["pm2_5_24h_ug_m3"]
    )
    sources["so2_ruim"] = (
        frame["sulphur_dioxide"].rolling(24, min_periods=24).mean()
        > limits["sulphur_dioxide_24h_ug_m3"]
    )
    sources["ozone_ruim"] = (
        frame["ozone"].rolling(8, min_periods=8).mean() > limits["ozone_8h_ug_m3"]
    )
    sources["no2_ruim"] = (
        frame["nitrogen_dioxide"] > limits["nitrogen_dioxide_1h_ug_m3"]
    )
    co_ppm = (
        frame["carbon_monoxide"].rolling(8, min_periods=8).mean()
        * target_config["co_ug_m3_to_ppm_factor"]
    )
    sources["co_ruim"] = co_ppm > limits["carbon_monoxide_8h_ppm"]
    return sources


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    featured = frame.copy()
    featured["hora"] = featured["timestamp"].dt.hour
    featured["dia_semana"] = featured["timestamp"].dt.dayofweek
    featured["mes"] = featured["timestamp"].dt.month

    for column in POLLUTANTS + WEATHER:
        featured[f"{column}_lag_1h"] = featured[column].shift(1)
        featured[f"{column}_lag_24h"] = featured[column].shift(24)
        featured[f"{column}_media_24h"] = (
            featured[column].shift(1).rolling(24, min_periods=24).mean()
        )
    return featured


def save_eda(train: pd.DataFrame) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    numeric = POLLUTANTS + WEATHER
    describe = train[numeric].describe().round(4)
    correlation = train[numeric].corr().round(4)
    describe.to_csv(REPORT_DIR / "estatisticas_treino.csv", encoding="utf-8")
    correlation.to_csv(REPORT_DIR / "correlacao_treino.csv", encoding="utf-8")

    outliers: dict[str, int] = {}
    for column in numeric:
        q1 = train[column].quantile(0.25)
        q3 = train[column].quantile(0.75)
        iqr = q3 - q1
        mask = (train[column] < q1 - 1.5 * iqr) | (train[column] > q3 + 1.5 * iqr)
        outliers[column] = int(mask.sum())

    daily = train.set_index("timestamp")[["pm2_5", "ozone"]].resample("D").mean()
    axes = daily.plot(subplots=True, figsize=(12, 6), title=["PM2,5 — média diária", "O₃ — média diária"])
    axes[-1].set_xlabel("Data")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "serie_temporal_treino.png", dpi=150)
    plt.close()

    train[POLLUTANTS].hist(bins=40, figsize=(12, 8))
    plt.suptitle("Histogramas dos poluentes — treino")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "histogramas_treino.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    train[POLLUTANTS].plot.box(ax=ax, rot=30)
    ax.set_title("Boxplots dos poluentes — treino")
    ax.set_ylabel("Concentração (µg/m³)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "boxplots_treino.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)), labels=numeric, rotation=60, ha="right")
    ax.set_yticks(range(len(numeric)), labels=numeric)
    for row in range(len(numeric)):
        for column in range(len(numeric)):
            ax.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax)
    ax.set_title("Correlação no treino")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlacao_treino.png", dpi=150)
    plt.close()

    pairs = [
        (left, right, float(correlation.loc[left, right]))
        for index, left in enumerate(POLLUTANTS)
        for right in POLLUTANTS[index + 1 :]
    ]
    strongest = max(pairs, key=lambda item: abs(item[2])) if pairs else None
    strongest_pair = [strongest[0], strongest[1]] if strongest else []
    strongest_value = strongest[2] if strongest else None
    return {
        "outliers_iqr_by_column": outliers,
        "strongest_pollutant_correlation_pair": strongest_pair,
        "strongest_pollutant_correlation": strongest_value,
    }


def main() -> None:
    config = load_config(CONFIG_PATH)
    sprint2 = config["sprint2"]
    raw = pd.read_csv(RAW_PATH)
    inspection = inspect_raw(raw, sprint2["sentinels"])
    cleaned, cleaning_log = clean(raw, sprint2["sentinels"])

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(INTERIM_DIR / "integrado_limpo.csv", index=False, encoding="utf-8")
    train, test, cutoff = temporal_split(cleaned, sprint2["test_fraction"])
    train.to_csv(INTERIM_DIR / "treino_limpo.csv", index=False, encoding="utf-8")
    test.to_csv(INTERIM_DIR / "teste_limpo.csv", index=False, encoding="utf-8")

    eda = save_eda(train)
    source_flags = target_sources(cleaned, sprint2["target"])
    bad_now = source_flags.any(axis=1)
    horizon = sprint2["forecast_horizon_hours"]
    target_name = sprint2["target"]["name"]
    target = bad_now.shift(-horizon).astype("Int64")

    featured = add_features(cleaned)
    featured[target_name] = target
    train_features = featured.loc[featured["timestamp"] < cutoff].copy()
    test_features = featured.loc[featured["timestamp"] >= cutoff].copy()
    train_target_valid = train_features.dropna(subset=[target_name])
    test_target_valid = test_features.dropna(subset=[target_name])
    model_feature_columns = [
        column for column in featured.columns if column not in ("timestamp", target_name)
    ]
    train_model = train_target_valid.dropna(subset=model_feature_columns)
    test_model = test_target_valid.dropna(subset=model_feature_columns)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_model.to_csv(PROCESSED_DIR / "treino_features.csv", index=False, encoding="utf-8")
    test_model.to_csv(PROCESSED_DIR / "teste_features.csv", index=False, encoding="utf-8")

    shifted_sources = source_flags.shift(-horizon)
    train_mask = cleaned["timestamp"] < cutoff
    train_trigger_counts = {
        key: int(value)
        for key, value in shifted_sources.loc[train_mask].sum().items()
    }
    train_counts = train_target_valid[target_name].value_counts().sort_index()
    test_counts = test_target_valid[target_name].value_counts().sort_index()
    report = {
        "inspection_before_cleaning": inspection,
        "cleaning": cleaning_log,
        "split": {
            "cutoff": cutoff.isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": train["timestamp"].min().isoformat(),
            "train_end": train["timestamp"].max().isoformat(),
            "test_start": test["timestamp"].min().isoformat(),
            "test_end": test["timestamp"].max().isoformat(),
        },
        "residual_missing": {
            key: int(value) for key, value in cleaned.isna().sum().items()
        },
        "eda_train": eda,
        "target": {
            "name": target_name,
            "horizon_hours": horizon,
            "definition": sprint2["target"],
            "train_valid_rows": len(train_target_valid),
            "train_negative": int(train_counts.get(0, 0)),
            "train_positive": int(train_counts.get(1, 0)),
            "train_positive_percent": round(float(train_target_valid[target_name].mean() * 100), 4),
            "test_valid_rows": len(test_target_valid),
            "test_negative": int(test_counts.get(0, 0)),
            "test_positive": int(test_counts.get(1, 0)),
            "test_positive_percent": round(float(test_target_valid[target_name].mean() * 100), 4),
            "rows_without_future_label": int(target.isna().sum()),
            "train_trigger_counts": train_trigger_counts,
        },
        "features": {
            "calendar": ["hora", "dia_semana", "mes"],
            "lags_hours": [1, 24],
            "rolling_window_hours": 24,
            "rolling_uses_shift": 1,
            "feature_columns_except_timestamp_and_target": len(featured.columns) - 2,
            "excluded_target_construction_columns": list(source_flags.columns),
            "model_train_rows": len(train_model),
            "model_test_rows": len(test_model),
            "train_rows_removed_for_initial_history": len(train_target_valid) - len(train_model),
            "test_rows_removed_for_missing_features": len(test_target_valid) - len(test_model),
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "relatorio_execucao.json").open("w", encoding="utf-8") as file:
        json.dump(json_ready(report), file, ensure_ascii=False, indent=2)
    print(json.dumps(json_ready(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
