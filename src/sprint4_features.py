"""Sprint 4 / Trilha B: features novas, selecao no treino, pipeline congelado e lift S3 -> S4.

Execute na raiz do repositorio:
    python src/sprint4_features.py

Entradas (Sprints 2 e 3, nao sao alteradas):
    data/interim/integrado_limpo.csv
    data/processed/treino_features.csv
    data/processed/teste_features.csv
    reports/sprint3/comparativo_baselines.csv
    config/params.yaml

Saidas:
    data/processed/treino_features_s4.csv, teste_features_s4.csv
    models/column_transformer_s4.joblib   (transformer CONGELADO, fit so no treino)
    models/naive_bayes_s4.joblib          (Pipeline completo do NB retreinado)
    models/features_s4.json               (lista congelada de colunas)
    reports/sprint4/*                     (erros, selecao, metricas, lift, matrizes)

Regras respeitadas:
    - mesmo corte da Sprint 2 (14/04/2026) e mesma purga de 24 h da Sprint 3;
    - toda escolha (features e normalizacao) e feita com validacao temporal DENTRO do treino;
    - o teste e usado uma unica vez, no final, para medir os tres baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sprint3_baselines as s3  # noqa: E402  (reaproveita persistencia, avaliacao e graficos)

INTERIM_FILE = ROOT / "data/interim/integrado_limpo.csv"
TRAIN_S2 = ROOT / "data/processed/treino_features.csv"
TEST_S2 = ROOT / "data/processed/teste_features.csv"
TRAIN_S4 = ROOT / "data/processed/treino_features_s4.csv"
TEST_S4 = ROOT / "data/processed/teste_features_s4.csv"
S3_RESULTS = ROOT / "reports/sprint3/comparativo_baselines.csv"
CONFIG_FILE = ROOT / "config/params.yaml"
REPORT_DIR = ROOT / "reports/sprint4"
MODEL_DIR = ROOT / "models"

ALVO = s3.ALVO
CUTOFF = s3.CUTOFF
HORIZON_H = s3.HORIZON_H
LAT, LON, UTC_OFFSET_H = -23.5489, -46.6388, -3

# Validacao temporal DENTRO do treino: 3 janelas consecutivas, cada uma treinada so no passado.
FOLDS = [("2025-07-01", "2025-10-01"), ("2025-10-01", "2026-01-01"), ("2026-01-01", None)]

NEW_FEATURES = [
    "ozone_media_8h", "ozone_max_8h",                       # estrutura da metrica CETESB do O3
    "ozone_8h_max_24h", "ozone_8h_max_24h_delta",           # pico diario e tendencia
    "horas_ruins_24h",                                      # duracao do episodio
    "temperature_max_24h", "temperature_max_24h_delta",     # calor do dia e tendencia
    "hora_sin", "hora_cos", "mes_sin", "mes_cos",           # calendario ciclico
    "elevacao_solar_alvo",                                  # sol na hora prevista (astronomico)
]

FAMILIES: dict[str, list[str]] = {
    **{c: [c, f"{c}_lag_1h", f"{c}_lag_24h", f"{c}_media_24h"] for c in s3.POLLUTANTS + s3.WEATHER},
    "calendario_bruto": ["hora", "dia_semana", "mes"],
    "ozone_8h": ["ozone_media_8h", "ozone_max_8h"],
    "pico_diario": ["ozone_8h_max_24h", "ozone_8h_max_24h_delta"],
    "episodio": ["horas_ruins_24h"],
    "temp_max": ["temperature_max_24h", "temperature_max_24h_delta"],
    "hora_ciclica": ["hora_sin", "hora_cos"],
    "mes_ciclico": ["mes_sin", "mes_cos"],
    "solar": ["elevacao_solar_alvo"],
}
SCALERS = {"standard": StandardScaler, "yeo_johnson": PowerTransformer}


# --------------------------------------------------------------------------- features
def solar_elevation(ts: pd.Series) -> np.ndarray:
    """Elevacao do sol (graus) em Sao Paulo, pela formula astronomica da NOAA.

    Depende so da data/hora e da coordenada: e conhecida com antecedencia,
    por isso pode ser calculada para o instante PREVISTO (t + 24 h) sem vazamento.
    """
    doy = ts.dt.dayofyear.to_numpy()
    hour = ts.dt.hour.to_numpy() + 0.5  # meio da hora
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    true_solar_min = hour * 60 + eqtime + 4 * LON - 60 * UTC_OFFSET_H
    hour_angle = np.radians(true_solar_min / 4 - 180)
    lat = np.radians(LAT)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(cos_zen, -1, 1)))


def estado_atual(clean: pd.DataFrame, config: dict) -> pd.Series:
    """Classe do ar no instante t (mesma regra do alvo), usando so t e o passado."""
    cfg = config["sprint2"]["target"]
    lim = cfg["thresholds"]
    regras = {
        "pm10": (24, lim["pm10_24h_ug_m3"], 1.0),
        "pm2_5": (24, lim["pm2_5_24h_ug_m3"], 1.0),
        "sulphur_dioxide": (24, lim["sulphur_dioxide_24h_ug_m3"], 1.0),
        "ozone": (8, lim["ozone_8h_ug_m3"], 1.0),
        "carbon_monoxide": (8, lim["carbon_monoxide_8h_ppm"], cfg["co_ug_m3_to_ppm_factor"]),
        "nitrogen_dioxide": (1, lim["nitrogen_dioxide_1h_ug_m3"], 1.0),
    }
    flags = []
    for col, (janela, limite, fator) in regras.items():
        media = (clean[col] * fator).rolling(janela, min_periods=janela).mean()
        flags.append((media > limite).where(media.notna()))
    flags = pd.concat(flags, axis=1)
    return flags.any(axis=1).astype(float).where(flags.notna().all(axis=1))


def build_new_features(clean: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cria as 12 features novas. Todas usam somente o instante t e o passado,
    exceto a elevacao solar, que e astronomica (conhecida com antecedencia)."""
    df = clean.sort_values("timestamp").reset_index(drop=True)
    if not df["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Serie com lacunas: janelas em horas exigem passo de 1 h")
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    o3_8h = df["ozone"].rolling(8, min_periods=8).mean()
    out["ozone_media_8h"] = o3_8h
    out["ozone_max_8h"] = df["ozone"].rolling(8, min_periods=8).max()
    out["ozone_8h_max_24h"] = o3_8h.rolling(24, min_periods=24).max()
    out["ozone_8h_max_24h_delta"] = out["ozone_8h_max_24h"] - out["ozone_8h_max_24h"].shift(24)
    out["horas_ruins_24h"] = estado_atual(df, config).rolling(24, min_periods=24).sum()
    out["temperature_max_24h"] = df["temperature_2m"].rolling(24, min_periods=24).max()
    out["temperature_max_24h_delta"] = out["temperature_max_24h"] - out["temperature_max_24h"].shift(24)
    hora = df["timestamp"].dt.hour
    mes = df["timestamp"].dt.month
    out["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    out["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    out["mes_sin"] = np.sin(2 * np.pi * (mes - 1) / 12)
    out["mes_cos"] = np.cos(2 * np.pi * (mes - 1) / 12)
    out["elevacao_solar_alvo"] = solar_elevation(df["timestamp"] + pd.Timedelta(hours=HORIZON_H))
    return out


def leakage_check(clean: pd.DataFrame, config: dict, n_points: int = 6) -> list[str]:
    """Recalcula as features apagando o futuro de t; o valor em t nao pode mudar."""
    full = build_new_features(clean, config)
    idx = np.linspace(100, len(clean) - 30, n_points).astype(int)
    checados = []
    for i in idx:
        cut = build_new_features(clean.iloc[: i + 1], config)
        a, b = full.iloc[i][NEW_FEATURES], cut.iloc[i][NEW_FEATURES]
        if not np.allclose(a.astype(float), b.astype(float), equal_nan=True):
            raise AssertionError(f"Vazamento temporal em {clean['timestamp'].iloc[i]}")
        checados.append(str(clean["timestamp"].iloc[i]))
    return checados


# --------------------------------------------------------------------------- pipeline
def make_transformer(columns: list[str], scaler: str = "yeo_johnson") -> ColumnTransformer:
    """Transformer da Sprint 4: mediana (robusta a caudas) + normalizacao escolhida na validacao."""
    steps = [("imputador", SimpleImputer(strategy="median"))]
    if scaler == "yeo_johnson":
        steps.append(("normalizador", PowerTransformer(method="yeo-johnson", standardize=True)))
    else:
        steps.append(("normalizador", StandardScaler()))
    return ColumnTransformer([("numericas", Pipeline(steps), list(columns))], remainder="drop")


def make_pipeline(columns: list[str], classifier, scaler: str = "yeo_johnson") -> Pipeline:
    return Pipeline([("column_transformer", make_transformer(columns, scaler)),
                     ("classificador", classifier)])


def cv_score(train: pd.DataFrame, columns: list[str], scaler: str) -> dict[str, float]:
    recalls, f1s = [], []
    for inicio, fim in FOLDS:
        a = pd.Timestamp(inicio)
        b = pd.Timestamp(fim) if fim else CUTOFF
        fit = train[train["timestamp"] + pd.Timedelta(hours=HORIZON_H) < a]
        val = train[(train["timestamp"] >= a) & (train["timestamp"] + pd.Timedelta(hours=HORIZON_H) < b)]
        model = make_pipeline(columns, GaussianNB(), scaler).fit(fit[columns], fit[ALVO])
        _, r, f, _ = precision_recall_fscore_support(
            val[ALVO], model.predict(val[columns]), labels=[0, 1], zero_division=0)
        recalls.append(r[1])
        f1s.append(f[1])
    return {"recall_medio": float(np.mean(recalls)), "f1_medio": float(np.mean(f1s)),
            "f1_por_janela": [round(float(x), 4) for x in f1s]}


def backward_selection(train: pd.DataFrame, scaler: str, tol: float = 0.002):
    """Remove familias de features enquanto o F1 medio da validacao temporal subir."""
    atual = list(FAMILIES)
    cols = lambda fams: [c for f in fams for c in FAMILIES[f]]  # noqa: E731
    melhor = cv_score(train, cols(atual), scaler)
    passos = [{"normalizacao": scaler, "passo": 0, "removida": "(todas as features S2 + novas)",
               "n_features": len(cols(atual)), **melhor}]
    while len(atual) > 1:
        tentativas = []
        for fam in atual:
            resto = [f for f in atual if f != fam]
            tentativas.append((cv_score(train, cols(resto), scaler), fam))
        score, fam = max(tentativas, key=lambda t: (t[0]["f1_medio"], t[0]["recall_medio"]))
        if score["f1_medio"] <= melhor["f1_medio"] + tol:
            break
        atual.remove(fam)
        melhor = score
        passos.append({"normalizacao": scaler, "passo": len(passos), "removida": fam,
                       "n_features": len(cols(atual)), **melhor})
    return atual, cols(atual), melhor, passos


# --------------------------------------------------------------------------- analise de erros S3
def s3_error_analysis(train: pd.DataFrame) -> pd.DataFrame:
    """Erros dos baselines da Sprint 3 na ULTIMA janela de validacao (dentro do treino)."""
    a = pd.Timestamp(FOLDS[-1][0])
    fit = train[train["timestamp"] + pd.Timedelta(hours=HORIZON_H) < a]
    val = train[(train["timestamp"] >= a)
                & (train["timestamp"] + pd.Timedelta(hours=HORIZON_H) < CUTOFF)].copy()
    nb = s3.pipeline_classico(GaussianNB()).fit(fit[s3.FEATURES], fit[ALVO])
    val["prev_nb_s3"] = nb.predict(val[s3.FEATURES])
    val["prev_persistencia"] = val["estado_atual"].astype(int)
    val["hora"] = val["timestamp"].dt.hour
    val["tipo_positivo"] = np.where(val["estado_atual"] == 1, "continuacao", "inicio_de_episodio")
    return val


# --------------------------------------------------------------------------- principal
def main() -> dict:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(INTERIM_FILE, parse_dates=["timestamp"]).sort_values("timestamp")
    clean = clean.reset_index(drop=True)
    novas = build_new_features(clean, config)
    novas["estado_atual"] = estado_atual(clean, config)
    checados = leakage_check(clean, config)

    treino = s3.ler_features(TRAIN_S2).merge(novas, on="timestamp", how="left", validate="one_to_one")
    teste = s3.ler_features(TEST_S2).merge(novas, on="timestamp", how="left", validate="one_to_one")
    if not (treino["timestamp"] < CUTOFF).all() or not (teste["timestamp"] >= CUTOFF).all():
        raise ValueError("Split diferente do da Sprint 2")
    TRAIN_S4.parent.mkdir(parents=True, exist_ok=True)
    treino.to_csv(TRAIN_S4, index=False, encoding="utf-8")
    teste.to_csv(TEST_S4, index=False, encoding="utf-8")

    n_csv = len(treino)
    treino = treino.loc[treino["timestamp"] + pd.Timedelta(hours=HORIZON_H) < CUTOFF].copy()

    # 1) Erros da Sprint 3 na validacao interna (motivam as features)
    val = s3_error_analysis(treino)
    pos = val[val[ALVO] == 1]
    erros_hora = pd.DataFrame({
        "positivos": pos.groupby("hora").size(),
        "fn_nb_s3": pos.groupby("hora")["prev_nb_s3"].apply(lambda s: int((s == 0).sum())),
        "fn_persistencia": pos.groupby("hora")["prev_persistencia"].apply(lambda s: int((s == 0).sum())),
    }).reindex(range(24), fill_value=0)
    erros_hora.index.name = "hora"
    erros_hora.to_csv(REPORT_DIR / "erros_s3_validacao_por_hora.csv")
    erros_tipo = pos.groupby("tipo_positivo").agg(
        positivos=(ALVO, "size"),
        recall_nb_s3=("prev_nb_s3", "mean"),
        recall_persistencia=("prev_persistencia", "mean"),
    ).round(4)
    erros_tipo.to_csv(REPORT_DIR / "erros_s3_validacao_por_tipo.csv")
    pos_tp_fn = pos.assign(grupo=np.where(pos["prev_nb_s3"] == 1, "TP", "FN")).groupby("grupo")[
        ["ozone", "ozone_media_8h", "ozone_lag_24h", "temperature_2m", "hora"]].median().round(1)
    pos_tp_fn.to_csv(REPORT_DIR / "erros_s3_validacao_tp_vs_fn.csv")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(erros_hora.index - 0.2, erros_hora["fn_nb_s3"], width=0.4, label="FN Naive Bayes S3")
    ax.bar(erros_hora.index + 0.2, erros_hora["fn_persistencia"], width=0.4, label="FN persistência")
    ax.plot(erros_hora.index, erros_hora["positivos"], "k.--", label="horas positivas")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora do dia (local)")
    ax.set_ylabel("Horas")
    ax.set_title("Falsos negativos da Sprint 3 por hora — validação interna (jan–abr/2026)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "erros_s3_validacao_por_hora.png", dpi=150)
    plt.close(fig)

    # 2) Selecao de features e da normalizacao — so no treino
    s2_ref = {sc: cv_score(treino, s3.FEATURES, sc) for sc in SCALERS}
    resultados, passos = {}, []
    for sc in SCALERS:
        fams, cols, score, hist = backward_selection(treino, sc)
        resultados[sc] = (fams, cols, score)
        passos.extend(hist)
    pd.DataFrame(passos).to_csv(REPORT_DIR / "selecao_validacao.csv", index=False)
    escolhido = max(resultados, key=lambda k: (resultados[k][2]["f1_medio"], resultados[k][2]["recall_medio"]))
    familias, features, score_val = resultados[escolhido]

    congeladas = config.get("sprint4", {}).get("features_congeladas")
    if congeladas and list(congeladas) != features:
        raise ValueError("A selecao reexecutada difere da lista congelada em params.yaml")
    norm_cfg = config.get("sprint4", {}).get("normalizacao")
    if norm_cfg and norm_cfg != escolhido:
        raise ValueError("A normalizacao escolhida difere da congelada em params.yaml")

    # 3) Retreino dos tres baselines no mesmo split, com o transformer congelado
    X_train, y_train = treino[features], treino[ALVO].to_numpy()
    X_test, y_test = teste[features], teste[ALVO].to_numpy()
    dummy = make_pipeline(features, DummyClassifier(strategy="most_frequent"), escolhido).fit(X_train, y_train)
    naive = make_pipeline(features, GaussianNB(), escolhido).fit(X_train, y_train)
    persist = s3.pipeline_persistencia().fit(treino[["estado_atual"]], y_train)

    transformer = naive.named_steps["column_transformer"]
    joblib.dump(transformer, MODEL_DIR / "column_transformer_s4.joblib")
    joblib.dump(naive, MODEL_DIR / "naive_bayes_s4.joblib")
    (MODEL_DIR / "features_s4.json").write_text(json.dumps(
        {"normalizacao": escolhido, "familias": familias, "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    modelos = [
        ("DummyClassifier", "dummy", dummy.predict(X_test)),
        ("Persistencia temporal", "persistencia", persist.predict(teste[["estado_atual"]])),
        ("Gaussian Naive Bayes", "naive_bayes", naive.predict(X_test)),
    ]
    resumo, por_classe, erros = [], [], []
    s3.REPORT_DIR = REPORT_DIR  # graficos da Sprint 3 gravados na pasta da Sprint 4
    for nome, stem, pred in modelos:
        gerais, classes, matriz, err = s3.avaliar(nome, y_test, pred, teste["timestamp"])
        resumo.append(gerais)
        por_classe.extend(classes)
        erros.append(err)
        s3.salvar_grafico(matriz, f"{nome} (S4)", f"{stem}_s4")
    resumo_df = pd.DataFrame(resumo)
    resumo_df.to_csv(REPORT_DIR / "comparativo_baselines_s4.csv", index=False)
    pd.DataFrame(por_classe).to_csv(REPORT_DIR / "metricas_por_classe_s4.csv", index=False)
    erros_df = pd.concat(erros)
    erros_df.to_csv(REPORT_DIR / "erros_por_horario_s4.csv", index=False)

    s3_df = pd.read_csv(S3_RESULTS).set_index("modelo")
    lift = resumo_df.set_index("modelo")[["recall_classe_1", "f1_classe_1", "fn", "fp"]].join(
        s3_df[["recall_classe_1", "f1_classe_1", "fn", "fp"]], lsuffix="_s4", rsuffix="_s3")
    lift["delta_recall"] = lift["recall_classe_1_s4"] - lift["recall_classe_1_s3"]
    lift["delta_f1"] = lift["f1_classe_1_s4"] - lift["f1_classe_1_s3"]
    lift["delta_fn"] = lift["fn_s4"] - lift["fn_s3"]
    lift = lift[["recall_classe_1_s3", "recall_classe_1_s4", "f1_classe_1_s3", "f1_classe_1_s4",
                 "delta_recall", "delta_f1", "fn_s3", "fn_s4", "delta_fn", "fp_s3", "fp_s4"]].round(4)
    lift.to_csv(REPORT_DIR / "lift_s3_s4.csv")

    fn_hora = (erros_df[erros_df["erro"] == "FN"]
               .assign(hora=lambda d: d["timestamp"].dt.hour)
               .groupby(["modelo", "hora"]).size().unstack(0).reindex(range(24), fill_value=0).fillna(0))
    fn_hora.astype(int).to_csv(REPORT_DIR / "fn_teste_por_hora_s4.csv")

    resultado = {
        "split_corte": str(CUTOFF),
        "horizonte_horas": HORIZON_H,
        "n_treino_csv": n_csv,
        "n_treino_utilizado": len(treino),
        "n_teste": len(teste),
        "janelas_validacao": FOLDS,
        "teste_vazamento_instantes": checados,
        "features_novas_criadas": NEW_FEATURES,
        "validacao_s2_por_normalizacao": s2_ref,
        "selecao_por_normalizacao": {k: {"familias": v[0], "n_features": len(v[1]), **v[2]}
                                     for k, v in resultados.items()},
        "normalizacao_escolhida": escolhido,
        "features_congeladas": features,
        "n_features": len(features),
        "random_state_referencia": s3.SEED,
        "resultados_teste": resumo,
        "lift_s3_s4": lift.reset_index().to_dict(orient="records"),
        "versoes": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    (REPORT_DIR / "resultado_execucao.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== SPRINT 4 — features novas, pipeline congelado e lift ===")
    print(f"Treino utilizado: {len(treino)} | teste: {len(teste)} | features: {len(features)} ({escolhido})")
    print("Validacao interna (media de 3 janelas):")
    print(f"  S2 (39 features, standard): recall {s2_ref['standard']['recall_medio']:.3f} | F1 {s2_ref['standard']['f1_medio']:.3f}")
    print(f"  S4 ({len(features)} features, {escolhido}): recall {score_val['recall_medio']:.3f} | F1 {score_val['f1_medio']:.3f}")
    print("\nLift no teste (classe 1):")
    print(lift.to_string())
    return resultado


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError, AssertionError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
