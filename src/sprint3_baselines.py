"""Sprint 3 / Trilha B: Dummy, persistencia temporal e Gaussian Naive Bayes.

Execute na raiz do repositorio Projeto-TrilhaB:
    python src/sprint3_baselines.py

Entradas obrigatorias (produzidas pela Sprint 2):
    data/processed/treino_features.csv
    data/processed/teste_features.csv
    data/interim/integrado_limpo.csv
    config/params.yaml

Apenas grava novos relatorios em reports/sprint3. Nao modifica dados anteriores.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data/processed/treino_features.csv"
TEST_FILE = ROOT / "data/processed/teste_features.csv"
INTERIM_FILE = ROOT / "data/interim/integrado_limpo.csv"
CONFIG_FILE = ROOT / "config/params.yaml"
REPORT_DIR = ROOT / "reports/sprint3"
ALVO = "qualidade_inadequada_24h"
CUTOFF = pd.Timestamp("2026-04-14 00:00:00")
HORIZON_H = 24
SEED = 42  # Registrada para reproducibilidade; os baselines usados sao deterministicos.
POLLUTANTS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"
]
WEATHER = ["temperature_2m", "precipitation", "wind_speed_10m"]
CALENDAR = ["hora", "dia_semana", "mes"]
FEATURES = (
    POLLUTANTS + WEATHER + CALENDAR
    + [f"{name}_{suffix}" for name in POLLUTANTS + WEATHER
       for suffix in ("lag_1h", "lag_24h", "media_24h")]
)


class PersistenciaClassifier(ClassifierMixin, BaseEstimator):
    """Baseline de persistencia: a classe do ar observada agora vale para t+24 h.

    A coluna 'estado_atual' e calculada SEM consultar o alvo futuro,
    aplicando os limiares da Sprint 2 aos dados historicos ate o instante t.
    """

    def fit(self, X, y):
        x = np.asarray(X)
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("Persistencia espera somente uma coluna: estado_atual")
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != 1 or not np.isin(x[:, 0], [0, 1]).all():
            raise ValueError("estado_atual deve ser um vetor binario (0/1)")
        return x[:, 0].astype(int)

    def predict_proba(self, X):
        y = self.predict(X)
        return np.column_stack((1 - y, y)).astype(float)


def ler_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo ausente: {path}. Execute a Sprint 2 antes.")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = ["timestamp", ALVO] + FEATURES
    absent = sorted(set(required).difference(df.columns))
    if absent:
        raise ValueError(f"Faltam colunas em {path.name}: {absent}")
    if df["timestamp"].isna().any() or df["timestamp"].duplicated().any():
        raise ValueError(f"Timestamps invalidos ou duplicados em {path.name}")
    if df[ALVO].isna().any() or not df[ALVO].isin([0, 1]).all():
        raise ValueError(f"Alvo ausente ou nao binario em {path.name}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[ALVO] = df[ALVO].astype(int)
    return df


def preparar_estado_atual(config: dict, all_timestamps: pd.Series) -> pd.DataFrame:
    """Gera classe no instante t com as mesmas regras de alvo da Sprint 2.

    rolling(24/8) avalia somente amostras ate t; NA de historico
    incompleto e excluido da persistencia, nunca interpretado como 0.
    """
    if not INTERIM_FILE.is_file():
        raise FileNotFoundError(f"Arquivo necessario para persistencia: {INTERIM_FILE}")
    air = pd.read_csv(INTERIM_FILE, parse_dates=["timestamp"])
    air = air.sort_values("timestamp").reset_index(drop=True)
    if air["timestamp"].isna().any() or air["timestamp"].duplicated().any():
        raise ValueError("Dados horarios intermediarios com timestamps invalidos ou duplicados")
    if not air["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Ha lacunas na base horaria: rolling(24/8) exige horas consecutivas")
    cfg = config["sprint2"]["target"]
    lim = cfg["thresholds"]
    needed = POLLUTANTS
    for col in needed:
        if col not in air.columns:
            raise ValueError(f"A base interim nao contem {col}")

    # Estado no instante t. Nao usar y_t nem y_{t+24} para construir a persistencia.
    medias = {
        "pm10": air.pm10.rolling(24, min_periods=24).mean(),
        "pm2_5": air.pm2_5.rolling(24, min_periods=24).mean(),
        "so2": air.sulphur_dioxide.rolling(24, min_periods=24).mean(),
        "ozone": air.ozone.rolling(8, min_periods=8).mean(),
        "co": air.carbon_monoxide.rolling(8, min_periods=8).mean()
             * cfg["co_ug_m3_to_ppm_factor"],
    }
    condicoes = [
        medias["pm10"] > lim["pm10_24h_ug_m3"],
        medias["pm2_5"] > lim["pm2_5_24h_ug_m3"],
        medias["so2"] > lim["sulphur_dioxide_24h_ug_m3"],
        medias["ozone"] > lim["ozone_8h_ug_m3"],
        air.nitrogen_dioxide > lim["nitrogen_dioxide_1h_ug_m3"],
        medias["co"] > lim["carbon_monoxide_8h_ppm"],
    ]
    estado = np.logical_or.reduce(condicoes).astype(float)
    # Os 23 primeiros registros nao possuem historico de 24 h.
    estado[:23] = np.nan
    obs = pd.DataFrame({"timestamp": air["timestamp"], "estado_atual": estado})
    aligned = pd.DataFrame({"timestamp": all_timestamps}).merge(
        obs, on="timestamp", how="left", validate="one_to_one"
    )
    if aligned.estado_atual.isna().any():
        miss = aligned.loc[aligned.estado_atual.isna(), "timestamp"].head(3).tolist()
        raise ValueError(f"Estado atual indisponivel para registros: {miss}")
    return aligned


def pipeline_classico(modelo) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("numericas", Pipeline([
                ("imputador", SimpleImputer(strategy="median")),
                ("normalizador", StandardScaler()),
            ]), FEATURES)
        ],
        remainder="drop",
    )
    return Pipeline([("column_transformer", transformer), ("classificador", modelo)])


def pipeline_persistencia() -> Pipeline:
    transformer = ColumnTransformer(
        [("classe_observada", "passthrough", ["estado_atual"])],
        remainder="drop",
    )
    return Pipeline([
        ("column_transformer", transformer),
        ("classificador", PersistenciaClassifier()),
    ])


def avaliar(nome: str, y_true: np.ndarray, y_pred: np.ndarray, tempos: pd.Series):
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matriz.ravel())
    precision, recall, f1, suporte = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    gerais = {
        "modelo": nome,
        "n_teste": int(len(y_true)),
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "precisao_classe_1": float(precision[1]),
        "recall_classe_1": float(recall[1]),
        "f1_classe_1": float(f1[1]),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    classes = [
        {"modelo": nome, "classe": classe, "precisao": float(precision[classe]),
         "recall": float(recall[classe]), "f1": float(f1[classe]), "suporte": int(suporte[classe])}
        for classe in [0, 1]
    ]
    erros = pd.DataFrame({
        "timestamp": tempos,
        "modelo": nome,
        "real": y_true,
        "previsto": y_pred,
    })
    erros["erro"] = np.select(
        [(erros.real.eq(1) & erros.previsto.eq(0)),
         (erros.real.eq(0) & erros.previsto.eq(1))],
        ["FN", "FP"], default="acerto"
    )
    return gerais, classes, matriz, erros


def salvar_grafico(matriz: np.ndarray, nome: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks([0, 1], ["Adequada (0)", "Inadequada (1)"])
    ax.set_yticks([0, 1], ["Adequada (0)", "Inadequada (1)"])
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"Matriz de confusao — {nome}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matriz[i, j])), ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=.85)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"matriz_{stem}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if int(config["sprint2"]["forecast_horizon_hours"]) != HORIZON_H:
        raise ValueError("Horizonte da configuracao mudou: Sprint 3 exige 24 horas")
    if config["sprint2"]["target"]["name"] != ALVO:
        raise ValueError("O nome da variavel-alvo mudou em relacao a Sprint 2")

    treino = ler_features(TRAIN_FILE)
    teste = ler_features(TEST_FILE)
    if not (treino.timestamp < CUTOFF).all() or not (teste.timestamp >= CUTOFF).all():
        raise ValueError("Split divergente: esperado corte em 2026-04-14 00:00 (Sprint 2)")
    if set(treino.timestamp) & set(teste.timestamp):
        raise ValueError("Vazamento: timestamps presentes simultaneamente em treino e teste")
    n_original = len(treino)
    # Purga de 24 h: y(t) foi calculado a partir da condicao em t+24h.
    # Portanto, os instantes de treino que usariam t+24h dentro do teste
    # sao removidos SOMENTE da modelagem, sem mudar o corte dos CSVs.
    treino = treino.loc[treino.timestamp + pd.Timedelta(hours=HORIZON_H) < CUTOFF].copy()
    n_purgados = n_original - len(treino)
    if not len(treino) or not len(teste):
        raise ValueError("Treino ou teste vazio apos exclusao de fronteira")
    if set(treino[ALVO].unique()) != {0, 1}:
        raise ValueError("Treino deve conter as classes 0 e 1")
    X_train, y_train = treino[FEATURES], treino[ALVO].to_numpy()
    X_test, y_test = teste[FEATURES], teste[ALVO].to_numpy()

    # Mesmas 39 features iniciais da Sprint 2 para os classificadores aprendidos.
    dummy = pipeline_classico(DummyClassifier(strategy="most_frequent"))
    naive = pipeline_classico(GaussianNB())
    dummy.fit(X_train, y_train)
    naive.fit(X_train, y_train)

    # Persistencia usa uma REGRA: condicao observada em t para prever t+24h.
    # Calculada independentemente de y_t, apenas sobre dados observaveis em t.
    todos = pd.concat([treino["timestamp"], teste["timestamp"]], ignore_index=True)
    atual = preparar_estado_atual(config, todos)
    treino_p = pd.DataFrame({"estado_atual": atual.estado_atual.iloc[:len(treino)].to_numpy()})
    teste_p = pd.DataFrame({"estado_atual": atual.estado_atual.iloc[len(treino):].to_numpy()})
    persistencia = pipeline_persistencia()
    persistencia.fit(treino_p, y_train)

    modelos = [
        ("DummyClassifier", "dummy", dummy.predict(X_test)),
        ("Persistencia temporal", "persistencia", persistencia.predict(teste_p)),
        ("Gaussian Naive Bayes", "naive_bayes", naive.predict(X_test)),
    ]
    resumo = []
    class_rows = []
    predicao_df = pd.DataFrame({"timestamp": teste.timestamp, "classe_real": y_test})
    erro_rows = []
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for nome, stem, pred in modelos:
        gerais, classe, matrix, errors = avaliar(nome, y_test, pred, teste.timestamp)
        resumo.append(gerais)
        class_rows.extend(classe)
        predicao_df[f"prev_{stem}"] = pred
        erro_rows.append(errors)
        salvar_grafico(matrix, nome, stem)

    pd.DataFrame(resumo).to_csv(REPORT_DIR / "comparativo_baselines.csv", index=False)
    pd.DataFrame(class_rows).to_csv(REPORT_DIR / "metricas_por_classe.csv", index=False)
    # Texto pronto para colar na secao 3 do diario da Sprint 3.
    lines = [
        "# Resultados de modelagem — Sprint 3 (calculados com CSV reais)", "",
        f"**Corte temporal:** {CUTOFF:%d/%m/%Y %H:%M}; "
        f"**treino original:** {n_original}; **treino apos purga de {HORIZON_H}h:** {len(treino)}; "
        f"**teste:** {len(teste)}.", "",
        "## Comparativo dos tres baselines", "",
        "| Modelo | Acuracia | Precisao classe 1 | Recall classe 1 | F1 classe 1 | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in resumo:
        lines.append(
            f"| {r['modelo']} | {r['acuracia']:.4f} | "
            f"{r['precisao_classe_1']:.4f} | {r['recall_classe_1']:.4f} | "
            f"{r['f1_classe_1']:.4f} | {r['fp']} | {r['fn']} |"
        )
    lines += ["", "## Metricas por classe", "",
        "| Modelo | Classe | Precisao | Recall | F1 | Suporte |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in class_rows:
        lines.append(f"| {r['modelo']} | {r['classe']} | {r['precisao']:.4f} | "
                     f"{r['recall']:.4f} | {r['f1']:.4f} | {r['suporte']} |")
    melhor_recall = max(resumo, key=lambda x: x["recall_classe_1"])
    lines += [
        "", "## Interpretacao dos erros", "",
        f"O maior recall da classe 1 nesta comparacao foi o do modelo **{melhor_recall['modelo']}** "
        f"({melhor_recall['recall_classe_1']:.4f}), com **{melhor_recall['fn']} "
        f"falsos negativos** e **{melhor_recall['fp']} falsos positivos**.",
        "Falso negativo significa classificar como adequada (0) uma hora que estava "
        "inadequada (1), erro de maior custo para o projeto conforme o RFC da Sprint 1.",
        "A acuracia do Dummy, que prediz a classe mais frequente, nao demonstra "
        "capacidade de detectar episodios inadequados.",
        "O Naive Bayes e um baseline probabilistico e supoe independencia "
        "condicional dos atributos; PM2.5/PM10 e certas variaveis meteorologicas "
        "sao correlacionadas, portanto a hipotese e uma simplificacao.",
        "Os tres modelos foram comparados no mesmo teste, sem SMOTE nem "
        "ajuste de limiar (Sprint 5). A analise dos FN pode orientar features "
        "da Sprint 4; nenhuma selecao de modelo final e feita aqui.",
        "", "## Evidencias", "",
        "- `reports/sprint3/comparativo_baselines.csv`",
        "- `reports/sprint3/metricas_por_classe.csv`",
        "- `reports/sprint3/erros_por_horario.csv`",
        "- `reports/sprint3/matriz_dummy.png`",
        "- `reports/sprint3/matriz_persistencia.png`",
        "- `reports/sprint3/matriz_naive_bayes.png`",
        "- `src/sprint3_baselines.py` e `notebooks/03_baseline.ipynb`",
    ]
    (REPORT_DIR / "relatorio_modelagem_para_sprint3.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    pd.concat(erro_rows).to_csv(REPORT_DIR / "erros_por_horario.csv", index=False)
    predicao_df.to_csv(REPORT_DIR / "previsoes_teste.csv", index=False)
    metadata = {
        "split_corte": str(CUTOFF),
        "horizonte_horas": HORIZON_H,
        "random_state_referencia": SEED,
        "observacao_seed": "Nao ha sorteio nos tres baselines; random_state nao se aplica aos estimadores escolhidos.",
        "features_modelos_aprendidos": FEATURES,
        "n_features": len(FEATURES),
        "n_treino_csv": n_original,
        "n_treino_utilizado": len(treino),
        "n_treino_purgado_fronteira_24h": n_purgados,
        "n_teste": len(teste),
        "taxa_positivos_treino_utilizado": float(y_train.mean()),
        "taxa_positivos_teste": float(y_test.mean()),
        "versoes": {"pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "politica_vazamento": "fit so no treino; purga de 24h na fronteira; persistencia observa somente t e passado",
        "resultados": resumo,
        "metricas_por_classe": class_rows,
        "nota": "As tres abordagens foram avaliadas nos MESMOS timestamps do teste; persistencia e regra, nao modelo treinado.",
    }
    (REPORT_DIR / "resultado_execucao.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== SPRINT 3 - TRES BASELINES (RESULTADOS REAIS) ===")
    print(f"Treino CSV: {n_original} | treino utilizado: {len(treino)} (purga: {n_purgados}) | teste: {len(teste)}")
    print(pd.DataFrame(resumo).round(4).to_string(index=False))
    print(f"\nArquivos salvos em {REPORT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
