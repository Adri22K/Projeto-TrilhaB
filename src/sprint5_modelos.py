"""Sprint 5 / Trilha B: comparativo no pipeline final, limiar na validacao, selecao e model card.

Execute na raiz do repositorio:
    python src/sprint5_modelos.py

Contrato:
    - split da Sprint 2 (corte 14/04/2026) e purga de 24 h, iguais as Sprints 3 e 4;
    - ColumnTransformer CONGELADO da Sprint 4 (25 features, mediana + Yeo-Johnson) para todos os modelos;
    - hiperparametros e limiar escolhidos SO na validacao temporal dentro do treino (3 janelas da Sprint 4);
    - regra de selecao registrada antes do teste; o teste e medido uma unica vez.

Saidas:
    reports/sprint5/*                      (busca, curva de limiar, caderno, comparativo, erros)
    models/modelo_final_s5.joblib          (Pipeline inteiro + limiar, FixedThresholdClassifier)
    models/modelo_final_s5.json            (metadados do modelo final)
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
import sklearn
import yaml
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sprint3_baselines as s3  # noqa: E402
import sprint4_features as s4  # noqa: E402

TRAIN_S4 = ROOT / "data/processed/treino_features_s4.csv"
TEST_S4 = ROOT / "data/processed/teste_features_s4.csv"
INTERIM_FILE = ROOT / "data/interim/integrado_limpo.csv"
LIFT_S4 = ROOT / "reports/sprint4/comparativo_baselines_s4.csv"
CONFIG_FILE = ROOT / "config/params.yaml"
REPORT_DIR = ROOT / "reports/sprint5"
MODEL_DIR = ROOT / "models"

ALVO, CUTOFF, HORIZON_H, SEED = s4.ALVO, s4.CUTOFF, s4.HORIZON_H, 42
BETA = 2  # F2: recall pesa 2x a precisao, porque o FN e o erro mais caro (RFC)
GRADE_LIMIAR = np.round(np.arange(0.05, 0.951, 0.05), 2)
TOLERANCIA_EMPATE = 0.005  # empate em F2 na validacao -> modelo mais simples

# Ordem = complexidade crescente (usada no desempate)
CANDIDATOS = {
    "Naive Bayes": {
        "familia": "baseline S4",
        "grade": {"padrao": GaussianNB()},
    },
    "Regressao Logistica": {
        "familia": "novo",
        "grade": {f"C={c}": LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
                  for c in (0.01, 0.1, 1.0)},
    },
    "Decision Tree": {
        "familia": "novo",
        "grade": {f"max_depth={d}": DecisionTreeClassifier(
            max_depth=d, min_samples_split=20, min_samples_leaf=15,
            class_weight="balanced", random_state=SEED) for d in (4, 6, 8)},
    },
    "Random Forest": {
        "familia": "novo",
        "grade": {
            f"max_depth={d}, min_samples_leaf={leaf}": RandomForestClassifier(
                n_estimators=200, max_depth=d, min_samples_leaf=leaf, min_samples_split=10,
                class_weight="balanced", random_state=SEED, n_jobs=-1)
            for d, leaf in ((8, 20), (12, 8), (16, 20))
        },
    },
    "HistGradientBoosting": {
        "familia": "novo",
        "grade": {f"learning_rate={lr}, max_depth={d}": HistGradientBoostingClassifier(
            learning_rate=lr, max_depth=d, max_iter=300, class_weight="balanced", random_state=SEED)
            for lr in (0.05, 0.1) for d in (3, 4)},
    },
}


# --------------------------------------------------------------------------- utilidades
def janelas(treino: pd.DataFrame):
    for inicio, fim in s4.FOLDS:
        a = pd.Timestamp(inicio)
        b = pd.Timestamp(fim) if fim else CUTOFF
        fit = treino[treino["timestamp"] + pd.Timedelta(hours=HORIZON_H) < a]
        val = treino[(treino["timestamp"] >= a) & (treino["timestamp"] + pd.Timedelta(hours=HORIZON_H) < b)]
        yield inicio, fit, val


def metricas(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {"precisao_c1": float(p[1]), "recall_c1": float(r[1]), "f1_c1": float(f[1]),
            "f2_c1": float(fbeta_score(y, pred, beta=BETA, zero_division=0)),
            "recall_c0": float(r[0]), "acuracia": float((tp + tn) / len(y)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def oof(estimador, treino, features):
    """Probabilidades fora da amostra nas 3 janelas de validacao (cada uma treinada so no passado)."""
    ys, ps, aps = [], [], []
    for _, fit, val in janelas(treino):
        modelo = s4.make_pipeline(features, clone(estimador), "yeo_johnson").fit(fit[features], fit[ALVO])
        prob = modelo.predict_proba(val[features])[:, 1]
        ys.append(val[ALVO].to_numpy())
        ps.append(prob)
        aps.append(average_precision_score(val[ALVO], prob))
    oof.janelas = (ys, ps)
    return np.concatenate(ys), np.concatenate(ps), aps


# --------------------------------------------------------------------------- principal
def main() -> dict:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    features = list(config["sprint4"]["features_congeladas"])
    if config["sprint4"]["normalizacao"] != "yeo_johnson":
        raise ValueError("Normalizacao diferente da congelada na Sprint 4")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    treino = s3.ler_features(TRAIN_S4)
    teste = s3.ler_features(TEST_S4)
    if not (treino["timestamp"] < CUTOFF).all() or not (teste["timestamp"] >= CUTOFF).all():
        raise ValueError("Split diferente do da Sprint 2")
    n_csv = len(treino)
    treino = treino.loc[treino["timestamp"] + pd.Timedelta(hours=HORIZON_H) < CUTOFF].copy()
    vazamento = s4.leakage_check(
        pd.read_csv(INTERIM_FILE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True),
        config)

    # 1) Busca de hiperparametros (AP medio nas 3 janelas; nao depende de limiar)
    busca, escolhidos = [], {}
    for nome, spec in CANDIDATOS.items():
        melhor = None
        for rotulo, est in spec["grade"].items():
            y_val, p_val, aps = oof(est, treino, features)
            linha = {"modelo": nome, "hiperparametros": rotulo, "ap_medio": float(np.mean(aps)),
                     **{f"ap_janela_{i + 1}": float(a) for i, a in enumerate(aps)}}
            busca.append(linha)
            if melhor is None or linha["ap_medio"] > melhor[0]["ap_medio"]:
                melhor = (linha, est, y_val, p_val, oof.janelas)
        escolhidos[nome] = melhor
    pd.DataFrame(busca).round(4).to_csv(REPORT_DIR / "busca_hiperparametros.csv", index=False)

    # 2) Curva de limiar na validacao e limiar congelado (max F2)
    curva, limiares = [], {}
    for nome, (_, _, y_val, p_val, _) in escolhidos.items():
        for t in GRADE_LIMIAR:
            curva.append({"modelo": nome, "limiar": float(t), **metricas(y_val, (p_val >= t).astype(int))})
        sub = [c for c in curva if c["modelo"] == nome]
        melhor = max(sub, key=lambda c: (round(c["f2_c1"], 6), c["recall_c1"]))
        limiares[nome] = {"limiar": melhor["limiar"], "val": melhor,
                          "val_padrao": metricas(y_val, (p_val >= 0.5).astype(int))}
    por_janela = []
    for nome, (_, _, _, _, (ys, ps)) in escolhidos.items():
        t = limiares[nome]["limiar"]
        for (inicio, _, _), y, p in zip(janelas(treino), ys, ps):
            m = metricas(y, (p >= t).astype(int))
            por_janela.append({"modelo": nome, "janela_inicio": inicio, "limiar": t,
                               "recall_c1": m["recall_c1"], "f2_c1": m["f2_c1"], "fn": m["fn"], "fp": m["fp"]})
    pd.DataFrame(por_janela).round(4).to_csv(REPORT_DIR / "validacao_por_janela.csv", index=False)
    curva_df = pd.DataFrame(curva)
    curva_df.round(4).to_csv(REPORT_DIR / "curva_limiar_validacao.csv", index=False)

    # persistencia na validacao (sem limiar)
    y_pers, p_pers = [], []
    for _, _, val in janelas(treino):
        y_pers.append(val[ALVO].to_numpy())
        p_pers.append(val["estado_atual"].astype(int).to_numpy())
    val_persist = metricas(np.concatenate(y_pers), np.concatenate(p_pers))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for nome in escolhidos:
        sub = curva_df[curva_df["modelo"] == nome]
        ax.plot(sub["limiar"], sub["f2_c1"], marker=".", label=f"{nome} (t={limiares[nome]['limiar']:.2f})")
    ax.axhline(val_persist["f2_c1"], color="k", ls="--", lw=1, label="Persistência")
    ax.set_xlabel("Limiar de probabilidade da classe 1")
    ax.set_ylabel("F2 da classe 1 (validação)")
    ax.set_title("Curva de limiar na validação temporal (3 janelas dentro do treino)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "curva_limiar_validacao.png", dpi=150)
    plt.close(fig)

    # 3) Regra de selecao (registrada antes do teste): maior F2 na validacao; empate -> mais simples
    ordem = list(CANDIDATOS)
    f2_val = {n: limiares[n]["val"]["f2_c1"] for n in ordem}
    topo = max(f2_val.values())
    selecionado = next(n for n in ordem if f2_val[n] >= topo - TOLERANCIA_EMPATE)

    # 4) Treino final no treino completo e teste UMA vez (limiar padrao e limiar congelado)
    X_tr, y_tr = treino[features], treino[ALVO].to_numpy()
    X_te, y_te = teste[features], teste[ALVO].to_numpy()
    comparativo, por_classe, previsoes = [], [], {}

    def registrar(nome, familia, hp, limiar_txt, pred, prob=None, val_info=None):
        m = metricas(y_te, pred)
        comparativo.append({"modelo": nome, "familia": familia, "hiperparametros": hp,
                            "limiar": limiar_txt, **{k: m[k] for k in m},
                            "recall_val": val_info["recall_c1"] if val_info else None,
                            "f2_val": val_info["f2_c1"] if val_info else None})
        p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=[0, 1], zero_division=0)
        for c in (0, 1):
            por_classe.append({"modelo": nome, "limiar": limiar_txt, "classe": c, "precisao": p[c],
                               "recall": r[c], "f1": f[c], "suporte": int(s[c])})
        previsoes[(nome, limiar_txt)] = pred
        return m

    dummy = s4.make_pipeline(features, DummyClassifier(strategy="most_frequent"), "yeo_johnson").fit(X_tr, y_tr)
    registrar("Dummy", "baseline S4", "most_frequent", "padrao (predict)", dummy.predict(X_te))
    persist = s3.pipeline_persistencia().fit(treino[["estado_atual"]], y_tr)
    registrar("Persistencia", "baseline S4", "regra estado(t) -> t+24h", "—",
              persist.predict(teste[["estado_atual"]]), val_info=val_persist)

    finais = {}
    for nome in ordem:
        linha, est, _, _, _ = escolhidos[nome]
        pipe = s4.make_pipeline(features, clone(est), "yeo_johnson").fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        t = limiares[nome]["limiar"]
        fam = CANDIDATOS[nome]["familia"]
        registrar(nome, fam, linha["hiperparametros"], "padrao (predict)", pipe.predict(X_te),
                  val_info=limiares[nome]["val_padrao"])
        registrar(nome, fam, linha["hiperparametros"], f"{t:.2f} (validação)", (prob >= t).astype(int),
                  val_info=limiares[nome]["val"])
        finais[nome] = (pipe, prob)

    comp = pd.DataFrame(comparativo)
    comp.round(4).to_csv(REPORT_DIR / "comparativo_teste.csv", index=False)
    pd.DataFrame(por_classe).round(4).to_csv(REPORT_DIR / "metricas_por_classe.csv", index=False)

    # conferencia: baselines S4 reproduzidos
    s4_ref = pd.read_csv(LIFT_S4).set_index("modelo")
    nb_rep = comp[(comp.modelo == "Naive Bayes") & (comp.limiar == "padrao (predict)")].iloc[0]
    if abs(nb_rep["recall_c1"] - s4_ref.loc["Gaussian Naive Bayes", "recall_classe_1"]) > 1e-9:
        raise ValueError("Naive Bayes nao reproduz a Sprint 4")
    pers_rep = comp[comp.modelo == "Persistencia"].iloc[0]
    if int(pers_rep["fn"]) != int(s4_ref.loc["Persistencia temporal", "fn"]):
        raise ValueError("Persistencia nao reproduz a Sprint 4")

    # 5) Teste do modelo selecionado contra a persistencia (regra do template)
    sel_row = comp[(comp.modelo == selecionado) & (comp.limiar != "padrao (predict)")].iloc[0]
    persistencia_vence = (pers_rep["f2_c1"] > sel_row["f2_c1"]) and (pers_rep["fn"] <= sel_row["fn"])
    modelo_final = "Persistencia" if persistencia_vence else selecionado

    # matrizes do modelo final e da persistencia
    s3.REPORT_DIR = REPORT_DIR
    t_final = limiares[selecionado]["limiar"]
    for nome, chave, stem in [(selecionado, (selecionado, f"{t_final:.2f} (validação)"), "final_limiar"),
                              (selecionado, (selecionado, "padrao (predict)"), "final_padrao"),
                              ("Persistencia", ("Persistencia", "—"), "persistencia")]:
        s3.salvar_grafico(confusion_matrix(y_te, previsoes[chave], labels=[0, 1]), f"{nome} ({chave[1]})", stem)

    # 6) Artefato final: Pipeline inteiro + limiar
    pipe_sel, prob_sel = finais[selecionado]
    final = FixedThresholdClassifier(
        s4.make_pipeline(features, clone(escolhidos[selecionado][1]), "yeo_johnson"),
        threshold=t_final, response_method="predict_proba").fit(X_tr, y_tr)
    if not np.array_equal(final.predict(X_te), (prob_sel >= t_final).astype(int)):
        raise ValueError("Artefato final nao reproduz as previsoes avaliadas")
    joblib.dump(final, MODEL_DIR / "modelo_final_s5.joblib")
    meta = {"modelo": selecionado, "hiperparametros": escolhidos[selecionado][0]["hiperparametros"],
            "limiar": t_final, "origem_limiar": "max F2 na validacao temporal (3 janelas dentro do treino)",
            "features": features, "normalizacao": "yeo_johnson", "random_state": SEED,
            "corte_split": str(CUTOFF), "sklearn": sklearn.__version__}
    (MODEL_DIR / "modelo_final_s5.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7) Analise de erros do modelo final (episodios concretos)
    clean = pd.read_csv(INTERIM_FILE, parse_dates=["timestamp"]).set_index("timestamp")
    o3_8h = clean["ozone"].rolling(8, min_periods=8).mean()
    pm25_24h = clean["pm2_5"].rolling(24, min_periods=24).mean()
    pm10_24h = clean["pm10"].rolling(24, min_periods=24).mean()
    pred_final = (prob_sel >= t_final).astype(int)
    err = teste[["timestamp", ALVO, "estado_atual", "ozone_media_8h", "temperature_max_24h"]].copy()
    err["prob"] = prob_sel
    err["previsto"] = pred_final
    err["tipo"] = np.select([(err[ALVO] == 1) & (err.previsto == 0), (err[ALVO] == 0) & (err.previsto == 1)],
                            ["FN", "FP"], "acerto")
    alvo_ts = err["timestamp"] + pd.Timedelta(hours=HORIZON_H)
    err["hora_alvo"] = alvo_ts
    err["o3_8h_no_alvo"] = o3_8h.reindex(alvo_ts).to_numpy()
    err["pm2_5_24h_no_alvo"] = pm25_24h.reindex(alvo_ts).to_numpy()
    err["pm10_24h_no_alvo"] = pm10_24h.reindex(alvo_ts).to_numpy()
    err.to_csv(REPORT_DIR / "erros_modelo_final_teste.csv", index=False)

    e = err[err.tipo != "acerto"].copy()
    e["bloco"] = ((e["tipo"] != e["tipo"].shift()) |
                  (e["timestamp"].diff() != pd.Timedelta(hours=1))).cumsum()
    episodios = e.groupby("bloco").agg(
        tipo=("tipo", "first"), inicio_alvo=("hora_alvo", "min"), fim_alvo=("hora_alvo", "max"),
        horas=("tipo", "size"), prob_media=("prob", "mean"),
        o3_8h_max_no_alvo=("o3_8h_no_alvo", "max"), pm2_5_24h_max_no_alvo=("pm2_5_24h_no_alvo", "max"),
        o3_8h_em_t=("ozone_media_8h", "mean"), estado_em_t=("estado_atual", "mean"),
    ).sort_values(["tipo", "horas"], ascending=[True, False])
    episodios = episodios.round({c: 3 for c in episodios.columns if episodios[c].dtype.kind == "f"})
    episodios.to_csv(REPORT_DIR / "episodios_erro_modelo_final.csv", index=False)

    fn_hora = err[err.tipo == "FN"].groupby(err["hora_alvo"].dt.hour).size().reindex(range(24), fill_value=0)
    fp_hora = err[err.tipo == "FP"].groupby(err["hora_alvo"].dt.hour).size().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(fn_hora.index - 0.2, fn_hora.values, width=0.4, color="tab:red", label=f"FN ({fn_hora.sum()})")
    ax.bar(fp_hora.index + 0.2, fp_hora.values, width=0.4, color="tab:gray", label=f"FP ({fp_hora.sum()})")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora prevista (local)")
    ax.set_ylabel("Horas")
    ax.set_title(f"Erros do modelo final no teste por hora — {selecionado}, limiar {t_final:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "erros_modelo_final_por_hora.png", dpi=150)
    plt.close(fig)

    # 8) Caderno de experimentos
    caderno = []
    for i, row in enumerate(comp.itertuples(), start=1):
        caderno.append({"ID": f"E{i:02d}", "modelo": row.modelo, "hiperparametros": row.hiperparametros,
                        "features": "S4 (25, Yeo-Johnson)" if row.modelo != "Persistencia" else "estado_atual",
                        "limiar": row.limiar, "recall_val": row.recall_val, "f2_val": row.f2_val,
                        "recall_teste": row.recall_c1, "precisao_teste": row.precisao_c1,
                        "f1_teste": row.f1_c1, "f2_teste": row.f2_c1, "fn_teste": row.fn, "fp_teste": row.fp,
                        "random_state": SEED if row.modelo not in ("Dummy", "Persistencia", "Naive Bayes",
                                                                   "Regressao Logistica") else "—"})
    pd.DataFrame(caderno).round(4).to_csv(REPORT_DIR / "caderno_experimentos.csv", index=False)

    resultado = {
        "corte": str(CUTOFF), "n_treino_csv": n_csv, "n_treino_utilizado": len(treino), "n_teste": len(teste),
        "positivos_teste": int(y_te.sum()), "n_features": len(features), "features": features,
        "teste_vazamento_instantes": vazamento, "beta": BETA,
        "hiperparametros_escolhidos": {n: escolhidos[n][0]["hiperparametros"] for n in ordem},
        "limiares_validacao": {n: limiares[n]["limiar"] for n in ordem},
        "f2_validacao": f2_val, "persistencia_validacao": val_persist,
        "selecionado_na_validacao": selecionado, "persistencia_vence_no_teste": bool(persistencia_vence),
        "modelo_final": modelo_final, "limiar_final": t_final,
        "versoes": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (REPORT_DIR / "resultado_execucao.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== SPRINT 5 — comparativo, limiar e selecao ===")
    print(f"Treino {len(treino)} | teste {len(teste)} ({int(y_te.sum())} positivas) | {len(features)} features S4")
    print("Validacao (F2 no limiar congelado):", {k: round(v, 3) for k, v in f2_val.items()})
    print(f"Persistencia na validacao: F2 {val_persist['f2_c1']:.3f}")
    print(f"Selecionado na validacao: {selecionado} (limiar {t_final:.2f}) | modelo final: {modelo_final}")
    print(comp[["modelo", "limiar", "recall_c1", "precisao_c1", "f1_c1", "f2_c1", "fn", "fp"]].round(3).to_string())
    return resultado


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError, AssertionError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
