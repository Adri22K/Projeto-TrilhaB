"""Indicadores do dashboard, separados por protocolo de avaliação.

Nunca se escolhe o modelo oficial por uma ordenação no conjunto de teste.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def carregar_resultado_oficial() -> dict:
    caminho = ROOT / "reports/sprint5/resultado_execucao.json"
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.is_file() else {}


def carregar_comparativo_oficial() -> pd.DataFrame | None:
    pasta = ROOT / "reports/sprint5"
    caminho = pasta / "comparativo_teste.csv"
    if not caminho.is_file():
        return None
    quadro = pd.read_csv(caminho)
    quadro = quadro[quadro["limiar"].str.contains("valida", na=False)].copy()
    # Execuções novas já gravam as métricas; a reprodução independente só é
    # aceita se posterior à tabela e compatível com os ajustes congelados.
    metricas = ["roc_auc", "average_precision"]
    if all(c in quadro for c in metricas):
        return quadro
    adicional = pasta / "discriminacao_teste_reproduzida.csv"
    if adicional.is_file() and adicional.stat().st_mtime >= caminho.stat().st_mtime:
        quadro["limiar_validacao"] = quadro["limiar"].str.extract(r"^(\d+\.\d+)")[0].astype(float)
        chaves = ["modelo", "hiperparametros", "limiar_validacao"]
        extra = pd.read_csv(adicional)
        if set(chaves + metricas).issubset(extra.columns) and not extra.duplicated(chaves).any():
            quadro = quadro.drop(columns=metricas, errors="ignore").merge(
                extra[chaves + metricas], on=chaves, how="left", validate="one_to_one")
    return quadro


def carregar_resumo_sprint5() -> pd.DataFrame | None:
    """Uma linha por modelo, com limiar validado quando aplicável."""
    candidatos = carregar_comparativo_oficial()
    if candidatos is None:
        return None
    bruto = pd.read_csv(ROOT / "reports/sprint5/comparativo_teste.csv")
    baselines = bruto[bruto["modelo"].isin(["Dummy", "Persistencia"])]
    return pd.concat([baselines, candidatos], ignore_index=True)


def comparacao_exploratoria_com_nb(comparativo: pd.DataFrame | None) -> pd.DataFrame | None:
    """Mantém as árvores exploratórias e identifica a origem distinta do NB."""
    if comparativo is None:
        return None
    quadro = comparativo.copy()
    quadro["protocolo"] = "Exploratório (árvores e baselines)"
    resumo = carregar_resumo_sprint5()
    if resumo is not None:
        nb = resumo.loc[resumo["modelo"] == "Naive Bayes"].copy()
        nb["limiar"] = nb["limiar"].str.extract(r"^(\d+\.\d+)")[0].astype(float)
        nb["modelo"] = "Naive Bayes (Sprint 5)"
        nb["categoria"] = "Naive Bayes"
        nb["protocolo"] = "Sprint 5 (limiar validado)"
        quadro = pd.concat([quadro, nb.reindex(columns=quadro.columns)], ignore_index=True)
    return quadro


def selecionar_linha_calibrada(comparativo: pd.DataFrame | None, nome: str) -> pd.Series | None:
    if comparativo is None or "modelo" not in comparativo.columns:
        return None
    candidatas = comparativo.loc[
        comparativo["modelo"].astype(str).str.startswith(f"{nome} (Limiar F2", na=False)
    ]
    return candidatas.iloc[0] if not candidatas.empty else None


def linha_persistencia(comparativo: pd.DataFrame | None) -> pd.Series | None:
    if comparativo is None or "modelo" not in comparativo.columns:
        return None
    linhas = comparativo.loc[comparativo["modelo"] == "Persistencia Temporal"]
    return linhas.iloc[0] if not linhas.empty else None


def destaques_exploratorios(comparativo: pd.DataFrame | None) -> dict:
    linhas = {nome: selecionar_linha_calibrada(comparativo, nome)
              for nome in ("Decision Tree", "Random Forest", "XGBoost")}
    validas = {nome: linha for nome, linha in linhas.items() if linha is not None}
    if not validas:
        return {"linhas": linhas, "menor_fn": None, "empates_fn": [], "maior_auc": None, "maior_f2": None}
    menor_fn = min(int(linha["fn"]) for linha in validas.values())
    empates = [nome for nome, linha in validas.items() if int(linha["fn"]) == menor_fn]
    return {
        "linhas": linhas,
        "menor_fn": empates[0],
        "empates_fn": empates,
        "maior_auc": max(validas, key=lambda nome: float(validas[nome]["roc_auc"])),
        "maior_f2": max(validas, key=lambda nome: float(validas[nome]["f2_c1"])),
    }
