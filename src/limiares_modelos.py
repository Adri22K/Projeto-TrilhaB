"""Limiar de decisao dos modelos exploratorios: sempre originado da validacao temporal.

O arquivo de metadados e gravado junto com os .joblib. O CSV legado so e
consultado quando nao ha metadados; jamais inferimos o limiar do teste.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METADADOS = ROOT / "models/limiares_modelos_arvore.json"
COMPARATIVO_LEGADO = ROOT / "reports/modelos_decision_tree_random_forest/comparativo_completo_decision_tree_vs_random_forest.csv"


def carregar_limiares() -> dict[str, float]:
    if METADADOS.is_file():
        data = json.loads(METADADOS.read_text(encoding="utf-8"))
        candidatos = data["limiares_f2"]
        for valor in candidatos.values():
            if not 0 < float(valor) <= 1:
                raise ValueError("Limiar F2 invalido nos metadados de treinamento")
        return {nome: float(valor) for nome, valor in candidatos.items()}

    if COMPARATIVO_LEGADO.is_file():
        df = pd.read_csv(COMPARATIVO_LEGADO)
        return {
            nome: float(df.loc[mask, "limiar"].iloc[0])
            for nome in ("Decision Tree", "Random Forest", "XGBoost")
            if (mask := df["modelo"].str.startswith(nome + " (Limiar F2", na=False)).any()
        }
    return {}


def limiar_para_modelo(nome: str) -> float:
    if nome == "Naive Bayes":
        # O GaussianNB da S4 é reproduzido na S5; usar seu limiar OOF da S5.
        caminho = ROOT / "reports/sprint5/comparativo_teste.csv"
        if caminho.is_file():
            df = pd.read_csv(caminho)
            linha = df[(df["modelo"] == nome) & df["limiar"].str.contains("valida", na=False)]
            if len(linha) == 1:
                valor = float(linha.iloc[0]["limiar"].split()[0])
                if 0 < valor <= 1:
                    return valor
        raise ValueError("Limiar validado do Naive Bayes indisponível")
    if nome == "Modelo oficial (Sprint 5)":
        caminho = ROOT / "models/modelo_final_s5.json"
        if caminho.is_file():
            valor = json.loads(caminho.read_text(encoding="utf-8")).get("limiar")
            return float(valor) if valor is not None else 0.50
    if nome == "Persistência Temporal":
        return 0.50
    limiares = carregar_limiares()
    # Ausencia de calibracao: limiar neutro e explicito, sem inventar otimizacao.
    return limiares.get(nome, 0.50)
