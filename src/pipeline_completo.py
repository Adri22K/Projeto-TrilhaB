"""Pipeline Completo de Ponta a Ponta:
1. Coleta de dados abertos das APIs Open-Meteo (Air Quality e Historical Weather)
2. Inspecao, tratamento, split temporal e engenharia de atributos
3. Modelagem com Decision Tree, Random Forest e XGBoost no pipeline congelado
4. Comparacao das metricas contra as baselines (Persistencia Temporal e DummyClassifier)
5. Disponibilizacao dos artefatos e dados para o Dashboard interativo

Como executar:
    python src/pipeline_completo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import coleta_integracao
import prepara_sprint2
import sprint3_baselines
import sprint4_features
import sprint5_modelos
import modelos_arvore_random_forest


def executar_pipeline_completo(forcar_recoleta: bool = False) -> dict:
    inicio = time.time()
    print("=" * 80)
    print("   PIPELINE COMPLETO DE DADOS ABERTOS, MODELAGEM E COMPARACAO")
    print("   Trilha B — Qualidade do Ar Inadequada em Sao Paulo (24h a frente)")
    print("=" * 80)

    # 1. ETAPA DE COLETA DE DADOS ABERTOS
    print("\n>>> ETAPA 1/4: Coleta e Integracao de Dados Abertos (Open-Meteo)")
    argumentos = ["--config", str(ROOT / "config/params.yaml")]
    if forcar_recoleta:
        argumentos.append("--overwrite")
    coleta_integracao.main(argumentos)

    # Cada execucao reconstrui as dependencias. Assim uma nova coleta nunca
    # reutiliza matrizes antigas e o botao de retreino realmente roda S3/S4/S5.
    print("\n>>> ETAPA 2/4: Limpeza, split e engenharia de atributos (S2, S3, S4)")
    prepara_sprint2.main()
    sprint3_baselines.main()
    sprint4_features.main()

    # 3. ETAPA DE MODELAGEM: DECISION TREE, RANDOM FOREST E XGBOOST
    print("\n>>> ETAPA 3/4: Selecao oficial Sprint 5 e comparacao exploratoria")
    sprint5_modelos.main()
    print(" -> Reavaliando os tres algoritmos com calibracao temporal independente...")
    resumo_modelagem = modelos_arvore_random_forest.main()

    # 4. ETAPA DE COMPARACAO COM AS BASELINES
    print("\n>>> ETAPA 4/4: Consolidacao da Comparacao com as Baselines")
    comparativo = resumo_modelagem["comparativo"]

    tempo_total = time.time() - inicio
    print("\n" + "=" * 80)
    print(f"   PIPELINE EXECUTADO COM SUCESSO EM {tempo_total:.2f}s!")
    print("=" * 80)

    # Resumo para exibicao rápida
    print("\nRESUMO DO COMPARATIVO EXPLORATORIO NO CONJUNTO DE TESTE:")
    for r in comparativo:
        print(f" - {r['modelo']:<36} | Recall: {r['recall_c1']*100:5.2f}% | FN: {r['fn']:3d} | FP: {r['fp']:3d} | F2: {r['f2_c1']:.4f} | Status: {r['status_aprovacao']}")

    print("\nArtefatos disponiveis em:")
    print(" - Modelos treinados: models/ (pipeline_decision_tree, pipeline_random_forest, pipeline_xgboost)")
    print(" - Relatorios e graficos: reports/modelos_decision_tree_random_forest/")
    print(" - Dashboard interativo: execute 'streamlit run app.py'")

    return {
        "status": "sucesso",
        "tempo_execucao_s": tempo_total,
        "comparativo": comparativo,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline completo de ponta a ponta")
    parser.add_argument("--recoletar", action="store_true", help="Forca nova consulta as APIs Open-Meteo")
    args = parser.parse_args()
    executar_pipeline_completo(forcar_recoleta=args.recoletar)

