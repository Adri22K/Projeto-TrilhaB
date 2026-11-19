"""Ponto de entrada raiz para execucao do pipeline completo."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline_completo import executar_pipeline_completo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline completo")
    parser.add_argument("--recoletar", action="store_true", help="Forca nova coleta historica")
    args = parser.parse_args()
    executar_pipeline_completo(forcar_recoleta=args.recoletar)

