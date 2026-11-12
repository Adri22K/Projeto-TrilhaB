"""Ponto de entrada raiz para execucao do pipeline completo."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline_completo import executar_pipeline_completo

if __name__ == "__main__":
    executar_pipeline_completo()

