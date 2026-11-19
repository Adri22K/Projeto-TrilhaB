"""Limites de CPU do treinamento, compartilhados entre as sprints."""
from joblib import cpu_count

# Usa todas as CPUs logicas disponiveis, sem teto artificial.
TRAINING_WORKERS = max(1, cpu_count())
