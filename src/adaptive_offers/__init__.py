"""Plataforma de experimentação adaptativa para próxima melhor ação.

Datathon POSTECH — Machine Learning Engineering (Fase 5).
"""

__version__ = "1.0.0"

from .config import ARMS, N_ARMS, N_SEGMENTS, describe_arm

__all__ = ["ARMS", "N_ARMS", "N_SEGMENTS", "describe_arm", "__version__"]
