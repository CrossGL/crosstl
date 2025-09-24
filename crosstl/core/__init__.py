"""
CrossTL Core Infrastructure.
Production-ready universal programming language translator.
"""

from .engine import TranslationEngine
from .ir import CrossGLIR
from .type_system import TypeSystem
from .errors import ErrorCollector, TranslationError

__all__ = [
    "TranslationEngine",
    "CrossGLIR",
    "TypeSystem",
    "ErrorCollector",
    "TranslationError",
]
