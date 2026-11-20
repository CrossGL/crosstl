"""The refusal of calls the WGSL generator would write with another language's built-in function name.

The importers carry the built-in functions of the languages they read into the IR as plain calls
(:data:`~crosstl.translator.semantics.builtin_function_names.BUILTIN_FUNCTION_NAMES`), and the WGSL generator lowers
the ones it knows. A call it had no lowering for was written with the name it had: ``EmitVertex()``, ``lit(...)`` or
``simd_sum(...)`` in WGSL output. WGSL declares none of them (WGSL, "Built-in Functions"), so the module fails to
compile, or calls a function of the same name the shader declares for another purpose.

:func:`is_foreign_wgsl_call` decides it for one call: a built-in function name the generator did not rename and
that is not a WGSL built-in function (:data:`~.wgsl_builtin_functions.WGSL_BUILTIN_FUNCTIONS`) has no WGSL callee.
:class:`WGSLForeignBuiltinCallError` is the refusal.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..semantics.builtin_function_names import BUILTIN_FUNCTION_NAMES
from .wgsl_builtin_functions import WGSL_BUILTIN_FUNCTIONS

__all__ = ["WGSLForeignBuiltinCallError", "is_foreign_wgsl_call"]


class WGSLForeignBuiltinCallError(TranslationDiagnostic):
    """A call to a built-in function of another language, which WGSL does not declare."""

    def __init__(self, function_name):
        self.function_name = function_name
        self.missing_capabilities = ("wgsl.builtin-function",)
        super().__init__(
            f"WGSL target cannot lower the call to {function_name}(): {function_name} is a built-in function of "
            "another language, and WGSL declares no built-in function of that name",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-foreign-builtin-call",
        )


def is_foreign_wgsl_call(function_name, written_name):
    """Whether a call to ``function_name``, written as ``written_name``, names no WGSL function.

    ``written_name`` is the name the generator writes; it differs from ``function_name`` when the generator renamed
    the call. Only a built-in function of a source language is refused: any other name is a function, constructor
    or type the program declares, which ``declared_functions`` has checked.
    """
    if (
        not isinstance(function_name, str)
        or function_name not in BUILTIN_FUNCTION_NAMES
    ):
        return False
    if written_name != function_name:
        return False
    return written_name not in WGSL_BUILTIN_FUNCTIONS
