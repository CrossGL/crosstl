"""Refuse cooperative matrices on targets that have no lowering for them.

A cooperative matrix (``CooperativeMatrix<...>`` and the ``cooperative_matrix_*`` operations) is spread across the
lanes of a subgroup, and the lane layout belongs to the target's matrix hardware. The CUDA, HIP, Slang, Rust, Mojo
and WGSL emitters have no lowering for one. They used to write the CrossGL type, or the Python representation of the
type or of the operation node, into the program and return it. Those emitters now call
:func:`refuse_cooperative_matrices` before generating, and the program is refused with every construct it uses.
"""

from __future__ import annotations

from ..ast import CooperativeMatrixOpNode, CooperativeMatrixType
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .ast_walk import walk_ast_values

__all__ = [
    "CooperativeMatrixLoweringUnsupportedError",
    "cooperative_matrix_constructs",
    "refuse_cooperative_matrices",
]


class CooperativeMatrixLoweringUnsupportedError(TranslationDiagnostic):
    """The program uses cooperative matrices, and the target has no lowering for them."""

    code = f"{CODE_PREFIX}cooperative-matrix-unsupported"

    def __init__(self, label: str, target: str, constructs: list[str]) -> None:
        self.constructs = tuple(constructs)
        self.missing_capabilities = (f"{target}.cooperative-matrix",)
        super().__init__(
            f"{label} has no lowering for subgroup cooperative matrices, so it cannot "
            f"translate {', '.join(self.constructs)}",
            target=target,
        )


def _part(value) -> str:
    for attribute in ("name", "value"):
        if hasattr(value, attribute):
            return str(getattr(value, attribute))
    return str(value)


def _type_spelling(matrix: CooperativeMatrixType) -> str:
    parts = (matrix.element_type, matrix.rows, matrix.cols, matrix.scope)
    return f"CooperativeMatrix<{', '.join(_part(part) for part in parts)}>"


def cooperative_matrix_constructs(ast) -> list[str]:
    """Name each distinct cooperative-matrix type and operation reachable from ``ast``, sorted."""
    found = set()
    for value in walk_ast_values(ast):
        if isinstance(value, CooperativeMatrixOpNode):
            found.add(f"operation {value.operation}")
        elif isinstance(value, CooperativeMatrixType):
            found.add(f"type {_type_spelling(value)}")
    return sorted(found)


def refuse_cooperative_matrices(ast, *, label: str, target: str) -> None:
    """Raise :class:`CooperativeMatrixLoweringUnsupportedError` when ``ast`` uses a cooperative matrix."""
    constructs = cooperative_matrix_constructs(ast)
    if constructs:
        raise CooperativeMatrixLoweringUnsupportedError(label, target, constructs)
