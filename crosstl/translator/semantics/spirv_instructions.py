"""SPIR-V operations the importer keeps as calls, refused for the targets that would print them verbatim.

The SPIR-V importer (:mod:`crosstl.backend.SPIRV.VulkanParser`) writes an instruction CrossGL has no operation for
as a call to a reserved name: ``spirv`` followed by an upper-case letter (``spirvImageSparseSample``,
``spirvSDot``, ``spirvImageTexelPointer``), or ``spirv_`` followed by an extended instruction set and instruction
(``spirv_ext_inst``). The CrossGL text keeps the instruction that way, so the import shows what it could not
express. No emitter defines those names, so generating any other target from such a program used to print the call
as it was, which no target compiler accepts and which hides the gap from the caller.

:func:`refuse_spirv_instructions` refuses the first such call that ``program`` does not itself declare a function
for with :class:`SpirvOperationUnsupportedError`, for every target but CrossGL. Passes that lower one of these
operations run before it (``spirvArrayLength`` in :mod:`.storage_blocks`).
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions

__all__ = [
    "SpirvOperationUnsupportedError",
    "is_spirv_operation_name",
    "refuse_spirv_instructions",
]

_RESERVED = re.compile(r"spirv(?:[A-Z][A-Za-z0-9_]*|_[A-Za-z0-9_]+)")
_CROSSGL_TARGETS = frozenset({"cgl", "crossgl"})


class SpirvOperationUnsupportedError(TranslationDiagnostic):
    """A SPIR-V operation the importer kept as a call, for a target whose emitter does not lower it."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.operation = name
        self.missing_capabilities = (f"{target}.spirv-operation.{name}",)
        super().__init__(
            f"SPIR-V operation '{name}' has no {target} translation: the SPIR-V importer keeps it as a call, "
            f"and the {target} emitter does not lower it",
            target=target,
            code=f"{CODE_PREFIX}{target}-spirv-operation-unsupported",
        )


def is_spirv_operation_name(name) -> bool:
    """Whether ``name`` is one the SPIR-V importer reserves for an operation it keeps as a call."""
    return isinstance(name, str) and _RESERVED.fullmatch(name) is not None


def _call_name(node):
    function = getattr(node, "function", None) or getattr(node, "name", None)
    return getattr(function, "name", function)


def _nodes(node, seen):
    """Every AST node under ``node``, depth first in field order, each once."""
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
            continue
        if isinstance(current, dict):
            stack.extend(reversed(list(current.values())))
            continue
        if not isinstance(current, A.ASTNode):
            continue
        yield current
        stack.extend(
            reversed(
                [
                    value
                    for key, value in vars(current).items()
                    if key not in {"parent", "source_location"}
                ]
            )
        )


def refuse_spirv_instructions(program, target) -> None:
    """Raise :class:`SpirvOperationUnsupportedError` for the first call to a reserved SPIR-V operation name.

    A name ``program`` declares a function for is the program's own function, not an operation. CrossGL targets
    keep the calls.
    """
    target = str(target or "").lower()
    if (
        not target
        or target in _CROSSGL_TARGETS
        or not isinstance(program, A.ShaderNode)
    ):
        return
    declared = {getattr(function, "name", None) for function in iter_functions(program)}
    for node in _nodes(program, set()):
        if not isinstance(node, A.FunctionCallNode):
            continue
        name = _call_name(node)
        if is_spirv_operation_name(name) and name not in declared:
            raise SpirvOperationUnsupportedError(
                target, name, source_location=getattr(node, "source_location", None)
            )
