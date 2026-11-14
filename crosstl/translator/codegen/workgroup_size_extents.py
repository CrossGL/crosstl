"""Array extents that a program writes with ``gl_WorkGroupSize``.

``gl_WorkGroupSize`` is a constant expression once a GLSL compute shader has declared its local size (GLSL 4.60
section 7.1.3), so ``shared float tile[gl_WorkGroupSize.x];`` declares a fixed-size array. The CrossGL import keeps
that spelling, and the built-in means the workgroup size of the program's compute, mesh or task entry.

MSL array bounds and HLSL array sizes must be integral constant expressions, and neither language has a constant
that names the declared workgroup size (Metal's ``threads_per_threadgroup`` is an entry argument). SPIR-V needs a
constant length for ``OpTypeArray``; without one the array became an ``OpTypeRuntimeArray`` in Workgroup storage,
which Vulkan forbids. These generators therefore replace every ``gl_WorkGroupSize`` component of an array extent with
the declared size before they render the program. A component the pipeline specializes (``local_size_x_id``), or one
the host chooses, has no value at translation time, and neither does a component on which the entries disagree, so
such an extent is refused rather than emitted with an undeclared name. GLSL keeps the extent as written and declares
the local size before the first declaration or helper that reads it (:func:`glsl_workgroup_size_declaration_layout`).
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Callable

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
)
from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name
from .array_utils import evaluate_literal_int_expression
from .ast_walk import walk_ast_values
from .host_workgroup_size_ids import is_host_sized_workgroup
from .specialized_workgroup_size import workgroup_size_specialization_ids
from .stage_utils import compute_local_size, function_stage_name, stage_matches

__all__ = [
    "WORKGROUP_SIZE_BUILTIN",
    "fold_workgroup_size_extents",
    "glsl_workgroup_size_declaration_layout",
    "program_reads_workgroup_size_outside_entries",
    "program_workgroup_size",
    "workgroup_size_axis",
]

WORKGROUP_SIZE_BUILTIN = "gl_WorkGroupSize"
WORKGROUP_SIZE_STAGES = frozenset({"compute", "mesh", "task"})
_AXES = {"x": 0, "y": 1, "z": 2}
_AXIS_NAMES = "xyz"
_TEXT_COMPONENT = re.compile(r"\bgl_WorkGroupSize\s*\.\s*([xyz])\b")


def _is_builtin(expr: Any) -> bool:
    return isinstance(expr, IdentifierNode) and expr.name == WORKGROUP_SIZE_BUILTIN


def workgroup_size_axis(expr: Any) -> int | None:
    """The component ``gl_WorkGroupSize.x`` (or ``[0]``, and so on) reads, or ``None`` for any other expression."""
    if isinstance(expr, MemberAccessNode) and _is_builtin(expr.object_expr):
        return _AXES.get(expr.member)
    if isinstance(expr, ArrayAccessNode) and _is_builtin(expr.array_expr):
        index = evaluate_literal_int_expression(expr.index_expr)
        return index if index in (0, 1, 2) else None
    return None


def _references_builtin(value: Any) -> bool:
    if isinstance(value, str):
        return WORKGROUP_SIZE_BUILTIN in value
    if isinstance(value, (int, float)) or value is None:
        return False
    return any(_is_builtin(node) for node in walk_ast_values(value))


def _declares_builtin_name(ast: Any) -> bool:
    """Whether the program declares its own ``gl_WorkGroupSize``, as a SPIR-V import of a specialized size does."""
    declarations = list(getattr(ast, "constants", None) or [])
    declarations += list(getattr(ast, "global_variables", None) or [])
    return any(
        getattr(node, "name", None) == WORKGROUP_SIZE_BUILTIN for node in declarations
    )


def _workgroup_sized_arrays(ast: Any) -> list:
    return [
        node
        for node in walk_ast_values(ast)
        if isinstance(node, ArrayType) and _references_builtin(node.size)
    ]


def program_reads_workgroup_size_outside_entries(ast: Any) -> bool:
    """Whether a declaration or helper function, rather than an entry, reads the built-in ``gl_WorkGroupSize``."""
    if _declares_builtin_name(ast):
        return False
    roots = [
        getattr(ast, name, None)
        for name in ("constants", "global_variables", "structs", "cbuffers")
    ]
    roots += [
        function
        for function in getattr(ast, "functions", None) or []
        if function_stage_name(function) not in STAGE_QUALIFIER_NAMES
    ]
    for stage in (getattr(ast, "stages", None) or {}).values():
        roots.append(getattr(stage, "local_variables", None))
        roots.append(getattr(stage, "local_functions", None))
    return any(_references_builtin(root) for root in roots)


def _workgroup_entries(ast: Any, target_stage: Any = None) -> list:
    """``(stage name, execution metadata)`` of each compute, mesh or task entry, in program order."""
    entries = []
    for function in getattr(ast, "functions", None) or []:
        stage_name = function_stage_name(function)
        if stage_name in WORKGROUP_SIZE_STAGES and stage_matches(
            target_stage, stage_name
        ):
            entries.append(
                (stage_name, dict(getattr(function, "execution_config", None) or {}))
            )
    for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
        stage_name = normalize_stage_name(stage_type)
        if stage_name not in WORKGROUP_SIZE_STAGES or not stage_matches(
            target_stage, stage_name
        ):
            continue
        if getattr(stage, "entry_point", None) is not None:
            entries.append(
                (stage_name, dict(getattr(stage, "execution_config", None) or {}))
            )
    return entries


def _entry_components(config: dict, target: str) -> list:
    """Each component of one entry's size: an ``int``, or the reason it has no value at translation time."""
    if is_host_sized_workgroup(config):
        return ["the host chooses it at dispatch"] * 3
    specialization_ids = workgroup_size_specialization_ids(config, target)
    components = []
    for size, spec_id in zip(compute_local_size(config), specialization_ids):
        if spec_id is not None:
            components.append(
                f"the pipeline specializes it (local_size_*_id = {spec_id})"
            )
            continue
        value = evaluate_literal_int_expression(size)
        components.append(
            value
            if isinstance(value, int) and value > 0
            else f"its size '{size}' is not an integer literal"
        )
    return components


def program_workgroup_size(ast: Any, target: str) -> list:
    """The program's workgroup size, one entry per component: an ``int``, or the reason it has no single value."""
    configs = [config for _stage_name, config in _workgroup_entries(ast)]
    if not configs:
        return ["the program has no compute, mesh or task entry"] * 3
    per_entry = [_entry_components(config, target) for config in configs]
    resolved = []
    for axis in range(3):
        values = [components[axis] for components in per_entry]
        reasons = [value for value in values if isinstance(value, str)]
        if reasons:
            resolved.append(reasons[0])
        elif len(set(values)) > 1:
            sizes = ", ".join(str(value) for value in sorted(set(values)))
            resolved.append(f"the program's entries declare different sizes ({sizes})")
        else:
            resolved.append(values[0])
    return resolved


def _uint_literal(value: int) -> LiteralNode:
    return LiteralNode(value, PrimitiveType("uint"))


def _substitute(expr: Any, sizes: list, refuse: Callable[[int], None]) -> Any:
    """``expr`` with each ``gl_WorkGroupSize`` component replaced by its size; ``refuse`` takes the unknown ones."""
    axis = workgroup_size_axis(expr)
    if axis is not None:
        if isinstance(sizes[axis], int):
            return _uint_literal(sizes[axis])
        refuse(axis)
        return _uint_literal(1)
    if _is_builtin(expr):
        refuse(None)
        return _uint_literal(1)
    if isinstance(expr, list):
        return [_substitute(item, sizes, refuse) for item in expr]
    if not hasattr(expr, "__dict__") or isinstance(expr, type):
        return expr
    for name, value in list(vars(expr).items()):
        if name == "parent" or not _references_builtin(value):
            continue
        setattr(expr, name, _substitute(value, sizes, refuse))
    if isinstance(expr, MemberAccessNode):
        # ``object`` aliases ``object_expr``.
        expr.object = expr.object_expr
    if isinstance(expr, ArrayAccessNode):
        expr.array, expr.index = expr.array_expr, expr.index_expr
    return expr


def _substitute_text(text: str, sizes: list, refuse: Callable[[int], None]) -> str:
    def component(match):
        axis = _AXES[match.group(1)]
        if isinstance(sizes[axis], int):
            return f"{sizes[axis]}u"
        refuse(axis)
        return "1u"

    substituted = _TEXT_COMPONENT.sub(component, text)
    if WORKGROUP_SIZE_BUILTIN in substituted:
        refuse(None)
    return substituted


def _evaluate_extent(expr: Any) -> tuple | None:
    """``(value, unsigned)`` of an integer extent built from literals, with GLSL's 32-bit ``uint`` wrap-around.

    :func:`evaluate_literal_int_expression` leaves unsigned arithmetic alone, and ``gl_WorkGroupSize`` is a
    ``uvec3``, so ``gl_WorkGroupSize.x * 2u`` needs the unsigned rules: an operation with a ``uint`` operand is a
    ``uint`` operation (GLSL 4.60 section 4.1.10, implicit conversions).
    """
    if isinstance(expr, LiteralNode):
        value = evaluate_literal_int_expression(expr)
        if not isinstance(value, int):
            return None
        unsigned = getattr(expr.literal_type, "name", expr.literal_type) == "uint"
        return int(value), unsigned
    left, right = getattr(expr, "left", None), getattr(expr, "right", None)
    operator = getattr(expr, "operator", getattr(expr, "op", None))
    if left is None or right is None or operator not in _UINT_OPERATORS:
        value = evaluate_literal_int_expression(expr)
        return (int(value), False) if isinstance(value, int) else None
    left, right = _evaluate_extent(left), _evaluate_extent(right)
    if left is None or right is None:
        return None
    unsigned = left[1] or right[1]
    if not unsigned:
        value = evaluate_literal_int_expression(expr)
        return (int(value), False) if isinstance(value, int) else None
    a, b = left[0] & _UINT_MASK, right[0] & _UINT_MASK
    if operator in ("/", "%") and b == 0:
        return None
    return _UINT_OPERATORS[operator](a, b) & _UINT_MASK, True


_UINT_MASK = 0xFFFFFFFF
_UINT_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "<<": lambda a, b: a << (b & 31),
    ">>": lambda a, b: a >> (b & 31),
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
}


def _fold_extent(size: Any, sizes: list, refuse: Callable[[int], None]) -> Any:
    if isinstance(size, str):
        folded = _substitute_text(size, sizes, refuse)
        value = evaluate_literal_int_expression(folded)
        return int(value) if isinstance(value, int) else folded
    folded = _substitute(size, sizes, refuse)
    evaluated = _evaluate_extent(folded)
    if evaluated is None:
        return folded
    # An extent is a count; every target spells it as a plain integer literal.
    return LiteralNode(evaluated[0], PrimitiveType("int"))


def fold_workgroup_size_extents(
    ast: Any, target: str, refuse_lowering: Callable[[str], Any]
) -> Any:
    """Return ``ast`` with every ``gl_WorkGroupSize`` component of an array extent replaced by the declared size.

    The input is left unchanged; a copy is made only when an extent reads the built-in. An extent whose component
    has no value at translation time goes to ``refuse_lowering`` with the reason.
    """
    if _declares_builtin_name(ast) or not _workgroup_sized_arrays(ast):
        return ast
    ast = deepcopy(ast)
    sizes = program_workgroup_size(ast, target)

    def refuse(axis: int | None) -> None:
        if axis is None:
            refuse_lowering(
                f"{target} array extent reads {WORKGROUP_SIZE_BUILTIN} as a whole vector; an array extent "
                "must be one integer component such as gl_WorkGroupSize.x"
            )
            return
        refuse_lowering(
            f"{target} array extent reads {WORKGROUP_SIZE_BUILTIN}.{_AXIS_NAMES[axis]}, which has no value at "
            f"translation time because {sizes[axis]}; {target} array sizes must be integer constants"
        )

    for array in _workgroup_sized_arrays(ast):
        array.size = _fold_extent(array.size, sizes, refuse)
    return ast


def glsl_workgroup_size_declaration_layout(
    codegen: Any, ast: Any, target_stage: Any
) -> str:
    """The local-size layout a GLSL program declares before the declarations and helpers that read ``gl_WorkGroupSize``.

    GLSL makes reading ``gl_WorkGroupSize`` before the shader declares its local size a compile-time error (GLSL 4.60
    section 7.1.3), and the generator writes the layout with the entry, after the global declarations and helper
    functions. A shader may repeat the declaration with the same sizes (section 4.4.1.4), so the entry keeps its own
    copy. With several workgroup entries of different sizes, a read outside them has no single size and is refused.
    """
    if not program_reads_workgroup_size_outside_entries(ast):
        return ""
    entries = _workgroup_entries(ast, target_stage)
    if not entries:
        return ""
    distinct = {
        repr(sorted(config.items(), key=lambda item: item[0]))
        for _name, config in entries
    }
    if len(distinct) > 1:
        codegen.refuse_lowering(
            "GLSL declaration outside the entries reads gl_WorkGroupSize, but the program's compute, mesh or "
            "task entries declare different workgroup sizes; translate one entry at a time"
        )
        return ""
    stage_name, config = entries[0]
    return codegen.wrap_stage_guard(codegen.generate_compute_layout(config), stage_name)
