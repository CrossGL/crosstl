"""Metal atomic functions that order memory, imported as relaxed CrossGL atomics between CrossGL fences.

CrossGL's atomic functions are relaxed: the generators write them as Metal ``memory_order_relaxed`` atomics, GLSL
atomic functions and SPIR-V atomics without memory semantics (:mod:`crosstl.translator.codegen.spirv_atomic_load_store`).
Metal's atomic functions without the ``_explicit`` suffix are sequentially consistent over all memory: the toolchain's
``metal_atomic`` header defines ``atomic_fetch_add(object, operand)`` as ``atomic_fetch_add_explicit(object, operand,
memory_order_seq_cst, <every memory flag>)``, and likewise ``atomic_load``, ``atomic_store``, ``atomic_exchange``,
``atomic_compare_exchange_weak``, the other ``atomic_fetch_*`` functions and the 64-bit ``atomic_max`` and
``atomic_min`` (MSL 4.1, section 6.15.4 "Atomic Functions").

The importer keeps that order. It writes the relaxed CrossGL atomic between two sequentially consistent CrossGL fences
over every memory flag at device scope, the scope of Metal's ``atomic_int``, ``atomic_uint``, ``atomic_float``,
``atomic_bool`` and ``atomic_ulong``. A sequentially consistent fence on each side of a relaxed operation orders it
at least as strongly as the sequentially consistent operation does; this is the leading-and-trailing fence mapping
of C++ ``memory_order_seq_cst`` atomics.

The fences are statements, so the call must be a whole statement whose other effects are local to the thread: an
expression statement, the initializer of a local declaration, or the value assigned to a local variable. Its operands
may not assign, increment, or call a source function or another atomic, because those effects would fall between the
fences instead of before the ordered operation. Anywhere else the importer refuses the call rather than weaken it.

Metal's ``_explicit`` read-modify-write functions also take a memory order. Their three-operand form accepts only
``memory_order_relaxed`` (the header's ``METAL_RELAXED_ORDER``), and the four-operand form adds memory flags, which a
relaxed operation does not use. :func:`require_relaxed_atomic_order` refuses any other order instead of dropping it.
"""

from __future__ import annotations

import copy
from typing import Any

from .MetalAst import (
    AssignmentNode,
    ASTNode,
    CallNode,
    FunctionCallNode,
    MethodCallNode,
    PostfixOpNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = [
    "SEQUENTIALLY_CONSISTENT_ATOMICS",
    "MetalAtomicOrderLoweringError",
    "fenced_atomic_statement",
    "refuse_unfenced_atomic_call",
    "require_relaxed_atomic_order",
    "sequentially_consistent_atomic_name",
]

# Metal atomic function -> (its ``_explicit`` form, operands before the memory orders, memory orders it takes).
SEQUENTIALLY_CONSISTENT_ATOMICS = {
    "atomic_load": ("atomic_load_explicit", 1, 1),
    "atomic_store": ("atomic_store_explicit", 2, 1),
    "atomic_exchange": ("atomic_exchange_explicit", 2, 1),
    "atomic_compare_exchange_weak": ("atomic_compare_exchange_weak_explicit", 3, 2),
    "atomic_fetch_add": ("atomic_fetch_add_explicit", 2, 1),
    "atomic_fetch_sub": ("atomic_fetch_sub_explicit", 2, 1),
    "atomic_fetch_and": ("atomic_fetch_and_explicit", 2, 1),
    "atomic_fetch_or": ("atomic_fetch_or_explicit", 2, 1),
    "atomic_fetch_xor": ("atomic_fetch_xor_explicit", 2, 1),
    "atomic_fetch_min": ("atomic_fetch_min_explicit", 2, 1),
    "atomic_fetch_max": ("atomic_fetch_max_explicit", 2, 1),
    "atomic_max": ("atomic_max_explicit", 2, 1),
    "atomic_min": ("atomic_min_explicit", 2, 1),
}

# The memory flags of Metal's ``mem_flags`` enumeration (MSL 4.1, section 6.9.1), which a sequentially consistent
# Metal atomic orders, and the fence the importer writes on each side of the relaxed CrossGL atomic.
_ALL_MEMORY_FLAGS = (
    "mem_device",
    "mem_threadgroup",
    "mem_threadgroup_imageblock",
    "mem_texture",
    "mem_object_data",
)
_SEQUENTIALLY_CONSISTENT_FENCE = f"atomicThreadFence({' | '.join(_ALL_MEMORY_FLAGS)}, memory_order_seq_cst, thread_scope_device)"
_RELAXED = "memory_order_relaxed"
_LOCAL_ADDRESS_SPACES = frozenset({"thread"})


class MetalAtomicOrderLoweringError(ValueError):
    """Raised when a Metal atomic's memory order cannot be kept in CrossGL."""

    project_diagnostic_code = "project.translate.metal-atomic-order-unsupported"
    missing_capabilities = ("metal.atomic-memory-order-lowering",)

    def __init__(self, function, reason, source_location=None):
        self.function = function
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot lower Metal {function} without changing its memory order: {reason}"
        )


def _unscoped_library_name(converter: Any, name: Any) -> str | None:
    """``name`` without a leading ``metal::``, or ``None`` when it names another namespace or a source function."""
    function_name = str(name).lstrip(":")
    if function_name.startswith("metal::"):
        function_name = function_name[len("metal::") :]
    if "::" in function_name:
        return None
    if function_name in getattr(converter, "user_function_names", ()):
        return None
    if converter.metal_user_function_overloads(name):
        return None
    return function_name


def sequentially_consistent_atomic_name(converter: Any, expr: Any) -> str | None:
    """The Metal function name when ``expr`` calls a sequentially consistent library atomic, else ``None``."""
    if not isinstance(expr, FunctionCallNode):
        return None
    if str(expr.name).rsplit("::", 1)[-1] not in SEQUENTIALLY_CONSISTENT_ATOMICS:
        return None
    name = _unscoped_library_name(converter, expr.name)
    return name if name in SEQUENTIALLY_CONSISTENT_ATOMICS else None


def refuse_unfenced_atomic_call(converter: Any, expr: Any) -> None:
    """Refuse a sequentially consistent atomic that :func:`fenced_atomic_statement` did not take as a statement."""
    name = sequentially_consistent_atomic_name(converter, expr)
    if name is not None:
        raise MetalAtomicOrderLoweringError(
            name,
            "a sequentially consistent atomic becomes a relaxed CrossGL atomic between two fences, which needs a "
            "statement of its own: an expression statement, a local declaration or an assignment to a local "
            "variable",
            getattr(expr, "source_location", None),
        )


def _order_name(converter: Any, expr: Any) -> str | None:
    name = getattr(expr, "name", expr) if isinstance(expr, VariableNode) else expr
    if not isinstance(name, str):
        return None
    name = name.lstrip(":")
    if name.startswith("metal::"):
        name = name[len("metal::") :]
    if (
        name in converter.current_variable_types
        or name in converter.global_variable_types
    ):
        return None
    return name


def require_relaxed_atomic_order(
    converter: Any, function: str, orders: list, source_location=None
) -> None:
    """Refuse an ``_explicit`` read-modify-write whose ``orders`` (the operands after the value) are not relaxed."""
    if len(orders) not in {1, 2} or _order_name(converter, orders[0]) != _RELAXED:
        raise MetalAtomicOrderLoweringError(
            function,
            "only memory_order_relaxed has a portable read-modify-write contract",
            source_location,
        )


def _child_nodes(node: Any):
    fields = getattr(node, "__dict__", None)
    if fields is None:
        fields = {
            slot: getattr(node, slot, None)
            for cls in type(node).__mro__
            for slot in getattr(cls, "__slots__", ())
        }
    for value in fields.values():
        if isinstance(value, ASTNode):
            yield value
        elif isinstance(value, (list, tuple)):
            yield from (item for item in value if isinstance(item, ASTNode))


def _operand_effect(converter: Any, node: Any) -> str | None:
    """A description of an effect in ``node`` that would fall between the fences, or ``None``."""
    if isinstance(node, AssignmentNode):
        return "an assignment"
    if isinstance(node, (UnaryOpNode, PostfixOpNode)) and str(node.op) in {"++", "--"}:
        return "an increment or decrement"
    if isinstance(node, (MethodCallNode, CallNode)):
        return "a member call"
    if isinstance(node, FunctionCallNode):
        name = _unscoped_library_name(converter, node.name)
        if name is None:
            return f"a call of the source function '{node.name}'"
        if name.startswith("atomic_"):
            return f"the atomic '{name}'"
    for child in _child_nodes(node):
        effect = _operand_effect(converter, child)
        if effect is not None:
            return effect
    return None


def _is_local_variable(converter: Any, target: Any) -> bool:
    if not isinstance(target, VariableNode):
        return False
    if getattr(target, "vtype", None):
        return True
    name = target.name
    if name not in converter.current_variable_types:
        return False
    declared_type = str(converter.current_variable_types.get(name) or "")
    qualifiers = set(converter.current_variable_type_qualifiers.get(name) or ())
    address_spaces = qualifiers & {
        "device",
        "constant",
        "threadgroup",
        "object_data",
        "threadgroup_imageblock",
    }
    return "&" not in declared_type and not address_spaces


def _statement_call(stmt: Any):
    """The call a statement evaluates as a whole, and the statement with that call replaced."""
    if isinstance(stmt, FunctionCallNode):
        return stmt, lambda call: call
    if isinstance(stmt, AssignmentNode) and isinstance(stmt.right, FunctionCallNode):

        def replace(call):
            replaced = copy.copy(stmt)
            replaced.right = call
            return replaced

        return stmt.right, replace
    return None, None


def fenced_atomic_statement(
    converter: Any, stmt: Any, indent: int, is_main: bool
) -> str | None:
    """A statement that is one sequentially consistent Metal atomic as CrossGL between fences, or ``None``."""
    call, replace = _statement_call(stmt)
    name = sequentially_consistent_atomic_name(converter, call)
    if name is None:
        return None
    location = getattr(call, "source_location", None)
    if isinstance(stmt, AssignmentNode) and not _is_local_variable(
        converter, stmt.left
    ):
        raise MetalAtomicOrderLoweringError(
            name,
            "its result must initialize or be assigned to a local variable, so that the trailing fence orders "
            "the atomic before any later memory access",
            location,
        )
    explicit_name, operand_count, order_count = SEQUENTIALLY_CONSISTENT_ATOMICS[name]
    if len(call.args) != operand_count:
        raise MetalAtomicOrderLoweringError(
            name, f"expects {operand_count} operand(s), got {len(call.args)}", location
        )
    for operand in call.args:
        effect = _operand_effect(converter, operand)
        if effect is not None:
            raise MetalAtomicOrderLoweringError(
                name,
                f"an operand contains {effect}, which the leading fence would no longer order",
                location,
            )
    if "atomicThreadFence" in getattr(converter, "user_function_names", ()):
        raise MetalAtomicOrderLoweringError(
            name,
            "the CrossGL fence atomicThreadFence conflicts with a source function",
            location,
        )
    explicit = FunctionCallNode(
        explicit_name,
        [*call.args, *(VariableNode("", _RELAXED) for _ in range(order_count))],
    )
    explicit.source_location = location
    body = converter.generate_function_body([replace(explicit)], indent, is_main)
    padding = "    " * indent
    return f"{_SEQUENTIALLY_CONSISTENT_FENCE};\n{body}{padding}{_SEQUENTIALLY_CONSISTENT_FENCE};\n"
