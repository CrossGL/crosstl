"""Left-to-right evaluation of call arguments in HLSL output.

CrossGL evaluates the arguments of a call exactly once, in order, from left to right, and copies an ``inout``
argument's value in when its turn comes (GLSL 4.60 section 6.1.1; MSL and SPIR-V output keep the same order).
DXC's DXIL code generator does not: checked with DXC 1.9.2602, it evaluates the arguments of function and
intrinsic calls from right to left, while constructors, initializer lists and operators stay left to right::

    two(x, bump(x))            // DXIL passes the updated x as the first argument
    max(bump(x), x)            // DXIL reads x before bump runs
    pair(bump(t[l]), t[l])     // DXIL copies t[l] in before bump writes it

When an argument has a side effect and another argument, not a constant, can read or write what that side effect
writes, the arguments up to the last such side effect are evaluated into function-scope temporaries, in order, in a
comma expression placed where the call was, so the call is still evaluated only where the source evaluates it (a
conditional branch, a short-circuit operand or a loop condition)::

    two(x, bump(x))            int((__crossgl_argument = x, __crossgl_argument_ = bump(x),
                                    two(__crossgl_argument, __crossgl_argument_)))

A used value is wrapped in a function-style cast to its own type, because DXC flags a parenthesized comma
expression that initializes a variable, a parameter or a return value (``-Wcomma-in-init``).

An ``out`` or ``inout`` argument in that range is passed through a temporary as well: its location is fixed and
an ``inout`` value is copied in at its turn, and the temporary is copied back after the call. A memory argument of
a built-in (the object or expected value of an atomic operation, a result it writes, ``&lvalue``) and a pointer
or resource view argument keep their place with their indices fixed at their turn; the callee reads that memory
after every argument is evaluated, whichever order DXC takes. Arguments after the last ordered side effect read
their operands after every side effect, which is the source order, so they stay in place.

A side effect writes the variables it names when it is an assignment, an increment or decrement, or a call to a
function of the program that writes only its locals and its ``out`` and ``inout`` parameters. Any other side
effect (``discard``, a built-in that writes memory, synchronizes or emits, see
:func:`~.wgsl_conditional_effects.is_effectful_builtin`, a write to a module-scope variable, a pointer or a
resource) can change anything, so every other argument that is not a constant depends on it.

A call is left alone when it has fewer than two arguments or no such dependence. A call whose ordered arguments
include one that HLSL cannot hold in a temporary (a value of unknown type, a pointer or resource view whose
location cannot be fixed) is refused rather than emitted in DXC's order.
"""

from __future__ import annotations

import copy
import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import split_array_type_suffix
from .wgsl_conditional_effects import (
    _callee_name,
    _nodes,
    _parameter_writes_caller,
    _root_name,
    is_effectful_builtin,
)

__all__ = ["hlsl_ordered_call_expression"]

_ACTIVE = "_hlsl_ordered_call_ids"
_EFFECTS = "_hlsl_ordered_call_function_effects"
# Built-in methods that write the resource they are called on, besides those is_effectful_builtin names.
_EFFECTFUL_METHODS = frozenset(
    {"append", "consume", "incrementcounter", "decrementcounter"}
)
# Vector, matrix and scalar constructors: DXC evaluates their operands from left to right.
_CONSTRUCTOR = re.compile(
    r"^(?:[biuhd]?vec[234]|d?mat[234](?:x[234])?|"
    r"(?:bool|int|uint|float|half|double|min16float|min16int|min16uint|"
    r"int16_t|uint16_t|int64_t|uint64_t|float16_t|float64_t)(?:[1-4](?:x[1-4])?)?)$"
)
# A type a function-style cast can name: no array extents, pointers or references.
_CAST_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*<[A-Za-z0-9_,\s<>]*>)?")
_FIXED_EXTENTS = re.compile(r"(?:\[\s*[0-9]+\s*\])+")
_CONSTANT_UNARY = frozenset({"-", "+", "!", "~"})
_CONSTANT_BINARY = frozenset(
    {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}
    | {"&&", "||", "==", "!=", "<", ">", "<=", ">="}
)
_OPAQUE_PREFIXES = (
    "Texture",
    "RWTexture",
    "FeedbackTexture",
    "RasterizerOrdered",
    "SamplerState",
    "SamplerComparisonState",
    "Buffer",
    "RWBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
    "ByteAddressBuffer",
    "RWByteAddressBuffer",
    "AppendStructuredBuffer",
    "ConsumeStructuredBuffer",
    "ConstantBuffer",
    "RaytracingAccelerationStructure",
    "RayQuery",
)


def _builtin_reference_positions(name, count):
    """Argument positions of the built-in ``name`` that name memory or receive a result, not values."""
    lowered = (name or "").lower()
    if lowered.startswith("atomic") and "compareexchange" in lowered.replace("_", ""):
        # The object and the expected value are both memory the operation reads and writes.
        return {0, 1}
    if lowered.startswith(("atomic", "interlocked")):
        return (
            {0, count - 1} if lowered.startswith("interlocked") and count > 2 else {0}
        )
    if lowered in ("modf", "frexp"):
        return {1}
    if lowered in ("uaddcarry", "usubborrow"):
        return {2}
    if lowered in ("umulextended", "imulextended"):
        return {2, 3}
    if lowered == "sincos":
        return {1, 2}
    if lowered.startswith("interpolateat"):
        return {0}
    return set()


def _builtin_result_type(name):
    """The result type of a built-in the generator cannot type on its own, or None."""
    lowered = (name or "").lower().replace("_", "")
    if lowered.startswith("atomic") and "compareexchange" in lowered:
        return "bool"
    return None


# Built-ins whose result has the type their arguments share (HLSL intrinsic functions reference, GLSL 4.60 section
# 8.3): fmod(float2, float2) is a float2.
_SAME_TYPE_BUILTINS = frozenset(
    {
        "atan2",
        "clamp",
        "fma",
        "fmod",
        "lerp",
        "mad",
        "max",
        "min",
        "mix",
        "mod",
        "pow",
        "smoothstep",
        "step",
    }
)


def _shared_argument_type(sequencer, name, arguments):
    """The type every argument of a same-type built-in has, or None."""
    if (name or "").lower() not in _SAME_TYPE_BUILTINS:
        return None
    types = [sequencer.source_type(argument) for argument in arguments]
    if any(source_type is None for source_type in types):
        return None
    mapped = {str(sequencer.codegen.map_type(source_type)) for source_type in types}
    return types[0] if len(mapped) == 1 else None


class _Effects:
    """Side effects of expressions and of the program's functions, for one HLSL generator."""

    def __init__(self, codegen):
        self.codegen = codegen
        self.functions = getattr(codegen, "hlsl_function_overloads_by_name", None) or {}
        self.cache = codegen.__dict__.setdefault(_EFFECTS, {})

    def call(self, call, visiting=None):
        name = _callee_name(call)
        if name is None:
            return True
        if isinstance(getattr(call, "function", None), MemberAccessNode):
            if name.lower() in _EFFECTFUL_METHODS:
                return True
            if name not in self.functions:
                return is_effectful_builtin(name)
        return self.function(name, visiting)

    def function(self, name, visiting=None):
        if name in self.cache:
            return self.cache[name]
        overloads = self.functions.get(name)
        if not overloads:
            return is_effectful_builtin(name)
        visiting = set() if visiting is None else visiting
        if name in visiting:
            return True
        visiting.add(name)
        result = any(self._body(function, visiting) for function in overloads)
        visiting.discard(name)
        self.cache[name] = result
        return result

    def _body(self, function, visiting):
        parameters = list(getattr(function, "parameters", None) or [])
        if any(_parameter_writes_caller(parameter) for parameter in parameters):
            return True
        body = getattr(function, "body", None)
        nodes = list(_nodes(body if not isinstance(body, list) else BlockNode(body)))
        locals_ = {getattr(parameter, "name", None) for parameter in parameters}
        locals_.update(node.name for node in nodes if isinstance(node, VariableNode))
        # A local that shares a module-scope name may be out of scope where it is written.
        locals_ -= set(getattr(self.codegen, "global_variable_types", None) or {})
        for node in nodes:
            if (
                isinstance(node, AssignmentNode)
                and _root_name(node.target) not in locals_
            ):
                return True
            if (
                isinstance(node, UnaryOpNode)
                and node.operator in ("++", "--")
                and _root_name(node.operand) not in locals_
            ):
                return True
            if isinstance(node, IdentifierNode) and node.name == "discard":
                return True
            if isinstance(node, FunctionCallNode) and self.call(node, visiting):
                return True
        return False

    def writes(self, expr):
        """The variables ``expr`` can write, by root name, or None when it can write memory it does not name."""
        written = set()
        for node in _nodes(expr):
            if isinstance(node, AssignmentNode):
                roots = {_root_name(node.target)}
            elif isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
                roots = {_root_name(node.operand)}
            elif isinstance(node, IdentifierNode) and node.name == "discard":
                return None
            elif isinstance(node, FunctionCallNode):
                roots = self._call_writes(node)
            else:
                continue
            if roots is None or None in roots or not all(map(self._named, roots)):
                return None
            written |= roots
        return written

    def _named(self, root):
        """Whether writing the variable ``root`` changes nothing but that variable: a value, not a view."""
        codegen = self.codegen
        if root in (getattr(codegen, "global_variable_types", None) or {}):
            # A function another argument calls can read a module-scope variable.
            return False
        local_type = (getattr(codegen, "local_variable_types", None) or {}).get(root)
        if local_type is None:
            return False
        return not _view_type(local_type)

    def _call_writes(self, call):
        if not self.call(call):
            return set()
        name = _callee_name(call)
        if (
            name is None
            or isinstance(getattr(call, "function", None), MemberAccessNode)
            or name not in self.functions
            or self._writes_beyond_parameters(name)
        ):
            return None
        arguments = list(getattr(call, "arguments", None) or [])
        roots = set()
        for function in self.functions[name]:
            parameters = getattr(
                function, "parameters", getattr(function, "params", None)
            )
            for index, parameter in enumerate(parameters or []):
                if index < len(arguments) and _parameter_writes_caller(parameter):
                    roots.add(_root_name(arguments[index]))
        return roots

    def _writes_beyond_parameters(self, name, visiting=None):
        """Whether a function of the program can write anything but its locals and write-back parameters."""
        key = ("beyond", name)
        if key in self.cache:
            return self.cache[key]
        visiting = set() if visiting is None else visiting
        if name in visiting:
            return True
        visiting.add(name)
        result = any(
            self._body_writes_beyond(function, visiting)
            for function in self.functions.get(name) or ()
        )
        visiting.discard(name)
        self.cache[key] = result
        return result

    def _body_writes_beyond(self, function, visiting):
        parameters = list(getattr(function, "parameters", None) or [])
        if any(
            _view_type(getattr(parameter, "param_type", None))
            for parameter in parameters
        ):
            return True
        body = getattr(function, "body", None)
        nodes = list(_nodes(body if not isinstance(body, list) else BlockNode(body)))
        locals_ = {getattr(parameter, "name", None) for parameter in parameters}
        locals_.update(node.name for node in nodes if isinstance(node, VariableNode))
        locals_ -= set(getattr(self.codegen, "global_variable_types", None) or {})
        for node in nodes:
            if isinstance(node, AssignmentNode):
                roots = {_root_name(node.target)}
            elif isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
                roots = {_root_name(node.operand)}
            elif isinstance(node, IdentifierNode) and node.name == "discard":
                return True
            elif isinstance(node, FunctionCallNode):
                callee = _callee_name(node)
                if not self.call(node, visiting):
                    continue
                if (
                    callee is None
                    or isinstance(getattr(node, "function", None), MemberAccessNode)
                    or callee not in self.functions
                    or self._writes_beyond_parameters(callee, visiting)
                ):
                    return True
                roots = self._call_writes(node)
                if roots is None:
                    return True
            else:
                continue
            if not roots <= locals_:
                return True
        return False

    def expression(self, expr):
        for node in _nodes(expr):
            if isinstance(node, AssignmentNode):
                return True
            if isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
                return True
            if isinstance(node, IdentifierNode) and node.name == "discard":
                return True
            if isinstance(node, FunctionCallNode) and self.call(node):
                return True
        return False


def _view_type(type_node):
    """Whether a value of this type refers to other storage (a pointer, reference or resource)."""
    if type_node is None:
        return False
    if isinstance(type_node, (PointerType, ReferenceType)):
        return True
    spelled = str(type_node)
    return "*" in spelled or "&" in spelled or spelled.startswith(_OPAQUE_PREFIXES)


def _reads_written(codegen, expr, written):
    """Whether ``expr`` can read one of the variables ``written`` (or anything, when it is None)."""
    if written is None:
        return True
    local_types = getattr(codegen, "local_variable_types", None) or {}
    for node in _nodes(expr):
        if isinstance(node, IdentifierNode):
            if node.name in written:
                return True
            # A pointer or reference may name a written variable under another name.
            if written and _view_type(local_types.get(node.name)):
                return True
        if (
            written
            and isinstance(node, UnaryOpNode)
            and node.operator == "*"
            and not node.is_postfix
        ):
            return True
    return False


def _address_operand(expr):
    """The operand whose address ``&operand`` takes, or None."""
    if isinstance(expr, UnaryOpNode) and expr.operator == "&" and not expr.is_postfix:
        return expr.operand
    return None


def _memory_lvalue(codegen, expr):
    """``expr`` as an l-value of the storage it names (a typed buffer load becomes an index), or None."""
    normalize = getattr(codegen, "hlsl_typed_buffer_atomic_lvalue", None)
    target = normalize(expr) if normalize is not None else expr
    return target if _is_lvalue(target) else None


def _is_lvalue(expr):
    return _root_name(expr) is not None and isinstance(
        expr, (IdentifierNode, MemberAccessNode, ArrayAccessNode, SwizzleNode)
    )


def _opaque(codegen, expr):
    try:
        source_type = codegen.expression_result_type(expr)
    except Exception:  # noqa: BLE001 - an untyped expression is simply not opaque
        return False
    if source_type is None:
        return False
    mapped = str(codegen.map_type(codegen.resource_base_type(source_type)))
    return mapped.startswith(_OPAQUE_PREFIXES)


def _constant(expr):
    """Whether ``expr`` is built from literals by operators and constructors only, so it reads no storage."""
    if isinstance(expr, LiteralNode):
        return True
    if isinstance(expr, UnaryOpNode):
        return (
            not expr.is_postfix
            and expr.operator in _CONSTANT_UNARY
            and _constant(expr.operand)
        )
    if isinstance(expr, BinaryOpNode):
        return (
            expr.operator in _CONSTANT_BINARY
            and _constant(expr.left)
            and _constant(expr.right)
        )
    if isinstance(expr, FunctionCallNode):
        name = _callee_name(expr)
        return (
            name is not None
            and not isinstance(expr.function, MemberAccessNode)
            and _CONSTRUCTOR.match(name) is not None
            and not getattr(expr, "generic_args", None)
            and all(_constant(argument) for argument in expr.arguments)
        )
    return False


def _stable(codegen, expr):
    """Whether no side effect can change the value ``expr`` evaluates to."""
    if _constant(expr):
        return True
    if codegen.literal_int_value(expr, {}) is not None:
        return True
    return _opaque(codegen, expr)


def _fixed_location(codegen, expr):
    """Whether the l-value ``expr`` names one location whatever runs first: no index other than a constant."""
    if isinstance(expr, (IdentifierNode, VariableNode)):
        return True
    if isinstance(expr, MemberAccessNode):
        return _fixed_location(codegen, expr.object_expr)
    if isinstance(expr, SwizzleNode):
        return _fixed_location(codegen, expr.vector_expr)
    if isinstance(expr, ArrayAccessNode):
        return codegen.literal_int_value(
            expr.index, {}
        ) is not None and _fixed_location(codegen, expr.array)
    return False


def _written_roots(arguments):
    """Root names of the variables the arguments assign, increment or decrement directly."""
    roots = set()
    for argument in arguments:
        for node in _nodes(argument):
            if isinstance(node, AssignmentNode):
                roots.add(_root_name(node.target))
            elif isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
                roots.add(_root_name(node.operand))
    roots.discard(None)
    return roots


def _callee_parameters(codegen, call, name, arguments):
    if isinstance(getattr(call, "function", None), MemberAccessNode):
        return None
    if name not in (getattr(codegen, "hlsl_function_overloads_by_name", None) or {}):
        return None
    try:
        function = codegen.resolve_hlsl_function_overload(
            name, arguments, call_node=call
        )
    except Exception:  # noqa: BLE001 - generation reports the overload error itself
        return None
    if function is None:
        return None
    return list(getattr(function, "parameters", getattr(function, "params", [])) or [])


def _special_parameter_indices(codegen, name):
    indices = set()
    for table in (
        getattr(codegen, "function_hlsl_workgroup_pointer_parameter_indices", {}),
        getattr(codegen, "function_private_pointer_parameter_indices", {}),
    ):
        indices.update((table or {}).get(name, {}) or ())
    for method in (
        "hlsl_resource_pointer_parameter_indices_for_function",
        "hlsl_resource_pointer_array_parameter_indices_for_function",
    ):
        lookup = getattr(codegen, method, None)
        if lookup is not None:
            indices.update(lookup(name) or ())
    return indices


def _ordered_argument(
    codegen, sequencer, name, index, arguments, parameters, special, memory
):
    """The replacement for one argument evaluated at its turn, None to keep it, or a refusal reason."""
    argument = arguments[index]
    address = _address_operand(argument)
    if special:
        # A pointer or resource view argument names memory; the callee reads that memory after
        # every argument is evaluated, so only the location itself has to be fixed at its turn.
        target = address if address is not None else argument
        others = arguments[:index] + arguments[index + 1 :]
        if (
            _is_lvalue(target)
            and _fixed_location(codegen, target)
            and _root_name(target) not in _written_roots(others)
        ):
            return None
        target = _memory_lvalue(codegen, address) if address is not None else None
        if target is not None:
            return UnaryOpNode("&", sequencer.location(target))
        return (
            f"argument {index + 1} of '{name}' is a pointer or resource view that cannot be "
            "evaluated ahead of the call"
        )
    if parameters is None and (address is not None or memory):
        # A built-in's memory argument (an atomic's object, a result it writes) keeps its place
        # with its indices fixed at its turn; copying its value would detach it from the memory.
        target = _memory_lvalue(codegen, address if address is not None else argument)
        if target is None:
            return (
                f"argument {index + 1} of '{name}' names memory through a form whose location "
                "cannot be fixed ahead of the call"
            )
        stable = sequencer.location(target)
        return UnaryOpNode("&", stable) if address is not None else stable
    qualifiers = set()
    if parameters is not None and index < len(parameters):
        qualifiers = set(codegen.hlsl_parameter_qualifiers(parameters[index]))
    if qualifiers & {"out", "inout"}:
        replacement = sequencer.reference(argument, copy_in="inout" in qualifiers)
    else:
        replacement = sequencer.value(argument)
    if replacement is None:
        return (
            f"argument {index + 1} of '{name}' has no type HLSL can hold in a temporary"
        )
    return replacement


class _Sequencer:
    """Builds the comma expression that evaluates one call's arguments in order."""

    def __init__(self, codegen, call):
        self.codegen = codegen
        self.call = call
        self.steps = []
        self.copy_backs = []

    def refuse(self, detail):
        location = getattr(self.call, "source_location", None)
        where = f" at {location}" if location else ""
        return self.codegen.refuse_lowering(
            "HLSL call argument order: DXC evaluates call arguments from right to left, and "
            f"{detail}{where}",
            placeholder="0",
        )

    def temporary(self, source_type):
        codegen = self.codegen
        mapped = str(
            codegen.map_type(codegen.type_name_string(source_type) or source_type)
        )
        base, suffix = split_array_type_suffix(mapped)
        # A temporary needs a complete type: no unsized or unresolved extent.
        if (
            not base
            or base == "void"
            or (suffix and not _FIXED_EXTENTS.fullmatch(suffix))
        ):
            return None
        used = codegen.current_identifier_reserved_names | set(
            codegen.local_variable_types
        )
        used.update(codegen.hlsl_wide_integer_reserved_names)
        used.update(codegen.global_variable_types)
        number = len(codegen.hlsl_wide_integer_compound_indices)
        name = f"__crossgl_argument{number}"
        while name in used or name in codegen.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES:
            number += 1
            name = f"__crossgl_argument{number}"
        codegen.current_identifier_reserved_names.add(name)
        codegen.local_variable_types[name] = source_type
        # The function's declaration list writes ``type name;``, so an array's extents follow the name.
        codegen.hlsl_wide_integer_compound_indices.append((base, f"{name}{suffix}"))
        return name

    def source_type(self, expr):
        try:
            source_type = self.codegen.expression_result_type(expr)
        except Exception:  # noqa: BLE001 - reported as an unknown type by the caller
            return None
        if source_type is None or not self.codegen.type_name_string(source_type):
            return None
        return source_type

    def value(self, argument):
        source_type = self.source_type(argument)
        name = self.temporary(source_type) if source_type is not None else None
        if name is None:
            return None
        rendered = self.codegen.generate_expression_with_expected(argument, None)
        self.steps.append(f"{name} = {rendered}")
        return IdentifierNode(name)

    def location(self, argument):
        """Fix the indices of the l-value ``argument`` now; return the stable l-value."""
        assignments, stable = self.codegen.hlsl_stabilize_inout_lvalue(
            self.call,
            argument,
            None,
            operation="reference argument",
            reason="reference-unstable-lvalue",
        )
        self.steps.extend(assignments)
        return stable

    def reference(self, argument, copy_in):
        stable = self.location(argument)
        source_type = self.source_type(stable)
        name = self.temporary(source_type) if source_type is not None else None
        if name is None:
            return None
        if copy_in:
            rendered = self.codegen.generate_expression_with_expected(stable, None)
            self.steps.append(f"{name} = {rendered}")
        # The generator's assignment lowering writes buffer elements and other lowered storage.
        self.copy_backs.append(
            self.codegen.generate_expression(
                AssignmentNode(copy.deepcopy(stable), IdentifierNode(name))
            )
        )
        return IdentifierNode(name)


def _needs_order(codegen, effects, arguments):
    """The index of the last side effect whose order against another argument matters, or None.

    The order matters between an argument with a side effect and another argument that is not a
    constant and can read (or write) what that side effect writes. Arguments after the last such
    side effect read their operands after it whichever order DXC takes, so they stay in place.
    """
    last = None
    for index, argument in enumerate(arguments):
        if not effects.expression(argument):
            continue
        written = effects.writes(argument)
        if any(
            other_index != index
            and not _stable(codegen, other)
            and _reads_written(codegen, other, written)
            for other_index, other in enumerate(arguments)
        ):
            last = index
    return last


def _render_unordered(codegen, expr, active):
    active.add(id(expr))
    try:
        return codegen.generate_expression(expr)
    finally:
        active.discard(id(expr))


def hlsl_ordered_call_expression(codegen, expr):
    """The HLSL for call ``expr`` with its arguments evaluated in source order, or None to emit it as is."""
    if not isinstance(expr, FunctionCallNode):
        return None
    arguments = list(getattr(expr, "arguments", None) or [])
    if len(arguments) < 2 or codegen.hlsl_wide_integer_compound_indices is None:
        return None
    active = codegen.__dict__.setdefault(_ACTIVE, set())
    if id(expr) in active:
        return None
    name = _callee_name(expr)
    if name is None or _CONSTRUCTOR.match(name):
        return None
    effects = _Effects(codegen)
    last = _needs_order(codegen, effects, arguments)
    if last is None:
        return None

    parameters = _callee_parameters(codegen, expr, name, arguments)
    special = _special_parameter_indices(codegen, name) if parameters else set()
    builtin_references = (
        set()
        if parameters is not None
        else _builtin_reference_positions(name, len(arguments))
    )
    sequencer = _Sequencer(codegen, expr)
    ordered = list(arguments)
    for index, argument in enumerate(arguments[: last + 1]):
        if _stable(codegen, argument):
            continue
        replacement = _ordered_argument(
            codegen,
            sequencer,
            name,
            index,
            arguments,
            parameters,
            index in special,
            index in builtin_references,
        )
        if isinstance(replacement, str):
            # The generator's own diagnostic for the call is more specific; let it raise first.
            _render_unordered(codegen, expr, active)
            return sequencer.refuse(replacement)
        if replacement is not None:
            ordered[index] = replacement

    original = expr.arguments, getattr(expr, "args", None)
    expr.arguments = expr.args = ordered
    try:
        rendered_call = _render_unordered(codegen, expr, active)
    finally:
        expr.arguments, expr.args = original
    steps = list(sequencer.steps)
    result_type = sequencer.source_type(expr)
    if result_type is None and parameters is None:
        result_type = _builtin_result_type(name) or _shared_argument_type(
            sequencer, name, arguments
        )
    mapped = str(codegen.map_type(result_type)) if result_type is not None else "void"
    if sequencer.copy_backs:
        if mapped == "void":
            steps.append(rendered_call)
            steps.extend(sequencer.copy_backs)
        else:
            result = sequencer.temporary(result_type)
            if result is None:
                return sequencer.refuse(
                    f"the result of '{name}' has no type HLSL can hold in a temporary"
                )
            steps.append(f"{result} = {rendered_call}")
            steps.extend(sequencer.copy_backs)
            steps.append(result)
    else:
        steps.append(rendered_call)
    sequence = "(" + ", ".join(steps) + ")"
    discarded = expr is getattr(codegen, "hlsl_statement_expression", None)
    if mapped != "void" and not discarded and _CAST_TYPE.fullmatch(mapped):
        # DXC flags a parenthesized comma expression that initializes a variable, a parameter
        # or a return value (-Wcomma-in-init, an error under -WX), but not the operand of a
        # function-style cast to its own type. A statement's discarded value stays bare, since
        # a discarded cast is an unused value (-Wunused-value).
        return f"{mapped}({sequence})"
    return sequence
