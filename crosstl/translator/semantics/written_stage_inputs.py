"""Entry-point inputs that the entry point writes, copied into locals for GLSL.

HLSL, MSL, WGSL and SPIR-V pass an entry point its stage inputs and built-in values as parameters, which the body
may assign like any other parameter: ``gid.x *= 2``, ``input.uv += offset``, ``++vertexId``. GLSL has no entry
parameters. A stage input becomes an ``in`` variable (``in vec2 uv;``) and a built-in value becomes the built-in
itself (``gl_GlobalInvocationID``, ``gl_FragCoord``), and GLSL forbids every write to either: glslang reports
``l-value required (can't modify shader input)``. The generator used to emit the write unchanged.

For the GLSL targets, each plain-value entry parameter that the body writes, increments or passes to an ``out`` or
``inout`` parameter is copied into a local at the top of the body, ``T <name>_copy = <name>;``, and the body
uses the local from then on. The parameter keeps its name, so the interface variable keeps the name the
neighbouring stage and the host link it by. The copy reads the input once, with the generator's usual
spelling of that parameter, so built-in conversions and flattened input structures keep working.

References inside a scope that declares a local of the same name are left alone, because they name that
local. Output parameters, resources, workgroup memory and other non-value parameters are left in place.
"""

from __future__ import annotations

from .. import ast as A
from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name
from .compute_entry_values import _is_plain_type, _unique
from .rewrite import child_slots, get_child, set_child
from .stage_varyings import stages

__all__ = ["TARGETS", "copy_written_stage_inputs"]

TARGETS = frozenset({"opengl", "glsl", "webgl", "gles", "essl"})

# Geometry and tessellation entry points take arrays of vertices that
# :mod:`.geometry_streams` and :mod:`.tessellation_patches` rewrite; their
# interfaces are handled there.
_STAGES = frozenset({"vertex", "fragment", "compute", "mesh", "task"})

_WRITING_UNARY_OPERATORS = frozenset({"++", "--", "&"})
_OUTPUT_QUALIFIERS = frozenset({"out", "inout"})
_NON_VALUE_QUALIFIERS = _OUTPUT_QUALIFIERS | frozenset(
    {
        "workgroup",
        "groupshared",
        "threadgroup",
        "shared",
        "uniform",
        "buffer",
        "device",
        "constant",
        "payload",
        "taskpayloadshared",
    }
)
_RESOURCE_ATTRIBUTES = frozenset(
    {
        "buffer",
        "texture",
        "sampler",
        "threadgroup",
        "register",
        "binding",
        "set",
        "group",
        "space",
        "push_constant",
        "uniform",
        "payload",
    }
)
# Built-in functions whose argument at these positions is an output.
_INTRINSIC_OUTPUT_ARGUMENTS = {
    "modf": (1,),
    "frexp": (1,),
    "sincos": (1, 2),
    "uaddCarry": (2,),
    "usubBorrow": (2,),
    "umulExtended": (2, 3),
    "imulExtended": (2, 3),
}


def _lower_names(values) -> set:
    return {str(getattr(value, "name", value)).lower() for value in values or []}


def _is_input_value(parameter, structs) -> bool:
    if _lower_names(getattr(parameter, "qualifiers", None)) & _NON_VALUE_QUALIFIERS:
        return False
    if _lower_names(getattr(parameter, "resource_qualifiers", None)) & (
        _NON_VALUE_QUALIFIERS
    ):
        return False
    if _lower_names(getattr(parameter, "attributes", None)) & _RESOURCE_ATTRIBUTES:
        return False
    return _is_plain_type(getattr(parameter, "param_type", None), structs)


def _function_stage(function):
    qualifiers = list(getattr(function, "qualifiers", None) or [])
    names = qualifiers[:1] + [
        getattr(attribute, "name", attribute)
        for attribute in getattr(function, "attributes", None) or []
    ]
    for name in names:
        stage = normalize_stage_name(name)
        if stage in STAGE_QUALIFIER_NAMES:
            return stage
    return None


def _entries(program):
    """``(function, functions_by_name, structs)`` for each stage entry point of ``program``."""
    functions = [
        function
        for function in getattr(program, "functions", None) or []
        if isinstance(function, A.FunctionNode)
    ]
    structs = {
        struct.name: struct
        for struct in getattr(program, "structs", None) or []
        if isinstance(struct, A.StructNode) and not getattr(struct, "is_cbuffer", False)
    }
    found = []
    for function in functions:
        if _function_stage(function) in _STAGES:
            found.append((function, functions, structs))
    for stage in stages(program):
        entry = getattr(stage, "entry_point", None)
        if not isinstance(entry, A.FunctionNode):
            continue
        if normalize_stage_name(getattr(stage, "stage", None)) not in _STAGES:
            continue
        local_structs = dict(structs)
        for struct in getattr(stage, "local_structs", None) or []:
            if isinstance(struct, A.StructNode):
                local_structs[struct.name] = struct
        local_functions = functions + [
            function
            for function in getattr(stage, "local_functions", None) or []
            if isinstance(function, A.FunctionNode)
        ]
        found.append((entry, local_functions, local_structs))
    return found


def _walk(node):
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if id(current) in seen or not isinstance(current, A.ASTNode):
            continue
        seen.add(id(current))
        yield current
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
        )


def _root_name(expression):
    while isinstance(expression, (A.MemberAccessNode, A.ArrayAccessNode)):
        expression = (
            expression.object_expr
            if isinstance(expression, A.MemberAccessNode)
            else expression.array_expr
        )
    return expression.name if isinstance(expression, A.IdentifierNode) else None


def _callee_name(call):
    callee = getattr(call, "function", None)
    if isinstance(callee, A.IdentifierNode):
        return callee.name
    return callee if isinstance(callee, str) else getattr(call, "name", None)


def _output_positions(name, functions_by_name) -> set:
    positions = set(_INTRINSIC_OUTPUT_ARGUMENTS.get(name, ()))
    for function in functions_by_name.get(name, ()):
        for index, parameter in enumerate(getattr(function, "parameters", None) or []):
            if _lower_names(getattr(parameter, "qualifiers", None)) & (
                _OUTPUT_QUALIFIERS
            ):
                positions.add(index)
    return positions


def _written_names(body, functions_by_name) -> set:
    written = set()
    for node in _walk(body):
        if isinstance(node, A.AssignmentNode):
            written.add(_root_name(node.target))
        elif (
            isinstance(node, A.UnaryOpNode)
            and str(node.operator) in _WRITING_UNARY_OPERATORS
        ):
            written.add(_root_name(node.operand))
        elif isinstance(node, A.FunctionCallNode):
            arguments = list(getattr(node, "arguments", None) or [])
            for index in _output_positions(_callee_name(node), functions_by_name):
                if index < len(arguments):
                    written.add(_root_name(arguments[index]))
    written.discard(None)
    return written


def _rename(node, renames) -> None:
    """Rename references in ``node``; a declaration hides its name for the rest of its scope."""
    for field, key in list(child_slots(node)):
        if not renames:
            return
        child = get_child(node, field, key)
        if isinstance(node, A.FunctionCallNode) and field == "function":
            continue
        if isinstance(child, A.IdentifierNode):
            if child.name in renames:
                replacement = A.IdentifierNode(renames[child.name])
                replacement.source_location = getattr(child, "source_location", None)
                set_child(node, field, key, replacement)
        elif isinstance(child, A.VariableNode):
            # The initializer still sees the outer name.
            _rename(child, dict(renames))
            renames.pop(child.name, None)
        elif isinstance(child, A.ASTNode):
            _rename(child, dict(renames))


def _declared_names(function) -> set:
    names = {p.name for p in getattr(function, "parameters", None) or []}
    names.update(
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.VariableNode)
    )
    return names


def _global_names(program) -> set:
    names = set()
    for field in ("global_variables", "constants", "functions", "structs", "cbuffers"):
        for node in getattr(program, field, None) or []:
            names.add(getattr(node, "name", None))
            for member in getattr(node, "members", None) or []:
                names.add(getattr(member, "name", None))
    for stage in stages(program):
        for field in ("local_variables", "local_functions", "local_structs"):
            for node in getattr(stage, field, None) or []:
                names.add(getattr(node, "name", None))
    return {name for name in names if isinstance(name, str)}


def _copy_entry(function, functions, structs, global_names) -> None:
    body = getattr(function, "body", None)
    statements = getattr(body, "statements", None)
    if not isinstance(statements, list):
        return
    functions_by_name = {}
    for candidate in functions:
        functions_by_name.setdefault(candidate.name, []).append(candidate)
    written = _written_names(body, functions_by_name)
    candidates = [
        parameter
        for parameter in getattr(function, "parameters", None) or []
        if parameter.name in written and _is_input_value(parameter, structs)
    ]
    if not candidates:
        return
    used = global_names | _declared_names(function)
    renames, copies = {}, []
    for parameter in candidates:
        local = _unique(f"{parameter.name}_copy", used)
        renames[parameter.name] = local
        copies.append(
            A.VariableNode(
                local, parameter.param_type, A.IdentifierNode(parameter.name)
            )
        )
    _rename(body, renames)
    statements[:0] = copies


def copy_written_stage_inputs(program, target) -> None:
    """Copy the entry-point inputs that ``program``'s entry points write into locals, for the GLSL ``target``\\s."""
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return
    entries = _entries(program)
    if not entries:
        return
    global_names = _global_names(program)
    for function, functions, structs in entries:
        _copy_entry(function, functions, structs, global_names)
