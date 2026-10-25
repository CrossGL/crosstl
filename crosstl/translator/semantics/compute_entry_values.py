"""Plain-value parameters of ``@ compute fn`` entries, passed as one constant block.

A CUDA, HIP or OpenCL kernel, and a CrossGL ``@ compute fn`` entry written in their style, takes plain values
(``i32 n``) beside its buffers, and its host sets them for each dispatch. Compute entry points of the graphics
targets have no such parameters: an HLSL or Slang entry parameter needs a system-value semantic, an MSL kernel
parameter needs a ``[[buffer]]`` attribute, and a SPIR-V ``GLCompute`` entry has no user-defined inputs. The
generators used to emit the parameter anyway, which the target compilers reject, or for SPIR-V an ``Input``
variable the host cannot set.

For those targets this pass moves an entry's plain-value parameters, in order, into one constant block named
``<entry>_Args``, bound at the first binding its group leaves free. That is the block the OpenCL bridge lowering
already gives CUDA and OpenCL kernels on these targets, so a host binds the same block whichever way the kernel
arrived. A parameter the body writes becomes a local copy of the block member, since the block is read-only. A
parameter whose name a global declaration already uses is refused with :class:`ComputeEntryValueNameError`,
because the block member would collide with it.

Buffers, textures, samplers, workgroup variables, built-ins and parameters with attributes are left in place;
the generators already bind them.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name
from .half import _is_half_matrix
from .hub_types import type_from_name
from .opaque_structs import _is_opaque
from .rewrite import child_slots, get_child, set_child

__all__ = [
    "ComputeEntryValueNameError",
    "TARGETS",
    "lower_compute_entry_values",
]

TARGETS = frozenset(
    {
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
        "metal",
        "vulkan",
        "spirv",
        "slang",
    }
)

_WRITING_UNARY_OPERATORS = frozenset({"++", "--", "&"})


class ComputeEntryValueNameError(TranslationDiagnostic):
    """A plain-value entry parameter shares its name with a global declaration its constant block would hide."""

    code = CODE_PREFIX + "compute-entry-value-name"


def _is_compute_entry(function) -> bool:
    qualifiers = list(getattr(function, "qualifiers", None) or [])
    names = qualifiers[:1] + [
        getattr(attribute, "name", attribute)
        for attribute in getattr(function, "attributes", None) or []
    ]
    for name in names:
        stage = normalize_stage_name(name)
        if stage in STAGE_QUALIFIER_NAMES:
            return stage == "compute"
    return False


def _plain_struct_names(program) -> dict:
    return {
        struct.name: struct
        for struct in getattr(program, "structs", None) or []
        if isinstance(struct, A.StructNode) and not getattr(struct, "is_cbuffer", False)
    }


def _is_plain_type(type_node, structs, visiting=()) -> bool:
    """Whether a value of this type is plain data: numbers, vectors, matrices, sized arrays and structs of them."""
    if isinstance(type_node, (A.PrimitiveType, A.VectorType, A.MatrixType)):
        name = str(getattr(type_node, "name", "") or "")
        return name.lower() != "void" and not _is_opaque(type_node)
    if isinstance(type_node, A.ArrayType):
        return getattr(type_node, "size", None) is not None and _is_plain_type(
            type_node.element_type, structs, visiting
        )
    if isinstance(type_node, A.NamedType):
        name = getattr(type_node, "name", None)
        struct = structs.get(name)
        if struct is None and not getattr(type_node, "generic_args", None):
            # Spellings the parser keeps as names, such as ``f16vec3`` and ``f16mat3``.
            return type_from_name(name) is not None or _is_half_matrix(str(name))
        if (
            struct is None
            or getattr(type_node, "generic_args", None)
            or name in visiting
        ):
            return False
        return all(
            _is_plain_type(
                getattr(member, "member_type", None), structs, visiting + (name,)
            )
            for member in getattr(struct, "members", None) or []
        )
    return False


def _is_plain_value_parameter(parameter, structs) -> bool:
    if getattr(parameter, "attributes", None) or getattr(
        parameter, "resource_qualifiers", None
    ):
        return False
    qualifiers = {str(q).lower() for q in getattr(parameter, "qualifiers", None) or []}
    if qualifiers & {"out", "inout", "workgroup", "groupshared", "threadgroup"}:
        return False
    return _is_plain_type(getattr(parameter, "param_type", None), structs)


def _attribute_int(attribute):
    arguments = list(getattr(attribute, "arguments", None) or [])
    if not arguments:
        return None
    value = getattr(arguments[0], "value", getattr(arguments[0], "name", arguments[0]))
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return None


def _binding_of(node) -> tuple[int, int | None]:
    """``(group, binding)`` of a declaration's ``@ group`` / ``@ binding`` attributes."""
    group, binding = 0, None
    for attribute in getattr(node, "attributes", None) or []:
        name = str(getattr(attribute, "name", "")).lower()
        if name in {"group", "set", "space"}:
            group = _attribute_int(attribute) or 0
        elif name in {"binding", "register", "buffer"}:
            value = _attribute_int(attribute)
            if value is not None:
                binding = value
    return group, binding


def _taken_bindings(program) -> set:
    taken = set()
    declarations = list(getattr(program, "global_variables", None) or [])
    declarations += list(getattr(program, "cbuffers", None) or [])
    for function in getattr(program, "functions", None) or []:
        declarations += list(getattr(function, "parameters", None) or [])
    for declaration in declarations:
        group, binding = _binding_of(declaration)
        if binding is not None:
            taken.add((group, binding))
    return taken


def _entry_group(function) -> int:
    """The one descriptor group an entry's bound parameters share, else group 0."""
    groups = {
        _binding_of(parameter)[0]
        for parameter in getattr(function, "parameters", None) or []
        if _binding_of(parameter)[1] is not None
    }
    return groups.pop() if len(groups) == 1 else 0


def _global_names(program) -> set:
    names = set()
    for field in ("global_variables", "constants", "functions", "structs"):
        for node in getattr(program, field, None) or []:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    for cbuffer in getattr(program, "cbuffers", None) or []:
        names.add(getattr(cbuffer, "name", None))
        for member in getattr(cbuffer, "members", None) or []:
            names.add(getattr(member, "name", None))
    return {name for name in names if isinstance(name, str)}


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


def _written_names(body) -> set:
    written = set()
    for node in _walk(body):
        if isinstance(node, A.AssignmentNode):
            written.add(_root_name(node.target))
        elif (
            isinstance(node, A.UnaryOpNode)
            and str(node.operator) in _WRITING_UNARY_OPERATORS
        ):
            written.add(_root_name(node.operand))
    written.discard(None)
    return written


def _declared_names(function) -> set:
    names = {p.name for p in getattr(function, "parameters", None) or []}
    names.update(
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.VariableNode)
    )
    return names


def _rename_identifiers(body, renames) -> None:
    for node in list(_walk(body)):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode) and child.name in renames:
                replacement = A.IdentifierNode(renames[child.name])
                replacement.source_location = getattr(child, "source_location", None)
                set_child(node, field, key, replacement)


def _unique(base, used) -> str:
    candidate, suffix = base, 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _int_attribute(name, value):
    return A.AttributeNode(name, [A.LiteralNode(value, A.PrimitiveType("int"))])


def _lower_entry(program, function, values, taken, global_names, target) -> None:
    clashes = sorted(p.name for p in values if p.name in global_names)
    if clashes:
        raise ComputeEntryValueNameError(
            f"compute entry {function.name} passes {', '.join(clashes)} by value, but a global declaration "
            "has the same name; its constant block member would collide with it",
            target=target,
        )
    group = _entry_group(function)
    binding = 0
    while (group, binding) in taken:
        binding += 1
    taken.add((group, binding))
    used = global_names | _declared_names(function)
    attributes = [_int_attribute("binding", binding)]
    if group:
        attributes.insert(0, _int_attribute("group", group))
    block = A.StructNode(
        _unique(f"{function.name}_Args", used),
        [A.StructMemberNode(p.name, p.param_type) for p in values],
        attributes=attributes,
    )
    block.is_cbuffer = True
    program.cbuffers.append(block)
    global_names.update({block.name, *(p.name for p in values)})

    body = getattr(function, "body", None)
    written = _written_names(body)
    renames, copies = {}, []
    for parameter in values:
        if parameter.name not in written:
            continue
        local = _unique(f"{parameter.name}_value", used)
        renames[parameter.name] = local
        copies.append(
            A.VariableNode(
                local, parameter.param_type, A.IdentifierNode(parameter.name)
            )
        )
    if copies and isinstance(getattr(body, "statements", None), list):
        _rename_identifiers(body, renames)
        body.statements[:0] = copies
    function.parameters = [
        p for p in function.parameters if not any(p is value for value in values)
    ]


def lower_compute_entry_values(program, target) -> None:
    """Move the plain-value parameters of ``program``'s ``@ compute fn`` entries into constant blocks for ``target``."""
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return
    structs = _plain_struct_names(program)
    entries = []
    for function in getattr(program, "functions", None) or []:
        if not isinstance(function, A.FunctionNode) or not _is_compute_entry(function):
            continue
        values = [
            parameter
            for parameter in getattr(function, "parameters", None) or []
            if _is_plain_value_parameter(parameter, structs)
        ]
        if values:
            entries.append((function, values))
    if not entries:
        return
    taken = _taken_bindings(program)
    global_names = _global_names(program)
    for function, values in entries:
        _lower_entry(program, function, values, taken, global_names, target)
