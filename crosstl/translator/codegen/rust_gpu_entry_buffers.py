"""Storage buffers in their other CrossGL spellings, as the declarations the rust-gpu output binds.

The rust-gpu output (:mod:`.rust_gpu_interface`) binds a storage buffer declared as a buffer pointer
(``buffer float* data;``, ``readonly buffer float* src;``) as a ``#[spirv(storage_buffer)]`` slice parameter of the
entry point, and a uniform value (``uniform float scale;``) as a ``#[spirv(uniform)]`` reference. CrossGL spells the
same resources in other ways, and this pass rewrites each into that form before generation:

- a structured buffer of elements, ``RWStructuredBuffer<T> b`` or ``StructuredBuffer<T> b``, declared at module or
  stage scope, is ``buffer T* b`` (``readonly`` for ``StructuredBuffer``); indexing ``b[i]`` reads element ``i`` in
  both spellings. The Rust and SPIR-V importers write runtime-sized buffers this way;
- a constant reference parameter of a compute entry point (MSL ``constant Params& p [[buffer(0)]]``), which the Metal
  importer writes, is a uniform value ``uniform Params p``;
- a structured buffer parameter of a compute entry point (``void main(RWStructuredBuffer<float> out @ set(0)
  @ binding(0))``), and a WGSL-style bound parameter of an ``@ compute fn`` (``var<storage, read_write> out:
  array<f32>``, ``var<storage, read>``, ``var<uniform> scale: f32``), which the CUDA, HIP and OpenCL importers
  write, becomes a declaration of the same name in the entry's stage, or at module scope for an ``@ compute fn``,
  keeping its ``@ set``/``@ group`` and ``@ binding`` attributes.

On those storage buffers the element calls ``buffer_load(b, i)`` and ``buffer_store(b, i, v)`` (statement) are the
element read ``b[i]`` and the store ``b[i] = v``, as the GLSL output writes them; on any other operand they are left
for the generator, which refuses them.

A structured buffer whose element type is not a plain value type (a texture, another buffer) is left alone, so the
generator refuses it by name. A hoisted parameter whose name another declaration already uses is refused with the
reason recorded on the generator, since the body's references would then resolve to the wrong resource.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    NamedType,
    PointerType,
    ReferenceType,
    VariableNode,
)
from ..semantics.rewrite import child_slots, get_child, set_child
from .stage_utils import normalize_stage_name

__all__ = [
    "normalize_rust_gpu_buffers",
    "rewrite_rust_gpu_element_calls",
    "structured_buffer_parts",
]

_STRUCTURED_BUFFERS = {"RWStructuredBuffer": True, "StructuredBuffer": False}
_ACCESS_QUALIFIERS = ("readonly", "writeonly", "readwrite")


def structured_buffer_parts(var_type):
    """``(element type, writable)`` for ``RWStructuredBuffer<T>`` or ``StructuredBuffer<T>``, else ``None``."""
    if not isinstance(var_type, NamedType):
        return None
    writable = _STRUCTURED_BUFFERS.get(str(getattr(var_type, "name", "")))
    arguments = list(getattr(var_type, "generic_args", None) or [])
    if writable is None or len(arguments) != 1:
        return None
    element = arguments[0]
    if isinstance(element, NamedType) and getattr(element, "generic_args", None):
        return None
    return element, writable


def _buffer_declaration(name, element, writable, attributes, template=None):
    """A ``buffer T* name`` declaration, ``readonly`` when the buffer is not ``writable``."""
    pointer = PointerType(element)
    node = template if template is not None else VariableNode(name, pointer)
    node.var_type = pointer
    node.vtype = pointer
    qualifiers = [
        q
        for q in list(getattr(node, "qualifiers", None) or [])
        if str(q).lower() not in ("buffer", *_ACCESS_QUALIFIERS)
    ]
    if not writable:
        qualifiers.insert(0, "readonly")
    node.qualifiers = qualifiers + ["buffer"]
    node.attributes = list(attributes or [])
    return node


def _rewrite_declarations(generator, declarations) -> None:
    for node in declarations or []:
        if not isinstance(node, VariableNode):
            continue
        parts = structured_buffer_parts(getattr(node, "var_type", None))
        if parts is None or getattr(node, "initial_value", None) is not None:
            continue
        element, writable = parts
        # An explicit memory qualifier or @access attribute decides the access over the buffer type's default.
        access = generator.explicit_resource_access(node)
        if access is not None:
            writable = access != "readonly"
        _buffer_declaration(
            node.name, element, writable, node.attributes, template=node
        )


def _resource_qualifiers(parameter) -> list[str]:
    return [
        str(q).lower() for q in getattr(parameter, "resource_qualifiers", None) or []
    ]


def _hoisted_declaration(parameter):
    """The module or stage declaration of a bound compute-entry ``parameter``, or ``None`` to keep it."""
    parameter_type = getattr(parameter, "param_type", None)
    attributes = list(getattr(parameter, "attributes", None) or [])
    parts = structured_buffer_parts(parameter_type)
    if parts is not None:
        element, writable = parts
        return _buffer_declaration(parameter.name, element, writable, attributes)
    qualifiers = _resource_qualifiers(parameter)
    if "storage" in qualifiers:
        if not isinstance(parameter_type, ArrayType) or parameter_type.size is not None:
            return None
        writable = "read" not in qualifiers or "read_write" in qualifiers
        return _buffer_declaration(
            parameter.name, parameter_type.element_type, writable, attributes
        )
    if qualifiers == ["uniform"]:
        return _uniform_declaration(parameter.name, parameter_type, attributes)
    if _is_constant_reference(parameter):
        return _uniform_declaration(
            parameter.name, parameter_type.referenced_type, attributes
        )
    return None


def _is_constant_reference(parameter) -> bool:
    """Whether ``parameter`` is a read-only reference into constant memory (MSL ``constant T& p [[buffer(n)]]``)."""
    parameter_type = getattr(parameter, "param_type", None)
    if not isinstance(parameter_type, ReferenceType) or getattr(
        parameter_type, "is_mutable", False
    ):
        return False
    qualifiers = {str(q).lower() for q in getattr(parameter, "qualifiers", None) or []}
    return (
        str(getattr(parameter_type, "address_space", "") or "").lower() == "constant"
        or "constant" in qualifiers
    )


def _uniform_declaration(name, value_type, attributes):
    node = VariableNode(name, value_type)
    node.qualifiers = ["uniform"]
    node.attributes = attributes
    return node


def _is_compute_function(function) -> bool:
    names = list(getattr(function, "qualifiers", None) or [])
    names.append(getattr(function, "qualifier", None))
    names.extend(
        getattr(attribute, "name", None)
        for attribute in getattr(function, "attributes", None) or []
    )
    return any(name and normalize_stage_name(str(name)) == "compute" for name in names)


def _compute_entries(ast):
    """``(function, declarations list)`` for every compute entry point and the scope its resources join."""
    stages = getattr(ast, "stages", None) or {}
    for stage_type, stage in stages.items():
        entry = getattr(stage, "entry_point", None)
        stage_name = normalize_stage_name(str(stage_type).split(".")[-1].lower())
        if entry is not None and stage_name == "compute":
            if getattr(stage, "local_variables", None) is None:
                stage.local_variables = []
            yield entry, stage.local_variables
    if getattr(ast, "global_variables", None) is None:
        ast.global_variables = []
    for function in getattr(ast, "functions", None) or []:
        if _is_compute_function(function):
            _qualify_compute_function(function)
            yield function, ast.global_variables


def _qualify_compute_function(function) -> None:
    """Give an ``@ compute fn`` the ``compute`` stage qualifier the generator reads entry points by."""
    qualifiers = list(getattr(function, "qualifiers", None) or [])
    if not any(normalize_stage_name(str(q)) == "compute" for q in qualifiers):
        function.qualifiers = ["compute"] + qualifiers


def _declared_names(ast) -> dict:
    names = {}
    for node in getattr(ast, "global_variables", None) or []:
        names[getattr(node, "name", None)] = node
    for stage in (getattr(ast, "stages", None) or {}).values():
        for node in getattr(stage, "local_variables", None) or []:
            names[getattr(node, "name", None)] = node
    return names


def _same_resource(first, second) -> bool:
    return (
        repr(getattr(first, "var_type", None))
        == repr(getattr(second, "var_type", None))
        and list(map(str, getattr(first, "qualifiers", None) or []))
        == list(map(str, getattr(second, "qualifiers", None) or []))
        and repr(getattr(first, "attributes", None))
        == repr(getattr(second, "attributes", None))
    )


def normalize_rust_gpu_buffers(generator, ast) -> None:
    """Rewrite ``ast``'s structured buffers and bound compute-entry parameters as rust-gpu storage declarations."""
    _rewrite_declarations(generator, getattr(ast, "global_variables", None))
    for stage in (getattr(ast, "stages", None) or {}).values():
        _rewrite_declarations(generator, getattr(stage, "local_variables", None))
    declared = _declared_names(ast)
    for function, scope in list(_compute_entries(ast)):
        kept = []
        for parameter in getattr(function, "parameters", None) or []:
            declaration = _hoisted_declaration(parameter)
            if declaration is None:
                kept.append(parameter)
                continue
            existing = declared.get(parameter.name)
            if existing is not None:
                if not _same_resource(existing, declaration):
                    generator.refuse_lowering(
                        f"resource: compute entry parameter '{parameter.name}' of '{function.name}' has the name "
                        "of another declaration, which rust-gpu would bind in its place"
                    )
                continue
            scope.append(declaration)
            declared[parameter.name] = declaration
        function.parameters = kept
    rewrite_rust_gpu_element_calls(ast)


def _call_name(node):
    if not isinstance(node, FunctionCallNode):
        return None
    function = getattr(node, "function", None)
    return getattr(function, "name", function if isinstance(function, str) else None)


def _storage_operand(call, storage_names):
    arguments = list(getattr(call, "arguments", None) or [])
    if not arguments or not isinstance(arguments[0], IdentifierNode):
        return None
    return arguments if arguments[0].name in storage_names else None


def _element_access(arguments):
    return ArrayAccessNode(arguments[0], arguments[1])


def _rewrite_element_calls(node, storage_names) -> None:
    """Rewrite ``buffer_load``/``buffer_store`` on the buffers ``storage_names`` below ``node``, in place."""
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        name = _call_name(child)
        arguments = _storage_operand(child, storage_names) if name else None
        if name == "buffer_load" and arguments is not None and len(arguments) == 2:
            set_child(node, field, key, _element_access(arguments))
            child = get_child(node, field, key)
        elif name == "buffer_store" and arguments is not None and len(arguments) == 3:
            statement = isinstance(node, ExpressionStatementNode) or isinstance(
                getattr(node, field), list
            )
            if statement:
                set_child(
                    node,
                    field,
                    key,
                    AssignmentNode(_element_access(arguments), arguments[2]),
                )
                child = get_child(node, field, key)
        _rewrite_element_calls(child, storage_names)


def _storage_names(ast) -> set:
    scopes = [getattr(ast, "global_variables", None)]
    scopes += [
        getattr(stage, "local_variables", None)
        for stage in (getattr(ast, "stages", None) or {}).values()
    ]
    names = set()
    for declarations in scopes:
        for node in declarations or []:
            if isinstance(node, VariableNode) and isinstance(
                getattr(node, "var_type", None), PointerType
            ):
                if "buffer" in {
                    str(q).lower() for q in getattr(node, "qualifiers", None) or []
                }:
                    names.add(node.name)
    return names


def rewrite_rust_gpu_element_calls(ast) -> None:
    """``buffer_load``/``buffer_store`` on storage buffers as element accesses (see the module docstring)."""
    names = _storage_names(ast)
    if names:
        _rewrite_element_calls(ast, names)
