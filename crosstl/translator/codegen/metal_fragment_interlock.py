"""Fragment shader interlock critical sections in Metal output.

GLSL's GL_ARB_fragment_shader_interlock, and SPIR-V's SPV_EXT_fragment_shader_interlock that CrossTL imports as the
same two calls, order a fragment stage's memory accesses with a critical section: the accesses between
``beginInvocationInterlockARB()`` and ``endInvocationInterlockARB()`` of the fragments that cover one pixel run one
fragment at a time, in primitive order for the ``*_ordered`` modes. MSL has no critical section. It orders the
accesses that overlapping fragments make to a resource with the ``raster_order_group(index)`` attribute on the
fragment function's device buffer or texture argument (Metal Shading Language Specification, "Rasterization order
groups"), which CrossGL writes as ``@rasterizer_ordered`` (:mod:`.metal_raster_order_groups`).

This pass lowers the critical section to that attribute: every global resource, fragment-stage resource or fragment
entry parameter that the section reaches, directly or through the functions it calls, is marked
``@rasterizer_ordered``, and the two calls are removed. The marked resources share group 0, so their accesses are
ordered by primitive order across all of them, as the section orders them; an ``*_unordered`` mode and the
per-sample modes need less ordering than that, so they keep every result the source can produce. Accesses to a
marked resource outside the section become ordered as well, which only removes interleavings.

The extension places each call once, directly in the fragment ``main()`` and outside control flow. When the two
calls are not one such pair of sibling statements, the section is taken to be the whole entry function, so every
resource the fragment stage reaches is ordered. Calls outside the fragment stages stay in place and are refused as
built-ins Metal does not have.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    ArrayType,
    AttributeNode,
    BlockNode,
    ExpressionStatementNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    NamedType,
    ParameterNode,
    PointerType,
    ReferenceType,
    VariableNode,
)
from .metal_raster_order_groups import RASTERIZER_ORDERED
from .stage_utils import collect_stage_entry_records, normalize_stage_name

__all__ = [
    "BEGIN_INTERLOCK",
    "END_INTERLOCK",
    "canonicalize_rasterizer_ordered_types",
    "lower_metal_fragment_interlocks",
]

BEGIN_INTERLOCK = "beginInvocationInterlockARB"
END_INTERLOCK = "endInvocationInterlockARB"
_INTERLOCK_FUNCTIONS = frozenset({BEGIN_INTERLOCK, END_INTERLOCK})
_GLSL_BUFFER_BLOCK = "glsl_buffer_block"
_ORDERED_PREFIX = "RasterizerOrdered"
# HLSL's ordered texture views, after the prefix, and the CrossGL storage image each one orders. The buffer views
# become ``RW*`` views, which the byte-address and typed-buffer passes lower with their subscripts.
_ORDERED_IMAGE_VIEWS = {
    "Texture1D": "image1D",
    "Texture1DArray": "image1DArray",
    "Texture2D": "image2D",
    "Texture2DArray": "image2DArray",
    "Texture3D": "image3D",
}
# Entry parameter types a fragment function takes as a device buffer or texture argument.
_RESOURCE_TYPE_PREFIXES = (
    "buffer",
    "image",
    "iimage",
    "uimage",
    "texture",
    "rwbuffer",
    "rwstructuredbuffer",
    "rwtexture",
    "rasterizerordered",
)


def _call_name(node):
    if not isinstance(node, FunctionCallNode):
        return None
    function = getattr(node, "function", None)
    if isinstance(function, str):
        return function
    name = getattr(function, "name", None)
    return name if isinstance(name, str) else None


def _interlock_call_name(statement):
    """The interlock function ``statement`` calls on its own, or ``None``."""
    if not isinstance(statement, ExpressionStatementNode):
        return None
    name = _call_name(getattr(statement, "expression", None))
    return name if name in _INTERLOCK_FUNCTIONS else None


def _calls_interlock(root):
    return any(
        _call_name(node) in _INTERLOCK_FUNCTIONS
        for node in root.walk()
        if isinstance(node, FunctionCallNode)
    )


def _attribute_names(node):
    return {
        str(getattr(attribute, "name", attribute)).lower()
        for attribute in getattr(node, "attributes", None) or ()
    }


def _functions_by_name(ast):
    functions = {}
    candidates = list(getattr(ast, "functions", []) or [])
    for stage in (getattr(ast, "stages", {}) or {}).values():
        candidates.extend(getattr(stage, "local_functions", []) or [])
    for function in candidates:
        if isinstance(function, FunctionNode) and function.name:
            functions.setdefault(str(function.name), []).append(function)
    return functions


def _critical_section(entry):
    """The statements between the entry's sibling begin and end calls, or the whole body."""
    body = getattr(entry, "body", None)
    statements = getattr(body, "statements", None)
    if isinstance(statements, list):
        names = [_interlock_call_name(statement) for statement in statements]
        if names.count(BEGIN_INTERLOCK) == 1 and names.count(END_INTERLOCK) == 1:
            begin = names.index(BEGIN_INTERLOCK)
            end = names.index(END_INTERLOCK)
            if begin < end:
                return statements[begin + 1 : end]
    return [body] if body is not None else []


def _reached_names(roots, functions):
    """Every identifier ``roots`` and the user functions they call, transitively, name."""
    names = set()
    visited = set()
    pending = list(roots)
    while pending:
        root = pending.pop()
        for node in root.walk():
            if isinstance(node, IdentifierNode) and isinstance(node.name, str):
                names.add(node.name)
            name = _call_name(node)
            if name is None:
                continue
            names.add(name)
            for function in functions.get(name, ()):
                if id(function) not in visited and function.body is not None:
                    visited.add(id(function))
                    pending.append(function.body)
    return names


def _block_member_names(ast, declaration):
    """The members a GLSL buffer block declaration lets the program name without its instance name."""
    if _GLSL_BUFFER_BLOCK not in _attribute_names(declaration):
        return set()
    type_name = str(getattr(getattr(declaration, "var_type", None), "name", "") or "")
    structs = list(getattr(ast, "structs", []) or [])
    for stage in (getattr(ast, "stages", {}) or {}).values():
        structs.extend(getattr(stage, "local_structs", []) or [])
    for struct in structs:
        if str(getattr(struct, "name", "")) == type_name:
            return {
                str(member.name)
                for member in getattr(struct, "members", []) or []
                if getattr(member, "name", None)
            }
    return set()


def _is_resource_parameter(parameter):
    """Whether the entry ``parameter`` is a buffer or texture argument, which a raster order group can hold."""
    parameter_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
    if isinstance(parameter_type, (PointerType, ReferenceType)):
        return True
    type_name = str(getattr(parameter_type, "name", parameter_type) or "").lower()
    return type_name.startswith(_RESOURCE_TYPE_PREFIXES)


def _mark_ordered(declaration):
    if RASTERIZER_ORDERED in _attribute_names(declaration):
        return
    attributes = list(getattr(declaration, "attributes", None) or [])
    attributes.append(AttributeNode(RASTERIZER_ORDERED))
    declaration.attributes = attributes


def _remove_interlock_calls(root):
    for node in root.walk():
        if isinstance(node, BlockNode) and isinstance(node.statements, list):
            node.statements = [
                statement
                for statement in node.statements
                if _interlock_call_name(statement) is None
            ]


def lower_metal_fragment_interlocks(ast):
    """Return ``ast`` with each fragment interlock critical section written as raster-ordered resources.

    Returns ``ast`` itself when no fragment stage calls the interlock functions; the input AST is not modified.
    """
    if not _calls_interlock(ast):
        return ast
    ast = deepcopy(ast)
    functions = _functions_by_name(ast)
    declarations = list(getattr(ast, "global_variables", []) or [])
    for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
        if normalize_stage_name(stage_type) == "fragment":
            declarations.extend(getattr(stage, "local_variables", []) or [])
    for _entry_id, _stage, entry in collect_stage_entry_records(
        ast, None, {"fragment"}
    ):
        if entry.body is None:
            continue
        bodies = [entry.body] + [
            function.body
            for name in sorted(_reached_names([entry.body], functions))
            for function in functions.get(name, ())
            if function.body is not None
        ]
        if not any(_calls_interlock(body) for body in bodies):
            continue
        names = _reached_names(_critical_section(entry), functions)
        parameters = [
            parameter
            for parameter in entry.parameters or []
            if _is_resource_parameter(parameter)
        ]
        for declaration in declarations + parameters:
            declared = {str(getattr(declaration, "name", "") or "")}
            declared |= _block_member_names(ast, declaration)
            if declared & names:
                _mark_ordered(declaration)
        for body in bodies:
            _remove_interlock_calls(body)
    return ast


def _ordered_type(declaration):
    """The ``RasterizerOrdered*`` resource type ``declaration`` declares, or ``None``."""
    declared_type = getattr(
        declaration,
        "var_type" if isinstance(declaration, VariableNode) else "param_type",
        None,
    )
    while isinstance(declared_type, ArrayType):
        declared_type = declared_type.element_type
    if isinstance(declared_type, NamedType) and str(declared_type.name).startswith(
        _ORDERED_PREFIX
    ):
        return declared_type
    return None


def _image_prefix(texel_type):
    """``u``, ``i`` or nothing: the CrossGL image type prefix of a view whose texels are ``texel_type``."""
    name = str(getattr(texel_type, "name", texel_type) or "").lower()
    if name.startswith(("uint", "uvec", "u16vec", "u64vec")):
        return "u"
    if name.startswith(("int", "ivec", "i16vec", "i64vec")):
        return "i"
    return ""


def _canonicalize(node, ordered_type):
    """Declare ``node``'s ``RasterizerOrdered*`` view ``ordered_type`` as the CrossGL storage view it orders."""
    view = str(ordered_type.name)[len(_ORDERED_PREFIX) :]
    if view in _ORDERED_IMAGE_VIEWS:
        texel_type = (ordered_type.generic_args or [None])[0]
        ordered_type.name = _image_prefix(texel_type) + _ORDERED_IMAGE_VIEWS[view]
        ordered_type.generic_args = []
        attributes = list(getattr(node, "attributes", None) or [])
        if texel_type is not None:
            # The importers' storage-image spelling: ``@format(unknown) @texel(T) image2D``.
            attributes.append(AttributeNode("format", [IdentifierNode("unknown")]))
            texel_name = getattr(texel_type, "name", texel_type)
            attributes.append(AttributeNode("texel", [IdentifierNode(str(texel_name))]))
        node.attributes = attributes
    else:
        ordered_type.name = "RW" + view
    _mark_ordered(node)


def canonicalize_rasterizer_ordered_types(ast):
    """Return ``ast`` with each ``RasterizerOrdered*`` view declared as its storage view and ``@rasterizer_ordered``.

    CrossGL spells a rasterizer-ordered view either way: HLSL's ``RasterizerOrderedStructuredBuffer``,
    ``RasterizerOrderedByteAddressBuffer``, ``RasterizerOrderedBuffer`` and ``RasterizerOrderedTexture*`` are the
    ``RWStructuredBuffer``, ``RWByteAddressBuffer``, ``RWBuffer`` and ``image*`` views with ordered access. The
    Metal generator binds the storage views and writes the ordering as ``[[raster_order_group(0)]]``. Returns ``ast``
    itself when it declares no such view; the input is not modified.
    """
    if not any(
        isinstance(node, (VariableNode, ParameterNode)) and _ordered_type(node)
        for node in ast.walk()
    ):
        return ast
    ast = deepcopy(ast)
    for node in list(ast.walk()):
        if isinstance(node, (VariableNode, ParameterNode)):
            ordered_type = _ordered_type(node)
            if ordered_type is not None:
                _canonicalize(node, ordered_type)
    return ast
