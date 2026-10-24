"""Rename program identifiers that HLSL reads as keywords.

GLSL and Metal programs may name variables, parameters, functions, structs,
struct members, constant buffers or resources ``in``, ``out``, ``line``,
``point``, ``register``, ``vector``, ``Buffer`` and so on. DXC parses those
spellings as storage classes, parameter modifiers, primitive types, effect
framework tokens, reserved C++ words or built-in type names, so the emitted
declaration does not compile. Every declaration with such a name, and every
reference to it, is renamed on a copy of the AST before HLSL emission. The
rename appends underscores until the name is unused, which keeps the program's
own spellings recognisable.

Value names are the ones DXC 1.9 rejects in at least one of these positions:
local variable, struct member, function parameter, static global, function
name, ``cbuffer`` member and resource name; type names also include the ones it
rejects only as a struct or ``cbuffer`` name. Both were measured
compiling for ``cs_6_6`` with ``-enable-16bit-types``.
"""

from __future__ import annotations

import re
from copy import deepcopy
from enum import Enum

from ..ast import (
    ArrayType,
    AttributeNode,
    BufferNode,
    ConstantNode,
    EnumNode,
    EnumVariantNode,
    ForInNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LayoutQualifierNode,
    MatrixType,
    MemberAccessNode,
    NamedType,
    ParameterNode,
    PointerAccessNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    SamplerNode,
    StructMemberNode,
    StructNode,
    TextureResourceNode,
    VariableNode,
    VectorType,
)

# Storage classes, parameter and interpolation modifiers, and HLSL statements.
_HLSL_MODIFIER_NAMES = frozenset(
    {
        "centroid",
        "column_major",
        "const",
        "discard",
        "export",
        "extern",
        "groupshared",
        "in",
        "inline",
        "inout",
        "linear",
        "nointerpolation",
        "noperspective",
        "out",
        "packoffset",
        "register",
        "row_major",
        "shared",
        "snorm",
        "static",
        "uniform",
        "unorm",
        "volatile",
    }
)

# Geometry-shader primitive types.
_HLSL_PRIMITIVE_NAMES = frozenset(
    {"line", "lineadj", "point", "triangle", "triangleadj"}
)

# Effect-framework and buffer-declaration keywords.
_HLSL_EFFECT_NAMES = frozenset(
    {
        "cbuffer",
        "interface",
        "sampler_state",
        "string",
        "tbuffer",
        "technique",
        "technique10",
        "technique11",
    }
)

# C++ words that HLSL 2021 reserves.
_HLSL_RESERVED_WORDS = frozenset(
    {
        "asm",
        "auto",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "class",
        "const_cast",
        "continue",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "operator",
        "private",
        "protected",
        "public",
        "reinterpret_cast",
        "return",
        "short",
        "signed",
        "sizeof",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "throw",
        "true",
        "try",
        "typedef",
        "typename",
        "uint",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "while",
    }
)

# Built-in resource, ray-tracing, state-object, work-graph and packed types.
_HLSL_OBJECT_TYPE_NAMES = frozenset(
    {
        "AppendStructuredBuffer",
        "Buffer",
        "ConstantBuffer",
        "ConsumeStructuredBuffer",
        "DispatchNodeInputRecord",
        "FeedbackTexture2D",
        "FeedbackTexture2DArray",
        "GroupNodeInputRecords",
        "GroupNodeOutputRecords",
        "InputPatch",
        "int8_t4_packed",
        "LineStream",
        "matrix",
        "NodeOutput",
        "NodeOutputArray",
        "OutputPatch",
        "PointStream",
        "RasterizerOrderedBuffer",
        "RasterizerOrderedStructuredBuffer",
        "RasterizerOrderedTexture1D",
        "RasterizerOrderedTexture1DArray",
        "RasterizerOrderedTexture2D",
        "RasterizerOrderedTexture2DArray",
        "RasterizerOrderedTexture3D",
        "RayQuery",
        "RWBuffer",
        "RWDispatchNodeInputRecord",
        "RWGroupNodeInputRecords",
        "RWStructuredBuffer",
        "RWTexture1D",
        "RWTexture1DArray",
        "RWTexture2D",
        "RWTexture2DArray",
        "RWTexture2DMS",
        "RWTexture2DMSArray",
        "RWTexture3D",
        "RWThreadNodeInputRecord",
        "sampler",
        "StructuredBuffer",
        "Texture1D",
        "Texture1DArray",
        "Texture2D",
        "Texture2DArray",
        "Texture2DMS",
        "Texture2DMSArray",
        "Texture3D",
        "TextureBuffer",
        "TextureCube",
        "TextureCubeArray",
        "ThreadNodeInputRecord",
        "ThreadNodeOutputRecords",
        "TriangleStream",
        "uint8_t4_packed",
        "vector",
    }
)

# Scalar spellings whose vector (``T1``-``T4``) and matrix (``TNxM``) forms
# are also built-in type names. The 16-bit spellings are types under
# ``-enable-16bit-types``, which half-precision storage needs.
_HLSL_NUMERIC_BASE_NAMES = (
    "bool",
    "double",
    "dword",
    "float",
    "float16_t",
    "float32_t",
    "float64_t",
    "half",
    "int",
    "int16_t",
    "int32_t",
    "int64_t",
    "min10float",
    "min12int",
    "min16float",
    "min16int",
    "min16uint",
    "uint",
    "uint16_t",
    "uint32_t",
    "uint64_t",
)


# Spellings DXC accepts as variable, member, parameter, function and resource
# names but rejects as a struct or ``cbuffer`` name: modifiers that only
# qualify types or mesh-shader parameters, and built-in record, buffer, sampler
# and state-object types.
_HLSL_TYPE_ONLY_NAMES = frozenset(
    {
        "globallycoherent",
        "indices",
        "payload",
        "precise",
        "primitives",
        "sample",
        "vertices",
        "BuiltInTriangleIntersectionAttributes",
        "ByteAddressBuffer",
        "EmptyNodeInput",
        "EmptyNodeOutput",
        "EmptyNodeOutputArray",
        "GlobalRootSignature",
        "LocalRootSignature",
        "ProceduralPrimitiveHitGroup",
        "RasterizerOrderedByteAddressBuffer",
        "RayDesc",
        "RaytracingAccelerationStructure",
        "RaytracingPipelineConfig",
        "RaytracingPipelineConfig1",
        "RaytracingShaderConfig",
        "RWByteAddressBuffer",
        "SamplerComparisonState",
        "SamplerState",
        "StateObjectConfig",
        "SubobjectToExportsAssociation",
        "TriangleHitGroup",
    }
)


def _numeric_type_names():
    names = set()
    for base in _HLSL_NUMERIC_BASE_NAMES:
        names.add(base)
        for rows in range(1, 5):
            names.add(f"{base}{rows}")
            for columns in range(1, 5):
                names.add(f"{base}{rows}x{columns}")
    return frozenset(names)


HLSL_KEYWORD_NAMES = (
    _HLSL_MODIFIER_NAMES
    | _HLSL_PRIMITIVE_NAMES
    | _HLSL_EFFECT_NAMES
    | _HLSL_RESERVED_WORDS
    | _HLSL_OBJECT_TYPE_NAMES
    | _numeric_type_names()
)
HLSL_TYPE_KEYWORD_NAMES = HLSL_KEYWORD_NAMES | _HLSL_TYPE_ONLY_NAMES
# Built-in type spellings DXC accepts as a local variable, a value parameter or
# a struct member, and rejects only at program scope: as a global, a function,
# a ``cbuffer`` member or a resource.
HLSL_SCOPED_KEYWORD_NAMES = (
    _HLSL_OBJECT_TYPE_NAMES | _numeric_type_names() | {"string"}
) - _HLSL_RESERVED_WORDS

# Declarations that name values: variables, constants, resources, functions,
# parameters, struct members and enumerants.
_VALUE_DECLARATION_TYPES = (
    BufferNode,
    ConstantNode,
    EnumVariantNode,
    FunctionNode,
    ParameterNode,
    SamplerNode,
    StructMemberNode,
    TextureResourceNode,
    VariableNode,
)
# Declarations that name types; ``cbuffer`` blocks are struct nodes too.
_TYPE_DECLARATION_TYPES = (EnumNode, StructNode)
# Attribute and layout arguments are target vocabulary such as
# ``@interpolate(linear)``, never references to program declarations.
_OPAQUE_TYPES = (AttributeNode, LayoutQualifierNode)
_SKIPPED_FIELDS = frozenset({"parent", "annotations"})
_NAME_FIELDS = ("name", "identifier", "member", "pattern")


def _iter_nodes(root, *, include_opaque=False):
    """Yield every AST node reachable from ``root`` once."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, int, float, bool, Enum)):
            continue
        if isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
            continue
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
            continue
        if not hasattr(node, "__dict__") or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, _OPAQUE_TYPES) and not include_opaque:
            continue
        yield node
        for key, value in reversed(list(vars(node).items())):
            if key not in _SKIPPED_FIELDS:
                stack.append(value)


def _member_declarations(node):
    if isinstance(node, StructNode):
        return getattr(node, "members", None) or []
    if isinstance(node, EnumNode):
        return getattr(node, "variants", None) or []
    return []


_NUMERIC_SCALAR = re.compile(
    r"(bool|u?int(8|16|32|64)?(_t)?|float(16|32|64)?(_t)?|double|half|"
    r"u?short|u?char|u?long|min(10|16)float|min(12|16)u?int)"
)
# Function-scope declarations in these address spaces become program-scope
# HLSL declarations (``groupshared`` arrays, hoisted constant tables).
_PROGRAM_SCOPE_QUALIFIERS = frozenset(
    {"constant", "groupshared", "shared", "threadgroup", "workgroup", "static"}
)


def _is_value_type(type_node, struct_names):
    """Whether a parameter of this type stays a parameter in HLSL."""
    if isinstance(type_node, PrimitiveType):
        return bool(_NUMERIC_SCALAR.fullmatch(str(type_node.name)))
    if isinstance(type_node, (VectorType, MatrixType)):
        return True
    if isinstance(type_node, ArrayType):
        return _is_value_type(type_node.element_type, struct_names)
    if isinstance(type_node, NamedType):
        return not type_node.generic_args and type_node.name in struct_names
    if isinstance(type_node, (PointerType, ReferenceType)):
        return getattr(type_node, "address_space", None) in (None, "thread")
    return False


def _function_local_ids(ast):
    """Ids of variables declared in a function body at function scope."""
    local_ids = set()
    for node in _iter_nodes(ast):
        if not isinstance(node, FunctionNode):
            continue
        for inner in _iter_nodes(getattr(node, "body", None)):
            if isinstance(inner, VariableNode) and not (
                set(getattr(inner, "qualifiers", None) or ())
                & _PROGRAM_SCOPE_QUALIFIERS
            ):
                local_ids.add(id(inner))
    return local_ids


def _constant_buffers(ast):
    blocks = list(getattr(ast, "cbuffers", None) or [])
    stages = getattr(ast, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        blocks.extend(getattr(stage, "local_cbuffers", None) or [])
    return blocks


class _Declarations:
    """The names a program declares, split by namespace and scope."""

    def __init__(self, ast):
        self.values = set()
        self.members = set()
        self.types = set()
        # Value names declared at HLSL program scope.
        self.program_scope = set()
        struct_names = set()
        for node in _iter_nodes(ast):
            if isinstance(node, StructNode) and isinstance(node.name, str):
                struct_names.add(node.name)
        cbuffer_members = {
            id(member)
            for block in _constant_buffers(ast)
            for member in getattr(block, "members", None) or []
        }
        local_ids = _function_local_ids(ast)
        for node in _iter_nodes(ast):
            name = getattr(node, "name", None)
            if isinstance(node, _VALUE_DECLARATION_TYPES) and isinstance(name, str):
                self.values.add(name)
                if self._at_program_scope(
                    node, struct_names, cbuffer_members, local_ids
                ):
                    self.program_scope.add(name)
            elif isinstance(node, _TYPE_DECLARATION_TYPES) and isinstance(name, str):
                self.types.add(name)
            elif isinstance(node, ForInNode):
                pattern = getattr(node, "pattern", None)
                if isinstance(pattern, str):
                    self.values.add(pattern)
            for member in _member_declarations(node):
                member_name = getattr(member, "name", None)
                if isinstance(member_name, str):
                    self.members.add(member_name)

    @staticmethod
    def _at_program_scope(node, struct_names, cbuffer_members, local_ids):
        if isinstance(node, VariableNode):
            return id(node) not in local_ids
        if isinstance(node, ParameterNode):
            return not _is_value_type(getattr(node, "param_type", None), struct_names)
        if isinstance(node, StructMemberNode):
            return id(node) in cbuffer_members
        return True

    def keyword_values(self):
        names = (self.values | self.members) & HLSL_KEYWORD_NAMES
        return {
            name
            for name in names
            if name not in HLSL_SCOPED_KEYWORD_NAMES or name in self.program_scope
        }

    def keyword_types(self):
        return self.types & HLSL_TYPE_KEYWORD_NAMES


def hlsl_keyword_declarations(ast):
    """The value names and type names ``ast`` declares that HLSL reserves."""
    declarations = _Declarations(ast)
    return declarations.keyword_values(), declarations.keyword_types()


def _program_names(ast):
    names = set()
    for node in _iter_nodes(ast, include_opaque=True):
        for field in _NAME_FIELDS:
            value = getattr(node, field, None)
            if isinstance(value, str):
                names.add(value)
    return names


def _rename_map(declared, used):
    renames = {}
    for name in sorted(declared):
        candidate = f"{name}_"
        while candidate in used or candidate in HLSL_TYPE_KEYWORD_NAMES:
            candidate = f"{candidate}_"
        used.add(candidate)
        renames[name] = candidate
    return renames


def _callee_member_ids(ast):
    """Ids of member accesses that name a method, such as ``tex.sample``."""
    callees = set()
    for node in _iter_nodes(ast):
        if isinstance(node, FunctionCallNode):
            function = getattr(node, "function", None)
            if isinstance(function, (MemberAccessNode, PointerAccessNode)):
                callees.add(id(function))
    return callees


class _Renamer:
    def __init__(self, ast, declarations):
        used = _program_names(ast)
        self.values = _rename_map(declarations.keyword_values(), used)
        self.types = _rename_map(declarations.keyword_types(), used)
        self.members = {
            name: self.values[name]
            for name in declarations.members
            if name in self.values
        }
        # A struct constructor call names its type where a value would stand.
        self.identifiers = dict(self.values)
        for name, new_name in self.types.items():
            if name not in declarations.values and name not in declarations.members:
                self.identifiers.setdefault(name, new_name)
        self.callee_ids = _callee_member_ids(ast)

    def rename(self, node):
        name = getattr(node, "name", None)
        if isinstance(node, _VALUE_DECLARATION_TYPES):
            if name in self.values:
                node.name = self.values[name]
        elif isinstance(node, _TYPE_DECLARATION_TYPES):
            if name in self.types:
                node.name = self.types[name]
        elif isinstance(node, IdentifierNode):
            if name in self.identifiers:
                node.name = self.identifiers[name]
                node.identifier = node.name
        elif isinstance(node, NamedType):
            if name in self.types:
                node.name = self.types[name]
        elif isinstance(node, (MemberAccessNode, PointerAccessNode)):
            member = getattr(node, "member", None)
            if member in self.members and id(node) not in self.callee_ids:
                node.member = self.members[member]
        elif isinstance(node, FunctionCallNode):
            function = getattr(node, "function", None)
            if isinstance(function, str) and function in self.identifiers:
                node.function = self.identifiers[function]
                node.name = node.function
        elif isinstance(node, ForInNode):
            pattern = getattr(node, "pattern", None)
            if pattern in self.values:
                node.pattern = self.values[pattern]


def legalize_hlsl_keyword_names(ast):
    """Return ``ast``, or a renamed copy when it declares an HLSL keyword name."""
    if not any(hlsl_keyword_declarations(ast)):
        return ast
    ast = deepcopy(ast)
    renamer = _Renamer(ast, _Declarations(ast))
    for node in _iter_nodes(ast):
        renamer.rename(node)
    # Fixed-sampler records name module-scope samplers; they must follow the
    # rename, or the generator no longer finds the sampler and drops its record.
    annotations = getattr(ast, "annotations", None) or {}
    for record in annotations.get("fixed_samplers") or ():
        if record["name"] in renamer.values:
            record["name"] = renamer.values[record["name"]]
    return ast
