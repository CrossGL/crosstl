"""Scalar and vector types of CrossGL expressions, inferred with lexical scopes.

Rule carrying needs operand types: whether ``a + b`` overflows like a signed 32-bit integer depends on the types of
``a`` and ``b``. This module infers them from declarations: globals, buffer-block members, constants, structure
members, parameters, block-scoped locals and loop variables. Inference is partial by design. It returns ``None``
when a type is not known, and the caller decides what an unknown type means for its rule.

Types are :class:`HubType` values (kind, bit width, lanes); aggregates (arrays, buffers, structures) are kept as the
AST's ``TypeNode`` objects and resolved on element and member access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import ast as A
from .rule_table import CONVERSION_SIGNATURES, CONVERT_SAT_GENERIC


@dataclass(frozen=True)
class HubType:
    """A scalar or vector value type: ``kind`` is ``i`` (signed), ``u`` (unsigned), ``f`` (float) or ``b`` (bool)."""

    kind: str
    bits: int
    lanes: int = 1

    @property
    def scalar(self) -> HubType:
        return HubType(self.kind, self.bits)

    def with_lanes(self, lanes: int) -> HubType:
        return HubType(self.kind, self.bits, lanes)


INT32 = HubType("i", 32)
UINT32 = HubType("u", 32)
FLOAT32 = HubType("f", 32)
BOOL = HubType("b", 1)

_SCALARS = {
    "int": INT32,
    "i32": INT32,
    "int32_t": INT32,
    "int32": INT32,
    "uint": UINT32,
    "u32": UINT32,
    "uint32_t": UINT32,
    "uint32": UINT32,
    "unsigned": UINT32,
    "float": FLOAT32,
    "f32": FLOAT32,
    "half": HubType("f", 16),
    "f16": HubType("f", 16),
    "float16_t": HubType("f", 16),
    "min16float": HubType("f", 16),
    "double": HubType("f", 64),
    "f64": HubType("f", 64),
    "bool": BOOL,
    "char": HubType("i", 8),
    "i8": HubType("i", 8),
    "int8": HubType("i", 8),
    "int8_t": HubType("i", 8),
    "uchar": HubType("u", 8),
    "u8": HubType("u", 8),
    "uint8": HubType("u", 8),
    "uint8_t": HubType("u", 8),
    "short": HubType("i", 16),
    "i16": HubType("i", 16),
    "int16": HubType("i", 16),
    "int16_t": HubType("i", 16),
    "min16int": HubType("i", 16),
    "ushort": HubType("u", 16),
    "u16": HubType("u", 16),
    "uint16": HubType("u", 16),
    "uint16_t": HubType("u", 16),
    "min16uint": HubType("u", 16),
    "long": HubType("i", 64),
    "i64": HubType("i", 64),
    "int64": HubType("i", 64),
    "int64_t": HubType("i", 64),
    "ulong": HubType("u", 64),
    "u64": HubType("u", 64),
    "uint64": HubType("u", 64),
    "uint64_t": HubType("u", 64),
}
_GLSL_VECTOR = re.compile(r"(ivec|uvec|vec|bvec|dvec|i64vec|u64vec|f16vec)([234])")
_GLSL_VECTOR_KIND = {
    "ivec": INT32,
    "uvec": UINT32,
    "vec": FLOAT32,
    "bvec": BOOL,
    "dvec": HubType("f", 64),
    "i64vec": HubType("i", 64),
    "u64vec": HubType("u", 64),
    "f16vec": HubType("f", 16),
}
_C_VECTOR = re.compile(r"([a-z0-9_]+?)([234])")
_SWIZZLE = re.compile(r"[xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4}")
_BUFFER_WRAPPERS = frozenset(
    {
        "StructuredBuffer",
        "RWStructuredBuffer",
        "RasterizerOrderedStructuredBuffer",
        "AppendStructuredBuffer",
        "ConsumeStructuredBuffer",
        "Buffer",
        "RWBuffer",
        "array",
        "device",
        "constant",
    }
)
# Built-in inputs the GLSL importer leaves undeclared.
_BUILTIN_VARIABLES = {
    "gl_GlobalInvocationID": UINT32.with_lanes(3),
    "gl_LocalInvocationID": UINT32.with_lanes(3),
    "gl_WorkGroupID": UINT32.with_lanes(3),
    "gl_WorkGroupSize": UINT32.with_lanes(3),
    "gl_NumWorkGroups": UINT32.with_lanes(3),
    "gl_LocalInvocationIndex": UINT32,
    "gl_SubgroupInvocationID": UINT32,
    "gl_SubgroupSize": UINT32,
    "gl_VertexIndex": INT32,
    "gl_InstanceIndex": INT32,
    "gl_VertexID": INT32,
    "gl_InstanceID": INT32,
    "gl_PrimitiveID": INT32,
    "gl_FragCoord": FLOAT32.with_lanes(4),
    "gl_Position": FLOAT32.with_lanes(4),
}
_BOOL_RESULT_CALLS = frozenset({"any", "all", "isnan", "isinf", "isfinite"})
_FLOAT_RESULT_CALLS = frozenset({"length", "distance"})
_SAME_AS_ARGUMENTS = frozenset(
    {
        "min",
        "max",
        "clamp",
        "abs",
        "sign",
        "mix",
        "lerp",
        "select",
        "floor",
        "ceil",
        "trunc",
        "round",
        "fract",
        "sqrt",
        "rsqrt",
        "inversesqrt",
        "exp",
        "exp2",
        "log",
        "log2",
        "pow",
        "sin",
        "cos",
        "tan",
        "atan",
        "atan2",
        "fmod",
        "mod",
        "saturate",
        "normalize",
        "fma",
        "mad",
        "reversebits",
        "bitfieldReverse",
        "roundEven",
    }
)
_CONVERSION_RESULTS = {
    name: {"int": INT32, "uint": UINT32}[result]
    for name, result in [
        *((n, signature[1]) for n, signature in CONVERSION_SIGNATURES.items()),
        *((n, generic[0]) for n, generic in CONVERT_SAT_GENERIC.items()),
    ]
}


def type_from_name(name) -> HubType | None:
    """The value type a type spelling denotes, or ``None`` for aggregates and unknown names."""
    name = str(name or "").replace(" ", "")
    if name in _SCALARS:
        return _SCALARS[name]
    match = _GLSL_VECTOR.fullmatch(name)
    if match:
        return _GLSL_VECTOR_KIND[match.group(1)].with_lanes(int(match.group(2)))
    match = re.fullmatch(r"vec([234])<(\w+)>", name)
    if match and match.group(2) in _SCALARS:
        return _SCALARS[match.group(2)].with_lanes(int(match.group(1)))
    match = _C_VECTOR.fullmatch(name)
    if match and match.group(1) in _SCALARS:
        return _SCALARS[match.group(1)].with_lanes(int(match.group(2)))
    return None


def type_from_node(node) -> HubType | None:
    """The value type of a ``TypeNode`` (or spelling); ``None`` for aggregates, matrices and unknown types."""
    if node is None:
        return None
    if isinstance(node, str):
        return type_from_name(node)
    if isinstance(node, A.PrimitiveType):
        return type_from_name(node.name)
    if isinstance(node, A.VectorType):
        element = type_from_node(node.element_type)
        size = node.size
        if element is None or not str(size).isdigit():
            return None
        return element.with_lanes(int(size))
    if isinstance(node, A.NamedType) and not node.generic_args:
        return type_from_name(node.name)
    if isinstance(node, A.ReferenceType):
        return type_from_node(node.referenced_type)
    return None


def element_type_node(node):
    """The element ``TypeNode`` of an array, buffer, pointer or reference type, or ``None``."""
    if isinstance(node, A.ArrayType):
        return node.element_type
    if isinstance(node, A.PointerType):
        return node.pointee_type
    if isinstance(node, A.ReferenceType):
        return element_type_node(node.referenced_type)
    if (
        isinstance(node, A.NamedType)
        and node.name in _BUFFER_WRAPPERS
        and node.generic_args
    ):
        return node.generic_args[0]
    return None


def _struct_name(node):
    if isinstance(node, A.ReferenceType):
        return _struct_name(node.referenced_type)
    if isinstance(node, (A.NamedType, A.PrimitiveType)):
        return node.name
    return None


_TEXEL_READ_CALLS = frozenset(
    {
        "imageLoad",
        "subpassLoad",
        "texelFetch",
        "texelFetchOffset",
        "texture",
        "textureGather",
        "textureGatherOffset",
        "textureGrad",
        "textureGradOffset",
        "textureLod",
        "textureLodOffset",
        "textureOffset",
        "textureProj",
        "textureProjGrad",
        "textureProjLod",
        "textureProjOffset",
    }
)
# Calls that return a four-component texel even from a shadow (depth) texture: gathers, and texel fetches, which
# read the stored value instead of comparing it.
_SHADOW_GATHER_CALLS = frozenset(
    {"textureGather", "textureGatherOffset", "texelFetch", "texelFetchOffset"}
)
_TEXEL_RESOURCE = re.compile(
    r"([iu]?)(image|sampler|texture|subpassInput)\w*?(Shadow)?"
)
_TEXEL_KINDS = {"": FLOAT32, "i": INT32, "u": UINT32}


class TypeScope:
    """Lexical scopes of declared names, plus the program's structures, constants and functions."""

    def __init__(self, program):
        self.structs: dict[str, dict[str, object]] = {}
        self.functions: dict[str, object] = {}
        self.constants: dict[str, object] = {}
        self._fallback: dict[str, object] = {}
        self._frames: list[dict[str, object]] = [{}]
        for struct in iter_structs(program):
            members = {}
            for member in getattr(struct, "members", None) or []:
                member_type = getattr(member, "member_type", None)
                if member_type is None:
                    member_type = getattr(member, "var_type", None)
                if getattr(member, "name", None):
                    members[member.name] = member_type
            self.structs[struct.name] = members
        overloaded = set()
        for function in iter_functions(program):
            known = self.functions.get(function.name)
            if known is not None and _type_key(known) != _type_key(
                function.return_type
            ):
                overloaded.add(function.name)
            self.functions[function.name] = function.return_type
        for name in overloaded:
            self.functions[name] = None

    # Scopes ---------------------------------------------------------------------------------------------------
    def push(self):
        self._frames.append({})

    def pop(self):
        self._frames.pop()

    def declare(self, name, type_node):
        if name:
            self._frames[-1][name] = type_node

    def declare_block_members(self, variable):
        """Members of an unnamed interface block are visible as names; they rank below every declaration."""
        struct = self.structs.get(_struct_name(getattr(variable, "var_type", None)))
        if struct is None:
            return
        for name, member_type in struct.items():
            self._fallback.setdefault(name, member_type)

    def declare_cbuffer(self, cbuffer):
        for member in getattr(cbuffer, "members", None) or []:
            member_type = getattr(member, "member_type", None) or getattr(
                member, "var_type", None
            )
            self.declare(getattr(member, "name", None), member_type)

    def lookup(self, name):
        for frame in reversed(self._frames):
            if name in frame:
                return frame[name]
        return self._fallback.get(name)

    def is_declared(self, name) -> bool:
        return any(name in frame for frame in self._frames) or name in self._fallback

    # Inference ------------------------------------------------------------------------------------------------
    def node_type(self, expr):
        """The ``TypeNode`` of a named, indexed or member expression (aggregates included), or ``None``."""
        if isinstance(expr, A.IdentifierNode):
            return self.lookup(expr.name)
        if isinstance(expr, A.ArrayAccessNode):
            base = self.node_type(expr.array_expr)
            return element_type_node(base) if base is not None else None
        if isinstance(expr, A.MemberAccessNode):
            members = self.structs.get(_struct_name(self.node_type(expr.object_expr)))
            if members is not None:
                return members.get(expr.member)
        return None

    def infer(self, expr) -> HubType | None:
        """The value type of an expression, or ``None`` when it cannot be inferred."""
        if isinstance(expr, A.LiteralNode):
            return _literal_type(expr)
        if isinstance(expr, A.IdentifierNode):
            declared = self.lookup(expr.name)
            if declared is not None:
                return type_from_node(declared)
            if not self.is_declared(expr.name):
                return _BUILTIN_VARIABLES.get(expr.name)
            return None
        if isinstance(expr, A.BinaryOpNode):
            return self._infer_binary(expr)
        if isinstance(expr, A.UnaryOpNode):
            if expr.op == "!":
                operand = self.infer(expr.operand)
                return BOOL.with_lanes(operand.lanes) if operand else BOOL
            return self.infer(expr.operand)
        if isinstance(expr, A.TernaryOpNode):
            return arithmetic_result(
                self.infer(expr.true_expr), self.infer(expr.false_expr)
            )
        if isinstance(expr, A.ArrayAccessNode):
            return self._infer_index(expr)
        if isinstance(expr, A.MemberAccessNode):
            return self._infer_member(expr.object_expr, expr.member, expr)
        if isinstance(expr, A.SwizzleNode):
            return self._infer_member(expr.vector_expr, expr.components, None)
        if isinstance(expr, A.CastNode):
            return type_from_node(expr.target_type)
        if isinstance(expr, A.ConstructorNode):
            return type_from_node(expr.constructor_type)
        if isinstance(expr, A.FunctionCallNode):
            return self._infer_call(expr)
        if isinstance(expr, A.AssignmentNode):
            return self.infer(expr.target)
        return None

    def _infer_binary(self, expr):
        op = expr.op
        left, right = self.infer(expr.left), self.infer(expr.right)
        if op in ("&&", "||", "^^"):
            return BOOL
        if op in ("<", "<=", ">", ">=", "==", "!="):
            lanes = max(left.lanes if left else 1, right.lanes if right else 1)
            return BOOL.with_lanes(lanes)
        if op in ("<<", ">>"):
            if left is None:
                return None
            promoted = promote(left)
            if right is not None and right.lanes > promoted.lanes:
                return promoted.with_lanes(right.lanes)
            return promoted
        if op in ("+", "-", "*", "/", "%", "&", "|", "^"):
            return arithmetic_result(left, right)
        return None

    def _infer_index(self, expr):
        base = self.node_type(expr.array_expr)
        if base is not None:
            element = element_type_node(base)
            if element is not None:
                return type_from_node(element)
            matrix = _matrix_column(base)
            if matrix is not None:
                return matrix
        value = self.infer(expr.array_expr)
        if value is not None and value.lanes > 1:
            return value.scalar
        return None

    def _infer_member(self, object_expr, member, member_expr):
        if member_expr is not None:
            declared = self.node_type(member_expr)
            if declared is not None:
                return type_from_node(declared)
        value = self.infer(object_expr)
        if value is not None and value.lanes > 1 and _SWIZZLE.fullmatch(member or ""):
            return value.with_lanes(len(member)) if len(member) > 1 else value.scalar
        return None

    def _infer_call(self, expr):
        carried = expr.annotations.get("cgl_rule_type")
        if carried is not None:
            return carried
        name = call_name(expr)
        arguments = expr.arguments or []
        converted = _CONVERSION_RESULTS.get(name)
        if converted is not None:
            # A rule conversion's result has its own scalar type and the operand's lanes.
            first = self.infer(arguments[0]) if arguments else None
            return converted.with_lanes(first.lanes) if first is not None else None
        if name.startswith("__cgl_"):
            result = self.infer(arguments[0]) if arguments else None
            for argument in arguments[1:]:
                result = arithmetic_result(result, self.infer(argument))
            return result
        direct = type_from_name(name)
        if direct is not None:
            return direct
        first = self.infer(arguments[0]) if arguments else None
        lanes = first.lanes if first is not None else 1
        if name in ("asuint", "floatBitsToUint"):
            return UINT32.with_lanes(lanes)
        if name in ("asint", "floatBitsToInt"):
            return INT32.with_lanes(lanes)
        if name in ("asfloat", "intBitsToFloat", "uintBitsToFloat"):
            return FLOAT32.with_lanes(lanes)
        if name in _BOOL_RESULT_CALLS:
            return BOOL.with_lanes(lanes) if name.startswith("is") else BOOL
        if name in _FLOAT_RESULT_CALLS:
            return first.scalar if first is not None else None
        if name == "dot":
            return first.scalar if first is not None else None
        if name in _SAME_AS_ARGUMENTS and arguments:
            result = first
            for argument in arguments[1:]:
                result = arithmetic_result(result, self.infer(argument))
            return result
        texel = self._infer_texel_call(name, arguments)
        if texel is not None:
            return texel
        if name in self.functions:
            return type_from_node(self.functions[name])
        return None

    def _infer_texel_call(self, name, arguments):
        """The four-component texel a texture or storage-image read returns.

        GLSL 4.60 sections 8.9 and 8.12: reads return a ``gvec4`` of the
        resource's component type; a depth comparison returns a ``float``, a texel fetch from a depth texture its
        stored texel.
        """
        if name not in _TEXEL_READ_CALLS or not arguments:
            return None
        resource = self.node_type(arguments[0])
        match = _TEXEL_RESOURCE.fullmatch(str(getattr(resource, "name", "") or ""))
        if match is None:
            return None
        if match.group(3) and name not in _SHADOW_GATHER_CALLS:
            return FLOAT32
        return _TEXEL_KINDS[match.group(1)].with_lanes(4)


def is_pure_callee(name: str) -> bool:
    """Value constructors, bit casts, rule intrinsics and math built-ins that neither write nor read memory."""
    return (
        type_from_name(name) is not None
        or name.startswith(("__cgl_", "cgl_rule_"))
        or name in ("asint", "asuint", "asfloat", "floatBitsToInt", "floatBitsToUint")
        or name in ("intBitsToFloat", "uintBitsToFloat", "dot", "length", "distance")
        or name in _SAME_AS_ARGUMENTS
    )


def call_name(expr) -> str:
    function = getattr(expr, "function", None)
    if isinstance(function, A.IdentifierNode):
        return function.name
    if isinstance(function, str):
        return function
    return ""


def promote(value: HubType) -> HubType:
    """Integer promotion: booleans and integers narrower than 32 bits compute as ``int``."""
    if value.kind == "b" or (value.kind in ("i", "u") and value.bits < 32):
        return INT32.with_lanes(value.lanes)
    return value


def arithmetic_result(left: HubType | None, right: HubType | None) -> HubType | None:
    """The usual arithmetic conversions of two operand types; ``None`` when either is unknown."""
    if left is None or right is None:
        return None
    lanes = max(left.lanes, right.lanes)
    if "f" in (left.kind, right.kind):
        bits = max(
            left.bits if left.kind == "f" else 0, right.bits if right.kind == "f" else 0
        )
        return HubType("f", bits, lanes)
    left, right = promote(left), promote(right)
    if left.kind == right.kind:
        return HubType(left.kind, max(left.bits, right.bits), lanes)
    unsigned, signed = (left, right) if left.kind == "u" else (right, left)
    if unsigned.bits >= signed.bits:
        return HubType("u", unsigned.bits, lanes)
    return HubType("i", signed.bits, lanes)


def _literal_type(literal):
    declared = type_from_node(literal.literal_type)
    if declared is not None:
        return declared
    value = literal.value
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return INT32
    if isinstance(value, float):
        return FLOAT32
    text = str(value)
    if re.fullmatch(r"-?\d+[uU]", text):
        return UINT32
    if re.fullmatch(r"-?\d+", text):
        return INT32
    return None


def _matrix_column(node):
    if isinstance(node, A.MatrixType):
        element = type_from_node(node.element_type)
        return element.with_lanes(int(node.rows)) if element is not None else None
    return None


def _type_key(node):
    return repr(node)


def iter_structs(program):
    yield from getattr(program, "structs", None) or []
    yield from getattr(program, "cbuffers", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_structs", None) or []
        yield from getattr(stage, "local_cbuffers", None) or []


def iter_functions(program):
    yield from getattr(program, "functions", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_functions", None) or []
        if getattr(stage, "entry_point", None) is not None:
            yield stage.entry_point


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)
