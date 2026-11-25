"""Storage-image formats of CUDA and HIP surface objects, and the byte offsets of their accesses.

A CUDA or HIP surface is untyped memory. ``surf2Dread<T>(s, x, y)`` (or ``surf2Dread(&v, s, x, y)``) reads, and
``surf2Dwrite(v, s, x, y)`` writes, the ``sizeof(T)`` bytes of one ``T`` at byte offset ``x`` of row ``y``, with no
format conversion (CUDA C++ Programming Guide, "Surface Object API"; HIP mirrors it in amd_surface_functions.h).
A CrossGL storage image is typed by its format: ``imageLoad`` returns and ``imageStore`` takes a four-component
value that the format converts to and from the texel memory holds (GLSL 4.60, 8.12 "Image Functions").

The element type ``T`` of a surface's accesses therefore is the texel as memory holds it, and the importers declare
the surface with the format of the same channels: ``float`` is ``r32f``, ``float2`` ``rg32f``, ``float4``
``rgba32f``, ``int`` ``r32i``, ``uint4`` ``rgba32ui``, ``uchar4`` ``rgba8ui``, ``short2`` ``rg16i``. The integer
formats keep each channel's integer value, so ``rgba8ui`` holds the four bytes ``surf2Dwrite(uchar4)`` stores, and
an integer image is declared ``iimage*`` or ``uimage*``. A load takes the element's leading components of the four
``imageLoad`` returns and narrows 8- and 16-bit channels to their type; a store widens the element to four
components, green and blue 0 and alpha 1, of the image's 32-bit component type, which the format narrows back.

Every access of one surface must use one element type, since a typed image cannot reinterpret its texels: a
surface accessed as two element types, or by an access whose element type the importer cannot tell, is refused.
Element types with no storage-image format (``double``, the three-component vectors, 64-bit integers) are refused
too. A surface whose accesses all name ``float4`` may also be declared without a format, as an image whose
format the importer cannot see (a structure member, say) is: ``rgba32f`` is the four-component default.

The x byte offset becomes a texel coordinate by dividing it by ``sizeof(T)``, structurally
(:class:`SurfaceByteOffsetMixin`): only an offset that is provably a multiple of the element size divides, so
``x * sizeof(float)`` is ``x``, ``(x + 1) * 8`` with an 8-byte element is ``x + 1`` and ``0`` is ``0``; anything
else is refused instead of addressing the wrong texel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...translator.integer_literals import IntegerLiteralError, integer_literal_parts
from ..assignment_operands import grouped_operand
from ..common_ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = [
    "SURFACE_READS",
    "SURFACE_WRITES",
    "SurfaceAccessMixin",
    "SurfaceByteOffsetMixin",
    "SurfaceElement",
    "SurfaceElementConflict",
    "SurfaceElementKey",
    "SurfaceDeclaredKey",
    "declared_variable_types",
    "is_address_of",
    "merge_surface_element",
    "normalize_surface_type",
    "surface_element",
    "surface_element_refusal",
    "surface_type_size",
    "surface_value_type",
]

# Bytes, signedness and CrossGL spelling of each canonical scalar.
_SCALARS = {
    "bool": (1, None, "bool"),
    "char": (1, "int", "i8"),
    "uchar": (1, "uint", "u8"),
    "short": (2, "int", "i16"),
    "ushort": (2, "uint", "u16"),
    "half": (2, "float", "f16"),
    "int": (4, "int", "i32"),
    "uint": (4, "uint", "u32"),
    "float": (4, "float", "f32"),
    "longlong": (8, "int", "i64"),
    "ulonglong": (8, "uint", "u64"),
    "double": (8, "float", "f64"),
}

# CUDA, HIP, <cstdint> and CrossGL spellings of the canonical scalars. Plain ``char`` is signed, as the members of
# CUDA's ``char1``..``char4`` are.
_SCALAR_SPELLINGS = {
    "bool": "bool",
    "char": "char",
    "signed char": "char",
    "int8_t": "char",
    "i8": "char",
    "unsigned char": "uchar",
    "uchar": "uchar",
    "uint8_t": "uchar",
    "u8": "uchar",
    "short": "short",
    "signed short": "short",
    "short int": "short",
    "int16_t": "short",
    "i16": "short",
    "unsigned short": "ushort",
    "ushort": "ushort",
    "uint16_t": "ushort",
    "u16": "ushort",
    "half": "half",
    "__half": "half",
    "f16": "half",
    "int": "int",
    "signed": "int",
    "signed int": "int",
    "int32_t": "int",
    "i32": "int",
    "unsigned": "uint",
    "unsigned int": "uint",
    "uint": "uint",
    "uint32_t": "uint",
    "u32": "uint",
    "float": "float",
    "f32": "float",
    # ``long`` is 8 bytes on the LP64 hosts nvcc and hipcc target on Linux.
    "long": "longlong",
    "long int": "longlong",
    "signed long": "longlong",
    "unsigned long": "ulonglong",
    "unsigned long int": "ulonglong",
    "ulong": "ulonglong",
    "long long": "longlong",
    "long long int": "longlong",
    "longlong": "longlong",
    "int64_t": "longlong",
    "i64": "longlong",
    "unsigned long long": "ulonglong",
    "unsigned long long int": "ulonglong",
    "ulonglong": "ulonglong",
    "uint64_t": "ulonglong",
    "u64": "ulonglong",
    "double": "double",
    "f64": "double",
}

# CUDA vector type prefixes (``uchar4``) and the CrossGL component spellings of ``vecN<T>``.
_VECTOR_PREFIXES = {
    "char": "char",
    "uchar": "uchar",
    "short": "short",
    "ushort": "ushort",
    "half": "half",
    "__half": "half",
    "int": "int",
    "uint": "uint",
    "float": "float",
    "long": "longlong",
    "ulong": "ulonglong",
    "longlong": "longlong",
    "ulonglong": "ulonglong",
    "double": "double",
}
_CROSSGL_COMPONENTS = {
    "bool": "bool",
    "i8": "char",
    "u8": "uchar",
    "i16": "short",
    "u16": "ushort",
    "f16": "half",
    "i32": "int",
    "u32": "uint",
    "f32": "float",
    "i64": "longlong",
    "u64": "ulonglong",
    "f64": "double",
}
_VECTOR = re.compile(r"([A-Za-z_]+?)([1-4])")
_CROSSGL_VECTOR = re.compile(r"vec([1-4])<\s*([A-Za-z0-9]+)\s*>")
_QUALIFIERS = re.compile(
    r"\b(?:const|volatile|__restrict__|__restrict|restrict|struct|static|register)\b"
)

COMPONENTS = "xyzw"

_KIND_SCALARS = {"float": "f32", "int": "i32", "uint": "u32"}
_KIND_FILL = {"float": ("0.0", "1.0"), "int": ("0", "1"), "uint": ("0u", "1u")}
_KIND_IMAGE_PREFIXES = {"float": "", "int": "i", "uint": "u"}
_FORMAT_SUFFIXES = {"float": "f", "int": "i", "uint": "ui"}
_FORMAT_CHANNELS = {1: "r", 2: "rg", 4: "rgba"}


def normalize_surface_type(type_name):
    """The canonical scalar or vector name of a CUDA, HIP or CrossGL value type (``uchar4``), or ``None``."""
    if not type_name:
        return None
    text = _QUALIFIERS.sub(" ", str(type_name))
    text = " ".join(text.replace("::", " ").split())
    if text.startswith("std "):
        text = text[4:]
    if text in _SCALAR_SPELLINGS:
        return _SCALAR_SPELLINGS[text]
    match = _CROSSGL_VECTOR.fullmatch(text)
    if match is not None:
        component = _CROSSGL_COMPONENTS.get(match.group(2))
        if component is None or component == "bool":
            return None
        return f"{component}{match.group(1)}"
    match = _VECTOR.fullmatch(text.replace(" ", ""))
    if match is not None and match.group(1) in _VECTOR_PREFIXES:
        return f"{_VECTOR_PREFIXES[match.group(1)]}{match.group(2)}"
    return None


def _scalar_and_count(canonical):
    if canonical in _SCALARS:
        return canonical, 1
    match = _VECTOR.fullmatch(canonical or "")
    if match is None or match.group(1) not in _SCALARS:
        return None, 0
    return match.group(1), int(match.group(2))


def surface_type_size(type_name):
    """``sizeof`` the CUDA or HIP value type ``type_name`` names, or ``None`` when it is not a known value type."""
    scalar, count = _scalar_and_count(normalize_surface_type(type_name))
    if scalar is None:
        return None
    return _SCALARS[scalar][0] * count


@dataclass(frozen=True)
class SurfaceElement:
    """The element type of a surface's accesses and the CrossGL storage-image format that holds it."""

    cuda_type: str
    kind: str
    count: int
    bits: int
    component: str

    @property
    def format(self):
        """The CrossGL image format: ``r32f``, ``rgba8ui``."""
        return f"{_FORMAT_CHANNELS[self.count]}{self.bits}{_FORMAT_SUFFIXES[self.kind]}"

    @property
    def size(self):
        """``sizeof`` the element, the stride of the x byte offset."""
        return self.bits // 8 * self.count

    def image_type(self, image_type):
        """``image_type`` (``image2D``) with the prefix of this element's component kind: ``uimage2D``."""
        base = str(image_type)
        if base[:1] in {"i", "u"} and base[1:].startswith("image"):
            base = base[1:]
        return f"{_KIND_IMAGE_PREFIXES[self.kind]}{base}"

    def declaration_type(self, image_type):
        """The declared type of a surface holding this element: ``@format(r32f) image2D``."""
        return f"@format({self.format}) {self.image_type(image_type)}"

    def load(self, image_load):
        """The element value of ``image_load``, a four-component ``imageLoad`` of an image of this format."""
        if self.count == 4:
            value = image_load
        else:
            value = f"{image_load}.{COMPONENTS[: self.count]}"
        if self.bits == 32:
            return value
        if self.count == 1:
            return f"{self.component}({value})"
        return f"vec{self.count}<{self.component}>({value})"

    def store(self, value):
        """The four-component value ``imageStore`` writes for an element ``value``: missing channels 0, alpha 1."""
        scalar = _KIND_SCALARS[self.kind]
        zero, one = _KIND_FILL[self.kind]
        if self.count == 1:
            # imageStore splats a scalar; the format keeps the first channel.
            return value if self.bits == 32 else f"{scalar}({value})"
        if self.count == 4:
            return value if self.bits == 32 else f"vec4<{scalar}>({value})"
        if self.bits != 32:
            value = f"vec2<{scalar}>({value})"
        return f"vec4<{scalar}>({value}, {zero}, {one})"


@dataclass(frozen=True)
class SurfaceElementConflict:
    """A surface whose accesses name more than one element type."""

    types: tuple

    @property
    def reason(self):
        return (
            "it is accessed as " + " and ".join(self.types) + ", and a CrossGL storage "
            "image holds texels of one format"
        )


@dataclass(frozen=True)
class SurfaceElementKey:
    """The resource-hint key of a surface's element (the plain name keys its image type).

    A dataclass, so it never equals a plain name or the other key of the same name.
    """

    name: str


@dataclass(frozen=True)
class SurfaceDeclaredKey:
    """The resource-hint key marking a surface whose declaration carries its element's format."""

    name: str


def surface_element(type_name):
    """The :class:`SurfaceElement` of a surface accessed as ``type_name``, or ``None`` when no format holds it."""
    canonical = normalize_surface_type(type_name)
    scalar, count = _scalar_and_count(canonical)
    if scalar is None or count not in _FORMAT_CHANNELS:
        return None
    size, kind, component = _SCALARS[scalar]
    if kind is None or size == 8 or (kind == "float" and size != 4):
        return None
    return SurfaceElement(canonical, kind, count, size * 8, component)


def surface_element_refusal(type_name):
    """Why a surface accessed as ``type_name`` has no CrossGL storage-image format."""
    spelled = " ".join(str(type_name).split())
    scalar, count = _scalar_and_count(normalize_surface_type(type_name))
    if scalar is None:
        return f"{spelled} is not a value type a surface holds"
    if count == 3:
        return f"{spelled} has three components, and no storage-image format has three"
    if _SCALARS[scalar][0] == 8:
        return f"{spelled} has 64-bit components, and no CrossGL storage-image format has them"
    if scalar == "half":
        return f"{spelled} has 16-bit float components, which no CUDA surface function reads"
    return f"{spelled} has no CrossGL storage-image format"


def merge_surface_element(hints, name, element):
    """Record that surface ``name`` is accessed as ``element``; two different elements become a conflict."""
    if not name or element is None:
        return
    key = SurfaceElementKey(name)
    previous = hints.get(key)
    if previous is None or previous == element:
        hints[key] = element
        return
    types = (
        previous.types
        if isinstance(previous, SurfaceElementConflict)
        else (previous.cuda_type,)
    )
    if element.cuda_type not in types:
        types = (*types, element.cuda_type)
    hints[key] = SurfaceElementConflict(tuple(sorted(types)))


def declared_variable_types(function):
    """``{name: type}`` of a function's parameters and the variables its body declares (shadowing ignored)."""
    types = {}
    for param in getattr(function, "params", None) or []:
        if isinstance(param, VariableNode) and param.name:
            types[param.name] = param.vtype
        elif isinstance(param, dict) and param.get("name"):
            # The HIP parser keeps parameters as ``{"type": ..., "name": ...}``.
            types[param["name"]] = param.get("type")
    stack = [getattr(function, "body", None)]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
            continue
        if not hasattr(current, "__dict__") or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, VariableNode) and current.name:
            types.setdefault(current.name, current.vtype)
        stack.extend(reversed(list(vars(current).values())))
    return types


# ---------------------------------------------------------------------------------------------------------------
# Value types of surface write operands and read destinations.

_FLOAT_LITERAL = re.compile(
    r"(?:\d+\.\d*|\.\d+|\d+(?=[eE]))(?:[eE][+-]?\d+)?([fFlL]?)|\d+[eE][+-]?\d+([fFlL]?)"
)
_ARITHMETIC = {"+", "-", "*", "/", "%", "&", "|", "^"}
_SHIFTS = {"<<", ">>"}
_RANKS = ("int", "uint", "longlong", "ulonglong", "float", "double")


def _literal_type(text):
    spelling = str(text).strip().replace("'", "")
    if spelling in {"true", "false"}:
        return "bool"
    match = _FLOAT_LITERAL.fullmatch(spelling)
    if match is not None:
        suffix = (match.group(1) or match.group(2) or "").lower()
        return "float" if suffix == "f" else "double" if suffix == "" else None
    try:
        _digits, type_name = integer_literal_parts(spelling, legacy_octal=True)
    except IntegerLiteralError:
        return None
    return normalize_surface_type(type_name)


def _promoted(scalar):
    """C++ integral promotion: the 1- and 2-byte integers and bool compute as int."""
    if scalar in {"bool", "char", "uchar", "short", "ushort"}:
        return "int"
    return scalar


def _arithmetic(left, right):
    """The usual arithmetic conversions of two canonical scalars (C++ [expr.arith.conv]), or ``None``."""
    if left is None or right is None:
        return None
    if left == right == "half":
        # cuda_fp16.h and hip_fp16.h declare the __half operators.
        return "half"
    left, right = _promoted(left), _promoted(right)
    if left not in _RANKS or right not in _RANKS:
        return None
    return max(left, right, key=_RANKS.index)


def surface_value_type(node, variable_type, member_type=None):
    """The canonical value type of a CUDA or HIP expression, or ``None`` when it cannot be told.

    ``variable_type(name)`` gives a variable's declared type and ``member_type(object type, member)`` the declared
    type of a structure member, in CUDA, HIP or CrossGL spelling; ``member_type`` may be ``None``. Vectors only
    combine with the same vector type, through the operators a program declares for them.
    """

    def declared(current):
        """The declared type of a named object, element, member or dereference, as written."""
        if isinstance(current, str):
            if _literal_type(current) is not None:
                return None
            return variable_type(current)
        if isinstance(current, CastNode):
            return current.target_type
        if isinstance(current, (ArrayAccessNode, UnaryOpNode)):
            if isinstance(current, UnaryOpNode) and current.op != "*":
                return None
            inner = (
                current.array
                if isinstance(current, ArrayAccessNode)
                else current.operand
            )
            outer = declared(inner)
            return _strip_level(outer) if outer else None
        if isinstance(current, MemberAccessNode):
            object_type = declared(current.object)
            if object_type is None:
                return None
            if getattr(current, "is_pointer", False):
                object_type = _strip_level(object_type)
                if object_type is None:
                    return None
            scalar, count = _scalar_and_count(normalize_surface_type(object_type))
            if scalar is not None and count > 1:
                return scalar if current.member in COMPONENTS[:count] else None
            if member_type is None:
                return None
            return member_type(object_type, current.member)
        return None

    def value(current):
        if isinstance(current, str):
            literal = _literal_type(current)
            if literal is not None:
                return literal
        if isinstance(current, (str, CastNode, ArrayAccessNode, MemberAccessNode)) or (
            isinstance(current, UnaryOpNode) and current.op == "*"
        ):
            return normalize_surface_type(declared(current))
        if isinstance(current, UnaryOpNode):
            if current.op in {"+", "-", "~"}:
                operand = value(current.operand)
                return _promoted(operand) if operand in _SCALARS else operand
            if current.op == "!":
                return "bool"
            return None
        if isinstance(current, BinaryOpNode):
            if current.op in {"<", ">", "<=", ">=", "==", "!=", "&&", "||"}:
                return "bool"
            left, right = value(current.left), value(current.right)
            if current.op in _SHIFTS:
                return _promoted(left) if left in _SCALARS else None
            if current.op not in _ARITHMETIC:
                return None
            if left is not None and left == right and left not in _SCALARS:
                return left
            return _arithmetic(left, right)
        if isinstance(current, TernaryOpNode):
            true_type = value(current.true_expr)
            false_type = value(current.false_expr)
            if true_type == false_type:
                return true_type
            return _arithmetic(true_type, false_type)
        if isinstance(current, FunctionCallNode) and isinstance(current.name, str):
            name = current.name.removeprefix("::")
            if name.startswith("make_"):
                return normalize_surface_type(name[len("make_") :])
            return normalize_surface_type(name)
        return None

    return value(node)


def _strip_level(type_name):
    """``type_name`` with one array extent or pointer level removed: ``float*`` -> ``float``."""
    text = " ".join(str(type_name).split())
    match = re.fullmatch(r"(?:array|ptr)<\s*(.+?)\s*(?:,\s*[^<>,]+)?>", text)
    if match is not None:
        return match.group(1)
    if text.endswith("]") and "[" in text:
        base, _, rest = text.partition("[")
        tail = rest.split("]", 1)[1]
        return f"{base.strip()}{tail}".strip()
    if text.endswith("*"):
        return text[:-1].strip()
    return None


# ---------------------------------------------------------------------------------------------------------------
# Byte offsets.


class SurfaceByteOffsetMixin:
    """Dividing a surface access's x byte offset by the element size, structurally.

    The converter provides ``visit`` and ``strip_type_qualifiers``.
    """

    def divide_surface_byte_offset(self, node, byte_size, value_type):
        """The texel coordinate of byte offset ``node``, as ``(text, precedence)``, or ``None``.

        ``None`` means the offset cannot be shown to be a multiple of ``byte_size``. ``precedence`` is ``"atom"``
        for a single operand, ``"mul"`` for a product and ``"add"`` for a sum or difference, so an embedding
        operator knows when to group the result. Every operand is kept as written: terms are never combined or
        dropped, so a call or an increment inside the offset still evaluates exactly once.
        """
        if node is None:
            return None
        if isinstance(node, str):
            literal = self.parse_surface_integer_literal(node)
            if literal is None:
                return None
            return self.divide_surface_constant(literal, byte_size)
        if isinstance(node, CastNode):
            return self.divide_surface_byte_offset(
                node.expression, byte_size, value_type
            )
        if isinstance(node, UnaryOpNode):
            if node.op not in {"-", "+"}:
                return None
            inner = self.divide_surface_byte_offset(node.operand, byte_size, value_type)
            if inner is None:
                return None
            text, precedence = inner
            if node.op == "+":
                return (text, precedence)
            grouped = f"({text})" if precedence != "atom" else text
            return (f"(-{grouped})", "atom")
        if isinstance(node, FunctionCallNode):
            if node.name != "sizeof" or len(node.args) != 1:
                return None
            return self.divide_surface_sizeof(node.args[0], byte_size, value_type)
        if isinstance(node, BinaryOpNode):
            if node.op == "*":
                return self.divide_surface_product(node, byte_size, value_type)
            if node.op in {"+", "-"}:
                left = self.divide_surface_byte_offset(node.left, byte_size, value_type)
                right = self.divide_surface_byte_offset(
                    node.right, byte_size, value_type
                )
                if left is None or right is None:
                    return None
                left_text, _ = left
                right_text, right_precedence = right
                if node.op == "-":
                    if right_precedence == "add":
                        right_text = f"({right_text})"
                    return (f"{left_text} - {right_text}", "add")
                return (f"{left_text} + {right_text}", "add")
        return None

    def divide_surface_constant(self, value, byte_size):
        """Divide a constant byte count by the element size, or return None."""
        if value == 0:
            return ("0", "atom")
        if byte_size is None or value % byte_size != 0:
            return None
        return (str(value // byte_size), "atom")

    def is_surface_element_sizeof(self, arg, value_type):
        """Whether ``sizeof(arg)`` is the size of the element type ``value_type``, however either is spelled."""
        if value_type is None:
            return False
        if self.strip_type_qualifiers(arg) == self.strip_type_qualifiers(value_type):
            return True
        canonical = normalize_surface_type(arg)
        return canonical is not None and canonical == normalize_surface_type(value_type)

    def divide_surface_sizeof(self, arg, byte_size, value_type):
        """Divide a ``sizeof`` byte count by the element size, or return None."""
        if not isinstance(arg, str):
            return None
        if self.is_surface_element_sizeof(arg, value_type):
            return ("1", "atom")
        size = surface_type_size(arg)
        if size is None:
            return None
        return self.divide_surface_constant(size, byte_size)

    def divide_surface_product(self, node, byte_size, value_type):
        """Divide a product by the element size, or return None.

        One side must be a byte factor: an integer literal, or ``sizeof`` of the element type (one element size,
        whatever its byte count) or of a type with a known size. The other side is kept as written, so calls and
        increments inside it still evaluate exactly once.
        """
        left = self.surface_byte_factor(node.left, value_type)
        right = self.surface_byte_factor(node.right, value_type)
        if left is not None and right is not None:
            if left[0] == "size" and right[0] == "size":
                if byte_size is None:
                    return None
                return (str(byte_size), "atom")
            if left[0] == "size":
                return (str(right[1]), "atom")
            if right[0] == "size":
                return (str(left[1]), "atom")
            return self.divide_surface_constant(left[1] * right[1], byte_size)
        if left is not None:
            return self.divide_surface_product_factor(left, node.right, byte_size)
        if right is not None:
            return self.divide_surface_product_factor(right, node.left, byte_size)
        return None

    def divide_surface_product_factor(self, factor, other, byte_size):
        """Divide ``other * factor`` by the element size, or return None."""
        if factor[0] == "bytes" and factor[1] == 0:
            return ("0", "atom")
        other_text = grouped_operand(self, other)
        if factor[0] == "size":
            return (other_text, "atom")
        count = factor[1]
        if byte_size is None or count % byte_size != 0:
            return None
        quotient = count // byte_size
        if quotient == 1:
            return (other_text, "atom")
        return (f"{other_text} * {quotient}", "mul")

    def surface_byte_factor(self, node, value_type):
        """``("bytes", count)`` for a literal or a sized ``sizeof``, ``("size",)`` for ``sizeof`` the element."""
        if isinstance(node, str):
            literal = self.parse_surface_integer_literal(node)
            if literal is not None:
                return ("bytes", literal)
            return None
        if isinstance(node, CastNode):
            return self.surface_byte_factor(node.expression, value_type)
        if (
            isinstance(node, FunctionCallNode)
            and node.name == "sizeof"
            and len(node.args) == 1
            and isinstance(node.args[0], str)
        ):
            if self.is_surface_element_sizeof(node.args[0], value_type):
                return ("size",)
            size = surface_type_size(node.args[0])
            if size is not None:
                return ("bytes", size)
        return None

    def parse_surface_integer_literal(self, text):
        """The value of a CUDA or HIP integer literal, or ``None`` when ``text`` is not one."""
        spelling = str(text).replace("'", "")
        try:
            digits, _type = integer_literal_parts(spelling, legacy_octal=True)
        except IntegerLiteralError:
            return None
        try:
            return int(digits, 0)
        except ValueError:
            return None


# ---------------------------------------------------------------------------------------------------------------
# Surface calls.

SURFACE_READS = frozenset(
    {
        "surf1Dread",
        "surf1DLayeredread",
        "surf2Dread",
        "surf3Dread",
        "surf2DLayeredread",
        "surfCubemapread",
        "surfCubemapLayeredread",
    }
)
SURFACE_WRITES = frozenset(name.replace("read", "write") for name in SURFACE_READS)


def is_address_of(node):
    """Whether ``node`` is ``&operand``, the destination of a pointer-form surface read."""
    return isinstance(node, UnaryOpNode) and node.op == "&"


class SurfaceAccessMixin(SurfaceByteOffsetMixin):
    """The element types of CUDA and HIP surface accesses and the formats of the surfaces they access.

    The converter provides ``parse_cpp_template``, ``is_user_defined_function``, ``lookup_variable_type``,
    ``lookup_resource_object_type_hint``, ``get_resource_object_expression_name``, ``refuse_import``, ``visit``
    and ``strip_type_qualifiers``. ``SURFACE_LANGUAGE`` names the source language in diagnostics.
    """

    SURFACE_LANGUAGE = "CUDA"

    def surface_write_value_is_pointer(self, base_name):
        """Whether ``base_name`` takes a pointer to the value it writes (HIP's ``surfCubemapLayeredwrite``)."""
        return False

    def surface_member_type(self, object_type, member):
        """The declared type of member ``member`` of a structure of type ``object_type``, or ``None``.

        The converters that know their programs' structures override it.
        """
        return None

    def surface_call_operands(self, call):
        """``(base name, surface operand, value operand, template type)`` of a surface call, or ``None``.

        The value operand is the destination of a pointer-form read (``v`` of ``surf2Dread(&v, s, x, y)``), the
        value a write stores, or ``None`` for a value-form read.
        """
        if not isinstance(call, FunctionCallNode) or not isinstance(call.name, str):
            return None
        base_name, template_args = self.parse_cpp_template(call.name)
        base_name = base_name.removeprefix("::")
        if self.is_user_defined_function(base_name):
            return None
        args = list(getattr(call, "args", None) or [])
        template_type = template_args[0] if template_args else None
        if base_name in SURFACE_READS:
            if args and is_address_of(args[0]):
                if len(args) < 2:
                    return None
                return base_name, args[1], args[0].operand, template_type
            if not args:
                return None
            return base_name, args[0], None, template_type
        if base_name in SURFACE_WRITES:
            if len(args) < 2:
                return None
            value = args[0]
            if self.surface_write_value_is_pointer(base_name) and is_address_of(value):
                value = value.operand
            return base_name, args[1], value, template_type
        return None

    def surface_call_element_type(self, operands, variable_type):
        """The type ``T`` a surface call reads or writes: its template argument, or its value operand's type."""
        _base_name, _surface, value, template_type = operands
        if template_type:
            return template_type
        if value is None:
            return None
        return surface_value_type(value, variable_type, self.surface_member_type)

    def collect_surface_element_hint(self, hints, call, variable_types, names=None):
        """Record the element type of surface call ``call`` in ``hints``; ``names`` limits the surfaces recorded."""
        operands = self.surface_call_operands(call)
        if operands is None:
            return
        name = self.get_resource_object_expression_name(operands[1])
        if not name or (names is not None and name not in names):
            return
        call_type = self.surface_call_element_type(operands, variable_types.get)
        element = surface_element(call_type) if call_type else None
        if element is not None:
            merge_surface_element(hints, name, element)

    def surface_access_element(self, function_name, surface, call_type):
        """The :class:`SurfaceElement` an access of ``surface`` as ``call_type`` converts through.

        Returns ``(element, size)``: ``element`` is ``None`` for a ``float4`` access of a surface declared
        without a format, whose four-component default holds it. Returns ``None`` after refusing an access whose
        element type is unknown, has no format, or differs from the surface's.
        """
        name = self.get_resource_object_expression_name(surface)
        declared = self.lookup_resource_object_type_hint(SurfaceElementKey(name))
        language = self.SURFACE_LANGUAGE
        if isinstance(declared, SurfaceElementConflict):
            self.refuse_import(
                f"{language} {function_name} on surface '{name}': {declared.reason}"
            )
            return None
        if not call_type:
            self.refuse_import(
                f"{language} {function_name} on surface '{name}': the type of the value it "
                "reads or writes is unknown, so neither the texel format nor the byte stride "
                "of x is known"
            )
            return None
        element = surface_element(call_type)
        if element is None:
            self.refuse_import(
                f"{language} {function_name} on surface '{name}': "
                f"{surface_element_refusal(call_type)}"
            )
            return None
        if declared is None or (
            element.format != "rgba32f"
            and not self.lookup_resource_object_type_hint(SurfaceDeclaredKey(name))
        ):
            # rgba32f is the format of an image declared without one.
            if element.format == "rgba32f":
                return None, element.size
            self.refuse_import(
                f"{language} {function_name} on surface '{name}': the surface's declaration "
                f"cannot be given the format of its {element.cuda_type} texels"
            )
            return None
        if declared != element:
            self.refuse_import(
                f"{language} {function_name} on surface '{name}': it is accessed as "
                f"{element.cuda_type} here and as {declared.cuda_type} elsewhere, and a "
                "CrossGL storage image holds texels of one format"
            )
            return None
        return element, element.size

    def mark_surface_declared(self, name):
        """Record that surface ``name``'s declaration carries its element's format, in the innermost hint scope."""
        scopes = self.resource_object_hint_scopes
        scope = scopes[-1] if scopes else self.global_resource_object_type_hints
        scope[SurfaceDeclaredKey(name)] = True

    def surface_declaration_type(self, crossgl_type, name):
        """``crossgl_type`` of parameter ``name`` with its surface element's format and component kind, if any."""
        element = self.lookup_resource_object_type_hint(SurfaceElementKey(name))
        text = str(crossgl_type)
        if not isinstance(element, SurfaceElement) or not text.startswith(
            ("image", "iimage", "uimage")
        ):
            return crossgl_type
        self.mark_surface_declared(name)
        return element.declaration_type(text)

    def surface_format_attribute(self, name):
        """``@format(r32f) `` for a global surface whose element has a format, else ``""``."""
        element = self.lookup_resource_object_type_hint(SurfaceElementKey(name))
        if isinstance(element, SurfaceElement):
            self.mark_surface_declared(name)
            return f"@format({element.format}) "
        return ""

    def surface_image_type(self, crossgl_type, name):
        """``crossgl_type`` with the component kind of surface ``name``'s element (``uimage2D``), no format."""
        element = self.lookup_resource_object_type_hint(SurfaceElementKey(name))
        text = str(crossgl_type)
        if isinstance(element, SurfaceElement) and text.startswith(
            ("image", "iimage", "uimage")
        ):
            return element.image_type(text)
        return crossgl_type

    def surface_global_declaration(self, crossgl_type, name, output_name):
        """The declaration of a namespace-scope surface, or ``None`` when ``crossgl_type`` is not an image.

        A ``__device__ cudaSurfaceObject_t`` is a handle the host sets through the symbol API, and a surface
        reference (``surface<void, 2> s;``) one it binds; both become the image resource the host binds, with the
        format of the texels the program accesses, instead of a storage buffer holding an image.
        """
        if not str(crossgl_type).startswith(("image", "iimage", "uimage")):
            return None
        image_type = self.surface_image_type(crossgl_type, name)
        return f"{self.surface_format_attribute(name)}var {output_name}: {image_type};"
