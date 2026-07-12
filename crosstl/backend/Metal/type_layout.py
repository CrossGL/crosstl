"""Host-independent object sizes for concrete Metal source types.

The layouts in this module describe Metal Shading Language objects.  They do
not depend on the Python host ABI or on any CrossGL target backend layout.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

__all__ = ["metal_type_layout", "metal_type_size"]

_MAX_TYPE_TEXT_LENGTH = 4096
_MAX_LAYOUT_SIZE = (1 << 63) - 1


@dataclass(frozen=True)
class _Layout:
    size: int
    alignment: int
    scalar_name: Optional[str] = None


_SCALAR_SIZES = {
    "bool": 1,
    "char": 1,
    "uchar": 1,
    "int8_t": 1,
    "uint8_t": 1,
    "short": 2,
    "ushort": 2,
    "int16_t": 2,
    "uint16_t": 2,
    "half": 2,
    "bfloat": 2,
    "bfloat16": 2,
    "bfloat16_t": 2,
    "float16_t": 2,
    "int": 4,
    "uint": 4,
    "int32_t": 4,
    "uint32_t": 4,
    "float": 4,
    "float32_t": 4,
    "long": 8,
    "ulong": 8,
    "int64_t": 8,
    "uint64_t": 8,
    "double": 8,
    "float64_t": 8,
    "size_t": 8,
    "ptrdiff_t": 8,
}

_VECTOR_SCALARS = set(_SCALAR_SIZES) - {"double", "float64_t"}
_PACKED_VECTOR_SCALARS = _VECTOR_SCALARS - {"bool"}
_MATRIX_SCALARS = {"half", "float"}

_PREFIX_QUALIFIERS = {
    "const",
    "volatile",
    "device",
    "constant",
    "thread",
    "threadgroup",
    "threadgroup_imageblock",
    "ray_data",
    "object_data",
}
_CV_QUALIFIERS = {"const", "volatile"}
_ADDRESS_SPACE_QUALIFIERS = _PREFIX_QUALIFIERS - _CV_QUALIFIERS

_VECTOR_PATTERN = re.compile(
    r"^(bool|char|uchar|short|ushort|int|uint|long|ulong|half|bfloat|float)([2-4])$"
)
_PACKED_VECTOR_PATTERN = re.compile(
    r"^packed_(char|uchar|short|ushort|int|uint|long|ulong|half|bfloat|float)([2-4])$"
)
_MATRIX_PATTERN = re.compile(r"^(half|float)([2-4])x([2-4])$")
_TOKEN_PATTERN = re.compile(
    r"\s+|::|<<|>>|[A-Za-z_][A-Za-z0-9_]*|[0-9][A-Za-z0-9']*|" r"[][<>,()+\-*/%&^|]"
)
_INTEGER_PATTERN = re.compile(
    r"^(?P<body>"
    r"0[xX][0-9A-Fa-f](?:'?[0-9A-Fa-f])*|"
    r"0[bB][01](?:'?[01])*|"
    r"0(?:'?[0-7])+|"
    r"0|"
    r"[1-9](?:'?[0-9])*"
    r")(?P<suffix>[uU](?:[lL]{1,2})?|[lL]{1,2}[uU]?)?$"
)

_BINARY_PRECEDENCE = {
    "|": 1,
    "^": 2,
    "&": 3,
    "<<": 4,
    ">>": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
}


class _ParseError(ValueError):
    pass


class _TypeParser:
    def __init__(self, type_text: str):
        self.tokens = self._tokenize(type_text)
        self.position = 0

    @staticmethod
    def _tokenize(type_text: str) -> List[str]:
        tokens = []
        position = 0
        while position < len(type_text):
            match = _TOKEN_PATTERN.match(type_text, position)
            if match is None:
                raise _ParseError("unsupported character")
            token = match.group(0)
            if not token.isspace():
                tokens.append(token)
            position = match.end()
        if not tokens:
            raise _ParseError("empty type")
        return tokens

    def parse(self) -> _Layout:
        layout = self._parse_type("all", allow_arrays=True)
        if self._peek() is not None:
            raise _ParseError("trailing type text")
        return layout

    def _parse_type(self, qualifier_mode: str, allow_arrays: bool) -> _Layout:
        qualifiers = self._consume_prefix_qualifiers(qualifier_mode)
        layout = self._parse_core_type()

        if qualifier_mode != "none":
            while self._peek() in _CV_QUALIFIERS:
                qualifier = self._take()
                if qualifier in qualifiers:
                    raise _ParseError("duplicate qualifier")
                qualifiers.add(qualifier)

        if allow_arrays:
            while self._accept("["):
                extent = self._parse_positive_integer_expression()
                self._expect("]")
                layout = self._array_layout(layout, extent)

        return layout

    def _consume_prefix_qualifiers(self, qualifier_mode: str) -> Set[str]:
        if qualifier_mode == "all":
            allowed = _PREFIX_QUALIFIERS
        elif qualifier_mode == "cv":
            allowed = _CV_QUALIFIERS
        else:
            allowed = set()

        qualifiers = set()
        address_space = None
        while self._peek() in allowed:
            qualifier = self._take()
            if qualifier in qualifiers:
                raise _ParseError("duplicate qualifier")
            if qualifier in _ADDRESS_SPACE_QUALIFIERS:
                if address_space is not None:
                    raise _ParseError("multiple address spaces")
                address_space = qualifier
            qualifiers.add(qualifier)
        return qualifiers

    def _parse_core_type(self) -> _Layout:
        namespaced = False
        if self._accept("::"):
            self._expect("metal")
            self._expect("::")
            namespaced = True
        elif self._peek() == "metal" and self._peek(1) == "::":
            self._take()
            self._take()
            namespaced = True

        base_name = self._take_identifier()
        if base_name in {"signed", "unsigned"}:
            if namespaced:
                raise _ParseError("qualified signed type")
            return self._parse_signed_scalar(base_name)

        if not namespaced and base_name in {"short", "long"} and self._peek() == "int":
            self._take()

        if self._accept("<"):
            return self._parse_generic_type(base_name)

        scalar_size = _SCALAR_SIZES.get(base_name)
        if scalar_size is not None:
            return _Layout(scalar_size, scalar_size, base_name)

        vector_match = _VECTOR_PATTERN.fullmatch(base_name)
        if vector_match is not None:
            scalar_name, width_text = vector_match.groups()
            return self._vector_layout(
                self._scalar_layout(scalar_name), int(width_text)
            )

        packed_match = _PACKED_VECTOR_PATTERN.fullmatch(base_name)
        if packed_match is not None:
            scalar_name, width_text = packed_match.groups()
            return self._packed_vector_layout(
                self._scalar_layout(scalar_name), int(width_text)
            )

        matrix_match = _MATRIX_PATTERN.fullmatch(base_name)
        if matrix_match is not None:
            scalar_name, columns_text, rows_text = matrix_match.groups()
            return self._matrix_layout(
                self._scalar_layout(scalar_name),
                int(columns_text),
                int(rows_text),
            )

        raise _ParseError("unsupported type")

    def _parse_signed_scalar(self, signedness: str) -> _Layout:
        base_name = "int"
        if self._peek() in {"char", "short", "int", "long"}:
            base_name = self._take()
            if base_name in {"short", "long"} and self._peek() == "int":
                self._take()

        if signedness == "unsigned":
            base_name = {
                "char": "uchar",
                "short": "ushort",
                "int": "uint",
                "long": "ulong",
            }[base_name]
        return self._scalar_layout(base_name)

    def _parse_generic_type(self, base_name: str) -> _Layout:
        if base_name in {"vector", "vec", "packed_vec"}:
            scalar = self._parse_type("none", allow_arrays=False)
            if scalar.scalar_name not in _VECTOR_SCALARS:
                raise _ParseError("vector element is not scalar")
            self._expect(",")
            width = self._parse_positive_integer_expression()
            self._expect(">")
            if width not in {2, 3, 4}:
                raise _ParseError("invalid vector width")
            if base_name == "packed_vec":
                if scalar.scalar_name not in _PACKED_VECTOR_SCALARS:
                    raise _ParseError("invalid packed vector scalar")
                return self._packed_vector_layout(scalar, width)
            return self._vector_layout(scalar, width)

        if base_name == "matrix":
            scalar = self._parse_type("none", allow_arrays=False)
            if scalar.scalar_name not in _MATRIX_SCALARS:
                raise _ParseError("invalid matrix scalar")
            self._expect(",")
            columns = self._parse_positive_integer_expression()
            self._expect(",")
            rows = self._parse_positive_integer_expression()
            self._expect(">")
            if columns not in {2, 3, 4} or rows not in {2, 3, 4}:
                raise _ParseError("invalid matrix dimensions")
            return self._matrix_layout(scalar, columns, rows)

        if base_name == "array":
            element = self._parse_type("cv", allow_arrays=True)
            self._expect(",")
            extent = self._parse_positive_integer_expression()
            self._expect(">")
            return self._array_layout(element, extent)

        raise _ParseError("unsupported generic type")

    def _parse_positive_integer_expression(self) -> int:
        value = self._parse_integer_expression(1)
        if value <= 0:
            raise _ParseError("non-positive extent")
        return value

    def _parse_integer_expression(self, minimum_precedence: int) -> int:
        value = self._parse_integer_unary()
        while True:
            operator = self._peek()
            precedence = _BINARY_PRECEDENCE.get(operator, 0)
            if precedence < minimum_precedence:
                return value
            self._take()
            right = self._parse_integer_expression(precedence + 1)
            value = self._apply_integer_operator(operator, value, right)

    def _parse_integer_unary(self) -> int:
        if self._accept("+"):
            return self._parse_integer_unary()
        if self._accept("-"):
            return -self._parse_integer_unary()
        if self._accept("("):
            value = self._parse_integer_expression(1)
            self._expect(")")
            return value

        token = self._take()
        match = _INTEGER_PATTERN.fullmatch(token)
        if match is None:
            raise _ParseError("dependent or malformed extent")
        body = match.group("body").replace("'", "")
        if body.lower().startswith("0x"):
            value = int(body[2:], 16)
        elif body.lower().startswith("0b"):
            value = int(body[2:], 2)
        elif len(body) > 1 and body.startswith("0"):
            value = int(body, 8)
        else:
            value = int(body, 10)
        return self._bounded_integer(value)

    @staticmethod
    def _apply_integer_operator(operator: str, left: int, right: int) -> int:
        if operator == "+":
            value = left + right
            return _TypeParser._bounded_integer(value)
        if operator == "-":
            value = left - right
            return _TypeParser._bounded_integer(value)
        if operator == "*":
            value = left * right
            return _TypeParser._bounded_integer(value)
        if operator in {"/", "%"}:
            if right == 0:
                raise _ParseError("division by zero")
            quotient = abs(left) // abs(right)
            if (left < 0) != (right < 0):
                quotient = -quotient
            value = quotient if operator == "/" else left - quotient * right
            return _TypeParser._bounded_integer(value)
        if operator in {"<<", ">>"}:
            if left < 0 or right < 0:
                raise _ParseError("non-portable signed shift")
            if operator == ">>" and right >= _MAX_LAYOUT_SIZE.bit_length():
                return 0
            if operator == "<<" and right >= _MAX_LAYOUT_SIZE.bit_length():
                if left == 0:
                    return 0
                raise _ParseError("layout expression exceeds supported size")
            value = left << right if operator == "<<" else left >> right
            return _TypeParser._bounded_integer(value)
        if operator in {"&", "^", "|"}:
            if left < 0 or right < 0:
                raise _ParseError("non-portable signed bitwise expression")
            if operator == "&":
                value = left & right
            elif operator == "^":
                value = left ^ right
            else:
                value = left | right
            return _TypeParser._bounded_integer(value)
        raise _ParseError("unsupported integer operator")

    @staticmethod
    def _bounded_integer(value: int) -> int:
        if abs(value) > _MAX_LAYOUT_SIZE:
            raise _ParseError("layout expression exceeds supported size")
        return value

    @staticmethod
    def _scalar_layout(scalar_name: str) -> _Layout:
        size = _SCALAR_SIZES[scalar_name]
        return _Layout(size, size, scalar_name)

    @staticmethod
    def _vector_layout(scalar: _Layout, width: int) -> _Layout:
        if scalar.scalar_name not in _SCALAR_SIZES:
            raise _ParseError("invalid vector scalar")
        storage_width = 4 if width == 3 else width
        size = scalar.size * storage_width
        return _Layout(size, size)

    @staticmethod
    def _packed_vector_layout(scalar: _Layout, width: int) -> _Layout:
        return _Layout(scalar.size * width, scalar.alignment)

    @classmethod
    def _matrix_layout(cls, scalar: _Layout, columns: int, rows: int) -> _Layout:
        column = cls._vector_layout(scalar, rows)
        stride = cls._align_up(column.size, column.alignment)
        size = cls._bounded_integer(stride * columns)
        return _Layout(size, column.alignment)

    @classmethod
    def _array_layout(cls, element: _Layout, extent: int) -> _Layout:
        stride = cls._align_up(element.size, element.alignment)
        size = cls._bounded_integer(stride * extent)
        return _Layout(size, element.alignment)

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    def _peek(self, offset: int = 0) -> Optional[str]:
        position = self.position + offset
        if position >= len(self.tokens):
            return None
        return self.tokens[position]

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise _ParseError("unexpected end of type")
        self.position += 1
        return token

    def _take_identifier(self) -> str:
        token = self._take()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) is None:
            raise _ParseError("expected type name")
        return token

    def _accept(self, token: str) -> bool:
        if self._peek() != token:
            return False
        self.position += 1
        return True

    def _expect(self, token: str) -> None:
        if not self._accept(token):
            raise _ParseError("unexpected token")


def metal_type_layout(type_text: str) -> Optional[Tuple[int, int]]:
    """Return a concrete Metal object's ``(size, alignment)`` in bytes.

    Supported inputs are Metal scalar and fixed-width aliases, ordinary and
    packed vectors, matrices, and nested fixed arrays.  Valid top-level Metal
    address-space/CV qualifiers and the ``metal::`` namespace do not affect the
    result.  Dependent extents, pointers, references, user aggregates, and
    malformed or unsupported types deliberately return ``None``.
    """

    if not isinstance(type_text, str) or len(type_text) > _MAX_TYPE_TEXT_LENGTH:
        return None
    try:
        layout = _TypeParser(type_text).parse()
        return layout.size, layout.alignment
    except (ValueError, OverflowError, RecursionError):
        return None


def metal_type_size(type_text: str) -> Optional[int]:
    """Return a concrete Metal object's size in bytes, or ``None`` if unknown."""

    layout = metal_type_layout(type_text)
    return layout[0] if layout is not None else None
