"""HLSL structured-buffer layouts on std430 targets: buffers of 32-bit words with explicit element offsets.

An HLSL structured buffer packs its elements tightly: every 32-bit scalar sits at a multiple of four bytes, a
``float3`` takes 12 bytes, ``struct { float a; float3 b; }`` puts ``b`` at byte 4 and takes 16 bytes, and an array
member's elements follow each other without padding (D3D11.3 functional specification, section 5.3.7, structured
buffers; DXC's ``-fvk-use-dx-layout``). GLSL's std430 rules, which OpenGL storage blocks and the SPIR-V emitter use,
align a three- or four-component vector to 16 bytes and a two-component one to 8, and align a struct to its widest
member (GLSL 4.60 section 7.6.2.2), so ``vec3 v[]`` has a 16-byte stride and ``b`` moves to byte 16. The generators
declared the HLSL element type unchanged, so a translated program read and wrote other bytes than the host's.

:func:`legalize_structured_layouts` rewrites, for OpenGL and SPIR-V, every buffer marked
``@buffer_layout(hlsl_structured)`` whose element type is laid out differently by the two rule sets into a buffer of
``uint`` words, and every access to it into word accesses at the HLSL offsets:

* a scalar read becomes ``asfloat(b[k])`` (``asint`` for ``int``, the word itself for ``uint``), a scalar write
  ``b[k] = asuint(value)``; a compound assignment, ``++`` or ``--`` statement reads and writes the word;
* a vector, struct or multi-component swizzle is read and written through generated helper functions that take the
  element's first word, so the index is evaluated once;
* an atomic on an ``int`` or ``uint`` scalar operates on its word;
* ``buffer_dimensions`` reports elements and the HLSL stride.

The element type may be built from 32-bit ``float``, ``int`` and ``uint`` scalars, vectors, structs and arrays of
those. Any other element type with a different std430 layout (matrices, booleans, 16- or 64-bit values), and any
access the rewrite cannot carry (the buffer passed to a function, a compound assignment whose index has effects, a
``min``/``max`` atomic on an ``int``, a float atomic), raises :class:`StructuredLayoutUnsupportedError` instead of
reading the wrong bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions, iter_structs, type_from_node
from .rewrite import call, child_slots, get_child, is_pure, set_child

__all__ = [
    "TARGETS",
    "StructuredLayoutUnsupportedError",
    "legalize_structured_layouts",
]

TARGETS = frozenset({"opengl", "glsl", "vulkan", "spirv"})
# The OpenGL generator stores struct elements as scalarized struct copies itself (glsl_hlsl_structured).
_GENERATOR_LOWERS_STRUCTS = frozenset({"opengl", "glsl"})
_WRAPPERS = frozenset({"StructuredBuffer", "RWStructuredBuffer"})
_KINDS = {"f": "float", "i": "int", "u": "uint", "b": "bool"}
_VECTOR_PREFIX = {"f": "vec", "i": "ivec", "u": "uvec", "b": "bvec"}
_READ = {"f": "asfloat", "i": "asint"}
_COMPONENTS = {"x": 0, "y": 1, "z": 2, "w": 3, "r": 0, "g": 1, "b": 2, "a": 3}
_ATOMICS = frozenset(
    {
        "atomicAdd",
        "atomicMin",
        "atomicMax",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
        "atomicExchange",
        "atomicCompSwap",
    }
)
_SIGNED_ORDER_ATOMICS = frozenset({"atomicMin", "atomicMax"})
_OUT_PARAMETER_BUILTINS = frozenset({"modf", "frexp", "buffer_dimensions"})


class StructuredLayoutUnsupportedError(TranslationDiagnostic):
    """An HLSL structured buffer whose layout or access the std430 target cannot reproduce."""

    code = f"{CODE_PREFIX}structured-layout-unsupported"


# -- shapes ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """A value in words: ``scalar`` (kind), ``vector`` (kind, lanes), ``struct`` (name, members) or ``array``."""

    form: str
    kind: str = ""
    lanes: int = 1
    name: str = ""
    members: tuple = ()  # (member name, shape, word offset)
    element: _Shape | None = None
    count: int = 0
    # An ``opaque`` value has no word lowering; ``rules`` maps "hlsl" and "std430" to its (size, alignment).
    rules: tuple = ()

    @property
    def lowerable(self) -> bool:
        if self.form == "opaque":
            return False
        if self.form == "array":
            return self.element.lowerable
        return all(member.lowerable for _name, member, _offset in self.members)

    @property
    def words(self) -> int:
        if self.form == "scalar":
            return 1
        if self.form == "vector":
            return self.lanes
        if self.form == "array":
            return self.count * self.element.words
        return sum(member.words for _name, member, _offset in self.members)

    def type_text(self) -> str:
        if self.form == "scalar":
            return _KINDS[self.kind]
        if self.form == "vector":
            return f"{_VECTOR_PREFIX[self.kind]}{self.lanes}"
        if self.form == "struct":
            return self.name
        raise StructuredLayoutUnsupportedError(
            "an array value of a structured-buffer element cannot be read or written whole"
        )

    def key(self) -> str:
        if self.form in ("scalar", "vector"):
            return self.type_text()
        if self.form == "struct":
            return self.name
        return f"{self.element.key()}_{self.count}"


def _round(value, alignment):
    return -(-value // alignment) * alignment


def _layout(shape: _Shape, rules: str) -> tuple[int, int, tuple]:
    """``(size, alignment, leaves)`` of ``shape`` under ``rules`` ("hlsl" or "std430"); leaves are (offset, size)."""
    if shape.form == "scalar":
        return 4, 4, ((0, 4),)
    if shape.form == "vector":
        alignment = 4 if rules == "hlsl" else (8 if shape.lanes == 2 else 16)
        return (
            4 * shape.lanes,
            alignment,
            tuple((4 * lane, 4) for lane in range(shape.lanes)),
        )
    if shape.form == "opaque":
        size, alignment = dict(shape.rules)[rules]
        return size, alignment, ((0, size),)
    if shape.form == "array":
        size, alignment, leaves = _layout(shape.element, rules)
        stride = _round(size, alignment)
        return (
            stride * shape.count,
            alignment,
            tuple(
                (index * stride + offset, width)
                for index in range(shape.count)
                for offset, width in leaves
            ),
        )
    offset, alignment, leaves = 0, 1, []
    for _name, member, _word in shape.members:
        size, member_alignment, member_leaves = _layout(member, rules)
        offset = _round(offset, member_alignment)
        leaves.extend((offset + start, width) for start, width in member_leaves)
        offset += size
        alignment = max(alignment, member_alignment)
    return _round(offset, alignment), alignment, tuple(leaves)


def _same_layout(shape: _Shape) -> bool:
    """Whether a runtime array of ``shape`` has the same element stride and leaf offsets under both rules."""
    hlsl_size, hlsl_alignment, hlsl_leaves = _layout(shape, "hlsl")
    glsl_size, glsl_alignment, glsl_leaves = _layout(shape, "std430")
    return (
        _round(hlsl_size, hlsl_alignment) == _round(glsl_size, glsl_alignment)
        and hlsl_leaves == glsl_leaves
    )


def _opaque(type_node, value=None, matrix=None) -> _Shape:
    """A value the rewrite cannot split into words, with its HLSL structured and std430 sizes."""
    if matrix is not None:
        columns, rows, bits = matrix
        scalar = bits // 8
        column_alignment = scalar * (2 if rows == 2 else 4)
        column_stride = _round(scalar * rows, column_alignment)
        return _Shape(
            "opaque",
            name=str(type_node),
            rules=(
                ("hlsl", (scalar * rows * columns, scalar)),
                ("std430", (column_stride * columns, column_alignment)),
            ),
        )
    scalar = max(value.bits, 32) // 8 if value.kind == "b" else value.bits // 8
    lanes = value.lanes
    glsl_alignment = scalar * (1 if lanes == 1 else 2 if lanes == 2 else 4)
    return _Shape(
        "opaque",
        name=str(type_node),
        rules=(
            ("hlsl", (scalar * lanes, scalar)),
            ("std430", (scalar * lanes, glsl_alignment)),
        ),
    )


class _Shapes:
    def __init__(self, program):
        self.structs = {}
        for struct in iter_structs(program):
            if isinstance(struct, A.StructNode) and getattr(struct, "members", None):
                self.structs.setdefault(struct.name, struct)
        self.cache = {}

    def of(self, type_node, context) -> _Shape:
        if isinstance(type_node, A.ArrayType):
            size = type_node.size
            count = size.value if isinstance(size, A.LiteralNode) else size
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise StructuredLayoutUnsupportedError(
                    f"{context}: an array without a constant element count has no structured-buffer layout"
                )
            return _Shape(
                "array", element=self.of(type_node.element_type, context), count=count
            )
        value = type_from_node(type_node)
        if value is not None:
            if value.kind not in _KINDS or (value.bits != 32 and value.kind != "b"):
                return _opaque(type_node, value=value)
            if value.lanes == 1:
                return _Shape("scalar", kind=value.kind)
            return _Shape("vector", kind=value.kind, lanes=value.lanes)
        if isinstance(type_node, A.MatrixType):
            element = type_from_node(type_node.element_type)
            if (
                element is not None
                and element.kind == "f"
                and str(type_node.rows).isdigit()
            ):
                return _opaque(
                    type_node,
                    matrix=(int(type_node.cols), int(type_node.rows), element.bits),
                )
        name = (
            getattr(type_node, "name", None)
            if not isinstance(type_node, str)
            else type_node
        )
        matrix = _matrix_spelling(name)
        if matrix is not None:
            return _opaque(type_node, matrix=matrix)
        struct = self.structs.get(name)
        if struct is None:
            raise StructuredLayoutUnsupportedError(
                f"{context}: the element type {name or type_node!r} has no word layout"
            )
        if name not in self.cache:
            members, offset = [], 0
            for member in struct.members:
                shape = self.of(member.member_type, f"{context}, member {member.name}")
                members.append((member.name, shape, offset))
                offset += shape.words
            self.cache[name] = _Shape("struct", name=name, members=tuple(members))
        return self.cache[name]


def _matrix_spelling(name):
    """``(columns, rows, bits)`` of a float matrix spelling (``float3x4`` is 3 rows of 4, ``mat3x4`` 3 columns)."""
    import re

    text = str(name or "")
    match = re.fullmatch(r"(?:float|half|double)([234])x([234])", text)
    if match:
        bits = {"f": 32, "h": 16, "d": 64}[text[0]]
        # HLSL matrices are column_major by default in buffers: rows x columns spelled, stored by column.
        return int(match.group(2)), int(match.group(1)), bits
    match = re.fullmatch(r"(d?)mat([234])(?:x([234]))?", text)
    if match:
        columns = int(match.group(2))
        rows = int(match.group(3) or match.group(2))
        return columns, rows, 64 if match.group(1) else 32
    return None


# -- accesses ----------------------------------------------------------------------------


@dataclass
class _Buffer:
    name: str
    shape: _Shape
    readonly: bool


@dataclass
class _Access:
    """A place inside a lowered buffer: its shape, constant word offset, dynamic terms and swizzle words."""

    buffer: _Buffer
    index: object
    shape: _Shape
    offset: int = 0
    terms: list = field(default_factory=list)  # (expression, words per step)
    swizzle: tuple | None = None  # word offsets of a multi-component swizzle

    def dynamic(self):
        return [self.index, *(expression for expression, _words in self.terms)]


def _uint(expression):
    return call("uint", [expression])


def _literal(value):
    return A.LiteralNode(int(value), A.PrimitiveType("uint"))


def _sum(left, right):
    return A.BinaryOpNode(left, "+", right)


def _product(left, right):
    return A.BinaryOpNode(left, "*", right)


class _Rewriter:
    def __init__(self, program, buffers, target):
        self.program = program
        self.buffers = buffers
        self.target = target
        self.helpers = {}
        self.functions = {
            function.name: function
            for function in iter_functions(program)
            if isinstance(getattr(function, "name", None), str)
        }

    def refuse(self, reason):
        raise StructuredLayoutUnsupportedError(reason, target=self.target)

    # Access chains ---------------------------------------------------------------------

    def access(self, node):
        """The :class:`_Access` ``node`` denotes, or ``None`` when it is not inside a lowered buffer."""
        steps = []
        while True:
            if isinstance(node, A.MemberAccessNode):
                steps.append(("member", node.member))
                node = node.object_expr
            elif isinstance(node, A.SwizzleNode):
                steps.append(("member", node.components))
                node = node.vector_expr
            elif isinstance(node, A.ArrayAccessNode):
                if (
                    isinstance(node.array_expr, A.IdentifierNode)
                    and node.array_expr.name in self.buffers
                ):
                    break
                steps.append(("index", node.index_expr))
                node = node.array_expr
            else:
                return None
        buffer = self.buffers[node.array_expr.name]
        place = _Access(buffer, node.index_expr, buffer.shape)
        for step, value in reversed(steps):
            self.step(place, step, value)
        return place

    def step(self, place, step, value):
        shape = place.shape
        if place.swizzle is not None:
            self.refuse(
                f"a swizzle of a swizzle of {place.buffer.name} is not supported"
            )
        if step == "member" and shape.form == "struct":
            for name, member, offset in shape.members:
                if name == value:
                    place.offset += offset
                    place.shape = member
                    return
            self.refuse(f"{shape.name} has no member {value!r}")
        if step == "member" and shape.form == "vector":
            words = [_COMPONENTS.get(character) for character in str(value)]
            if not words or None in words or max(words) >= shape.lanes:
                self.refuse(f"{value!r} is not a swizzle of a {shape.type_text()}")
            if len(words) == 1:
                place.offset += words[0]
                place.shape = _Shape("scalar", kind=shape.kind)
            else:
                place.swizzle = tuple(words)
                place.shape = _Shape("vector", kind=shape.kind, lanes=len(words))
            return
        if step == "index" and shape.form in ("array", "vector"):
            element = (
                shape.element
                if shape.form == "array"
                else _Shape("scalar", kind=shape.kind)
            )
            if (
                isinstance(value, A.LiteralNode)
                and str(value.value).rstrip("uU").isdigit()
            ):
                place.offset += int(str(value.value).rstrip("uU")) * element.words
            else:
                place.terms.append((value, element.words))
            place.shape = element
            return
        self.refuse(
            f"an access to {place.buffer.name} that the word layout cannot follow"
        )

    def base(self, place, rewritten_index, rewritten_terms):
        """The word index of ``place``'s first word."""
        expression = _product(
            _uint(rewritten_index), _literal(place.buffer.shape.words)
        )
        for term, (_original, words) in zip(rewritten_terms, place.terms):
            expression = _sum(expression, _product(_uint(term), _literal(words)))
        if place.offset:
            expression = _sum(expression, _literal(place.offset))
        return expression

    def place_base(self, place):
        index = self.expression(place.index)
        terms = [self.expression(term) for term, _words in place.terms]
        return self.base(place, index, terms)

    # Reads and writes ------------------------------------------------------------------

    def word(self, place, base, offset=0):
        index = base if not offset else _sum(base, _literal(offset))
        return A.ArrayAccessNode(A.IdentifierNode(place.buffer.name), index)

    def read(self, place):
        base = self.place_base(place)
        if place.shape.form == "scalar" and place.swizzle is None:
            return _from_word(place.shape.kind, self.word(place, base))
        return call(self.load_helper(place), [base])

    def write(self, place, value):
        base = self.place_base(place)
        if place.shape.form == "scalar" and place.swizzle is None:
            return A.AssignmentNode(
                self.word(place, base), _to_word(place.shape.kind, value)
            )
        return call(self.store_helper(place), [base, value])

    def helper_name(self, verb, place):
        if place.swizzle is not None:
            suffix = f"{place.shape.kind}_{'_'.join(map(str, place.swizzle))}"
        else:
            suffix = place.shape.key()
        return f"__cgl_{place.buffer.name}_{verb}_{suffix}"

    def load_helper(self, place):
        name = self.helper_name("load", place)
        if name not in self.helpers:
            if place.swizzle is not None:
                body = f"return {place.shape.type_text()}({_word_reads(place.buffer.name, place.shape.kind, place.swizzle)});"
            else:
                body = f"return {_value_text(place.buffer.name, place.shape, 0)};"
            self.helpers[name] = (
                f"{place.shape.type_text()} {name}(uint base) {{ {body} }}"
            )
        return name

    def store_helper(self, place):
        if place.buffer.readonly:
            self.refuse(f"{place.buffer.name} is a read-only structured buffer")
        name = self.helper_name("store", place)
        if name not in self.helpers:
            if place.swizzle is not None:
                lines = [
                    f"{place.buffer.name}[base + {offset}u] = {_word_write(place.shape.kind, f'value[{lane}]')};"
                    for lane, offset in enumerate(place.swizzle)
                ]
            else:
                lines = _store_lines(place.buffer.name, place.shape, 0, "value")
            self.helpers[name] = f"void {name}(uint base, {place.shape.type_text()} value) {{ {' '.join(lines)} }}"
        return name

    # Traversal ---------------------------------------------------------------------------

    def expression(self, node):
        """Rewrite ``node`` read as a value; returns its replacement."""
        if isinstance(node, A.IdentifierNode) and node.name in self.buffers:
            self.refuse(
                f"the structured buffer {node.name} is used other than through an element"
            )
        place = (
            self.access(node)
            if isinstance(node, (A.MemberAccessNode, A.SwizzleNode, A.ArrayAccessNode))
            else None
        )
        if place is not None:
            return self.read(place)
        if isinstance(node, A.FunctionCallNode):
            return self.function_call(node)
        if isinstance(node, A.AssignmentNode):
            self.refuse(
                "an assignment to a structured-buffer element used as a value is not supported"
            )
        if isinstance(node, A.UnaryOpNode) and node.operator in ("++", "--"):
            if self.access(node.operand) is not None:
                self.refuse(
                    "++ or -- on a structured-buffer element used as a value is not supported"
                )
        self.children(node)
        return node

    def children(self, node):
        for slot_field, key in list(child_slots(node)):
            child = get_child(node, slot_field, key)
            if isinstance(child, A.ASTNode) and not isinstance(
                child, (A.StatementNode, A.VariableNode)
            ):
                set_child(node, slot_field, key, self.expression(child))
            else:
                self.statement(child)

    def statement(self, node):
        if isinstance(node, A.ExpressionStatementNode):
            set_child(
                node, "expression", None, self.statement_expression(node.expression)
            )
            return
        if isinstance(node, A.VariableNode):
            if node.initial_value is not None:
                node.initial_value = self.expression(node.initial_value)
            return
        if not isinstance(node, A.ASTNode):
            return
        for slot_field, key in list(child_slots(node)):
            child = get_child(node, slot_field, key)
            if isinstance(child, A.AssignmentNode):
                set_child(node, slot_field, key, self.statement_expression(child))
            elif isinstance(child, (A.StatementNode, A.VariableNode)):
                self.statement(child)
            elif isinstance(child, A.ASTNode):
                set_child(node, slot_field, key, self.expression(child))

    def statement_expression(self, node):
        if isinstance(node, A.AssignmentNode):
            place = self.access(node.target)
            if place is None:
                self.children(node)
                return node
            value = self.expression(node.value)
            if node.operator == "=":
                return self.write(place, value)
            return self.update(place, node.operator[:-1], value)
        if isinstance(node, A.UnaryOpNode) and node.operator in ("++", "--"):
            place = self.access(node.operand)
            if place is not None:
                one = A.LiteralNode(
                    1.0 if place.shape.kind == "f" else 1,
                    A.PrimitiveType(_KINDS.get(place.shape.kind, "int")),
                )
                return self.update(place, node.operator[0], one)
        return self.expression(node)

    def update(self, place, operator, value):
        if not all(is_pure(expression) for expression in place.dynamic()):
            self.refuse(
                f"a compound assignment to {place.buffer.name} whose index has side effects is not supported"
            )
        current = self.read(place)
        return self.write(place, A.BinaryOpNode(current, operator, value))

    def function_call(self, node):
        name = call_name(node)
        arguments = list(node.arguments or [])
        if (
            name == "buffer_dimensions"
            and arguments
            and isinstance(arguments[0], A.IdentifierNode)
        ):
            buffer = self.buffers.get(arguments[0].name)
            if buffer is not None:
                return self.dimensions(buffer, arguments)
        if name in _ATOMICS and arguments:
            place = self.access(arguments[0])
            if place is not None:
                return self.atomic(node, name, place, arguments)
        out_positions = self.out_positions(name, len(arguments))
        for index, argument in enumerate(arguments):
            if index in out_positions and self.access(argument) is not None:
                self.refuse(
                    f"a structured-buffer element passed to an out parameter of {name}"
                )
            arguments[index] = self.expression(argument)
        node.arguments[:] = arguments
        return node

    def out_positions(self, name, count):
        function = self.functions.get(name)
        if function is None:
            return set(range(count)) if name in _OUT_PARAMETER_BUILTINS else set()
        positions = set()
        for index, parameter in enumerate(getattr(function, "parameters", None) or []):
            qualifiers = {str(q) for q in getattr(parameter, "qualifiers", None) or []}
            if qualifiers & {"out", "inout"}:
                positions.add(index)
        return positions

    def atomic(self, node, name, place, arguments):
        kind = place.shape.kind
        if (
            place.shape.form != "scalar"
            or place.swizzle is not None
            or kind not in ("i", "u")
        ):
            self.refuse(f"{name} on {place.buffer.name} needs an int or uint scalar")
        if kind == "i" and name in _SIGNED_ORDER_ATOMICS:
            self.refuse(
                f"{name} on an int member of {place.buffer.name} would compare the words unsigned"
            )
        if place.buffer.readonly:
            self.refuse(f"{place.buffer.name} is a read-only structured buffer")
        target = self.word(place, self.place_base(place))
        values = [self.expression(argument) for argument in arguments[1:]]
        if kind == "i":
            values = [_uint(value) for value in values]
        result = call(name, [target, *values])
        return call("int", [result]) if kind == "i" else result

    def dimensions(self, buffer, arguments):
        count = A.BinaryOpNode(
            _uint(call("buffer_dimensions", [A.IdentifierNode(buffer.name)])),
            "/",
            _literal(buffer.shape.words),
        )
        if len(arguments) == 1:
            return count
        name = f"__cgl_{buffer.name}_dimensions"
        if name not in self.helpers:
            self.helpers[name] = (
                f"void {name}(out uint count, out uint stride) {{ count = uint(buffer_dimensions({buffer.name})) "
                f"/ {buffer.shape.words}u; stride = {4 * buffer.shape.words}u; }}"
            )
        if len(arguments) == 3:
            return call(name, arguments[1:])
        if len(arguments) == 2:
            return A.AssignmentNode(arguments[1], count)
        self.refuse("buffer_dimensions takes one to three arguments")


def _from_word(kind, word):
    if kind == "b":
        return A.BinaryOpNode(word, "!=", _literal(0))
    return call(_READ[kind], [word]) if kind in _READ else word


def _to_word(kind, value):
    # HLSL stores bool as a 32-bit 0 or 1, which uint(bool) produces.
    return call("asuint", [value]) if kind in _READ else _uint(value)


def _word_read(buffer, kind, offset):
    word = f"{buffer}[base + {offset}u]"
    if kind == "b":
        return f"({word} != 0u)"
    return f"{_READ[kind]}({word})" if kind in _READ else word


def _word_write(kind, value):
    return f"asuint({value})" if kind in _READ else f"uint({value})"


def _word_reads(buffer, kind, offsets):
    return ", ".join(_word_read(buffer, kind, offset) for offset in offsets)


def _value_text(buffer, shape, offset):
    if shape.form == "scalar":
        return _word_read(buffer, shape.kind, offset)
    if shape.form == "vector":
        return f"{shape.type_text()}({_word_reads(buffer, shape.kind, range(offset, offset + shape.lanes))})"
    if shape.form == "struct":
        members = [
            _value_text(buffer, member, offset + start)
            for _name, member, start in shape.members
        ]
        return f"{shape.name}({', '.join(members)})"
    element = shape.element
    if element.form == "array":
        raise StructuredLayoutUnsupportedError(
            "a structured-buffer member that is an array of arrays cannot be read whole"
        )
    elements = [
        _value_text(buffer, element, offset + index * element.words)
        for index in range(shape.count)
    ]
    return f"{element.type_text()}[{shape.count}]({', '.join(elements)})"


def _store_lines(buffer, shape, offset, value):
    if shape.form == "scalar":
        return [f"{buffer}[base + {offset}u] = {_word_write(shape.kind, value)};"]
    if shape.form == "vector":
        return [
            f"{buffer}[base + {offset + lane}u] = {_word_write(shape.kind, f'{value}[{lane}]')};"
            for lane in range(shape.lanes)
        ]
    if shape.form == "struct":
        lines = []
        for name, member, start in shape.members:
            lines.extend(
                _store_lines(buffer, member, offset + start, f"{value}.{name}")
            )
        return lines
    lines = []
    for index in range(shape.count):
        lines.extend(
            _store_lines(
                buffer,
                shape.element,
                offset + index * shape.element.words,
                f"{value}[{index}]",
            )
        )
    return lines


# -- entry point -------------------------------------------------------------------------


def _declared_layout(declaration):
    for attribute in getattr(declaration, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() != "buffer_layout":
            continue
        for argument in getattr(attribute, "arguments", None) or []:
            value = getattr(argument, "name", getattr(argument, "value", argument))
            return str(value).strip().strip('"').lower()
    return None


def _lowered_buffers(program, shapes, target):
    buffers = {}
    for declaration in getattr(program, "global_variables", None) or []:
        var_type = getattr(declaration, "var_type", None)
        if (
            _declared_layout(declaration) != "hlsl_structured"
            or not isinstance(var_type, A.NamedType)
            or var_type.name not in _WRAPPERS
            or len(var_type.generic_args) != 1
        ):
            continue
        element = var_type.generic_args[0]
        context = f"structured buffer {declaration.name}"
        shape = shapes.of(element, context)
        if _same_layout(shape) and not _holds_bool(shape):
            continue
        if shape.form == "struct" and target in _GENERATOR_LOWERS_STRUCTS:
            continue
        if not shape.lowerable:
            raise StructuredLayoutUnsupportedError(
                f"{context}: its element type is laid out differently by HLSL structured buffers and std430, and "
                "holds a matrix, boolean, 16-bit or 64-bit value that cannot be split into 32-bit words",
                target=target,
            )
        buffers[declaration.name] = _Buffer(
            declaration.name, shape, var_type.name == "StructuredBuffer"
        )
        declaration.var_type = A.NamedType(var_type.name, [A.PrimitiveType("uint")])
        declaration.vtype = declaration.var_type
    return buffers


def _holds_bool(shape):
    if shape.form in ("scalar", "vector"):
        return shape.kind == "b"
    if shape.form == "array":
        return _holds_bool(shape.element)
    return any(_holds_bool(member) for _name, member, _offset in shape.members)


def _parse_helpers(texts):
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    source = "shader __cgl_structured_layouts {\n" + "\n".join(texts) + "\n}\n"
    return list(SOURCE_REGISTRY.get("cgl").parse(source).functions)


def legalize_structured_layouts(program, target) -> int:
    """Rewrite HLSL structured buffers whose layout std430 changes into word buffers; returns how many."""
    target = str(target or "")
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    shapes = _Shapes(program)
    buffers = _lowered_buffers(program, shapes, target)
    if not buffers:
        return 0
    rewriter = _Rewriter(program, buffers, target)
    for function in list(iter_functions(program)):
        if getattr(function, "body", None) is not None:
            rewriter.statement(function.body)
    if rewriter.helpers:
        functions = getattr(program, "functions", None)
        if not isinstance(functions, list):
            raise StructuredLayoutUnsupportedError(
                "the program has no function list to receive the structured-buffer helpers",
                target=target,
            )
        functions[0:0] = _parse_helpers(
            [rewriter.helpers[name] for name in sorted(rewriter.helpers)]
        )
    return len(buffers)
