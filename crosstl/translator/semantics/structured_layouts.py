"""Declared buffer layouts on targets with other buffer rules: buffers of 32-bit words with explicit offsets.

A structured buffer's element layout belongs to the language that declared it, and the generators declared the
element type unchanged, so a translated program read and wrote other bytes than the host's:

* an HLSL structured buffer packs its elements tightly: every 32-bit scalar sits at a multiple of four bytes, a
  ``float3`` takes 12 bytes, ``struct { float a; float3 b; }`` puts ``b`` at byte 4 and takes 16 bytes, and an
  array member's elements follow each other without padding (D3D11.3 functional specification, section 5.3.7;
  DXC's ``-fvk-use-dx-layout``). CrossGL marks such buffers ``@buffer_layout(hlsl_structured)``;
* an MSL ``device`` buffer uses the natural layout of its element type: ``float2`` is 8-byte and ``float3`` and
  ``float4`` 16-byte sized and aligned, so ``device float3*`` has a 16-byte stride and ``struct { float3 a;
  float b; }`` puts ``b`` at byte 16 and takes 32 bytes (MSL 3.2 section 2.2 and table 2.4). A packed vector member
  (``packed_float3``, ``@ packed vec3`` in CrossGL) is aligned to its scalar and takes 12 bytes (section 2.2.3).
  CrossGL marks such buffers ``@buffer_layout(metal)`` where that differs from the HLSL or std430 layout;
* GLSL's std430 rules, which OpenGL storage blocks and the SPIR-V emitter use, align a three- or four-component
  vector to 16 bytes and a two-component one to 8, size a ``vec3`` 12 bytes and align a struct to its widest member
  (GLSL 4.60 section 7.6.2.2), so ``vec3 v[]`` has a 16-byte stride and the two structs above take 16 bytes.

:func:`legalize_structured_layouts` rewrites, for OpenGL and SPIR-V (std430) and for HLSL (structured-buffer
rules), every marked buffer whose element type the target lays out differently into a buffer of ``uint`` words,
and every access to it into word accesses at the declared offsets:

* a scalar read becomes ``asfloat(b[k])`` (``asint`` for ``int``, the word itself for ``uint``), a scalar write
  ``b[k] = asuint(value)``; a compound assignment, ``++`` or ``--`` statement reads and writes the word;
* a vector, struct or multi-component swizzle is read and written through generated helper functions that take the
  element's first word, so the index is evaluated once;
* an atomic on an ``int`` or ``uint`` scalar operates on its word;
* ``buffer_dimensions`` reports elements and the declared stride.

An append or consume buffer cannot be indexed, and its hidden counter counts elements, so it keeps its kind and
holds a struct of the element's words instead (``struct __cgl_words3 { uint w[3]; }``, which every rule set lays
out as 12 tightly packed bytes): ``buffer_append(b, v)`` appends ``__cgl_pack_hlsl_vec3(v)`` and
``buffer_consume(b)`` becomes ``__cgl_unpack_hlsl_vec3(buffer_consume(b))``.

An array of buffers (``RWStructuredBuffer<T> b[N]``) becomes an array of word buffers; an access ``b[k][i]`` reads
the words of ``b[k]``, and a helper function takes ``k`` before the first word.

The element type may be built from 32-bit ``float``, ``int`` and ``uint`` scalars, vectors, structs and arrays of
those. Any other element type the target lays out differently (matrices, MSL's one-byte booleans, 16- or 64-bit
values), and any access the rewrite cannot carry (the buffer passed to a function, a compound assignment whose index
has effects, a ``min``/``max`` atomic on an ``int``, a float atomic), raises
:class:`StructuredLayoutUnsupportedError` instead of reading the wrong bytes. Padding words of an MSL layout are
never written.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

from .. import ast as A
from ..codegen.buffer_member_layout import member_is_packed
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions, iter_structs, type_from_node
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child

__all__ = [
    "TARGETS",
    "StructuredLayoutUnsupportedError",
    "element_stride",
    "legalize_structured_layouts",
]

HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
TARGETS = frozenset({"opengl", "glsl", "vulkan", "spirv"}) | HLSL_TARGETS
# ``@buffer_layout`` values this pass reproduces, and the rules they name.
_SOURCE_RULES = {"hlsl_structured": "hlsl", "metal": "metal"}
_RULE_NAMES = {
    "hlsl": "HLSL structured buffers",
    "std430": "std430",
    "metal": "MSL",
}
# The OpenGL generator stores the struct elements of global buffers as scalarized struct copies itself
# (glsl_hlsl_structured); buffer parameters are lowered here.
_GENERATOR_LOWERS_STRUCTS = frozenset({"opengl", "glsl"})
_WRAPPERS = frozenset({"StructuredBuffer", "RWStructuredBuffer"})
# Buffers reached only through buffer_append, buffer_consume and buffer_dimensions, whose counter counts elements.
_QUEUES = frozenset({"AppendStructuredBuffer", "ConsumeStructuredBuffer"})
_QUEUE_CALLS = {
    "buffer_append": "AppendStructuredBuffer",
    "buffer_consume": "ConsumeStructuredBuffer",
}
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
# Fields that hold a statement, where a block may replace it (the slot of a loop header holds an expression).
_STATEMENT_SLOTS = frozenset(
    {
        "statements",
        "then_branch",
        "else_branch",
        "if_body",
        "else_body",
        "else_if_bodies",
        "body",
    }
)


class StructuredLayoutUnsupportedError(TranslationDiagnostic):
    """A buffer with a declared layout whose layout or access the target cannot reproduce."""

    code = f"{CODE_PREFIX}structured-layout-unsupported"


# -- shapes ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """A value in words: ``scalar`` (kind), ``vector`` (kind, lanes), ``matrix`` (float columns: ``count`` columns
    of ``lanes`` rows), ``struct`` (name, members) or ``array``.

    ``source`` names the rules (``hlsl`` or ``metal``) the member word offsets and :attr:`words` follow.
    """

    form: str
    kind: str = ""
    lanes: int = 1
    name: str = ""
    members: tuple = ()  # (member name, shape, word offset)
    element: _Shape | None = None
    count: int = 0
    # An ``opaque`` value has no word lowering: (kind, bits, rows, columns), columns 0 for a scalar or vector.
    opaque: tuple = ()
    source: str = "hlsl"
    # A vector stored in MSL's packed form (a ``@ packed`` member): aligned to its scalar under the MSL rules.
    packed: bool = False

    @property
    def lowerable(self) -> bool:
        if self.form == "opaque":
            return False
        if self.form == "array":
            return self.element.lowerable
        return all(member.lowerable for _name, member, _offset in self.members)

    @property
    def words(self) -> int:
        """The element stride of an array of this value under its source rules, in 32-bit words."""
        size, alignment, _leaves = _layout(self, self.source)
        return _round(size, alignment) // 4

    def type_text(self) -> str:
        if self.form == "scalar":
            return _KINDS[self.kind]
        if self.form == "vector":
            return f"{_VECTOR_PREFIX[self.kind]}{self.lanes}"
        if self.form == "struct":
            return self.name
        if self.form == "matrix":
            return (
                f"mat{self.count}"
                if self.count == self.lanes
                else f"mat{self.count}x{self.lanes}"
            )
        raise StructuredLayoutUnsupportedError(
            "an array value of a structured-buffer element cannot be read or written whole"
        )

    def key(self) -> str:
        if self.form in ("scalar", "vector", "matrix"):
            return self.type_text()
        if self.form == "struct":
            return self.name
        return f"{self.element.key()}_{self.count}"


def _round(value, alignment):
    return -(-value // alignment) * alignment


def _vector_layout(scalar, lanes, rules, packed=False):
    """``(size, alignment, leaves)`` of a vector of ``lanes`` scalars of ``scalar`` bytes under ``rules``.

    HLSL structured buffers align a vector to its scalar (D3D11.3 section 5.3.7); std430 aligns a two-component
    vector to twice and a three- or four-component one to four times its scalar (GLSL 4.60 section 7.6.2.2); MSL
    sizes and aligns ``T3`` like ``T4`` (MSL 3.2 table 2.4), and a ``packed`` vector like its scalars. A target
    declares a packed vector as a plain one, so only the MSL rules read ``packed``.
    """
    leaves = tuple((scalar * lane, scalar) for lane in range(lanes))
    if rules == "hlsl" or lanes == 1 or (packed and rules == "metal"):
        return scalar * lanes, scalar, leaves
    if rules == "metal":
        size = scalar * (4 if lanes == 3 else lanes)
        return size, size, leaves
    return scalar * lanes, scalar * (2 if lanes == 2 else 4), leaves


def _opaque_layout(opaque, rules):
    kind, bits, rows, columns = opaque
    # MSL bool is one byte; HLSL and GLSL store it in 32 bits.
    scalar = (1 if rules == "metal" else 4) if kind == "b" else bits // 8
    if not columns:
        return _vector_layout(scalar, rows, rules)
    size, alignment, leaves = _vector_layout(scalar, rows, rules)
    stride = _round(size, alignment)
    return (
        stride * columns,
        alignment,
        tuple(
            (column * stride + offset, width)
            for column in range(columns)
            for offset, width in leaves
        ),
    )


@lru_cache(maxsize=None)
def _layout(shape: _Shape, rules: str) -> tuple[int, int, tuple]:
    """``(size, alignment, leaves)`` of ``shape`` under ``rules`` (``hlsl``, ``std430`` or ``metal``); leaves are
    ``(offset, size)`` pairs of its scalars."""
    if shape.form == "scalar":
        return _vector_layout(4, 1, rules)
    if shape.form == "vector":
        return _vector_layout(4, shape.lanes, rules, shape.packed)
    if shape.form == "opaque":
        return _opaque_layout(shape.opaque, rules)
    if shape.form == "matrix":
        return _opaque_layout(("f", 32, shape.lanes, shape.count), rules)
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


def _same_layout(shape: _Shape, target_rules: str) -> bool:
    """Whether a runtime array of ``shape`` has the same element stride and leaf offsets under its source rules
    and ``target_rules``."""
    source_size, source_alignment, source_leaves = _layout(shape, shape.source)
    target_size, target_alignment, target_leaves = _layout(shape, target_rules)
    return (
        _round(source_size, source_alignment) == _round(target_size, target_alignment)
        and source_leaves == target_leaves
    )


class _Shapes:
    """The shapes of element types under one source layout (``hlsl`` or ``metal``)."""

    def __init__(self, program, source):
        self.source = source
        self.structs = {}
        for struct in iter_structs(program):
            if isinstance(struct, A.StructNode) and getattr(struct, "members", None):
                self.structs.setdefault(struct.name, struct)
        self.cache = {}

    def opaque(self, type_node, kind, bits, rows, columns=0):
        return _Shape(
            "opaque",
            name=str(type_node),
            opaque=(kind, bits, rows, columns),
            source=self.source,
        )

    def matrix(self, type_node, bits, rows, columns):
        # An MSL matrix is an array of column vectors, the columns CrossGL indexes, so its words can be addressed;
        # the orientation of an HLSL structured-buffer matrix depends on its packing and is not split.
        if self.source == "metal" and bits == 32:
            return _Shape(
                "matrix", kind="f", lanes=rows, count=columns, source=self.source
            )
        return self.opaque(type_node, "f", bits, rows, columns)

    def of(self, type_node, context) -> _Shape:
        if isinstance(type_node, A.ArrayType):
            size = type_node.size
            count = size.value if isinstance(size, A.LiteralNode) else size
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise StructuredLayoutUnsupportedError(
                    f"{context}: an array without a constant element count has no structured-buffer layout"
                )
            return _Shape(
                "array",
                element=self.of(type_node.element_type, context),
                count=count,
                source=self.source,
            )
        value = type_from_node(type_node)
        if value is not None:
            if (
                value.kind not in _KINDS
                or (value.bits != 32 and value.kind != "b")
                or (value.kind == "b" and self.source == "metal")
            ):
                return self.opaque(type_node, value.kind, value.bits, value.lanes)
            if value.lanes == 1:
                return _Shape("scalar", kind=value.kind, source=self.source)
            return _Shape(
                "vector", kind=value.kind, lanes=value.lanes, source=self.source
            )
        if isinstance(type_node, A.MatrixType):
            element = type_from_node(type_node.element_type)
            if (
                element is not None
                and element.kind == "f"
                and str(type_node.rows).isdigit()
            ):
                return self.matrix(
                    type_node, element.bits, int(type_node.rows), int(type_node.cols)
                )
        name = (
            getattr(type_node, "name", None)
            if not isinstance(type_node, str)
            else type_node
        )
        matrix = _matrix_spelling(name)
        if matrix is not None:
            columns, rows, bits = matrix
            return self.matrix(type_node, bits, rows, columns)
        struct = self.structs.get(name)
        if struct is None:
            raise StructuredLayoutUnsupportedError(
                f"{context}: the element type {name or type_node!r} has no word layout"
            )
        if name not in self.cache:
            members, offset = [], 0
            for member in struct.members:
                member_context = f"{context}, member {member.name}"
                shape = self.of(member.member_type, member_context)
                if member_is_packed(member):
                    shape = _packed_shape(shape, member_context)
                size, alignment, _leaves = _layout(shape, self.source)
                offset = _round(offset, alignment)
                members.append((member.name, shape, offset // 4))
                offset += size
            self.cache[name] = _Shape(
                "struct", name=name, members=tuple(members), source=self.source
            )
        return self.cache[name]


def element_stride(program, type_node, rules, context, cache=None) -> int:
    """The array stride in bytes of ``type_node`` under ``rules`` (``hlsl``, ``std430`` or ``metal``).

    ``cache`` maps rules to the :class:`_Shapes` of ``program`` already built for them. Raises
    :class:`StructuredLayoutUnsupportedError` for a type without a layout (an unsized array member, an unknown
    struct).
    """
    cache = {} if cache is None else cache
    source = "metal" if rules == "metal" else "hlsl"
    if source not in cache:
        cache[source] = _Shapes(program, source)
    size, alignment, _leaves = _layout(cache[source].of(type_node, context), rules)
    return _round(size, alignment)


def _packed_shape(shape, context):
    """``shape`` stored as MSL packed vectors: a vector, or an array of vectors, of a ``@ packed`` member."""
    if shape.form == "vector":
        return replace(shape, packed=True)
    if shape.form == "array" and shape.element.form == "vector":
        return replace(shape, element=replace(shape.element, packed=True))
    raise StructuredLayoutUnsupportedError(
        f"{context}: only vectors and arrays of vectors have a packed form"
    )


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
    inline: bool = False  # a parameter, which generated helper functions cannot name
    queue: str = ""  # AppendStructuredBuffer or ConsumeStructuredBuffer: elements are word structs
    array: bool = False  # an array of buffers, whose elements are selected before they are indexed


@dataclass
class _Access:
    """A place inside a lowered buffer: its shape, constant word offset, dynamic terms and swizzle words."""

    buffer: _Buffer
    index: object
    shape: _Shape
    offset: int = 0
    terms: list = field(default_factory=list)  # (expression, words per step)
    swizzle: tuple | None = None  # word offsets of a multi-component swizzle
    selector: object = None  # the index of the buffer in an array of buffers
    rewritten_selector: object = None

    def dynamic(self):
        selector = [] if self.selector is None else [self.selector]
        return [
            *selector,
            self.index,
            *(expression for expression, _words in self.terms),
        ]


def _uint(expression):
    return call("uint", [expression])


def _literal(value):
    return A.LiteralNode(int(value), A.PrimitiveType("uint"))


def _sum(left, right):
    return A.BinaryOpNode(left, "+", right)


def _product(left, right):
    return A.BinaryOpNode(left, "*", right)


class _Rewriter:
    def __init__(self, program, buffers, target, parameter_plans=None):
        self.program = program
        self.buffers = buffers
        self.target = target
        # Function name -> the plan of each parameter of each overload (a :class:`_Buffer` where lowered).
        self.parameter_plans = parameter_plans or {}
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
                selected = self.selected_buffer(node.array_expr)
                if selected is not None:
                    buffer, selector = selected
                    break
                if (
                    isinstance(node.array_expr, A.IdentifierNode)
                    and node.array_expr.name in self.buffers
                ):
                    buffer, selector = self.buffers[node.array_expr.name], None
                    if buffer.array:
                        # One buffer of an array of buffers, not an element.
                        return None
                    break
                steps.append(("index", node.index_expr))
                node = node.array_expr
            else:
                return None
        if buffer.queue:
            self.refuse(f"the {buffer.queue} {buffer.name} cannot be indexed")
        place = _Access(buffer, node.index_expr, buffer.shape, selector=selector)
        for step, value in reversed(steps):
            self.step(place, step, value)
        return place

    def selected_buffer(self, node):
        """``(buffer, selector)`` when ``node`` is ``b[k]`` for a lowered array of buffers ``b``, else ``None``."""
        if (
            isinstance(node, A.ArrayAccessNode)
            and isinstance(node.array_expr, A.IdentifierNode)
            and node.array_expr.name in self.buffers
            and self.buffers[node.array_expr.name].array
        ):
            return self.buffers[node.array_expr.name], node.index_expr
        return None

    def place_selector(self, place):
        """The rewritten selector of ``place``'s buffer, or ``None`` outside an array of buffers."""
        if place.selector is None:
            return None
        if place.rewritten_selector is None:
            place.rewritten_selector = self.expression(place.selector)
            return place.rewritten_selector
        return clone(place.rewritten_selector)

    def buffer_reference(self, place):
        name = A.IdentifierNode(place.buffer.name)
        selector = self.place_selector(place)
        return name if selector is None else A.ArrayAccessNode(name, selector)

    def helper_arguments(self, place, base, *rest):
        selector = self.place_selector(place)
        leading = [] if selector is None else [_uint(selector)]
        return [*leading, base, *rest]

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
                place.shape = _Shape("scalar", kind=shape.kind, source=shape.source)
            else:
                place.swizzle = tuple(words)
                place.shape = _Shape(
                    "vector", kind=shape.kind, lanes=len(words), source=shape.source
                )
            return
        if step == "index" and shape.form in ("array", "vector", "matrix"):
            if shape.form == "array":
                element = shape.element
            elif shape.form == "matrix":
                element = _Shape(
                    "vector", kind=shape.kind, lanes=shape.lanes, source=shape.source
                )
            else:
                element = _Shape("scalar", kind=shape.kind, source=shape.source)
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
        return A.ArrayAccessNode(self.buffer_reference(place), index)

    def read(self, place):
        base = self.place_base(place)
        if place.shape.form == "scalar" and place.swizzle is None:
            return _from_word(place.shape.kind, self.word(place, base))
        if place.buffer.inline:
            return self.inline_read(place, base)
        return call(self.load_helper(place), self.helper_arguments(place, base))

    def write(self, place, value):
        """The assignment, helper call or (for a parameter) block statement that stores ``value``."""
        base = self.place_base(place)
        if place.shape.form == "scalar" and place.swizzle is None:
            if place.buffer.readonly:
                self.refuse(f"{place.buffer.name} is a read-only structured buffer")
            return A.AssignmentNode(
                self.word(place, base), _to_word(place.shape.kind, value)
            )
        if place.buffer.inline:
            return self.inline_write(place, base, value)
        return call(self.store_helper(place), self.helper_arguments(place, base, value))

    # A helper function cannot name a buffer parameter, so parameter accesses are expanded in place: a read repeats
    # its (side-effect free) first-word index, and a write becomes a block that evaluates the index and the value
    # once into reserved temporaries.

    def inline_read(self, place, base):
        if not is_pure(base):
            self.refuse(
                f"a read of a vector or struct from the buffer parameter {place.buffer.name} at an index with side "
                "effects is not supported"
            )
        if place.swizzle is not None:
            words = _word_reads(place.buffer.name, place.shape.kind, place.swizzle)
            text = f"{place.shape.type_text()}({words})"
        else:
            text = _value_text(place.buffer.name, place.shape, 0)
        function = _parse_helpers(
            [f"{place.shape.type_text()} __cgl_inline(uint base) {{ return {text}; }}"]
        )[0]
        statement = function.body.statements[0]
        return _substitute(statement.value, {"base": base})

    def inline_write(self, place, base, value):
        if place.buffer.readonly:
            self.refuse(f"{place.buffer.name} is a read-only structured buffer")
        if place.swizzle is not None:
            lines = [
                f"{place.buffer.name}[__cgl_base + {offset}u] = "
                f"{_word_write(place.shape.kind, f'__cgl_value[{lane}]')};"
                for lane, offset in enumerate(place.swizzle)
            ]
        else:
            lines = _store_lines(
                place.buffer.name, place.shape, 0, "__cgl_value", "__cgl_base"
            )
        type_text = place.shape.type_text()
        function = _parse_helpers(
            [
                f"void __cgl_inline() {{ {{ uint __cgl_base = __cgl_base_in; {type_text} __cgl_value = "
                f"__cgl_value_in; {' '.join(lines)} }} }}"
            ]
        )[0]
        block = function.body.statements[0]
        return _substitute(block, {"__cgl_base_in": base, "__cgl_value_in": value})

    def helper_name(self, verb, place):
        if place.swizzle is not None:
            suffix = f"{place.shape.kind}_{'_'.join(map(str, place.swizzle))}"
        else:
            suffix = place.shape.key()
        return f"__cgl_{place.buffer.name}_{verb}_{suffix}"

    def load_helper(self, place):
        name = self.helper_name("load", place)
        if name not in self.helpers:
            buffer, selector = _helper_buffer(place.buffer)
            if place.swizzle is not None:
                body = f"return {place.shape.type_text()}({_word_reads(buffer, place.shape.kind, place.swizzle)});"
            else:
                body = f"return {_value_text(buffer, place.shape, 0)};"
            self.helpers[name] = (
                f"{place.shape.type_text()} {name}({selector}uint base) {{ {body} }}"
            )
        return name

    def store_helper(self, place):
        if place.buffer.readonly:
            self.refuse(f"{place.buffer.name} is a read-only structured buffer")
        name = self.helper_name("store", place)
        if name not in self.helpers:
            buffer, selector = _helper_buffer(place.buffer)
            if place.swizzle is not None:
                lines = [
                    f"{buffer}[base + {offset}u] = {_word_write(place.shape.kind, f'value[{lane}]')};"
                    for lane, offset in enumerate(place.swizzle)
                ]
            else:
                lines = _store_lines(buffer, place.shape, 0, "value")
            self.helpers[name] = f"void {name}({selector}uint base, {place.shape.type_text()} value) {{ {' '.join(lines)} }}"
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
            result = self.function_call(node)
            if isinstance(result, A.BlockNode):
                self.refuse(
                    "buffer_dimensions with count and stride outputs on a buffer parameter is supported only as "
                    "a statement"
                )
            return result
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
            replacement = self.statement_expression(node.expression)
            if isinstance(replacement, A.BlockNode):
                self.refuse(
                    "a store to a buffer parameter in a statement position that cannot hold a block is not "
                    "supported"
                )
            set_child(node, "expression", None, replacement)
            return
        if isinstance(node, A.VariableNode):
            if node.initial_value is not None:
                node.initial_value = self.expression(node.initial_value)
            return
        if not isinstance(node, A.ASTNode):
            return
        for slot_field, key in list(child_slots(node)):
            child = get_child(node, slot_field, key)
            holds_statements = isinstance(node, A.BlockNode) or (
                slot_field in _STATEMENT_SLOTS
            )
            if isinstance(child, A.ExpressionStatementNode) and holds_statements:
                replacement = self.statement_expression(child.expression)
                if isinstance(replacement, A.BlockNode):
                    set_child(node, slot_field, key, replacement)
                else:
                    set_child(child, "expression", None, replacement)
            elif isinstance(child, A.AssignmentNode):
                replacement = self.statement_expression(child)
                if isinstance(replacement, A.BlockNode) and not holds_statements:
                    self.refuse(
                        "a store to a buffer parameter in a loop header or other expression position is not "
                        "supported"
                    )
                set_child(node, slot_field, key, replacement)
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
        if isinstance(node, A.FunctionCallNode):
            return self.function_call(node)
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
            name in _QUEUE_CALLS
            and arguments
            and isinstance(arguments[0], A.IdentifierNode)
            and arguments[0].name in self.buffers
        ):
            return self.queue_call(node, name, self.buffers[arguments[0].name])
        if name == "buffer_dimensions" and arguments:
            selected = self.selected_buffer(arguments[0])
            if selected is not None:
                buffer, selector = selected
                return self.dimensions(buffer, arguments, self.expression(selector))
            if isinstance(arguments[0], A.IdentifierNode):
                buffer = self.buffers.get(arguments[0].name)
                if buffer is not None and not buffer.array:
                    return self.dimensions(buffer, arguments)
        if name in _ATOMICS and arguments:
            place = self.access(arguments[0])
            if place is not None:
                return self.atomic(node, name, place, arguments)
        out_positions = self.out_positions(name, len(arguments))
        for index, argument in enumerate(arguments):
            passed = self.passed_buffer(name, index, argument)
            if passed is not None:
                arguments[index] = passed
                continue
            if index in out_positions and self.access(argument) is not None:
                self.refuse(
                    f"a structured-buffer element passed to an out parameter of {name}"
                )
            arguments[index] = self.expression(argument)
        node.arguments[:] = arguments
        return node

    def passed_buffer(self, name, index, argument):
        """The argument that passes a lowered buffer, or a pointer into it, to buffer parameter ``index`` of the
        function ``name``, or ``None`` when ``argument`` is not such a buffer.

        The parameter must have been lowered to the same words, so the function reads the caller's words; an
        element offset becomes a word offset.
        """
        base, offset = argument, None
        if isinstance(argument, A.BinaryOpNode) and argument.operator in ("+", "-"):
            base, offset = argument.left, argument.right
        if not (isinstance(base, A.IdentifierNode) and base.name in self.buffers):
            return None
        buffer = self.buffers[base.name]
        plans = [
            parameters[index] if index < len(parameters) else None
            for parameters in self.parameter_plans.get(name, [])
        ]
        if not plans or not all(
            isinstance(plan, _Buffer) and plan.shape == buffer.shape for plan in plans
        ):
            self.refuse(
                f"the structured buffer {base.name} is passed to {name}, whose parameter does not take its "
                "layout"
            )
        if any(buffer.readonly and not plan.readonly for plan in plans):
            self.refuse(
                f"the read-only structured buffer {base.name} is passed to a writable parameter of {name}"
            )
        if offset is None:
            return base
        words = _product(_uint(self.expression(offset)), _literal(buffer.shape.words))
        return A.BinaryOpNode(base, argument.operator, words)

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

    def queue_call(self, node, name, buffer):
        """``buffer_append(b, v)`` with ``v`` packed into words, or ``buffer_consume(b)`` unpacked from words."""
        arguments = node.arguments
        if buffer.queue != _QUEUE_CALLS[name]:
            self.refuse(
                f"{name} cannot take {buffer.name}, which is declared as {buffer.queue}"
            )
        if len(arguments) != (2 if name == "buffer_append" else 1):
            self.refuse(f"{name} on {buffer.name} takes the wrong number of arguments")
        words = _words_struct_name(buffer.shape.words)
        type_text = buffer.shape.type_text()
        suffix = f"{buffer.shape.source}_{buffer.shape.key()}"
        if name == "buffer_append":
            helper = f"__cgl_pack_{suffix}"
            if helper not in self.helpers:
                lines = _store_lines("words.w", buffer.shape, 0, "value", "0u")
                if len(lines) != buffer.shape.words:
                    # Padding words are stored as zero rather than left undefined.
                    lines[0:0] = [
                        f"words.w[{word}] = 0u;" for word in range(buffer.shape.words)
                    ]
                self.helpers[helper] = f"{words} {helper}({type_text} value) {{ {words} words; {' '.join(lines)} return words; }}"
            node.arguments[1] = call(helper, [self.expression(arguments[1])])
            return node
        helper = f"__cgl_unpack_{suffix}"
        if helper not in self.helpers:
            value = _value_text("words.w", buffer.shape, 0, "0u")
            self.helpers[helper] = (
                f"{type_text} {helper}({words} words) {{ return {value}; }}"
            )
        return call(helper, [node])

    def dimensions(self, buffer, arguments, selector=None):
        reference = A.IdentifierNode(buffer.name)
        if selector is not None:
            reference = A.ArrayAccessNode(reference, selector)
        count = _uint(call("buffer_dimensions", [reference]))
        if not buffer.queue:
            # A queue's elements are word structs, so its length already counts elements.
            count = A.BinaryOpNode(count, "/", _literal(buffer.shape.words))
        if len(arguments) == 1:
            return count
        if len(arguments) == 3 and buffer.inline:
            stride = _literal(4 * buffer.shape.words)
            return A.BlockNode(
                [
                    A.AssignmentNode(arguments[1], count),
                    A.AssignmentNode(arguments[2], stride),
                ]
            )
        name = f"__cgl_{buffer.name}_dimensions"
        if name not in self.helpers:
            text, leading = _helper_buffer(buffer)
            per_element = "" if buffer.queue else f" / {buffer.shape.words}u"
            self.helpers[name] = (
                f"void {name}({leading}out uint count, out uint stride) {{ count = uint(buffer_dimensions({text}))"
                f"{per_element}; stride = {4 * buffer.shape.words}u; }}"
            )
        if len(arguments) == 3:
            leading = [] if selector is None else [_uint(clone(selector))]
            return call(name, [*leading, *arguments[1:]])
        if len(arguments) == 2:
            return A.AssignmentNode(arguments[1], count)
        self.refuse("buffer_dimensions takes one to three arguments")


def _words_struct_name(words):
    return f"__cgl_words{words}"


def _helper_buffer(buffer):
    """How a helper function names ``buffer``'s words, and the parameter that selects them in an array of buffers."""
    if buffer.array:
        return f"{buffer.name}[selector]", "uint selector, "
    return buffer.name, ""


def _from_word(kind, word):
    if kind == "b":
        return A.BinaryOpNode(word, "!=", _literal(0))
    return call(_READ[kind], [word]) if kind in _READ else word


def _to_word(kind, value):
    # HLSL stores bool as a 32-bit 0 or 1, which uint(bool) produces.
    return call("asuint", [value]) if kind in _READ else _uint(value)


def _word_read(buffer, kind, offset, base="base"):
    word = f"{buffer}[{base} + {offset}u]"
    if kind == "b":
        return f"({word} != 0u)"
    return f"{_READ[kind]}({word})" if kind in _READ else word


def _word_write(kind, value):
    return f"asuint({value})" if kind in _READ else f"uint({value})"


def _word_reads(buffer, kind, offsets, base="base"):
    return ", ".join(_word_read(buffer, kind, offset, base) for offset in offsets)


def _value_text(buffer, shape, offset, base="base"):
    if shape.form == "scalar":
        return _word_read(buffer, shape.kind, offset, base)
    if shape.form == "vector":
        offsets = range(offset, offset + shape.lanes)
        return f"{shape.type_text()}({_word_reads(buffer, shape.kind, offsets, base)})"
    if shape.form == "struct":
        members = [
            _value_text(buffer, member, offset + start, base)
            for _name, member, start in shape.members
        ]
        return f"{shape.name}({', '.join(members)})"
    if shape.form == "matrix":
        column = _Shape(
            "vector", kind=shape.kind, lanes=shape.lanes, source=shape.source
        )
        columns = [
            _value_text(buffer, column, offset + index * column.words, base)
            for index in range(shape.count)
        ]
        return f"{shape.type_text()}({', '.join(columns)})"
    element = shape.element
    if element.form == "array":
        raise StructuredLayoutUnsupportedError(
            "a structured-buffer member that is an array of arrays cannot be read whole"
        )
    elements = [
        _value_text(buffer, element, offset + index * element.words, base)
        for index in range(shape.count)
    ]
    return f"{element.type_text()}[{shape.count}]({', '.join(elements)})"


def _store_lines(buffer, shape, offset, value, base="base"):
    if shape.form == "scalar":
        return [f"{buffer}[{base} + {offset}u] = {_word_write(shape.kind, value)};"]
    if shape.form == "vector":
        return [
            f"{buffer}[{base} + {offset + lane}u] = {_word_write(shape.kind, f'{value}[{lane}]')};"
            for lane in range(shape.lanes)
        ]
    if shape.form == "struct":
        lines = []
        for name, member, start in shape.members:
            lines.extend(
                _store_lines(buffer, member, offset + start, f"{value}.{name}", base)
            )
        return lines
    element = shape.element
    if shape.form == "matrix":
        element = _Shape(
            "vector", kind=shape.kind, lanes=shape.lanes, source=shape.source
        )
    lines = []
    for index in range(shape.count):
        lines.extend(
            _store_lines(
                buffer,
                element,
                offset + index * element.words,
                f"{value}[{index}]",
                base,
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


def _buffer_type(declaration):
    """The ``(type slot, type, element type, source rules)`` of a structured-buffer declaration, buffer parameter,
    buffer pointer parameter or bound runtime-array parameter (``var<storage> x: array<T>``) with a layout.
    """
    slot = "param_type" if isinstance(declaration, A.ParameterNode) else "var_type"
    var_type = getattr(declaration, slot, None)
    source = _SOURCE_RULES.get(_declared_layout(declaration))
    if source is None:
        return None
    if slot == "param_type" and isinstance(var_type, A.PointerType):
        return slot, var_type, var_type.pointee_type, source
    if (
        slot == "param_type"
        and isinstance(var_type, A.ArrayType)
        and var_type.size is None
    ):
        return slot, var_type, var_type.element_type, source
    if _is_buffer_array(slot, var_type):
        return slot, var_type, var_type.element_type.generic_args[0], source
    if (
        not isinstance(var_type, A.NamedType)
        or var_type.name not in _WRAPPERS | _QUEUES
        or len(var_type.generic_args) != 1
    ):
        return None
    return slot, var_type, var_type.generic_args[0], source


def _is_buffer_array(slot, var_type):
    """Whether a declaration of ``var_type`` declares an array of structured buffers."""
    element = getattr(var_type, "element_type", None)
    return (
        slot == "var_type"
        and isinstance(var_type, A.ArrayType)
        and isinstance(element, A.NamedType)
        and element.name in _WRAPPERS
        and len(element.generic_args) == 1
    )


def _word_type(var_type):
    """``var_type`` with ``uint`` elements, and whether it is read-only."""
    if _is_buffer_array("var_type", var_type):
        word, readonly = _word_type(var_type.element_type)
        return A.ArrayType(word, var_type.size), readonly
    if isinstance(var_type, A.PointerType):
        word = A.PointerType(
            A.PrimitiveType("uint"),
            var_type.is_mutable,
            var_type.address_space,
            var_type.access_mode,
            list(var_type.resource_qualifiers),
        )
        return word, var_type.address_space == "constant"
    if isinstance(var_type, A.ArrayType):
        return A.ArrayType(A.PrimitiveType("uint"), None), False
    word = A.NamedType(var_type.name, [A.PrimitiveType("uint")])
    return word, var_type.name == "StructuredBuffer"


def _plan_buffer(declaration, shapes, target, inline, passed=False):
    """A :class:`_Buffer` for a declaration whose layout must change, ``False`` when the generator keeps it, or
    ``None`` when the target's buffer rules already have the declared layout.

    A global buffer ``passed`` to a function is lowered here even where the generator keeps others, so that the
    function's buffer parameter and its argument have the same words.
    """
    found = _buffer_type(declaration)
    if found is None:
        return None
    slot, var_type, element_type, source = found
    target_rules = "hlsl" if target in HLSL_TARGETS else "std430"
    if source == target_rules:
        return None
    kind = "buffer parameter" if inline else "structured buffer"
    context = f"{kind} {declaration.name}"
    shape = shapes[source].of(element_type, context)
    if _same_layout(shape, target_rules) and not (
        target_rules == "std430" and _holds_bool(shape)
    ):
        return None
    queue = (
        var_type.name
        if isinstance(var_type, A.NamedType) and var_type.name in _QUEUES
        else ""
    )
    array = _is_buffer_array(slot, var_type)
    # The generator's own struct copies cover single indexed buffers only, so queues and arrays are lowered here.
    if (
        source == "hlsl"
        and shape.form == "struct"
        and target in _GENERATOR_LOWERS_STRUCTS
        and not inline
        and not passed
        and not queue
        and not array
    ):
        return False
    if not shape.lowerable:
        raise StructuredLayoutUnsupportedError(
            f"{context}: its element type is laid out differently by {_RULE_NAMES[source]} and "
            f"{_RULE_NAMES[target_rules]}, and holds a matrix, boolean, 16-bit or 64-bit value that cannot be "
            "split into 32-bit words",
            target=target,
        )
    if queue:
        new_type = A.NamedType(queue, [A.NamedType(_words_struct_name(shape.words))])
        readonly = False
    else:
        new_type, readonly = _word_type(var_type)
    qualifiers = {str(q) for q in getattr(declaration, "qualifiers", None) or []}
    buffer = _Buffer(
        declaration.name,
        shape,
        readonly or "const" in qualifiers,
        inline,
        queue=queue,
        array=array,
    )
    setattr(declaration, slot, new_type)
    if slot == "var_type":
        declaration.vtype = new_type
    # 32-bit words have one layout under every rule set, so the declaration keeps no layout to reproduce.
    declaration.attributes = [
        attribute
        for attribute in getattr(declaration, "attributes", None) or []
        if str(getattr(attribute, "name", "")).lower() != "buffer_layout"
    ]
    return buffer


def _passed_names(node, functions, names):
    """Add to ``names`` the identifiers passed to one of ``functions``, alone or offset by ``+`` or ``-``."""
    if isinstance(node, A.FunctionCallNode) and call_name(node) in functions:
        for argument in node.arguments or []:
            if isinstance(argument, A.BinaryOpNode) and argument.operator in ("+", "-"):
                argument = argument.left
            if isinstance(argument, A.IdentifierNode):
                names.add(argument.name)
    for slot_field, key in child_slots(node):
        child = get_child(node, slot_field, key)
        if isinstance(child, A.ASTNode):
            _passed_names(child, functions, names)
    return names


def _lowered_buffers(program, shapes, target):
    """The lowered global buffers, each function's buffer parameters and parameter names, and the global buffers
    whose element calls are spelled as indexing.

    A parameter maps to its :class:`_Buffer`, or to ``False`` when the generator keeps its layout itself.
    """
    buffers, canonical = {}, set()
    functions = list(iter_functions(program))
    names = {getattr(function, "name", None) for function in functions}
    passed = set()
    for function in functions:
        if getattr(function, "body", None) is not None:
            _passed_names(function.body, names, passed)
    for declaration in getattr(program, "global_variables", None) or []:
        plan = _plan_buffer(
            declaration,
            shapes,
            target,
            inline=False,
            passed=getattr(declaration, "name", None) in passed,
        )
        if plan is not None:
            canonical.add(declaration.name)
        if plan:
            buffers[declaration.name] = plan
    scopes = {}
    for function in iter_functions(program):
        parameters, names = {}, set()
        for parameter in getattr(function, "parameters", None) or []:
            name = getattr(parameter, "name", None)
            if not isinstance(name, str):
                continue
            names.add(name)
            plan = _plan_buffer(parameter, shapes, target, inline=True)
            if plan is not None:
                parameters[name] = plan
        scopes[id(function)] = (parameters, names)
    return buffers, scopes, canonical


def _buffer_call(node, name, count, names):
    """Whether ``node`` is ``name(buffer, ...)`` with ``count`` arguments on one of ``names``."""
    arguments = getattr(node, "arguments", None) or []
    return (
        isinstance(node, A.FunctionCallNode)
        and call_name(node) == name
        and len(arguments) == count
        and isinstance(arguments[0], A.IdentifierNode)
        and arguments[0].name in names
    )


def _canonicalize_buffer_calls(node, names):
    """Spell ``buffer_load(b, i)`` as ``b[i]`` and a ``buffer_store(b, i, v)`` statement as ``b[i] = v``."""
    if not names:
        return
    for slot_field, key in list(child_slots(node)):
        child = get_child(node, slot_field, key)
        if isinstance(child, A.ExpressionStatementNode) and _buffer_call(
            child.expression, "buffer_store", 3, names
        ):
            buffer, index, value = child.expression.arguments
            child.expression = A.AssignmentNode(A.ArrayAccessNode(buffer, index), value)
        elif _buffer_call(child, "buffer_load", 2, names):
            child = A.ArrayAccessNode(*child.arguments)
            set_child(node, slot_field, key, child)
        if isinstance(child, A.ASTNode):
            _canonicalize_buffer_calls(child, names)


def _holds_bool(shape):
    if shape.form in ("scalar", "vector"):
        return shape.kind == "b"
    if shape.form == "array":
        return _holds_bool(shape.element)
    return any(_holds_bool(member) for _name, member, _offset in shape.members)


def _substitute(node, replacements):
    """``node`` with each identifier named in ``replacements`` replaced by a copy of its expression."""
    if isinstance(node, A.IdentifierNode) and node.name in replacements:
        return clone(replacements[node.name])
    for slot_field, key in list(child_slots(node)):
        child = get_child(node, slot_field, key)
        set_child(node, slot_field, key, _substitute(child, replacements))
    return node


def _parse(texts):
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    source = "shader __cgl_structured_layouts {\n" + "\n".join(texts) + "\n}\n"
    return SOURCE_REGISTRY.get("cgl").parse(source)


def _parse_helpers(texts):
    return list(_parse(texts).functions)


def _add_words_structs(program, buffers, scopes, target):
    """Declare the word struct of every lowered append or consume buffer and buffer parameter."""
    plans = list(buffers.values())
    for parameters, _names in scopes.values():
        plans.extend(plan for plan in parameters.values() if plan)
    counts = sorted({plan.shape.words for plan in plans if plan.queue})
    if not counts:
        return
    structs = getattr(program, "structs", None)
    if not isinstance(structs, list):
        raise StructuredLayoutUnsupportedError(
            "the program has no struct list to receive the append and consume buffer elements",
            target=target,
        )
    texts = [
        f"struct {_words_struct_name(count)} {{ uint w[{count}]; }}" for count in counts
    ]
    structs[0:0] = list(_parse(texts).structs)


def legalize_structured_layouts(program, target) -> int:
    """Rewrite marked buffers whose layout ``target`` changes into word buffers; returns how many."""
    target = str(target or "")
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    shapes = {source: _Shapes(program, source) for source in ("hlsl", "metal")}
    buffers, scopes, canonical = _lowered_buffers(program, shapes, target)
    if not canonical and not any(parameters for parameters, _names in scopes.values()):
        return 0
    parameter_plans = {}
    for function in iter_functions(program):
        parameters, _names = scopes.get(id(function), ({}, set()))
        parameter_plans.setdefault(getattr(function, "name", None), []).append(
            [
                parameters.get(getattr(parameter, "name", None))
                for parameter in getattr(function, "parameters", None) or []
            ]
        )
    _add_words_structs(program, buffers, scopes, target)
    rewriter = _Rewriter(program, {}, target, parameter_plans)
    count = len(buffers)
    for function in list(iter_functions(program)):
        parameters, names = scopes.get(id(function), ({}, set()))
        scope = {name: buffer for name, buffer in buffers.items() if name not in names}
        scope.update((name, buffer) for name, buffer in parameters.items() if buffer)
        count += sum(1 for buffer in parameters.values() if buffer)
        kept = {name for name in canonical if name not in names} | set(parameters)
        if getattr(function, "body", None) is None:
            continue
        _canonicalize_buffer_calls(function.body, kept)
        if scope:
            rewriter.buffers = scope
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
    return count
