"""Lower MetalPerformancePrimitives' ``matmul2d`` on cooperative tensors to SIMD-group shuffles.

Metal 4's ``mpp::tensor_ops::matmul2d<desc, execution_simdgroup>`` multiplies matrices whose elements the 32 lanes of
a SIMD-group hold in registers (cooperative tensors)::

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto a = op.get_left_input_cooperative_tensor<half, half, float>();      // M x K
    auto b = op.get_right_input_cooperative_tensor<half, half, float>();     // K x N
    auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();  // M x N
    ... a[i] = ...; ... a.get_multidimensional_index(i) ...;
    op.run(a, b, c);                                                         // c += op(a) * op(b)

CrossGL has no such operation, so before template materialization each operation is rewritten into code every
target can express:

- Each cooperative tensor becomes a thread array of its capacity, ``rows * columns / 32`` elements, initialised to
  zero. Its elements keep their subscripts; ``get_capacity()`` becomes that number, ``is_valid_element(i)`` becomes
  ``true`` and ``get_multidimensional_index(i)`` returns the element's ``{column, row}`` from a table of the layout.
  A transposed operand's coordinates are those of the stored matrix, as MPP reports them.
- ``run(a, b, c)`` becomes a block in which each lane computes its own destination elements: for every ``k`` it
  fetches ``A(row, k)`` and ``B(k, column)`` from the lanes that hold them with ``simd_shuffle`` and accumulates in
  float with ``fma``, in increasing ``k``. ``mode::multiply`` starts from zero and ``multiply_accumulate`` from the
  destination's value. Every lane runs the same shuffles, so the block needs the SIMD-group's lanes to be active
  together, as ``run`` does. The descriptor's ``relaxed_precision`` flag lets MPP use less precision; the lowering
  ignores it and keeps float products and sums. Destinations must be float, which a ``static_assert`` checks after
  materialization when the element type is a template parameter.
- The ``matmul2d_descriptor::mode`` type becomes ``int``, with ``multiply`` as 0 and ``multiply_accumulate`` as 1, so
  template parameters of that type keep their values.

Which lane holds which element is implementation defined. Two layouts are known:

- ``simdgroup_tiles_8x8``, the default: the logical operand is split into 8x8 tiles in row-major order; element ``e``
  belongs to tile ``e // 2`` and holds ``simdgroup_matrix``'s ``thread_elements()[e % 2]`` of that tile. An Apple M2
  Max running macOS 27 reports exactly this layout for all seven operand shapes MPP accepts for cooperative inputs,
  every transpose and half, bfloat and float inputs (208,896 elements in 112 configurations).
- ``fragments_16x16``: the stored operand is split into 16x16 fragments in row-major order, and element ``e`` holds
  entry ``e % 8`` of fragment ``e // 8``, where lane ``l`` holds rows ``r`` and ``r + 8`` and four adjacent columns
  starting at ``c``, with ``q = l / 4``, ``r = (q & 4) | ((l / 2) & 3)`` and ``c = 4 * ((q & 2) | (l & 1))``.
  Kernels written for Apple GPUs with neural accelerators assume it. It is defined for operands of one or two
  fragments.

A kernel that reads elements only through ``get_multidimensional_index`` works with any layout. One that indexes
elements by a fixed lane layout must name it with the ``cooperative_tensor_layout`` Metal source option; without
it, a subscript that does not sit in a block that queries ``get_multidimensional_index`` of the same index is
refused. Operations, tensors and uses the rewrite cannot prove exact, such as symbolic sizes, other execution
scopes, tensor handles or passing a cooperative tensor to a function, raise ``MetalMatmul2dError``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Tuple

from crosstl.translator.refusals import MetalSourceRefusal

if TYPE_CHECKING:
    from .preprocessor import MetalPreprocessor

SUBGROUP_SIZE = 32
DEFAULT_LAYOUT = "simdgroup_tiles_8x8"
LAYOUT_OPTION = "cooperative_tensor_layout"
_PREFIX = "crosstl_mpp"
_ROLES = ("left_input", "right_input", "destination")
_LANE = "__metal_get_thread_index_in_simdgroup(ushort())"
_IDENTIFIER = r"[A-Za-z_]\w*"
_ACCUMULATOR = "float"

Coordinate = Tuple[int, int]
Layout = Tuple[Tuple[Coordinate, ...], ...]


class MetalMatmul2dError(MetalSourceRefusal):
    """A ``matmul2d`` operation or cooperative-tensor use the Metal importer cannot lower exactly."""


# Layouts ----------------------------------------------------------------------------------------------------------


def _simdgroup_matrix_coordinate(lane: int) -> Coordinate:
    """Row and first column of ``simdgroup_matrix<T, 8, 8>``'s ``thread_elements()`` for ``lane``."""
    quad = lane // 4
    return (quad & 4) + (lane // 2) % 4, (quad & 2) * 2 + (lane % 2) * 2


def _tiles_8x8(rows: int, columns: int, transposed: bool) -> Layout:
    lanes = []
    for lane in range(SUBGROUP_SIZE):
        row, column = _simdgroup_matrix_coordinate(lane)
        elements = []
        for element in range(rows * columns // SUBGROUP_SIZE):
            tile, entry = divmod(element, 2)
            tile_row, tile_column = divmod(tile, columns // 8)
            logical = (tile_row * 8 + row, tile_column * 8 + column + entry)
            elements.append(logical[::-1] if transposed else logical)
        lanes.append(tuple(elements))
    return tuple(lanes)


def _fragments_16x16(rows: int, columns: int, transposed: bool) -> Layout:
    stored_rows, stored_columns = (columns, rows) if transposed else (rows, columns)
    if stored_rows * stored_columns > 2 * 16 * 16:
        raise MetalMatmul2dError(
            "the fragments_16x16 cooperative-tensor layout is defined for operands of "
            f"one or two 16x16 fragments, not {stored_rows}x{stored_columns}"
        )
    lanes = []
    for lane in range(SUBGROUP_SIZE):
        quad = lane >> 2
        first_row = (quad & 4) | ((lane >> 1) & 3)
        first_column = ((quad & 2) | (lane & 1)) * 4
        elements = []
        for element in range(rows * columns // SUBGROUP_SIZE):
            fragment, entry = divmod(element, 8)
            fragment_row, fragment_column = divmod(fragment, stored_columns // 16)
            elements.append(
                (
                    fragment_row * 16 + first_row + (entry >> 2) * 8,
                    fragment_column * 16 + first_column + entry % 4,
                )
            )
        lanes.append(tuple(elements))
    return tuple(lanes)


LAYOUTS: dict[str, Callable[[int, int, bool], Layout]] = {
    "simdgroup_tiles_8x8": _tiles_8x8,
    "fragments_16x16": _fragments_16x16,
}


def validate_cooperative_tensor_layout(name: object) -> str | None:
    """``name`` if it names a known layout, None for the default; anything else is refused."""
    if name is None:
        return None
    if not isinstance(name, str) or name not in LAYOUTS:
        raise MetalMatmul2dError(
            f"Metal {LAYOUT_OPTION} must be one of {', '.join(sorted(LAYOUTS))}"
        )
    return name


@dataclass(frozen=True)
class _Variant:
    """One operand's layout: its logical shape, whether it is stored transposed, and the layout's name."""

    layout: str
    rows: int
    columns: int
    transposed: bool

    @property
    def capacity(self) -> int:
        return self.rows * self.columns // SUBGROUP_SIZE

    @property
    def stem(self) -> str:
        suffix = "t" if self.transposed else "n"
        return f"{_PREFIX}_{self.layout}_{self.rows}x{self.columns}{suffix}"

    def stored(self) -> Layout:
        return LAYOUTS[self.layout](self.rows, self.columns, self.transposed)

    def logical(self) -> Layout:
        return tuple(
            tuple(
                coordinate[::-1] if self.transposed else coordinate
                for coordinate in lane
            )
            for lane in self.stored()
        )

    def owners(self) -> dict[Coordinate, tuple[int, int]]:
        owners: dict[Coordinate, tuple[int, int]] = {}
        for lane, coordinates in enumerate(self.logical()):
            for element, coordinate in enumerate(coordinates):
                owners[coordinate] = (lane, element)
        expected = {(r, c) for r in range(self.rows) for c in range(self.columns)}
        if set(owners) != expected or len(owners) != self.rows * self.columns:
            raise MetalMatmul2dError(
                f"cooperative-tensor layout {self.layout} does not place every element of a "
                f"{self.rows}x{self.columns} operand exactly once"
            )
        return owners


# Generated tables -------------------------------------------------------------------------------------------------


@dataclass
class _Tables:
    """Program-scope tables and helpers the rewritten operations use, emitted once each."""

    definitions: dict[str, str] = field(default_factory=dict)

    def array(self, name: str, values: list[int]) -> str:
        if name not in self.definitions:
            rows = [
                ", ".join(str(value) for value in values[start : start + 16])
                for start in range(0, len(values), 16)
            ]
            body = ",\n    ".join(rows)
            self.definitions[name] = (
                f"constant ushort {name}[{len(values)}] = {{\n    {body}}};\n"
            )
        return name

    def index_table(self, variant: _Variant) -> str:
        """Each lane's elements' stored ``(row << 8) | column``."""
        values = [
            (row << 8) | column for lane in variant.stored() for row, column in lane
        ]
        return self.array(f"{variant.stem}_index", values)

    def index_function(self, variant: _Variant) -> str:
        name = f"{variant.stem}_coordinate"
        if name not in self.definitions:
            table = self.index_table(variant)
            self.definitions[name] = (
                f"static inline int2 {name}(ushort element) {{\n"
                f"  const ushort packed = {table}[{_LANE} * {variant.capacity} + element];\n"
                "  return int2(int(packed & 255), int(packed >> 8));\n"
                "}\n"
            )
        return name

    def owner_table(self, variant: _Variant) -> str:
        """For each logical ``(row, column)``, in row-major order, ``lane | (element << 5)``."""
        owners = variant.owners()
        values = [
            owners[(row, column)][0] | (owners[(row, column)][1] << 5)
            for row in range(variant.rows)
            for column in range(variant.columns)
        ]
        return self.array(f"{variant.stem}_owner", values)

    def text(self) -> str:
        return "".join(self.definitions[name] for name in self.definitions)


# Source scanning --------------------------------------------------------------------------------------------------


def _masked(preprocessor: MetalPreprocessor, text: str) -> str:
    """``text`` with comments and literals blanked, keeping offsets."""
    characters = list(text)
    for start, end in preprocessor._find_comment_and_literal_spans(text):
        for index in range(start, end):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


def _split_arguments(
    text: str, masked: str, open_paren: int, close_paren: int
) -> list[str]:
    """The texts between ``open_paren`` and ``close_paren`` split at commas outside brackets."""
    arguments = []
    depth = 0
    start = open_paren + 1
    for index in range(open_paren + 1, close_paren):
        character = masked[index]
        if character in "([{<":
            depth += 1
        elif character in ")]}>":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:index])
            start = index + 1
    arguments.append(text[start:close_paren])
    return arguments


def _scope_masked(preprocessor: MetalPreprocessor, source: str) -> str:
    """``source`` with comments, literals and directive lines blanked, keeping offsets."""
    masked = list(_masked(preprocessor, source))
    for match in re.finditer(r"(?m)^[ \t]*#[^\n]*", source):
        for index in range(match.start(), match.end()):
            masked[index] = " "
    return "".join(masked)


def _enclosing_block(masked: str, position: int) -> tuple[int, int] | None:
    depth = 0
    for index in range(position - 1, -1, -1):
        character = masked[index]
        if character == "}":
            depth += 1
        elif character == "{":
            if depth == 0:
                return index, _matching(masked, index, "{", "}")
            depth -= 1
    return None


def _matching(masked: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    for index in range(start, len(masked)):
        if masked[index] == opener:
            depth += 1
        elif masked[index] == closer:
            depth -= 1
            if depth == 0:
                return index
    raise MetalMatmul2dError(f"unbalanced '{opener}' in a matmul2d operation's scope")


def _matching_angle(masked: str, start: int) -> int:
    depth = 0
    for index in range(start, len(masked)):
        character = masked[index]
        if character in "<([":
            depth += 1
        elif character in ">)]":
            depth -= 1
            if depth == 0:
                if character != ">":
                    break
                return index
    raise MetalMatmul2dError(
        "unbalanced template arguments in a cooperative-tensor declaration"
    )


def _declaration_start(masked: str, position: int) -> int:
    """The start of the line that begins the namespace-scope declaration containing ``position``."""
    depth = 0
    boundary = 0
    for index in range(position):
        character = masked[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                boundary = index + 1
        elif character == ";" and depth == 0:
            boundary = index + 1
    newline = masked.find("\n", boundary)
    return boundary if newline < 0 or newline >= position else newline + 1


class _Names:
    """Spellings of MPP's names, given the namespaces the unit's using-directives nominate."""

    def __init__(self, masked: str):
        prefixes = [r"(?:::)?\s*mpp\s*::\s*tensor_ops\s*::\s*"]
        if re.search(r"\busing\s+namespace\s+(?:::)?\s*mpp\s*;", masked):
            prefixes.append(r"\btensor_ops\s*::\s*")
        if re.search(
            r"\busing\s+namespace\s+(?:::)?\s*mpp\s*::\s*tensor_ops\s*;", masked
        ):
            prefixes.append(r"(?<![\w:])")
        self.prefix = "(?:" + "|".join(prefixes) + ")"

    def name(self, identifier: str) -> str:
        return self.prefix + identifier + r"\b"


def _rewrite_mode_spellings(source: str, masked: str, names: _Names) -> str:
    """``matmul2d_descriptor::mode`` as ``int`` and its enumerators as 0 and 1."""
    pattern = re.compile(
        names.name(r"matmul2d_descriptor\s*::\s*mode")
        + r"(?:\s*::\s*(?P<enumerator>multiply_accumulate|multiply)\b)?"
    )
    edits = []
    for match in pattern.finditer(masked):
        enumerator = match.group("enumerator")
        if enumerator is None and re.match(r"\s*::", masked[match.end() :]):
            raise MetalMatmul2dError(
                "matmul2d_descriptor::mode has only the enumerators multiply and "
                "multiply_accumulate"
            )
        replacement = {None: "int", "multiply": "0", "multiply_accumulate": "1"}
        edits.append((match.start(), match.end(), replacement[enumerator]))
    return _apply(source, edits)


def _apply(source: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


# Operations -------------------------------------------------------------------------------------------------------


@dataclass
class _Tensor:
    name: str
    role: str
    element_type: str
    type_arguments: tuple[str, ...]
    span: tuple[int, int]


@dataclass
class _Operation:
    name: str
    m: int
    n: int
    k: int
    transpose_left: str
    transpose_right: str
    mode: str
    scope: tuple[int, int]
    tensors: dict[str, _Tensor] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    edits: list[tuple[int, int, str]] = field(default_factory=list)

    def shape(self, role: str) -> tuple[int, int]:
        return {
            "left_input": (self.m, self.k),
            "right_input": (self.k, self.n),
            "destination": (self.m, self.n),
        }[role]

    def transpose(self, role: str) -> str:
        return {
            "left_input": self.transpose_left,
            "right_input": self.transpose_right,
            "destination": "false",
        }[role]


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _statement_end(masked: str, position: int, what: str) -> int:
    match = re.compile(r"\s*;").match(masked, position)
    if match is None:
        raise MetalMatmul2dError(f"{what} must be a complete statement")
    return match.end()


def _descriptor(source: str, masked: str, match: re.Match, names: _Names):
    open_paren = match.end() - 1
    close_paren = _matching(masked, open_paren, "(", ")")
    end = _statement_end(masked, close_paren + 1, "a matmul2d_descriptor declaration")
    arguments = [
        _normalized(argument)
        for argument in _split_arguments(source, masked, open_paren, close_paren)
    ]
    if not 3 <= len(arguments) <= 7:
        raise MetalMatmul2dError(
            "matmul2d with cooperative-tensor operands needs a descriptor with "
            "literal M, N and K"
        )
    sizes = []
    for label, argument in zip("MNK", arguments[:3]):
        if not re.fullmatch(r"\d+", argument):
            raise MetalMatmul2dError(
                f"matmul2d descriptor {label} must be an integer literal, not '{argument}'"
            )
        sizes.append(int(argument))
    if any(size not in (16, 32) for size in sizes) or 32 not in sizes:
        raise MetalMatmul2dError(
            "matmul2d with cooperative-tensor operands needs M, N and K of 16 or 32, "
            f"at least one of them 32 (MPP's rule), not {sizes}"
        )
    defaults = ["false", "false", "false", "0"]
    flags = arguments[3:] + defaults[len(arguments) - 3 :]
    for label, flag in zip(("transpose_left", "transpose_right"), flags[:2]):
        if not re.fullmatch(rf"true|false|{_IDENTIFIER}", flag):
            raise MetalMatmul2dError(
                f"matmul2d descriptor {label} must be true, false or a constant name"
            )
    if not re.fullmatch(rf"0|1|{_IDENTIFIER}", flags[3]):
        raise MetalMatmul2dError(
            "matmul2d descriptor mode must be an enumerator of "
            "matmul2d_descriptor::mode or a constant name"
        )
    return sizes, flags[0], flags[1], flags[3], (match.start(), end)


def _single(matches: list[re.Match], what: str) -> re.Match:
    if len(matches) != 1:
        raise MetalMatmul2dError(
            f"each matmul2d descriptor needs exactly one {what} in its scope"
        )
    return matches[0]


def _operation(source, masked, match, names) -> _Operation:
    sizes, transpose_left, transpose_right, mode, descriptor_span = _descriptor(
        source, masked, match, names
    )
    descriptor = match.group("name")
    scope = _enclosing_block(masked, match.start())
    if scope is None:
        raise MetalMatmul2dError(
            "a matmul2d descriptor with cooperative-tensor operands must be declared "
            "in a function"
        )
    open_brace, close_brace = scope
    declaration = re.compile(
        names.name("matmul2d") + rf"\s*<\s*{descriptor}\s*,(?P<scope>[^;{{}}]*)>\s*"
        rf"(?P<op>{_IDENTIFIER})\s*;"
    )
    op_match = _single(
        list(declaration.finditer(masked, descriptor_span[1], close_brace)),
        "matmul2d operation",
    )
    execution = _normalized(op_match.group("scope"))
    if not re.fullmatch(
        r"(?:(?:::)?metal::)?(?:execution_simdgroup|execution_simdgroups\s*<\s*1\s*>)",
        execution,
    ):
        raise MetalMatmul2dError(
            "matmul2d with cooperative-tensor operands is lowered for one SIMD-group "
            f"(execution_simdgroup), not {execution}"
        )
    m, n, k = sizes
    operation = _Operation(
        op_match.group("op"),
        m,
        n,
        k,
        transpose_left,
        transpose_right,
        mode,
        scope,
    )
    operation.edits += [(*descriptor_span, ""), (op_match.start(), op_match.end(), "")]
    for use in re.finditer(rf"\b{descriptor}\b", masked[open_brace:close_brace]):
        position = open_brace + use.start()
        if not (
            descriptor_span[0] <= position < descriptor_span[1]
            or op_match.start() <= position < op_match.end()
        ):
            raise MetalMatmul2dError(
                f"matmul2d descriptor '{descriptor}' is used other than to declare its "
                "operation"
            )
    return operation


_GETTER = r"\s*\.\s*(?:template\s+)?get_(?P<role>left_input|right_input|destination)_cooperative_tensor\s*<"


def _tensors(source: str, masked: str, operation: _Operation) -> None:
    open_brace, close_brace = operation.scope
    op = re.escape(operation.name)
    forms = (
        (re.compile(rf"\bauto\s+(?P<name>{_IDENTIFIER})\s*=\s*{op}{_GETTER}"), False),
        (
            re.compile(
                rf"\busing\s+(?P<name>{_IDENTIFIER})\s*=\s*decltype\s*\(\s*{op}{_GETTER}"
            ),
            True,
        ),
    )
    declarations = []
    for pattern, alias in forms:
        for match in pattern.finditer(masked, open_brace, close_brace):
            close_angle = _matching_angle(masked, match.end() - 1)
            call = re.compile(r"\s*\(\s*\)" + (r"\s*\)" if alias else "")).match(
                masked, close_angle + 1
            )
            if call is None:
                raise MetalMatmul2dError(
                    "a cooperative tensor must be created with an argument-less "
                    f"get_{match.group('role')}_cooperative_tensor call"
                )
            end = _statement_end(masked, call.end(), "a cooperative-tensor declaration")
            arguments = tuple(
                _normalized(argument)
                for argument in _split_arguments(
                    source, masked, match.end() - 1, close_angle
                )
            )
            if len(arguments) != 3:
                raise MetalMatmul2dError(
                    "cooperative-tensor getters take three type arguments"
                )
            declarations.append((match, alias, arguments, (match.start(), end)))
    for match, alias, arguments, span in sorted(declarations, key=lambda d: d[3]):
        role, name = match.group("role"), match.group("name")
        if name in operation.tensors or name in operation.aliases:
            raise MetalMatmul2dError(f"cooperative tensor '{name}' is declared twice")
        if role == "destination":
            for operand_role, argument in zip(_ROLES[:2], arguments[:2]):
                if _named_operand(operation, argument) != operand_role:
                    raise MetalMatmul2dError(
                        "a destination cooperative tensor's first two type arguments must "
                        f"name the operation's {operand_role} tensor type, not '{argument}'"
                    )
        element = (
            arguments[2] if role == "destination" else arguments[_ROLES.index(role)]
        )
        if alias:
            operation.aliases[name] = role
        else:
            operation.tensors[name] = _Tensor(name, role, element, arguments, span)
        rows, columns = operation.shape(role)
        replacement = (
            ""
            if alias
            else f"{element} {name}[{rows * columns // SUBGROUP_SIZE}] = {{}};"
        )
        operation.edits.append((*span, replacement))
    types = {}
    for tensor in operation.tensors.values():
        if tensor.role != "destination":
            types.setdefault("operands", tensor.type_arguments)
            if types["operands"] != tensor.type_arguments:
                raise MetalMatmul2dError(
                    "the left and right cooperative tensors of one matmul2d must name the "
                    "same element types"
                )
    for tensor in operation.tensors.values():
        if tensor.role == "destination" and "operands" in types:
            if tensor.element_type != types["operands"][2]:
                raise MetalMatmul2dError(
                    "a destination cooperative tensor's element type must be the "
                    "operands' destination type"
                )


def _named_operand(operation: _Operation, argument: str) -> str | None:
    if argument in operation.aliases:
        return operation.aliases[argument]
    match = re.fullmatch(
        rf"(?:(?:::)?metal::)?(?:__)?(?:remove_addrspace_t\s*<\s*)?decltype\s*\(\s*"
        rf"(?P<name>{_IDENTIFIER})\s*\)\s*>?",
        argument,
    )
    if match is None or argument.count("<") != argument.count(">"):
        return None
    tensor = operation.tensors.get(match.group("name"))
    return tensor.role if tensor is not None else None


def _inside(position: int, spans) -> bool:
    return any(start <= position < end for start, end, *_ in spans)


def _variants(
    operation: _Operation, role: str, layout: str
) -> list[tuple[str, _Variant]]:
    """``(value, variant)`` for each value the role's transpose flag can take."""
    rows, columns = operation.shape(role)
    flag = operation.transpose(role)
    values = [flag] if flag in ("true", "false") else ["true", "false"]
    return [
        (value, _Variant(layout, rows, columns, value == "true")) for value in values
    ]


def _select(flag: str, choices: list[tuple[str, str]], index: str) -> str:
    """``table[index]`` for a literal flag; a conditional over both tables for a named one."""
    if len(choices) == 1:
        return f"{choices[0][1]}[{index}]"
    names = dict(choices)
    return f"({flag} ? {names['true']}[{index}] : {names['false']}[{index}])"


def _uses(source, masked, operation: _Operation, layout: str, configured: bool, tables):
    open_brace, close_brace = operation.scope
    for match in re.finditer(rf"\b{re.escape(operation.name)}\b", masked[:close_brace]):
        position = match.start()
        if position < open_brace or _inside(position, operation.edits):
            continue
        run = re.compile(rf"{re.escape(operation.name)}\s*\.\s*run\s*\(").match(
            masked, position
        )
        if run is None:
            raise MetalMatmul2dError(
                f"matmul2d operation '{operation.name}' is used other than to create its "
                "cooperative tensors and run on them"
            )
        close_paren = _matching(masked, run.end() - 1, "(", ")")
        end = _statement_end(masked, close_paren + 1, "a matmul2d run call")
        arguments = [
            _normalized(argument)
            for argument in _split_arguments(source, masked, run.end() - 1, close_paren)
        ]
        roles = [
            operation.tensors[argument].role if argument in operation.tensors else None
            for argument in arguments
        ]
        if roles != list(_ROLES):
            raise MetalMatmul2dError(
                f"{operation.name}.run takes the operation's left, right and destination "
                "cooperative tensors, in that order"
            )
        operation.edits.append(
            (position, end, _run_block(operation, arguments, layout, tables))
        )
    accessors = re.compile(
        r"\s*\.\s*(?P<member>get_capacity|is_valid_element|get_multidimensional_index)\s*\("
    )
    for tensor in operation.tensors.values():
        rows, columns = operation.shape(tensor.role)
        capacity = rows * columns // SUBGROUP_SIZE
        for match in re.finditer(rf"(?<![\w.]){re.escape(tensor.name)}\b", masked):
            position = match.start()
            if not open_brace < position < close_brace or _inside(
                position, operation.edits
            ):
                continue
            if re.compile(r"\s*::").match(masked, match.end()) or re.search(
                r"(?:->|::)\s*$", masked[max(0, position - 8) : position]
            ):
                continue
            if re.compile(r"\s*\[").match(masked, match.end()):
                if not configured:
                    _require_queried_layout(masked, tensor.name, match.end(), operation)
                continue
            accessor = accessors.match(masked, match.end())
            if accessor is None:
                raise MetalMatmul2dError(
                    f"cooperative tensor '{tensor.name}' is used other than through its "
                    "subscript, get_capacity, is_valid_element and "
                    "get_multidimensional_index"
                )
            open_paren = accessor.end() - 1
            close_paren = _matching(masked, open_paren, "(", ")")
            argument = source[open_paren + 1 : close_paren].strip()
            member = accessor.group("member")
            if member == "get_capacity":
                if argument:
                    raise MetalMatmul2dError("get_capacity takes no arguments")
                replacement = str(capacity)
            else:
                if re.search(r"\+\+|--|(?<![=!<>])=(?!=)", argument) or not argument:
                    raise MetalMatmul2dError(
                        f"{member} needs an element index without side effects"
                    )
                if member == "is_valid_element":
                    replacement = "true"
                else:
                    choices = [
                        (value, tables.index_function(variant))
                        for value, variant in _variants(operation, tensor.role, layout)
                    ]
                    calls = [(value, f"{name}") for value, name in choices]
                    if len(calls) == 1:
                        replacement = f"{calls[0][1]}(ushort({argument}))"
                    else:
                        names = dict(calls)
                        flag = operation.transpose(tensor.role)
                        replacement = (
                            f"({flag} ? {names['true']}(ushort({argument})) : "
                            f"{names['false']}(ushort({argument})))"
                        )
            operation.edits.append((position, close_paren + 1, replacement))


def _require_queried_layout(masked, name, bracket_search_start, operation) -> None:
    """Refuse a subscript unless its block reads the same element's coordinate."""
    open_bracket = masked.index("[", bracket_search_start)
    close_bracket = _matching(masked, open_bracket, "[", "]")
    index = masked[open_bracket + 1 : close_bracket].strip()
    block = _enclosing_block(masked, open_bracket)
    queried = False
    if re.fullmatch(_IDENTIFIER, index) and block is not None:
        queried = (
            re.search(
                rf"\b{re.escape(name)}\s*\.\s*get_multidimensional_index\s*\(\s*"
                rf"{re.escape(index)}\s*\)",
                masked[block[0] : block[1]],
            )
            is not None
        )
    if not queried:
        raise MetalMatmul2dError(
            f"cooperative tensor '{name}' is indexed by a fixed lane layout ('{name}[{index}]' "
            "without reading get_multidimensional_index of that element); set the Metal "
            f"source option {LAYOUT_OPTION} to the layout the kernel assumes "
            f"({', '.join(sorted(LAYOUTS))})"
        )


def _fetch_plans(operation: _Operation, role: str, layout: str, tables: _Tables):
    """Owner and candidate tables for each value of the operand's transpose flag."""
    destination = _Variant(layout, operation.m, operation.n, False).logical()
    plans = []
    for value, variant in _variants(operation, role, layout):
        owners = variant.owners()
        slots = []
        for element in range(operation.m * operation.n // SUBGROUP_SIZE):
            for k in range(operation.k):
                needed = set()
                for lane in range(SUBGROUP_SIZE):
                    row, column = destination[lane][element]
                    coordinate = (row, k) if role == "left_input" else (k, column)
                    needed.add(owners[coordinate][1])
                slots.append(sorted(needed))
        plans.append((value, variant, tables.owner_table(variant), slots))
    return plans


def _fetch(operation, role, tensor, layout, tables, variable, loop) -> list[str]:
    plans = _fetch_plans(operation, role, layout, tables)
    flag = operation.transpose(role)
    if role == "left_input":
        index = f"{_PREFIX}_row * {operation.k} + {_PREFIX}_k"
    else:
        index = f"{_PREFIX}_k * {operation.n} + {_PREFIX}_column"
    owner = f"{variable}_owner"
    lines = [
        f"const ushort {owner} = "
        + _select(flag, [(value, table) for value, _, table, _ in plans], index)
        + ";"
    ]
    lane = f"ushort({owner} & 31)"
    if all(len(slot) == 1 for *_, slots in plans for slot in slots):
        lines.append(
            f"const {_ACCUMULATOR} {variable} = simd_shuffle("
            f"{_ACCUMULATOR}({tensor}[{owner} >> 5]), {lane});"
        )
        return lines
    # Every slot reads the same number of candidates, so the loop's trip count is
    # a constant and every lane runs the same shuffles; a short slot repeats its
    # first candidate, which selects the same element again.
    width = max(len(slot) for *_, slots in plans for slot in slots)
    candidates = []
    for value, variant, _, slots in plans:
        values = [
            element
            for slot in slots
            for element in slot + [slot[0]] * (width - len(slot))
        ]
        name = f"{variant.stem}_{role}_{operation.m}x{operation.n}x{operation.k}_candidates"
        candidates.append((value, tables.array(name, values)))
    slot = f"{_PREFIX}_slot"
    lines += [
        f"{_ACCUMULATOR} {variable} = 0.0f;",
        f"for (ushort {loop} = 0; {loop} < {width}; {loop}++) {{",
        f"  const ushort {variable}_element = "
        + _select(flag, candidates, f"{slot} * {width} + {loop}")
        + ";",
        f"  const {_ACCUMULATOR} {variable}_value = simd_shuffle("
        f"{_ACCUMULATOR}({tensor}[{variable}_element]), {lane});",
        f"  {variable} = ({owner} >> 5) == {variable}_element ? {variable}_value : {variable};",
        "}",
    ]
    return lines


def _run_block(operation: _Operation, arguments, layout: str, tables: _Tables) -> str:
    left, right, destination = arguments
    accumulator_type = operation.tensors[destination].element_type
    capacity = operation.m * operation.n // SUBGROUP_SIZE
    index = tables.index_table(_Variant(layout, operation.m, operation.n, False))
    if operation.mode == "1":
        start = f"{_ACCUMULATOR}({destination}[{_PREFIX}_e])"
    elif operation.mode == "0":
        start = "0.0f"
    else:
        start = f"({operation.mode} == 1) ? {_ACCUMULATOR}({destination}[{_PREFIX}_e]) : 0.0f"
    body = [
        *_fetch(
            operation,
            "left_input",
            left,
            layout,
            tables,
            f"{_PREFIX}_x",
            f"{_PREFIX}_i",
        ),
        *_fetch(
            operation,
            "right_input",
            right,
            layout,
            tables,
            f"{_PREFIX}_y",
            f"{_PREFIX}_j",
        ),
        f"{_PREFIX}_sum = fma({_PREFIX}_x, {_PREFIX}_y, {_PREFIX}_sum);",
    ]
    if any(re.search(rf"\b{_PREFIX}_slot\b", line) for line in body):
        slot = f"{_PREFIX}_e * {operation.k} + {_PREFIX}_k"
        body.insert(0, f"const ushort {_PREFIX}_slot = {slot};")
    lines = []
    if accumulator_type != _ACCUMULATOR:
        if re.fullmatch(
            r"half|bfloat|bfloat16_t|int|uint|short|ushort|char|uchar", accumulator_type
        ):
            raise MetalMatmul2dError(
                "matmul2d with cooperative-tensor operands is lowered for float "
                f"destinations, not {accumulator_type}"
            )
        lines.append(
            f"static_assert(metal::is_same_v<{accumulator_type}, {_ACCUMULATOR}>, "
            '"matmul2d with cooperative-tensor operands is lowered for float destinations");'
        )
    lines += [
        f"const ushort {_PREFIX}_lane = {_LANE};",
        f"for (ushort {_PREFIX}_e = 0; {_PREFIX}_e < {capacity}; {_PREFIX}_e++) {{",
        f"  const ushort {_PREFIX}_packed = {index}[{_PREFIX}_lane * {capacity} + {_PREFIX}_e];",
        f"  const ushort {_PREFIX}_row = {_PREFIX}_packed >> 8;",
        f"  const ushort {_PREFIX}_column = {_PREFIX}_packed & 255;",
        f"  {_ACCUMULATOR} {_PREFIX}_sum = {start};",
        f"  for (ushort {_PREFIX}_k = 0; {_PREFIX}_k < {operation.k}; {_PREFIX}_k++) {{",
        *(f"    {line}" for line in body),
        "  }",
        f"  {destination}[{_PREFIX}_e] = {_PREFIX}_sum;",
        "}",
    ]
    return "{\n" + "".join(f"  {line}\n" for line in lines) + "}"


def lower_matmul2d_operations(preprocessor: MetalPreprocessor, source: str) -> str:
    """``source`` with every ``matmul2d`` on cooperative tensors lowered as the module describes."""
    if "matmul2d" not in source and "MetalPerformancePrimitives" not in source:
        return source
    configured = validate_cooperative_tensor_layout(
        getattr(preprocessor, LAYOUT_OPTION, None)
    )
    layout = configured or DEFAULT_LAYOUT
    masked = _scope_masked(preprocessor, source)
    if re.search(rf"\b{_PREFIX}_", masked):
        raise MetalMatmul2dError(
            f"names starting with {_PREFIX}_ are reserved for lowered matmul2d operations"
        )
    names = _Names(masked)
    source = _rewrite_mode_spellings(source, masked, names)
    descriptor = re.compile(
        rf"\bconstexpr\s+auto\s+(?P<name>{_IDENTIFIER})\s*=\s*"
        + names.name("matmul2d_descriptor")
        + r"\s*\("
    )
    tables = _Tables()
    for _ in range(len(descriptor.findall(_scope_masked(preprocessor, source)))):
        masked = _scope_masked(preprocessor, source)
        match = descriptor.search(masked)
        operation = _operation(source, masked, match, names)
        _tensors(source, masked, operation)
        _uses(source, masked, operation, layout, configured is not None, tables)
        source = _apply(source, operation.edits)
    masked = _scope_masked(preprocessor, source)
    leftover = re.search(
        names.name(r"(?:matmul2d|matmul2d_descriptor|convolution2d)")
        + r"|\bget_(?:left_input|right_input|destination)_cooperative_tensor\b",
        masked,
    )
    if leftover is not None:
        raise MetalMatmul2dError(
            "MetalPerformancePrimitives operations are lowered only as "
            "`constexpr auto desc = matmul2d_descriptor(...)` followed by one "
            f"matmul2d<desc, execution_simdgroup> on cooperative tensors; found "
            f"'{_normalized(masked[leftover.start() : leftover.end()])}'"
        )
    source = _drop_library_declarations(source, masked)
    masked = _scope_masked(preprocessor, source)
    if tables.definitions:
        first_use = re.search(rf"\b{_PREFIX}_", masked)
        start = _declaration_start(masked, first_use.start())
        source = source[:start] + tables.text() + source[start:]
    return source


def _drop_library_declarations(source: str, masked: str) -> str:
    """Remove MPP's includes and using-directives, which nothing names once its operations are lowered.

    Without the header the ``mpp`` namespaces do not exist, so a using-directive that nominates them would not
    compile; a remaining qualified ``mpp::`` name keeps both.
    """
    if re.search(r"(?<![\w:])mpp\s*::", masked) or re.search(
        r"(?<![\w:])tensor_ops\s*::", masked
    ):
        return source
    edits = [
        (match.start(), match.end(), "")
        for match in re.finditer(
            r"(?m)^[ \t]*#[ \t]*include[ \t]*<MetalPerformancePrimitives/[\w./]+>[ \t]*$",
            source,
        )
    ]
    edits += [
        (match.start(), match.end(), "")
        for match in re.finditer(
            r"\busing\s+namespace\s+(?:::)?\s*mpp(?:\s*::\s*tensor_ops)?\s*;", masked
        )
    ]
    return _apply(source, edits)
