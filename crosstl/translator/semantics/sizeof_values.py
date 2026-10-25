"""``sizeof`` of a CrossGL type or value becomes its byte count, or is refused where the targets disagree.

The Metal and HLSL importers fold ``sizeof`` of concrete types with the source layout. ``sizeof`` of a template
parameter (``sizeof(T)``), or of a value of that type, stays in the CrossGL program, and hand-written CrossGL may
spell it too. The emitters printed the call unchanged: GLSL has no ``sizeof`` and SPIR-V refused it, while MSL and HLSL
evaluated it with their own layout, which is not the source's for every type, and Metal widens ``int8`` and
``int16`` to 32 bits. CrossGL's ``sizeof`` is defined where every target agrees:

* a scalar is its bit width over eight (``half`` and ``bfloat16`` 2, ``int`` and ``float`` 4, ``double`` and
  ``int64`` 8, ``int8`` 1);
* a 2- or 4-component vector is its component count times its scalar's size;
* a sized array is its element count times its element's size.

Other types are refused. ``bool``: C++ leaves its size to the implementation
([expr.sizeof]/1); MSL stores it in one byte and HLSL in four. 3-component vectors: MSL gives them the size of four
components (MSL specification, table 2.3) and HLSL three. Matrices: MSL pads ``float3x3`` columns to four components
and HLSL does not. Minimum-precision HLSL types: their storage width is the implementation's choice. Structures:
their member layout is the target's.

Other types are refused with :class:`SizeofUnsupportedError`. ``metal_sizeof`` is ``sizeof`` with MSL's layout
(MSL specification, tables 2.2 to 2.4), which the Metal importer emits for the operands it cannot size itself because
they name a template parameter: ``bool`` is one byte, a 3-component vector has the size of four components, and a
``floatCxR`` or ``halfCxR`` matrix is ``C`` columns of ``R``-component vectors. Its structures are still refused.

This pass replaces each foldable call with an ``int`` literal. A value operand whose type involves an enclosing
generic function's type parameter is rewritten to name that type, so the Metal, HLSL and GLSL generators fold it
after specialization (:mod:`..codegen.sizeof_calls`); for other targets such a call is refused. CrossGL
output keeps both spellings. The CUDA and HIP emitters, which print ``sizeof`` natively with C++ layout, keep the
portable calls this pass cannot fold.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import TypeScope, type_from_name, type_from_node
from .rewrite import child_slots, get_child, set_child

__all__ = [
    "SIZEOF_LAYOUTS",
    "SizeofUnsupportedError",
    "fold_sizeof_values",
    "is_sizeof_call",
    "sizeof_type_bytes",
]

# Each ``sizeof`` spelling, with the layout it measures.
SIZEOF_LAYOUTS = {"sizeof": "portable", "metal_sizeof": "metal"}
_FUNCTIONS = {layout: name for name, layout in SIZEOF_LAYOUTS.items()}
_KEEP_TARGETS = frozenset({"cgl", "crossgl"})
_NATIVE_SIZEOF_TARGETS = frozenset({"cuda", "hip"})
# Targets whose generators fold ``sizeof`` of a type parameter in each specialization (:mod:`..codegen.sizeof_calls`).
# The SPIR-V generator infers a specialization's types after binary16 and narrow integers are widened, so it would
# size ``half`` as four bytes.
_SPECIALIZING_TARGETS = frozenset(
    {
        "metal",
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
        "opengl",
        "glsl",
        "webgl",
    }
)
_SCOPES = (A.BlockNode, A.ForNode, A.ForInNode, A.FunctionNode)
_MIN_PRECISION = frozenset(
    {"min16float", "min10float", "min16int", "min12int", "min16uint"}
)
# Spellings the hub's scalar table does not list, with their width in bytes.
_EXTRA_SCALARS = {
    "float16": 2,
    "bfloat": 2,
    "bfloat16": 2,
    "bfloat16_t": 2,
    "float32": 4,
    "float32_t": 4,
    "float64": 8,
    "float64_t": 8,
}
_EXTRA_VECTOR = re.compile(r"(bfloat16vec|bfloat|i8vec|u8vec|i16vec|u16vec)([234])")
_EXTRA_VECTOR_BYTES = {
    "bfloat16vec": 2,
    "bfloat": 2,
    "i8vec": 1,
    "u8vec": 1,
    "i16vec": 2,
    "u16vec": 2,
}
_MATRIX = re.compile(r"(?:[a-z0-9]*mat[234](?:x[234])?|[a-z0-9_]+[234]x[234])")
# CrossGL matrix spellings follow GLSL: ``matCxR`` has ``C`` columns of ``R`` components.
_FLOAT_MATRIX = re.compile(r"(f16|h)?mat([234])(?:x([234]))?")
_ARRAY_SPELLING = re.compile(r"(.+?)\s*\[\s*(\d+)\s*\]")


class SizeofUnsupportedError(TranslationDiagnostic):
    """A ``sizeof`` whose operand has no size every target agrees on, or no type the translator knows."""

    code = CODE_PREFIX + "sizeof-unsupported"

    def __init__(self, operand, detail, *, target=None, function="sizeof"):
        self.operand = str(operand)
        self.detail = detail
        self.function = function
        super().__init__(f"{function}({operand}): {detail}", target=target)


def _literal_int(node):
    if isinstance(node, int) and not isinstance(node, bool):
        return node
    if isinstance(node, A.LiteralNode) and not isinstance(node.value, bool):
        text = str(node.value).strip().rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


def _spelling(type_value):
    """The CrossGL spelling of a type node, for diagnostics and for naming a type in a ``sizeof`` operand."""
    if isinstance(type_value, str):
        return type_value.strip()
    if isinstance(type_value, A.ArrayType):
        size = _literal_int(type_value.size)
        if size is None and isinstance(type_value.size, A.IdentifierNode):
            size = type_value.size.name
        suffix = "" if size is None else size
        return f"{_spelling(type_value.element_type)}[{suffix}]"
    if isinstance(type_value, A.VectorType):
        return f"vec{type_value.size}<{_spelling(type_value.element_type)}>"
    if isinstance(type_value, A.MatrixType):
        return f"mat{type_value.cols}x{type_value.rows}<{_spelling(type_value.element_type)}>"
    if isinstance(type_value, A.NamedType) and type_value.generic_args:
        arguments = ", ".join(_spelling(arg) for arg in type_value.generic_args)
        return f"{type_value.name}<{arguments}>"
    return str(getattr(type_value, "name", type_value))


def _scalar_or_vector_bytes(type_value, operand, layout):
    name = type_value.replace(" ", "") if isinstance(type_value, str) else None
    if isinstance(type_value, (A.PrimitiveType, A.NamedType)) and not getattr(
        type_value, "generic_args", None
    ):
        name = type_value.name
    if name in _MIN_PRECISION:
        raise SizeofUnsupportedError(
            operand,
            "a minimum-precision type's storage width is chosen by the implementation",
        )
    if name in _EXTRA_SCALARS:
        return _EXTRA_SCALARS[name]
    match = _EXTRA_VECTOR.fullmatch(name or "")
    if match:
        return _vector_bytes(
            int(match.group(2)), _EXTRA_VECTOR_BYTES[match.group(1)], operand, layout
        )
    if isinstance(type_value, A.VectorType):
        element = _scalar_or_vector_bytes(type_value.element_type, operand, layout)
        if element is None or not str(type_value.size).isdigit():
            return None
        return _vector_bytes(int(type_value.size), element, operand, layout)
    value = type_from_name(name) if name is not None else type_from_node(type_value)
    if value is None:
        return None
    if value.kind == "b":
        if layout == "metal":
            return _vector_bytes(value.lanes, 1, operand, layout)
        raise SizeofUnsupportedError(
            operand,
            "bool has no portable size: C++ leaves it to the implementation ([expr.sizeof]/1), MSL stores it in one "
            "byte and HLSL in four",
        )
    return _vector_bytes(value.lanes, value.bits // 8, operand, layout)


def _vector_bytes(lanes, scalar_bytes, operand, layout):
    if lanes == 3:
        if layout == "metal":
            return 4 * scalar_bytes
        raise SizeofUnsupportedError(
            operand,
            "a 3-component vector has the size of four components in MSL (MSL specification, table 2.3) and of "
            "three in HLSL",
        )
    return lanes * scalar_bytes


def _metal_matrix_bytes(type_value, operand):
    """MSL's size of a ``half`` or ``float`` matrix: its columns, each the size of its column vector."""
    if isinstance(type_value, A.MatrixType):
        element = type_from_node(type_value.element_type)
        columns, rows = type_value.cols, type_value.rows
    else:
        match = _FLOAT_MATRIX.fullmatch(type_value.strip())
        element = type_from_name("half" if match and match.group(1) else "float")
        columns = match and match.group(2)
        rows = match and (match.group(3) or match.group(2))
    if (
        element is None
        or element.kind != "f"
        or element.bits not in (16, 32)
        or not str(columns).isdigit()
        or not str(rows).isdigit()
    ):
        raise SizeofUnsupportedError(
            operand, "MSL has matrices of half and float components only"
        )
    return int(columns) * _vector_bytes(int(rows), element.bits // 8, operand, "metal")


def sizeof_type_bytes(type_value, *, constants=None, target=None, layout="portable"):
    """The byte size of a CrossGL type (a ``TypeNode`` or a spelling); raises :class:`SizeofUnsupportedError`.

    ``layout`` is ``"portable"`` for ``sizeof`` and ``"metal"`` for ``metal_sizeof``.
    """
    operand = _spelling(type_value)
    try:
        return _type_bytes(type_value, operand, constants or {}, layout)
    except SizeofUnsupportedError as error:
        raise SizeofUnsupportedError(
            error.operand, error.detail, target=target, function=_FUNCTIONS[layout]
        ) from None


def _type_bytes(type_value, operand, constants, layout):
    if isinstance(type_value, A.ReferenceType):
        return _type_bytes(type_value.referenced_type, operand, constants, layout)
    if isinstance(type_value, str):
        match = _ARRAY_SPELLING.fullmatch(type_value.strip())
        if match:
            return int(match.group(2)) * _type_bytes(
                match.group(1), operand, constants, layout
            )
    if isinstance(type_value, A.ArrayType):
        count = _literal_int(type_value.size)
        if count is None and isinstance(type_value.size, A.IdentifierNode):
            count = constants.get(type_value.size.name)
        if count is None:
            raise SizeofUnsupportedError(
                operand, "the array's element count is not a constant"
            )
        # Every layout here sizes an element to a multiple of its alignment, so the array has no padding.
        return count * _type_bytes(type_value.element_type, operand, constants, layout)
    if isinstance(type_value, A.MatrixType) or (
        isinstance(type_value, str) and _MATRIX.fullmatch(type_value.strip())
    ):
        if layout == "metal":
            return _metal_matrix_bytes(type_value, operand)
        raise SizeofUnsupportedError(
            operand,
            "a matrix's size depends on the target's column layout (MSL pads 3-component columns to four)",
        )
    if isinstance(type_value, A.PointerType):
        raise SizeofUnsupportedError(operand, "pointer sizes differ between targets")
    size = _scalar_or_vector_bytes(type_value, operand, layout)
    if size is not None:
        return size
    raise SizeofUnsupportedError(
        operand,
        (
            "the type is not a scalar, a vector, a matrix or a sized array of them; a structure's layout is the "
            "target's"
            if layout == "metal"
            else "the type is not a scalar, a 2- or 4-component vector or a sized array of them; a structure's layout "
            "is the target's"
        ),
    )


def is_sizeof_call(node) -> bool:
    callee = getattr(node, "function", None)
    return (
        isinstance(node, A.FunctionCallNode)
        and isinstance(callee, A.IdentifierNode)
        and callee.name in SIZEOF_LAYOUTS
    )


def _generic_names(function):
    names = set()
    for parameter in getattr(function, "generic_params", None) or []:
        name = getattr(parameter, "name", parameter)
        if isinstance(name, str):
            names.add(name)
    return names


def _mentions(type_value, names):
    if not names or type_value is None:
        return False
    spelling = _spelling(type_value)
    return any(re.search(rf"\b{re.escape(name)}\b", spelling) for name in names)


class _Folder:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.native = target in _NATIVE_SIZEOF_TARGETS
        self.scope = TypeScope(program)
        self.constants = {}
        self.generic_names = []
        self.folded = 0

    def run(self):
        for constant in getattr(self.program, "constants", None) or []:
            self.visit_children(constant)
            self.scope.declare(constant.name, constant.const_type)
            value = _literal_int(getattr(constant, "value", None))
            if value is not None:
                self.constants[constant.name] = value
        for cbuffer in getattr(self.program, "cbuffers", None) or []:
            self.scope.declare_cbuffer(cbuffer)
        self.declare_variables(getattr(self.program, "global_variables", None) or [])
        for function in getattr(self.program, "functions", None) or []:
            self.visit(function)
        stages = getattr(self.program, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            self.scope.push()
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                self.scope.declare_cbuffer(cbuffer)
            self.declare_variables(getattr(stage, "local_variables", None) or [])
            for function in getattr(stage, "local_functions", None) or []:
                self.visit(function)
            if getattr(stage, "entry_point", None) is not None:
                self.visit(stage.entry_point)
            self.scope.pop()
        return self.folded

    def declare_variables(self, variables):
        for variable in variables:
            if not isinstance(variable, A.VariableNode):
                continue
            self.visit_children(variable)
            self.scope.declare(variable.name, variable.var_type)
            if "const" in {str(q) for q in variable.qualifiers or []}:
                value = _literal_int(getattr(variable, "initial_value", None))
                if value is not None:
                    self.constants[variable.name] = value

    def visit(self, node):
        scoped = isinstance(node, _SCOPES)
        if scoped:
            self.scope.push()
        if isinstance(node, A.FunctionNode):
            self.generic_names.append(_generic_names(node))
            for parameter in node.parameters or []:
                self.scope.declare(parameter.name, parameter.param_type)
            self.visit_children(node, skip=("parameters", "return_type"))
            self.generic_names.pop()
        else:
            self.visit_children(node)
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        if scoped:
            self.scope.pop()

    def visit_children(self, node, skip=()):
        for field, key in list(child_slots(node)):
            if field in skip or field in ("var_type", "param_type", "const_type"):
                continue
            child = get_child(node, field, key)
            if is_sizeof_call(child):
                replacement = self.fold(child)
                if replacement is not child:
                    set_child(node, field, key, replacement)
                continue
            self.visit(child)

    def operand_type(self, operand):
        """The declared type of a value operand, the named type of a type operand, or ``None``."""
        if isinstance(operand, A.IdentifierNode):
            declared = self.scope.lookup(operand.name)
            return (declared, True) if declared is not None else (operand.name, False)
        declared = self.scope.node_type(operand)
        if declared is not None:
            return declared, True
        if isinstance(operand, A.ArrayAccessNode) and isinstance(
            operand.array_expr, A.IdentifierNode
        ):
            count = _literal_int(operand.index_expr)
            if count is None and isinstance(operand.index_expr, A.IdentifierNode):
                count = self.constants.get(operand.index_expr.name)
            if count is not None:
                return f"{operand.array_expr.name}[{count}]", False
        value = self.scope.infer(operand)
        if value is not None:
            spelling = {"i": "int", "u": "uint", "f": "float", "b": "bool"}[value.kind]
            spelling = f"{spelling}{value.bits}" if value.kind != "b" else spelling
            return (
                A.VectorType(A.NamedType(spelling), value.lanes)
                if value.lanes > 1
                else spelling
            ), True
        return None, True

    def fold(self, call):
        name = call.function.name
        if name in self.scope.functions:
            return call
        layout = SIZEOF_LAYOUTS[name]
        native = self.native and layout == "portable"
        if len(call.arguments or []) != 1:
            raise SizeofUnsupportedError(
                ", ".join(_spelling(a) for a in call.arguments or []),
                "it takes exactly one operand",
                target=self.target,
                function=name,
            )
        operand = call.arguments[0]
        type_value, is_value = self.operand_type(operand)
        generic = self.generic_names[-1] if self.generic_names else set()
        if type_value is None:
            if native:
                return call
            raise SizeofUnsupportedError(
                type(operand).__name__,
                "the operand's type is not known",
                target=self.target,
                function=name,
            )
        if _mentions(type_value, generic):
            if native:
                return call
            if self.target not in _SPECIALIZING_TARGETS:
                raise SizeofUnsupportedError(
                    _spelling(type_value),
                    "the operand names a type parameter, and this target's generator does not size it in each "
                    "specialization",
                    target=self.target,
                    function=name,
                )
            if is_value:
                # The generators substitute the type parameter and fold it after specialization.
                call.arguments[0] = A.IdentifierNode(_spelling(type_value))
            return call
        try:
            size = sizeof_type_bytes(
                type_value, constants=self.constants, target=self.target, layout=layout
            )
        except SizeofUnsupportedError:
            if native:
                return call
            raise
        self.folded += 1
        return A.LiteralNode(size, A.PrimitiveType("int"))


def fold_sizeof_values(program, target) -> int:
    """Replace ``sizeof`` and ``metal_sizeof`` calls by their byte count, refusing unsizable ones; returns how many."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target in _KEEP_TARGETS:
        return 0
    return _Folder(program, target).run()
