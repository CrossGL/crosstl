"""Integer bit functions: HLSL spellings for every target, and 8- and 16-bit operands computed at their width.

HLSL spellings. CrossGL accepts the HLSL names ``countbits``, ``reversebits``, ``firstbithigh`` and ``firstbitlow``
next to GLSL's ``bitCount``, ``bitfieldReverse``, ``findMSB`` and ``findLSB``, but only the HLSL and Slang emitters
know them. The others emitted them unchanged, which GLSL, WGSL, MSL, Rust and Mojo do not declare. The pairs differ
in result type: ``countbits``, ``firstbithigh`` and ``firstbitlow`` return ``uint`` lanes, while their GLSL
counterparts return ``int`` lanes (GLSL 4.60 section 8.8). So ``firstbithigh(0) > 5`` holds in HLSL (``0xFFFFFFFF``
compares as unsigned) but ``findMSB(0) > 5`` does not. For every target but HLSL, Slang and CrossGL the pass spells
``countbits(x)`` as ``U(bitCount(x))``, ``firstbithigh(x)`` as ``U(findMSB(x))`` and ``firstbitlow(x)`` as
``U(findLSB(x))``, where ``U`` is ``uint`` with ``x``'s lanes, and ``reversebits(x)`` as ``bitfieldReverse(x)``,
which has ``x``'s type in both languages. A call whose argument type it cannot infer is refused with
:class:`BitFunctionArgumentUntypedError`, because the lanes of ``U`` are unknown.

Narrow operands. ``bitCount`` counts the set bits of a value at its own width and ``bitfieldReverse`` reverses them.
GLSL 4.60 section 8.8 with ``GL_EXT_shader_explicit_arithmetic_types`` overloads both for 8- and 16-bit integers
(glslang emits ``OpBitCount`` and ``OpBitReverse`` on ``%v2short``). HLSL ``countbits`` and ``reversebits`` take
16-bit integers under ``-enable-16bit-types``, and MSL 3.2 section 6.13 defines ``popcount`` and ``reverse_bits``
for every integer type. Several emitters compute 8- and 16-bit values in 32-bit carriers instead:

- Metal keeps every narrow register in an ``int`` or ``uint`` carrier (``codegen/metal_narrow_vectors.py``);
- the DirectX emitter keeps its 8-bit values in carriers, which HLSL has no type for;
- GLSL output without the 16-bit extension keeps all of them in carriers;
- SPIR-V and WGSL output do so after :mod:`.narrow`.

A signed carrier holds its value sign-extended and an unsigned one zero-extended. So a 32-bit count of
``int16_t(-1)`` is 32, not 16, and a 32-bit reversal moves every bit out of the low 16: ``bitfieldReverse(int16_t(1))``
is ``-32768``, but the carrier gives 0. ``abs`` of the minimum value is that value again at the narrow width (each
language computes ``abs`` in the operand type), while a carrier holds 32768. The pass spells these calls on explicit
32-bit ``uint`` operands, which each target computes the same whether its narrow registers are native or carriers:

- ``bitCount(x)`` becomes ``S(bitCount(U(x) & M))``. ``M`` is the mask of the ``N`` value bits and ``S`` the signed
  ``N``-bit type with ``x``'s lanes, which is GLSL's result type. HLSL ``countbits`` keeps its ``uint`` result:
  ``countbits(U(x) & M)``.
- ``bitfieldReverse(x)`` and ``reversebits(x)`` become ``T(f(U(x)) >> (32 - N))``, with ``T`` the type of ``x``.
  The 32-bit reversal puts the low ``N`` bits, reversed, at the top, and the logical shift brings them down.
- ``abs(x)`` of a signed ``x`` becomes ``T(abs(x))``.

Each conversion to a narrow type reduces its value modulo 2^N (C++ [conv.integral]), which every emitter spells for
its carriers. A narrow scalar of a promoting source that results from arithmetic is an ``int`` already, so a call
on it is left alone (C++ [conv.prom]). A call of a program function with one of these names is never rewritten.
The pass runs for every target before the narrow-integer passes.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import HubType, call_name
from .narrow import NarrowLegalizer, narrow_from_name, narrow_of
from .narrow_arithmetic import type_name
from .rewrite import call

__all__ = ["BitFunctionArgumentUntypedError", "legalize_bit_functions"]

# Targets whose emitters spell the HLSL bit functions natively, and CrossGL output, which keeps the source's names.
_HLSL_SPELLING_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "slang", "cgl", "crossgl"}
)
# HLSL name: (GLSL name, whether the HLSL result is ``uint`` lanes where GLSL's is ``int`` lanes).
_HLSL_SPELLINGS = {
    "countbits": ("bitCount", True),
    "reversebits": ("bitfieldReverse", False),
    "firstbithigh": ("findMSB", True),
    "firstbitlow": ("findLSB", True),
}
# Count functions and whether their result has the argument's width (GLSL) or is ``uint`` (HLSL).
_COUNTS = {"bitCount": True, "countbits": False}
_REVERSALS = frozenset({"bitfieldReverse", "reversebits"})
_SIGNED_RESULTS = frozenset({"abs"})
_NARROW_NAMES = frozenset(_COUNTS) | _REVERSALS | _SIGNED_RESULTS


class BitFunctionArgumentUntypedError(TranslationDiagnostic):
    """An HLSL bit function whose argument type is unknown, so its ``uint`` result has no spelling elsewhere."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.hlsl-bit-function-untyped",)
        glsl_name = _HLSL_SPELLINGS[name][0]
        super().__init__(
            f"HLSL {name}() of an argument whose type is unknown: the target spells it {glsl_name}(), whose result "
            "type differs, and the lanes of the uint conversion that keeps HLSL's result are unknown",
            target=target,
            code=f"{CODE_PREFIX}hlsl-bit-function-untyped",
        )


def _uint_type(lanes):
    return "uint" if lanes == 1 else f"uvec{lanes}"


def _uint_literal(value, lanes):
    literal = A.LiteralNode(value, A.PrimitiveType("uint"))
    return literal if lanes == 1 else call(f"uvec{lanes}", [literal])


def _located(node, original):
    node.source_location = getattr(original, "source_location", None)
    return node


def count_bits(name, argument, narrow: HubType):
    """``bitCount``/``countbits`` of ``argument`` at the width of ``narrow``."""
    lanes = narrow.lanes
    masked = A.BinaryOpNode(
        call(_uint_type(lanes), [argument]),
        "&",
        _uint_literal((1 << narrow.bits) - 1, lanes),
    )
    counted = call(name, [masked])
    if not _COUNTS[name]:
        return counted
    return call(type_name(HubType("i", narrow.bits, lanes)), [counted])


def reverse_bits(name, argument, narrow: HubType):
    """``bitfieldReverse``/``reversebits`` of ``argument`` at the width of ``narrow``."""
    lanes = narrow.lanes
    reversed_word = call(name, [call(_uint_type(lanes), [argument])])
    shifted = A.BinaryOpNode(
        reversed_word, ">>", _uint_literal(32 - narrow.bits, lanes)
    )
    return call(type_name(narrow), [shifted])


class BitFunctions(NarrowLegalizer):
    """Rewrite bit-function calls; registers and all other code keep their types."""

    def __init__(self, program, source, target):
        super().__init__(program, source, target)
        self.spell_hlsl = str(target or "").lower() not in _HLSL_SPELLING_TARGETS
        self.narrow = False
        # Narrow types of operations, by node identity, so the AST carries no annotation into later passes.
        self.operation_types = {}

    def run(self):
        names = {
            call_name(node)
            for node in self.program.walk()
            if isinstance(node, A.FunctionCallNode)
        }
        hlsl = self.spell_hlsl and bool(names & set(_HLSL_SPELLINGS))
        self.narrow = bool(names & _NARROW_NAMES) and self._mentions_narrow()
        if not (hlsl or self.narrow):
            return
        program, scope = self.program, self.scope
        for constant in getattr(program, "constants", None) or []:
            scope.declare(constant.name, constant.const_type)
            self._initializer(constant, "const_type", "value")
        self._declare_variables(getattr(program, "global_variables", None) or [])
        for cbuffer in getattr(program, "cbuffers", None) or []:
            scope.declare_cbuffer(cbuffer)
        self._member_initializers()
        for function in getattr(program, "functions", None) or []:
            if isinstance(function, A.FunctionNode):
                self.function(function)
        for stage in self._stages():
            scope.push()
            self._declare_variables(getattr(stage, "local_variables", None) or [])
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                scope.declare_cbuffer(cbuffer)
            for function in list(getattr(stage, "local_functions", None) or []) + [
                getattr(stage, "entry_point", None)
            ]:
                if isinstance(function, A.FunctionNode):
                    self.function(function)
            scope.pop()

    # The base pass widens registers and normalizes writes; this one only rewrites calls.
    def statement(self, node):
        return self.visit(node)

    def _normalize_declaration(self, node, type_field, value_field):
        return None

    def rewrite(self, node):
        if isinstance(node, (A.BinaryOpNode, A.UnaryOpNode)):
            if self.narrow:
                narrow = self.arithmetic_narrow(node)
                if narrow is not None:
                    self.operation_types[id(node)] = narrow
            return node
        if isinstance(node, A.FunctionCallNode):
            name = call_name(node)
            if name in self.functions or len(node.arguments or []) != 1:
                return node
            if self.spell_hlsl and name in _HLSL_SPELLINGS:
                return _located(self.glsl_spelling(node, name), node)
            if self.narrow and name in _NARROW_NAMES:
                return _located(self.narrow_call(node, name), node)
        return node

    def glsl_spelling(self, node, name):
        """``node``, an HLSL bit-function call, spelled with the GLSL function and HLSL's result type."""
        argument = node.arguments[0]
        glsl_name, unsigned_result = _HLSL_SPELLINGS[name]
        renamed = call(glsl_name, [argument])
        if self.narrow and glsl_name in _NARROW_NAMES:
            renamed = self.narrow_call(renamed, glsl_name)
        if not unsigned_result:
            return renamed
        value = self.scope.infer(argument)
        if value is None or value.kind not in ("i", "u"):
            raise BitFunctionArgumentUntypedError(
                self.target, name, source_location=node.source_location
            )
        return call(_uint_type(value.lanes), [renamed])

    def narrow_call(self, node, name):
        """``node`` computed at the width of its narrow argument, or unchanged when the argument is not narrow."""
        argument = node.arguments[0]
        narrow = self.narrow_value(argument)
        if narrow is None:
            return node
        if name in _COUNTS:
            return count_bits(name, argument, narrow)
        if name in _REVERSALS:
            return reverse_bits(name, argument, narrow)
        if narrow.kind != "i":
            return node
        return call(type_name(narrow), [node])

    def narrow_value(self, expr):
        known = self.operation_types.get(id(expr))
        if known is not None:
            return known
        narrow = super().narrow_value(expr)
        if narrow is not None:
            return narrow
        if isinstance(expr, A.FunctionCallNode):
            return narrow_from_name(call_name(expr))
        if isinstance(expr, A.CastNode):
            return narrow_of(expr.target_type)
        if isinstance(expr, A.LiteralNode):
            return narrow_of(expr.literal_type)
        return None


def legalize_bit_functions(program, source, target) -> bool:
    """Spell HLSL bit functions and narrow-integer bit functions for ``target``; return whether the pass ran."""
    if not isinstance(program, A.ShaderNode):
        return False
    BitFunctions(program, str(source or ""), target).run()
    return True
