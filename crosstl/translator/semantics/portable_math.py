"""C math functions that only some targets declare: ``sinpi``, ``cospi``, ``fdim``, ``nextafter``, ``ilogb``, ``copysign``.

CrossGL takes these functions from C (C17 section 7.12, IEEE 754-2019 section 9.2 for ``sinPi`` and ``cosPi``), as
the CUDA and HIP math libraries, OpenCL C and the Metal standard library declare them, component by component on
vectors:

* ``sinpi(x)`` is ``sin(pi * x)`` and ``cospi(x)`` is ``cos(pi * x)``, exact at integers and half-integers:
  ``sinpi(n)`` is a zero with the sign of ``n``, ``cospi(n + 1/2)`` is ``+0``; infinities and NaN give NaN.
* ``fdim(x, y)`` is ``x - y`` when ``x > y`` and ``+0`` otherwise; a NaN operand gives NaN.
* ``nextafter(x, y)`` is the representable value next to ``x`` in the direction of ``y``: ``y`` when the two are
  equal, NaN when either is NaN.
* ``ilogb(x)`` is the unbiased exponent of ``x`` as an ``int`` (that of a subnormal as if it were normalised). Zero
  and NaN give ``INT_MIN`` and infinities ``INT_MAX``, the values the CUDA math library returns (C leaves
  ``FP_ILOGB0`` and ``FP_ILOGBNAN`` to the implementation).
* ``copysign(x, y)`` is the magnitude of ``x`` with the sign bit of ``y``.

GLSL, HLSL, SPIR-V, WGSL, Slang, Rust and Mojo lack some of them. For each target without the function,
:func:`lower_portable_math` rewrites the call into a call of a CrossGL helper that it adds to the program once, ahead
of the program's functions; targets that declare the function natively (:func:`register_native_portable_math`) keep
the call. The binary32 helpers use only operations every target defines exactly or bounds tightly: bit
reinterpretation, 32-bit integer arithmetic, comparison, select, ``floor`` and correctly rounded binary32 addition and
multiplication, so neither the target's own transcendental functions nor its treatment of NaN in floating-point
comparisons enters the result.

* ``sinpi`` and ``cospi`` reduce ``x`` exactly: ``n = round(2x)`` and ``t = x - n/2`` (Sterbenz), ``|t| <= 1/4``,
  and evaluate ``sin(pi t)`` or ``cos(pi t)`` by the quadrant ``n mod 4``. Operands of magnitude ``2^23`` and above
  are integers and take their values from the bits. ``sin(pi t)`` is ``t * pi`` with ``t`` and ``pi`` split so that
  the leading product is exact (12 significant bits each), plus an odd polynomial of degree 7; ``cos(pi t)`` is ``1``
  plus an even polynomial of degree 8 whose leading product is exact (8 significant bits of ``t`` and of its
  coefficient). The coefficients are Lawson minimax fits of the binary64 functions rounded to binary32. Evaluated in
  binary32 over every binary32 ``t`` with ``|t| <= 1/4``, with and without fused multiply-add, the largest error is
  0.81 ulp for the sine and 0.90 ulp for the cosine (the reduction adds none), within the 1 ulp the CUDA math library
  bounds ``sinpif`` and ``cospif`` by.
* ``fdim``, ``nextafter``, ``ilogb`` and ``copysign`` test NaN from the bits, and ``nextafter`` steps through the
  ordered integer image of the encoding, so subnormal operands are exact as long as the target keeps them.

``half`` operands compute in binary32 and convert back, which is exact for ``fdim``, ``ilogb`` and ``copysign``
(binary32 has more than twice the precision of binary16 plus two bits) and rounds ``sinpi`` and ``cospi`` once more.
``double`` operands have a helper for ``fdim`` only. A call the helpers cannot take (``nextafter`` of ``half``, the
other functions of ``double``, an operand whose type cannot be inferred or is not floating-point) is refused with
:class:`PortableMathError`, as is a program that defines a function with a helper's reserved name. A call of a
function the program defines itself under one of these names is left alone.
"""

from __future__ import annotations

import functools
import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .helpers import HELPER_MARK, is_helper
from .hub_types import HubType, arithmetic_result, call_name, iter_functions
from .rewrite import call, clone
from .scoped_walk import ScopedWalker

__all__ = [
    "FUNCTIONS",
    "PREFIX",
    "PortableMathError",
    "helper_program_source",
    "lower_portable_math",
    "native_portable_math",
    "register_native_portable_math",
]

PREFIX = "cgl_rule_pm_"
# CrossGL function -> number of operands.
FUNCTIONS = {
    "sinpi": 1,
    "cospi": 1,
    "fdim": 2,
    "nextafter": 2,
    "ilogb": 1,
    "copysign": 2,
}
_MARK = "portable_math"
_LANE_NAMES = "xyzw"
_F16 = ("f", 16)
_F32 = ("f", 32)
_F64 = ("f", 64)
ANY = None

# Target -> {function: the (kind, bits) operand scalars it declares, or ANY}. CrossGL output writes the calls as they
# stand. Metal (MSL 3.2 section 6.5), CUDA and HIP (their math libraries: sinpif/sinpi, fdimf/fdim, ...) and OpenCL C
# (section 6.15.2) declare every function; Slang its fdim, nextafter and copysign; Mojo's std.math nextafter and
# copysign; Rust's f32::copysign. The GLSL and HLSL generators write binary32 copysign from the bits.
_ALL = {name: ANY for name in FUNCTIONS}
_NATIVE: dict[str, dict[str, frozenset | None]] = {
    "": dict(_ALL),
    "cgl": dict(_ALL),
    "crossgl": dict(_ALL),
    "metal": dict(_ALL),
    "cuda": dict(_ALL),
    "hip": dict(_ALL),
    "opencl": dict(_ALL),
    "slang": {"fdim": ANY, "nextafter": ANY, "copysign": ANY},
    "mojo": {"nextafter": ANY, "copysign": ANY},
    "rust": {"copysign": ANY},
    "opengl": {"copysign": frozenset({_F32})},
    "webgl": {"copysign": frozenset({_F32})},
    "directx": {"copysign": frozenset({_F32})},
}


# Target -> the native functions it declares for scalars only: a vector call becomes a helper that calls the native
# function on each component. The CUDA and HIP math libraries overload fdim, nextafter, ilogb and copysign on float
# and double only (their generators already write sinpi and cospi per component).
_SCALAR_ONLY: dict[str, set[str]] = {
    "cuda": {"fdim", "nextafter", "ilogb", "copysign"},
    "hip": {"fdim", "nextafter", "ilogb", "copysign"},
}
# Target -> the native pi functions whose zeros lose their sign, so the call goes through a helper that keeps the
# native value and gives a zero result its C sign. Apple's Metal compiler, even with -fno-fast-math, returns
# sinpi(-0) = +0, sinpi(1) = -0 and cospi(1.5) = -0 on an Apple M2 Max (conformance case x_portable_pi); the nonzero
# values are within 1 ulp.
_SIGNED_ZERO: dict[str, frozenset] = {"metal": frozenset({"sinpi", "cospi"})}


def register_native_portable_math(target, functions, scalars=ANY, vectors=True):
    """Declare that ``target``'s generator writes ``functions`` itself, for ``scalars`` ((kind, bits) pairs) or all.

    With ``vectors=False`` the target's functions take scalars only, and a vector call computes each component with
    the native function.
    """
    target = str(target or "").lower()
    entry = _NATIVE.setdefault(target, {})
    for name in functions:
        if name not in FUNCTIONS:
            # A registration naming another function is a defect of the registering backend, not a refusal.
            raise AssertionError(f"{name} is not a portable math function")
        entry[name] = ANY if scalars is ANY else frozenset(scalars)
        if vectors:
            _SCALAR_ONLY.get(target, set()).discard(name)
        else:
            _SCALAR_ONLY.setdefault(target, set()).add(name)


def native_portable_math(target) -> dict:
    """The functions ``target`` declares natively, each with the operand scalars it takes (``None``: every one)."""
    return dict(_NATIVE.get(str(target or "").lower(), {}))


class PortableMathError(TranslationDiagnostic):
    """A call of a portable math function that the target lacks and no CrossGL helper computes."""

    code = CODE_PREFIX + "portable-math-unsupported"


# The binary32 coefficients (see the module docstring): pi = PI_A + PI_B with 12 significant bits in PI_A; the odd
# polynomial sin(pi t) / t - pi = z * (S0 + z * (S1 + z * S2)) and the even cos(pi t) - 1 = z * (C0 + z * (C1 + z *
# (C2 + z * C3))) in z = t^2, with C0 = C0A + C0B and 8 significant bits in C0A. Each literal is the exact binary32
# value, written so that a binary64 parse rounds to the same binary32.
_PI_A = "3.1416015625"
_PI_B = "-8.90890987648163e-06"
_SINE = ("-5.167708873748779", "2.549805164337158", "-0.5894187092781067")
_COSINE = (
    "-4.934802055358887",
    "4.0587077140808105",
    "-1.3350505828857422",
    "0.23136422038078308",
)
_C0A = "-4.9375"
_C0B = "0.0026979446411132812"
_INT_MIN = "(-2147483647 - 1)"
_NAN = "asfloat(0x7fc00000u)"

# Scalar binary32 definitions, in dependency order. ``{p}`` is the reserved prefix.
_DEFINITIONS = {
    "sin_kernel": (
        f"""float {{p}}sin_kernel(float t) {{
    float high = asfloat(asuint(t) & 0xfffff000u);
    float low = t - high;
    float z = t * t;
    float tail = z * ({_SINE[0]} + z * ({_SINE[1]} + z * {_SINE[2]}));
    return high * {_PI_A} + (low * {_PI_A} + t * ({_PI_B} + tail));
}}"""
    ),
    "cos_kernel": (
        f"""float {{p}}cos_kernel(float t) {{
    float high = asfloat(asuint(t) & 0xffff0000u);
    float low = t - high;
    float z = t * t;
    float square = high * high;
    float tail = z * ({_COSINE[1]} + z * ({_COSINE[2]} + z * {_COSINE[3]}));
    float rest = square * {_C0B} + low * (t + high) * {_COSINE[0]} + z * tail;
    return 1.0 + (square * {_C0A} + rest);
}}"""
    ),
    "sinpi": (
        f"""float {{p}}sinpi(float x) {{
    uint bits = asuint(x);
    uint magnitude = bits & 0x7fffffffu;
    uint sign = bits & 0x80000000u;
    if (magnitude >= 0x7f800000u) {{ return {_NAN}; }}
    if (magnitude >= 0x4b000000u) {{ return asfloat(sign); }}
    float a = asfloat(magnitude);
    float twice = a * 2.0;
    float nearest = floor(twice);
    if (twice - nearest > 0.5) {{ nearest = nearest + 1.0; }}
    float t = a - nearest * 0.5;
    int quadrant = int(nearest) & 3;
    if ((quadrant & 1) == 0 && t == 0.0) {{ return asfloat(sign); }}
    float value = (quadrant & 1) == 0 ? {{p}}sin_kernel(t) : {{p}}cos_kernel(t);
    return asfloat(asuint((quadrant & 2) == 0 ? value : -value) ^ sign);
}}"""
    ),
    "cospi": (
        f"""float {{p}}cospi(float x) {{
    uint magnitude = asuint(x) & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {{ return {_NAN}; }}
    if (magnitude >= 0x4b800000u) {{ return 1.0; }}
    if (magnitude >= 0x4b000000u) {{ return (magnitude & 1u) == 0u ? 1.0 : -1.0; }}
    float a = asfloat(magnitude);
    float twice = a * 2.0;
    float nearest = floor(twice);
    if (twice - nearest > 0.5) {{ nearest = nearest + 1.0; }}
    float t = a - nearest * 0.5;
    int quadrant = int(nearest) & 3;
    if ((quadrant & 1) != 0 && t == 0.0) {{ return 0.0; }}
    float value = (quadrant & 1) == 0 ? {{p}}cos_kernel(t) : {{p}}sin_kernel(t);
    return ((quadrant + 1) & 2) == 0 ? value : -value;
}}"""
    ),
    "fdim": (
        """float {p}fdim(float x, float y) {
    if ((asuint(x) & 0x7fffffffu) > 0x7f800000u) { return x; }
    if ((asuint(y) & 0x7fffffffu) > 0x7f800000u) { return y; }
    return x > y ? x - y : 0.0;
}"""
    ),
    "nextafter": (
        """float {p}nextafter(float x, float y) {
    uint xbits = asuint(x);
    uint ybits = asuint(y);
    if ((xbits & 0x7fffffffu) > 0x7f800000u) { return x; }
    if ((ybits & 0x7fffffffu) > 0x7f800000u) { return y; }
    int from = (xbits >> 31u) != 0u ? -int(xbits & 0x7fffffffu) : int(xbits);
    int to = (ybits >> 31u) != 0u ? -int(ybits & 0x7fffffffu) : int(ybits);
    if (from == to) { return y; }
    int next = from < to ? from + 1 : from - 1;
    if (next == 0) { return asfloat(xbits & 0x80000000u); }
    return asfloat(next < 0 ? (uint(-next) | 0x80000000u) : uint(next));
}"""
    ),
    "ilogb": (
        f"""int {{p}}ilogb(float x) {{
    uint magnitude = asuint(x) & 0x7fffffffu;
    if (magnitude == 0u || magnitude > 0x7f800000u) {{ return {_INT_MIN}; }}
    if (magnitude == 0x7f800000u) {{ return 2147483647; }}
    if (magnitude < 0x00800000u) {{ return int(asuint(float(magnitude)) >> 23u) - 276; }}
    return int(magnitude >> 23u) - 127;
}}"""
    ),
    "copysign": (
        """float {p}copysign(float x, float y) {
    return asfloat((asuint(x) & 0x7fffffffu) | (asuint(y) & 0x80000000u));
}"""
    ),
    "fdim_double": (
        """double {p}fdim_double(double x, double y) {
    if (isnan(x) || isnan(y)) { return x + y; }
    return x > y ? x - y : double(0.0);
}"""
    ),
}
# Parameters and locals take the reserved prefix too, so that no helper declaration shadows a program global.
_LOCALS = (
    "t x y a r high low z tail square rest bits magnitude sign twice nearest quadrant value xbits ybits from to next"
).split()
_LOCAL_PATTERN = re.compile(r"\b(" + "|".join(_LOCALS) + r")\b")
# Functions whose binary16 operands compute in binary32 (nextafter would need binary16 steps).
_WIDENED = frozenset({"sinpi", "cospi", "fdim", "ilogb", "copysign"})


def _source(text):
    return _LOCAL_PATTERN.sub(
        lambda match: PREFIX + "v_" + match.group(1), text
    ).replace("{p}", PREFIX)


def _scalar_spelling(kind_bits):
    return {_F16: "half", _F32: "float", _F64: "double", ("i", 32): "int"}[kind_bits]


def _type_spelling(value):
    key = (value.kind, value.bits)
    if value.lanes == 1:
        return _scalar_spelling(key)
    prefix = {_F16: "f16vec", _F32: "vec", _F64: "dvec", ("i", 32): "ivec"}[key]
    return f"{prefix}{value.lanes}"


def _result_scalar(name, operand):
    return ("i", 32) if name == "ilogb" else operand


def helper_name(name, kind_bits, lanes=1):
    """The reserved name of the helper computing ``name`` on ``lanes`` components of the ``kind_bits`` scalar."""
    suffix = {_F16: "_half", _F32: "", _F64: "_double"}[kind_bits]
    return f"{PREFIX}{name}{suffix}" + ("" if lanes == 1 else f"_v{lanes}")


def _widened_source(name):
    arity = FUNCTIONS[name]
    result = _scalar_spelling(_result_scalar(name, _F16))
    parameters = ", ".join(f"half x{index}" for index in range(arity))
    arguments = ", ".join(f"float(x{index})" for index in range(arity))
    value = f"{helper_name(name, _F32)}({arguments})"
    if result == "half":
        value = f"half({value})"
    return (
        f"{result} {helper_name(name, _F16)}({parameters}) {{\n    return {value};\n}}"
    )


def native_lift_name(name, kind_bits, lanes):
    """The reserved name of the helper computing a vector ``name`` with the target's scalar function."""
    return f"{PREFIX}native_{name}_{kind_bits[0]}{kind_bits[1]}_v{lanes}"


def signed_zero_name(name, kind_bits, lanes=1):
    """The reserved name of the helper that calls the target's ``name`` and gives its zeros their C sign."""
    suffix = "" if lanes == 1 else f"_v{lanes}"
    return f"{PREFIX}signed_{name}_{kind_bits[0]}{kind_bits[1]}{suffix}"


def _signed_zero_source(name, kind_bits):
    """``sinpi``'s zero takes the sign of the operand (sinPi(+n) = +0, sinPi(-n) = -0), ``cospi``'s is ``+0``."""
    spelling = _scalar_spelling(kind_bits)
    zero = f"{spelling}(0.0)"
    fixed = f"copysign({zero}, x)" if name == "sinpi" else zero
    return _source(
        f"{spelling} {signed_zero_name(name, kind_bits)}({spelling} x) {{\n"
        f"    {spelling} r = {name}(x);\n"
        f"    return r == {zero} ? {fixed} : r;\n}}"
    )


def _lift_source(name, kind_bits, lanes, scalar=None, lift=None):
    arity = FUNCTIONS[name]
    operand = HubType(*kind_bits, lanes)
    result = _type_spelling(HubType(*_result_scalar(name, kind_bits), lanes))
    parameters = ", ".join(
        f"{_type_spelling(operand)} x{index}" for index in range(arity)
    )
    scalar = scalar or helper_name(name, kind_bits)
    per_lane = ", ".join(
        f"{scalar}({', '.join(f'x{index}.{lane}' for index in range(arity))})"
        for lane in _LANE_NAMES[:lanes]
    )
    lift = lift or helper_name(name, kind_bits, lanes)
    return f"{result} {lift}({parameters}) {{\n    return {result}({per_lane});\n}}"


def _dependencies(name):
    """The binary32 definitions ``name``'s binary32 helper calls, in dependency order, ending with its own."""
    if name in ("sinpi", "cospi"):
        return ["sin_kernel", "cos_kernel", name]
    return [name]


def _keys(name, kind_bits, lanes):
    """The definitions a call of ``name`` on ``lanes`` components of ``kind_bits`` needs, in dependency order."""
    if kind_bits == _F64:
        keys = [f"{name}_double"]
    else:
        keys = list(_dependencies(name))
        if kind_bits == _F16:
            keys.append(("widen", name))
    if lanes > 1:
        keys.append(("lift", name, kind_bits, lanes))
    return keys


def _key_source(key):
    if isinstance(key, str):
        return _source(_DEFINITIONS[key])
    if key[0] == "widen":
        return _widened_source(key[1])
    if key[0] == "native":
        return _lift_source(*key[1:], scalar=key[1], lift=native_lift_name(*key[1:]))
    if key[0] == "signed":
        if len(key) == 3:
            return _signed_zero_source(*key[1:])
        return _lift_source(
            *key[1:], scalar=signed_zero_name(*key[1:3]), lift=signed_zero_name(*key[1:])
        )
    return _lift_source(*key[1:])


def _key_name(key):
    if isinstance(key, str):
        return PREFIX + key
    if key[0] == "widen":
        return helper_name(key[1], _F16)
    if key[0] == "native":
        return native_lift_name(key[1], key[2], key[3])
    if key[0] == "signed":
        return signed_zero_name(*key[1:])
    return helper_name(key[1], key[2], key[3])


def helper_program_source(names) -> str:
    """A CrossGL program defining the binary32 helpers of ``names`` and every helper they call."""
    keys = []
    for name in names:
        for key in _dependencies(name):
            if key not in keys:
                keys.append(key)
    return (
        "shader __cgl_rule_helpers {\n"
        + "\n\n".join(_key_source(key) for key in keys)
        + "\n}\n"
    )


@functools.lru_cache(maxsize=None)
def _parsed(key):
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader __cgl_rule_helpers {\n" + _key_source(key) + "\n}\n"
    )
    (function,) = program.functions
    return function


class _Calls(ScopedWalker):
    """Rewrites the calls in place and records the helpers they need."""

    def __init__(self, program, target, native, user_names):
        super().__init__(program)
        self.target = target
        self.native = native
        self.scalar_only = _SCALAR_ONLY.get(target, set())
        self.user_names = user_names
        self.needed = []
        self.count = 0

    def error(self, reason):
        return PortableMathError(reason, target=self.target)

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        name = call_name(node)
        if name not in FUNCTIONS or name in self.user_names:
            return
        arguments = list(node.arguments or [])
        if len(arguments) != FUNCTIONS[name]:
            raise self.error(
                f"{name} takes {FUNCTIONS[name]} operand{'s' if FUNCTIONS[name] > 1 else ''}, "
                f"not {len(arguments)}"
            )
        types = [self.scope.infer(argument) for argument in arguments]
        common = types[0]
        for value in types[1:]:
            common = arithmetic_result(common, value)
        supported = self.native.get(name, ())
        if common is None or any(value is None for value in types):
            if (
                supported is ANY
                and name not in self.scalar_only
                and name not in _SIGNED_ZERO.get(self.target, ())
            ):
                # The target's own function takes every operand type, so the call stands as written.
                return
            raise self.error(f"the operand types of {name} cannot be determined")
        scalar = (common.kind, common.bits)
        if name in self.native and (supported is ANY or scalar in supported):
            if name in _SIGNED_ZERO.get(self.target, ()) and scalar in (_F16, _F32):
                keys = [("signed", name, scalar)]
                if common.lanes > 1:
                    keys.append(("signed", name, scalar, common.lanes))
                self.rewrite(node, name, arguments, types, common, keys)
            elif common.lanes > 1 and name in self.scalar_only:
                self.rewrite(
                    node,
                    name,
                    arguments,
                    types,
                    common,
                    [("native", name, scalar, common.lanes)],
                )
            return
        if common.kind != "f" or any(
            value.kind not in ("f", "i", "u") for value in types
        ):
            raise self.error(
                f"{name} takes floating-point operands; this call has a {common.kind}{common.bits} one"
            )
        if common.lanes not in (1, 2, 3, 4):
            raise self.error(f"{name} on {common.lanes} components has no helper")
        if (
            scalar not in (_F16, _F32, _F64)
            or (scalar == _F16 and name not in _WIDENED)
            or (scalar == _F64 and f"{name}_double" not in _DEFINITIONS)
        ):
            if name in self.native:
                # The target's generator writes this function for some operand types and refuses the others with
                # its own typed diagnostic (the GLSL and HLSL binary32-only copysign).
                return
            raise self.error(
                f"{name} of {common.bits}-bit floating-point operands has no {self.target} function and no "
                "CrossGL helper"
            )
        self.rewrite(
            node, name, arguments, types, common, _keys(name, scalar, common.lanes)
        )

    def rewrite(self, node, name, arguments, types, common, keys):
        """Make ``node`` a call of the helper ``keys`` end with, its operands converted to ``common``."""
        for key in keys:
            if key not in self.needed:
                self.needed.append(key)
        node.function = node.name = A.IdentifierNode(_key_name(keys[-1]))
        node.arguments = node.args = [
            argument if value == common else call(_type_spelling(common), [argument])
            for argument, value in zip(arguments, types)
        ]
        node.annotations["cgl_rule_type"] = HubType(
            *_result_scalar(name, (common.kind, common.bits)), common.lanes
        )
        self.count += 1


def _order(key):
    """Definitions first, in table order, then the binary16 wrappers, then the vector lifts."""
    if isinstance(key, str):
        return (0, list(_DEFINITIONS).index(key), "", 0)
    if key[0] == "widen":
        return (1, list(FUNCTIONS).index(key[1]), "", 0)
    rank = {"lift": 2, "native": 3, "signed": 3}[key[0]]
    lanes = key[3] if len(key) > 3 else 1
    return (rank, list(FUNCTIONS).index(key[1]), f"{key[2][0]}{key[2][1]:02d}", lanes)


def lower_portable_math(program, target) -> int:
    """Rewrite calls of the portable math functions ``target`` lacks into CrossGL helpers; returns their count."""
    target = str(target or "").lower()
    native = _NATIVE.get(target, {})
    if (
        all(native.get(name, ()) is ANY for name in FUNCTIONS)
        and not _SCALAR_ONLY.get(target)
        and not _SIGNED_ZERO.get(target)
    ):
        return 0
    functions = list(iter_functions(program))
    existing = {getattr(function, "name", None): function for function in functions}
    for function in functions:
        name = getattr(function, "name", None) or ""
        if name.startswith(PREFIX) and not is_helper(function):
            raise PortableMathError(
                f"the program defines {name}, a name reserved for the portable math helpers",
                target=target,
            )
    user_names = {name for name in existing if name in FUNCTIONS}
    walker = _Calls(program, target, native, user_names)
    walker.run_program()
    if not walker.count:
        return 0
    program_functions = getattr(program, "functions", None)
    if not isinstance(program_functions, list):
        raise PortableMathError(
            f"the program has no function list to receive the portable math helpers for {target}",
            target=target,
        )
    added = []
    for key in sorted(walker.needed, key=_order):
        name = _key_name(key)
        if name in existing:
            continue
        function = clone(_parsed(key))
        function.annotations[HELPER_MARK] = _MARK
        added.append(function)
        existing[name] = function
    program_functions[0:0] = added
    return walker.count
