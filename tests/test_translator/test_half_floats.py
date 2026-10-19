"""``half`` keeps binary16 semantics on targets that compute it in binary32 (defect S7).

The helpers are checked by interpreting their CrossGL definitions with numpy binary32 arithmetic, against exact
rational references; division and square root are also checked when the binary32 operation is off by up to 2 ULP,
as Vulkan and WGSL allow. The translations are checked with spirv-val and naga.
"""

import math
import re
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics import (
    HalfFloatUnsupportedError,
)
from crosstl.translator.semantics import half_helpers as H
from crosstl.translator.semantics import legalize_half_floats
from crosstl.translator.semantics.half import (
    HALF_TARGETS,
    half_from_name,
    round_to_half,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from tests.conformance import half_cases

TARGETS = sorted(HALF_TARGETS)
MASK = 0xFFFFFFFF


# A CrossGL-subset interpreter for the helper bodies ------------------------------------------------------------
class _Return(Exception):
    def __init__(self, value):
        self.value = value


def _f32(value):
    return np.float32(value)


def _bits(value):
    return int(np.array([value], np.float32).view(np.uint32)[0])


def _from_bits(bits):
    return np.array([bits & MASK], np.uint32).view(np.float32)[0]


_FLOAT_OPS = {
    "+": lambda a, b: np.float32(a + b),
    "-": lambda a, b: np.float32(a - b),
    "*": lambda a, b: np.float32(a * b),
    "/": lambda a, b: np.float32(a / b),
}
_UINT_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
}


class HelperInterpreter:
    """Runs the helpers of :mod:`half_helpers` with binary32 arithmetic; ``skew`` perturbs ``/`` and ``sqrt``."""

    def __init__(self, skew=0):
        self.skew = skew
        self.calls = 0
        self.functions = {
            base: H.helper_function(base, 1) for base in H.SCALAR_HELPERS
        }

    def perturb(self, value):
        if not np.isfinite(value) or value == 0:
            return value
        self.calls += 1
        steps = (self.calls * 7919) % (2 * self.skew + 1) - self.skew if self.skew else 0
        direction = np.float32(np.inf if steps > 0 else -np.inf)
        for _ in range(abs(steps)):
            value = np.nextafter(value, direction, dtype=np.float32)
        return value

    def call(self, name, *arguments):
        function = self.functions[name]
        frame = {}
        for parameter, argument in zip(function.parameters, arguments):
            kind = str(parameter.param_type.name)
            frame[parameter.name] = (
                _f32(argument) if kind == "float" else int(argument) & MASK
            )
        try:
            self.block(function.body, frame)
        except _Return as result:
            return result.value
        raise AssertionError(f"{name} did not return")

    def block(self, body, frame):
        for statement in body.statements if isinstance(body, A.BlockNode) else body:
            self.statement(statement, frame)

    def statement(self, node, frame):
        if isinstance(node, A.ExpressionStatementNode):
            node = node.expression
        if isinstance(node, A.VariableNode):
            frame[node.name] = self.expr(node.initial_value, frame)
        elif isinstance(node, A.AssignmentNode):
            assert node.operator == "=" and isinstance(node.target, A.IdentifierNode)
            frame[node.target.name] = self.expr(node.value, frame)
        elif isinstance(node, A.ReturnNode):
            raise _Return(self.expr(node.value, frame))
        elif isinstance(node, A.IfNode):
            if self.expr(node.condition, frame):
                self.block(node.then_branch, frame)
                return
            for condition, body in zip(node.else_if_conditions, node.else_if_bodies):
                if self.expr(condition, frame):
                    self.block(body, frame)
                    return
            if node.else_branch is not None:
                branch = node.else_branch
                if isinstance(branch, A.IfNode):
                    self.statement(branch, frame)
                else:
                    self.block(branch, frame)
        elif isinstance(node, A.BlockNode):
            self.block(node, frame)
        else:
            raise AssertionError(f"unexpected statement {type(node).__name__}")

    def expr(self, node, frame):
        if isinstance(node, A.LiteralNode):
            kind = str(getattr(node.literal_type, "name", ""))
            if isinstance(node.value, bool):
                return node.value
            if kind == "float" or isinstance(node.value, float):
                return _f32(node.value)
            return int(node.value) & MASK
        if isinstance(node, A.IdentifierNode):
            return frame[node.name]
        if isinstance(node, A.TernaryOpNode):
            chosen = node.true_expr if self.expr(node.condition, frame) else node.false_expr
            return self.expr(chosen, frame)
        if isinstance(node, A.UnaryOpNode):
            value = self.expr(node.operand, frame)
            if node.op == "!":
                return not value
            assert node.op == "-" and isinstance(value, np.float32)
            return -value
        if isinstance(node, A.BinaryOpNode):
            return self.binary(node, frame)
        if isinstance(node, A.FunctionCallNode):
            return self.function_call(node, frame)
        raise AssertionError(f"unexpected expression {type(node).__name__}")

    def binary(self, node, frame):
        op = node.op
        if op == "&&":
            return bool(self.expr(node.left, frame)) and bool(self.expr(node.right, frame))
        if op == "||":
            return bool(self.expr(node.left, frame)) or bool(self.expr(node.right, frame))
        left, right = self.expr(node.left, frame), self.expr(node.right, frame)
        comparisons = {
            "<": lambda: left < right,
            "<=": lambda: left <= right,
            ">": lambda: left > right,
            ">=": lambda: left >= right,
            "==": lambda: left == right,
            "!=": lambda: left != right,
        }
        if op in comparisons:
            assert type(left) is type(right), (op, left, right)
            return bool(comparisons[op]())
        if isinstance(left, np.float32):
            assert isinstance(right, np.float32) and op in "+-*/", (op, left, right)
            with np.errstate(all="ignore"):
                result = _FLOAT_OPS[op](left, right)
            return self.perturb(result) if op == "/" else result
        assert isinstance(left, int) and isinstance(right, int), (op, left, right)
        if op in ("<<", ">>"):
            assert right < 32, (op, right)
        if op == "-":
            assert left >= right, "uint underflow in a helper"
        return _UINT_OPS[op](left, right) & MASK

    def function_call(self, node, frame):
        name = node.function.name
        arguments = [self.expr(argument, frame) for argument in node.arguments]
        if name == "asuint":
            return _bits(arguments[0])
        if name == "asfloat":
            return _from_bits(arguments[0])
        if name == "float":
            return np.float32(arguments[0])
        if name == "uint":
            value = arguments[0]
            assert isinstance(value, np.float32) and 0 <= value < 2**32
            return int(value)
        if name == "sqrt":
            with np.errstate(all="ignore"):
                return self.perturb(np.sqrt(arguments[0]))
        return self.call(name, *arguments)


# Exact references ----------------------------------------------------------------------------------------------
HALF_MAX_EDGE = Fraction(65520)


def round_half(value: Fraction) -> float:
    """``value`` rounded to binary16, ties to even, as a Python float."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    if magnitude >= HALF_MAX_EDGE:
        return sign * float("inf")
    exponent = max(magnitude.numerator.bit_length() - magnitude.denominator.bit_length(), -14)
    while Fraction(2) ** exponent > magnitude and exponent > -14:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    ulp = Fraction(2) ** (exponent - 10)
    return sign * float(round(magnitude / ulp) * ulp)


def halves(values):
    return [float(v) for v in np.asarray(values, np.float32).astype(np.float16)]


def finite_halves():
    every = np.arange(0, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    return np.concatenate([every, -every[1:]])


def assert_same(actual, expected, what):
    actual_bits = _bits(np.float32(actual))
    expected_bits = _bits(np.float32(expected))
    assert actual_bits == expected_bits, f"{what}: got {actual!r}, expected {expected!r}"


# Unit tests ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, lanes",
    [("half", 1), ("float16", 1), ("float16_t", 1), ("f16", 1), ("f16vec3", 3), ("half4", 4), ("vec2<f16>", 2)],
)
def test_half_spellings(name, lanes):
    assert half_from_name(name).lanes == lanes


@pytest.mark.parametrize("name", ["float", "min16float", "double", "half4x4", "bfloat16_t"])
def test_other_types_are_not_half(name):
    assert half_from_name(name) is None


def test_literal_rounding():
    assert round_to_half(0.1) == float(np.float16(0.1))
    assert round_to_half(65519.0) == 65504.0
    assert round_to_half(65520.0) is None
    assert round_to_half(2.0**-25) == 0.0


def test_round_helper_matches_numpy_conversion():
    interpreter = HelperInterpreter()
    rng = np.random.default_rng(16)
    patterns = np.concatenate(
        [
            rng.integers(0, 2**32, 4000, dtype=np.uint64),
            # binary16 midpoints and their neighbours, around subnormals, the normal range and overflow
            np.array(
                [
                    (base + offset) & MASK
                    for base in (0x33000000, 0x33800000, 0x38800000, 0x387FF000, 0x3F801000, 0x477FF000, 0x477FE000)
                    for offset in (-1, 0, 1, 0x1000, 0x2000)
                ]
                + [0x7F800000, 0xFF800000, 0x7FC00000, 0x80000000, 0, 1],
                dtype=np.uint64,
            ),
        ]
    ).astype(np.uint32)
    for value in patterns.view(np.float32):
        expected = np.float32(np.float16(value))
        actual = interpreter.call("cgl_half_round", value)
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert_same(actual, expected, f"round({value!r})")


def _midpoint_pairs(rng, count):
    """Pairs whose exact quotient lies next to a binary16 midpoint (or on one), plus random pairs."""
    values = halves(rng.uniform(-4, 4, count) * 2.0 ** rng.integers(-12, 12, count))
    pairs = list(zip(values, halves(rng.uniform(-4, 4, count) * 2.0 ** rng.integers(-12, 12, count))))
    for numerator in (1, 3, 5, 7, 1025, 2047, 683, 1.5, 0.75):
        for denominator in (3, 7, 11, 1023, 2045, 0.6669921875, 3.0517578125e-05):
            pairs.append((float(np.float16(numerator)), float(np.float16(denominator))))
    return [(a, b) for a, b in pairs if b != 0]


@pytest.mark.parametrize("skew", [0, 2])
def test_division_helper_is_correctly_rounded(skew):
    interpreter = HelperInterpreter(skew)
    rng = np.random.default_rng(1 + skew)
    for a, b in _midpoint_pairs(rng, 1500) + [(65504.0, 2.0**-24), (2.0**-24, 65504.0), (-0.0, 3.0)]:
        expected = round_half(Fraction(a) / Fraction(b))
        if a == 0:
            expected = float(np.float16(a) / np.float16(b))
        assert_same(interpreter.call("cgl_half_div", a, b), expected, f"{a!r} / {b!r}")


@pytest.mark.parametrize("skew", [0, 2])
def test_square_root_helper_is_correctly_rounded(skew):
    interpreter = HelperInterpreter(skew)
    values = finite_halves()
    values = values[values >= 0][:: 3 if skew else 1]
    for value in values:
        # A binary16 square root is never within 2^-23 (relative) of a binary16 midpoint, so rounding the binary64
        # root exactly gives the correctly rounded result.
        expected = round_half(Fraction(math.sqrt(float(value))))
        assert_same(interpreter.call("cgl_half_sqrt", value), expected, f"sqrt({value!r})")
    assert np.isnan(interpreter.call("cgl_half_sqrt", np.float32(-1.0)))
    assert_same(interpreter.call("cgl_half_sqrt", np.float32(np.inf)), np.inf, "sqrt(inf)")


def _fma_triples(rng, count):
    def sample():
        return halves(rng.uniform(-2, 2, count) * 2.0 ** rng.integers(-14, 8, count))

    triples = list(zip(sample(), sample(), sample()))
    # products on a binary16 midpoint plus a tiny addend, and cancellations
    triples += [
        (1.5, 0.6669921875, 2.0**-24),
        (1.5, 0.6669921875, -(2.0**-24)),
        (1.5, 0.6669921875, 0.0),
        (1.5, 0.6669921875, -1.00048828125),
        (2047.0, 2047.0, -4190208.0 / 2),
        (255.875, 256.0, -65504.0),
        (256.0, 256.0, -16.0),
        (-0.0, 3.0, 0.0),
        (2.0**-24, 2.0**-24, 2.0**-24),
        (2.0**-24, 0.5, 0.0),
        (2.0**-24, 0.75, 0.0),
        (65504.0, 1.0, 16.0),
        (65504.0, 1.0, 15.0),
    ]
    triples += [
        (a, b, float(np.float16(-a * b)))
        for a, b in zip(sample()[:300], sample()[:300])
    ]
    # sums next to a binary16 midpoint: m = (2j + 1) * 2^(e - 11), p = a * b close to m - c
    for _ in range(1500):
        exponent = int(rng.integers(-13, 15))
        midpoint = (2 * int(rng.integers(1024, 2048)) + 1) * 2.0 ** (exponent - 11)
        c = float(np.float16(rng.uniform(-1, 1) * 2.0 ** int(rng.integers(-24, exponent))))
        a = float(np.float16(rng.uniform(1, 2) * 2.0 ** int(rng.integers(-6, 6))))
        b = float(np.float16((midpoint - c) / a))
        triples.append((a, b, c))
    return [
        (a, b, float(np.float16(c)))
        for a, b, c in triples
        if np.isfinite(np.float16(c)) and np.isfinite(np.float16(a) * np.float32(b))
    ]


def test_fma_helper_is_correctly_rounded():
    interpreter = HelperInterpreter()
    rng = np.random.default_rng(3)
    for a, b, c in _fma_triples(rng, 3000):
        exact = Fraction(a) * Fraction(b) + Fraction(c)
        expected = round_half(exact)
        if exact == 0:
            expected = float(np.float32(np.float32(a) * np.float32(b) + np.float32(c)))
        assert_same(interpreter.call("cgl_half_fma", a, b, c), expected, f"fma({a!r}, {b!r}, {c!r})")


@pytest.mark.parametrize(
    "helper, inputs, reference, arity",
    [
        ("cgl_half_div", half_cases.div_inputs, half_cases.div_reference, 2),
        ("cgl_half_sqrt", half_cases.sqrt_inputs, half_cases.sqrt_reference, 1),
        ("cgl_half_fma", half_cases.fma_inputs, half_cases.fma_reference, 3),
        ("cgl_half_round", half_cases.round_inputs, half_cases.round_reference, 1),
    ],
)
def test_helpers_pass_the_conformance_inputs(helper, inputs, reference, arity):
    """The GPU conformance inputs, interpreted: the oracle there must hold for the helper itself."""
    values = inputs()
    interpreter = HelperInterpreter(skew=2 if helper in ("cgl_half_div", "cgl_half_sqrt") else 0)
    for operands, expected in zip(values.reshape(-1, arity), reference(values)):
        actual = interpreter.call(helper, *operands)
        assert_same(actual, expected, f"{helper}{tuple(float(v) for v in operands)}")


def test_bit_helpers_round_trip_every_encoding():
    interpreter = HelperInterpreter()
    for encoding in range(0, 0x10000, 7):
        value = interpreter.call("cgl_half_from_bits", encoding)
        expected = np.array([encoding], np.uint16).view(np.float16)[0]
        if np.isnan(expected):
            assert np.isnan(value)
        else:
            assert_same(value, np.float32(expected), f"from_bits({encoding:#x})")
        assert interpreter.call("cgl_half_to_bits", value) == encoding


# Translation tests ---------------------------------------------------------------------------------------------
METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void k(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {{
{body}
}}
"""
GLSL_KERNEL = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Inp {{ float a[]; }};
layout(std430, binding = 1) buffer Outp {{ float o[]; }};
{helpers}
void main() {{
    uint i = gl_GlobalInvocationID.x;
{body}
}}
"""


def translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    # WGSL from Metal kernels has an unrelated interface defect (scalar invocation-id builtins), so only WGSL from
    # GLSL is checked natively.
    if backend == "vulkan" or suffix == ".comp":
        validate(tmp_path, output, backend)
    return output


def validate(tmp_path, output, backend):
    if backend == "vulkan":
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            return
        (tmp_path / "out.spvasm").write_text(output)
        assembled = subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert assembled.returncode == 0, assembled.stderr
        checked = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", "out.spv"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert checked.returncode == 0, checked.stderr
    elif shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr


def metal(tmp_path, body, backend, helpers=""):
    return translate(
        tmp_path, METAL_KERNEL.format(body=body, helpers=helpers), backend, ".metal"
    )


def glsl(tmp_path, body, backend, helpers=""):
    return translate(
        tmp_path, GLSL_KERNEL.format(body=body, helpers=helpers), backend, ".comp"
    )


def test_other_targets_are_not_legalized():
    register_default_sources()
    program = SOURCE_REGISTRY.get("cgl").parse(
        "shader s { float16 f(float16 x) { return x * x; } }"
    )
    assert legalize_half_floats(program, "cgl", "metal") is False
    assert legalize_half_floats(program, "cgl", "opengl") is False


def helpers_in(tmp_path, build, body, helpers=""):
    """The binary16 helpers in the WGSL translation; the SPIR-V one validates and defines as many functions."""
    wgsl = build(tmp_path, body, "wgsl", helpers)
    vulkan = build(tmp_path, body, "vulkan", helpers)
    functions = re.findall(r"^fn (\w+)\(", wgsl, flags=re.M)
    assert vulkan.count(" OpFunction ") == len(functions)
    return wgsl, [name for name in functions if name.startswith("cgl_half")]


def test_half_multiplication_rounds_each_product(tmp_path):
    body = "    half h = half(a[i]);\n    h = h * half(3.0h);\n    o[i] = float(h);"
    wgsl, names = helpers_in(tmp_path, metal, body)
    assert names == ["cgl_half_round"]
    assert "h = cgl_half_round((h * 3.0));" in wgsl


def test_glsl_float16_arithmetic_uses_exact_helpers(tmp_path):
    body = (
        "    float16_t x = float16_t(a[i]);\n"
        "    float16_t y = float16_t(a[i + 1u]);\n"
        "    f16vec2 v = f16vec2(x, y);\n"
        "    float16_t q = x / y + sqrt(x);\n"
        "    v = fma(v, v, f16vec2(q));\n"
        "    v /= y;\n"
        "    q += float16_t(1);\n"
        "    o[i] = float(q) + float(v.x) + float(v.y);"
    )
    wgsl, names = helpers_in(tmp_path, glsl, body)
    for helper in ("cgl_half_div", "cgl_half_sqrt", "cgl_half_fma2", "cgl_half_div2"):
        assert helper in names
    assert "q = cgl_half_round((q + 1.0));" in wgsl


def test_half_float_mixed_arithmetic_is_binary32(tmp_path):
    body = "    float16_t h = float16_t(a[i]);\n    float f = a[i] * 2.0;\n    o[i] = h * f / f;"
    wgsl, names = helpers_in(tmp_path, glsl, body)
    assert names == ["cgl_half_round"]
    assert "((h * f) / f)" in wgsl


def test_half_parameters_returns_and_workgroup_arrays(tmp_path):
    helpers = (
        "shared float16_t tile[64];\n"
        "float16_t scale(float16_t x, float k) { return x * float16_t(k); }\n"
    )
    body = (
        "    tile[gl_LocalInvocationID.x] = float16_t(a[i]);\n"
        "    barrier();\n"
        "    o[i] = float(scale(tile[gl_LocalInvocationID.x], 0.1));"
    )
    wgsl, _ = helpers_in(tmp_path, glsl, body, helpers)
    assert "array<f32, 64>" in wgsl
    assert "fn scale(x: f32, k: f32) -> f32" in wgsl


def test_half_bit_casts_use_the_binary16_encoding(tmp_path):
    body = (
        "    half h = half(a[i]);\n"
        "    ushort b = as_type<ushort>(h);\n"
        "    half g = as_type<half>(ushort(b + 1));\n"
        "    o[i] = float(g);"
    )
    _, names = helpers_in(tmp_path, metal, body)
    assert "cgl_half_to_bits" in names and "cgl_half_from_bits" in names


@pytest.mark.parametrize(
    "body, reason",
    [
        ("    half4x4 m = half4x4(1.0h);\n    o[i] = float(m[0][0]);", "matrix"),
        ("    half h = half(a[i]);\n    o[i] = float(h++);", "step-value"),
        ("    half h = half(a[i]);\n    o[i] = float(fmod(h, half(2.0h)));", "remainder"),
        ("    half2 h = half2(a[i]);\n    o[i] = float(as_type<uint>(h));", "bitcast"),
    ],
)
@pytest.mark.parametrize("target", TARGETS)
def test_unsupported_half_constructs_are_refused(tmp_path, target, body, reason):
    with pytest.raises(HalfFloatUnsupportedError) as raised:
        metal(tmp_path, body, target)
    assert raised.value.reason_code == reason
    assert raised.value.code == f"project.translate.{target}-half-float-unsupported"


def test_reserved_helper_names_are_refused(tmp_path):
    helpers = "float cgl_half_round(float x) { return x; }\n"
    with pytest.raises(HalfFloatUnsupportedError) as raised:
        glsl(tmp_path, "    float16_t h = float16_t(a[i]);\n    o[i] = float(h);", "vulkan", helpers)
    assert raised.value.reason_code == "reserved-name"


def test_helpers_are_emitted_once_in_dependency_order(tmp_path):
    body = (
        "    float16_t x = float16_t(a[i]);\n"
        "    o[i] = float(x / float16_t(3.0) + sqrt(x) + x / float16_t(7.0));"
    )
    output = glsl(tmp_path, body, "wgsl")
    names = [line.split("(")[0][3:] for line in output.splitlines() if line.startswith("fn cgl_half")]
    assert names == [
        "cgl_half_round",
        "cgl_half_index",
        "cgl_half_value",
        "cgl_half_midpoint",
        "cgl_half_div",
        "cgl_half_sqrt",
    ]


def test_half_literals_keep_binary16_arithmetic(tmp_path):
    """``h * 0.5h + 1.0h`` rounds after each operation; ``h * 0.5`` is a float operation in MSL and GLSL."""
    metal_body = "    half h = half(a[i]);\n    h = h * 0.5h + 1.0h;\n    o[i] = float(h) + float(h * 0.5);"
    wgsl, _ = helpers_in(tmp_path, metal, metal_body)
    assert "h = cgl_half_round((cgl_half_round((h * 0.5)) + 1.0));" in wgsl
    assert "(f32(h) + f32((h * 0.5)))" in wgsl
    glsl_body = "    float16_t h = float16_t(a[i]);\n    h = h * 0.5hf + 1.0hf;\n    o[i] = float(h);"
    wgsl, _ = helpers_in(tmp_path, glsl, glsl_body)
    assert "h = cgl_half_round((cgl_half_round((h * 0.5)) + 1.0));" in wgsl


def test_hlsl_float_literals_adopt_the_half_type(tmp_path):
    source = (
        "RWStructuredBuffer<float> o : register(u0);\n"
        "[numthreads(64, 1, 1)]\n"
        "void main(uint3 id : SV_DispatchThreadID) {\n"
        "    half h = (half)o[id.x];\n"
        "    h = h * 0.1 + 1.0;\n"
        "    o[id.x] = (float)h;\n"
        "}\n"
    )
    wgsl = translate(tmp_path, source, "wgsl", ".hlsl")
    assert "cgl_half_round((cgl_half_round((h * 0.0999755859375)) + 1.0))" in wgsl
