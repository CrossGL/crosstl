"""CrossGL ``sinpi``, ``cospi``, ``fdim``, ``nextafter``, ``ilogb`` and ``copysign`` on targets that lack them.

See :mod:`crosstl.translator.semantics.portable_math`. The helpers' CrossGL text is evaluated over binary32 operands
by :mod:`tests.conformance.portable_math_eval`; ``x_portable_pi``, ``x_portable_pairs`` and ``x_portable_ilogb``
execute the translated helpers on the conformance runtimes.
"""

import re
import shutil
import subprocess

import numpy as np
import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import portable_math as P
from tests.conformance import portable_math_eval as E

PROBE = """shader Probe {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iout;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = fin[i];
            float y = fin[i + 1u];
            vec3 a = vec3(x, y, x);
            vec3 b = vec3(y, x, y);
            BODY
        }
    }
}
"""

ALL_CALLS = """
            outp[i] = sinpi(x) + cospi(x) + fdim(x, y) + nextafter(x, y) + copysign(x, y);
            vec3 c = sinpi(a) + cospi(a) + fdim(a, b) + nextafter(a, b) + copysign(a, b);
            ivec3 e = ilogb(a);
            iout[i] = ilogb(x) + e.y;
            outp[i + 1u] = c.z;
"""

LOWERING_TARGETS = {
    "opengl": {"sinpi", "cospi", "fdim", "nextafter", "ilogb"},
    "directx": {"sinpi", "cospi", "fdim", "nextafter", "ilogb"},
    "vulkan": set(P.FUNCTIONS),
    "wgsl": set(P.FUNCTIONS),
    "rust": {"sinpi", "cospi", "fdim", "nextafter", "ilogb"},
    "mojo": {"sinpi", "cospi", "fdim", "ilogb"},
}


def _translate(tmp_path, body, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(PROBE.replace("BODY", body))
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", sorted(LOWERING_TARGETS))
def test_targets_without_the_functions_call_helpers(tmp_path, backend):
    code = _translate(tmp_path, ALL_CALLS, backend)
    if backend == "vulkan":
        # The SPIR-V module names no functions; spirv-val checks it below.
        return
    lowered = LOWERING_TARGETS[backend]
    for name in P.FUNCTIONS:
        assert (P.PREFIX + name in code) == (name in lowered), name
    assert f"{P.PREFIX}ilogb_v3" in code


def test_slang_lowers_ilogb_and_refuses_its_differently_defined_pi_functions(tmp_path):
    body = """
            outp[i] = fdim(x, y) + nextafter(x, y) + copysign(x, y);
            vec3 c = fdim(a, b) + nextafter(a, b) + copysign(a, b);
            ivec3 e = ilogb(a);
            iout[i] = ilogb(x) + e.y;
            outp[i + 1u] = c.z;
"""
    code = _translate(tmp_path, body, "slang")
    for name in P.FUNCTIONS:
        assert (P.PREFIX + name in code) == (name == "ilogb"), name
    # Slang computes its own sinpi and cospi as sin(pi * x) and cos(pi * x) outside Metal, so CrossGL's exact
    # functions keep their refusal rather than a helper or the native call.
    for name in ("sinpi", "cospi"):
        with pytest.raises(Exception, match="exact at integers and half-integers"):
            _translate(tmp_path, f"outp[i] = {name}(x);", "slang")


def test_metal_keeps_the_calls(tmp_path):
    code = _translate(tmp_path, ALL_CALLS, "metal")
    for name in P.FUNCTIONS:
        # Metal declares every function; no CrossGL implementation is added.
        assert re.search(rf"\b{name}\(", code), name
        assert P.helper_name(name, ("f", 32)) + "(" not in code, name
    # Metal's sinpi and cospi only get their zeros signed as C does (sinpi(-1) = -0, cospi(0.5) = +0).
    assert P.signed_zero_name("sinpi", ("f", 32)) in code
    assert P.signed_zero_name("cospi", ("f", 32)) in code
    assert "copysign(float(0.0), " in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_hip_call_their_scalar_functions_per_component(tmp_path, backend):
    code = _translate(tmp_path, ALL_CALLS, backend)
    for name in ("fdim", "nextafter", "ilogb", "copysign"):
        lift = P.native_lift_name(name, ("f", 32), 3)
        assert lift in code
        body = code[code.index(lift + "(float3 x0") :].split("}", 1)[0]
        assert f"{name}(x0.z" in body
    # Scalars and the pi functions, which the generators write per component, keep the library calls.
    assert f"{P.PREFIX}fdim(" not in code and f"{P.PREFIX}sinpi" not in code


def test_glsl_helpers_compile(tmp_path):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "probe.comp"
    shader.write_text(_translate(tmp_path, ALL_CALLS, "opengl"))
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", str(shader)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_helpers_validate(tmp_path):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "probe.spvasm"
    source.write_text(_translate(tmp_path, ALL_CALLS, "vulkan"))
    binary = tmp_path / "probe.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_wgsl_helpers_validate(tmp_path):
    if shutil.which("naga") is None:
        pytest.skip("naga is not installed")
    shader = tmp_path / "probe.wgsl"
    shader.write_text(_translate(tmp_path, ALL_CALLS, "wgsl"))
    result = subprocess.run(["naga", str(shader)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_half_operands_compute_in_float(tmp_path):
    body = "half h = half(x); f16vec2 v = f16vec2(h, h); outp[i] = float(sinpi(h) + fdim(v, v).y + copysign(h, h)); iout[i] = ilogb(h);"
    code = _translate(tmp_path, body, "directx")
    for name in ("sinpi", "fdim", "copysign", "ilogb"):
        assert f"{P.PREFIX}{name}_half" in code
    assert f"{P.PREFIX}fdim_half_v2" in code


def test_double_fdim_has_a_helper(tmp_path):
    code = _translate(
        tmp_path, "outp[i] = float(fdim(double(x), double(y)));", "directx"
    )
    assert f"{P.PREFIX}fdim_double" in code


def test_source_environment_constants_are_binary32_operands(tmp_path):
    # INFINITY is a binary32 constant, so nextafter(x, INFINITY) takes the float helper.
    code = _translate(tmp_path, "outp[i] = nextafter(x, INFINITY);", "opengl")
    assert re.search(rf"{P.PREFIX}nextafter\w*\(x, ", code)
    assert "INFINITY" not in code


def test_a_native_function_of_any_operand_needs_no_operand_types():
    # Mojo's std.math nextafter takes every operand type, so an operand CrossGL cannot type is left to it; a target
    # whose call needs a helper refuses the same call.
    source = PROBE.replace("BODY", "outp[i] = nextafter(x, unknownOperand);")
    program = Parser(Lexer(source).get_tokens()).parse()
    assert P.lower_portable_math(program, "mojo") == 0
    program = Parser(Lexer(source).get_tokens()).parse()
    with pytest.raises(P.PortableMathError, match="cannot be determined"):
        P.lower_portable_math(program, "opengl")


def test_types_the_generator_owns_keep_its_refusal(tmp_path):
    # The HLSL generator writes binary32 copysign and refuses the double one with its own diagnostic.
    with pytest.raises(Exception) as raised:
        _translate(
            tmp_path,
            "outp[i] = float(copysign(double(x), double(y)));",
            "directx",
        )
    assert not isinstance(raised.value, P.PortableMathError)
    assert "copysign" in str(raised.value)


@pytest.mark.parametrize(
    "body, message",
    [
        ("outp[i] = float(nextafter(half(x), half(y)));", "nextafter of 16-bit"),
        ("iout[i] = ilogb(double(x));", "ilogb of 64-bit"),
        ("outp[i] = sinpi(i);", "floating-point operands"),
    ],
)
def test_calls_without_a_helper_are_refused(tmp_path, body, message):
    with pytest.raises(P.PortableMathError) as raised:
        _translate(tmp_path, body, "vulkan")
    assert message in str(raised.value)
    assert raised.value.code == "project.translate.portable-math-unsupported"


def test_a_program_function_with_the_name_is_kept(tmp_path):
    source = PROBE.replace(
        "    compute {",
        "    float fdim(float a, float b) { return a + b; }\n    compute {",
    ).replace("BODY", "outp[i] = fdim(x, y);")
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    code = crosstl.translate(str(path), backend="opengl")
    assert P.PREFIX not in code


def test_reserved_names_are_refused(tmp_path):
    source = PROBE.replace(
        "    compute {",
        f"    float {P.PREFIX}fdim(float a) {{ return a; }}\n    compute {{",
    ).replace("BODY", "outp[i] = fdim(x, y);")
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    with pytest.raises(P.PortableMathError, match="reserved"):
        crosstl.translate(str(path), backend="opengl")


def test_a_target_can_register_native_functions(tmp_path):
    saved = P.native_portable_math("wgsl")
    try:
        P.register_native_portable_math("wgsl", ["copysign"], [("f", 32)])
        assert P.native_portable_math("wgsl")["copysign"] == frozenset({("f", 32)})
        with pytest.raises(AssertionError):
            P.register_native_portable_math("wgsl", ["sin"])
    finally:
        P._NATIVE["wgsl"] = saved


# The helpers' own text, evaluated in binary32.


def test_pi_helpers_are_within_one_ulp_and_exact_at_special_values():
    bits = E.stratified_bits(64, seed=5)
    for name in ("sinpi", "cospi"):
        error, pattern, exact = E.pi_check(name, bits)
        assert exact, name
        assert error < 1.0, (name, error, hex(pattern))


def test_pi_helpers_at_integers_and_half_integers():
    x = np.array(
        [0.0, -0.0, 1.0, -1.0, 2.0, -3.0, 0.5, -0.5, 1.5, 2.5, 8388609.0, 16777216.0],
        np.float32,
    )
    sine = E.evaluate("sinpi", x)
    cosine = E.evaluate("cospi", x)
    assert sine.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        0.0,
        0.0,
    ]
    assert np.signbit(sine).tolist() == [False, True, False, True, False, True] + [
        False,
        True,
        True,
        False,
    ] + [False, False]
    assert cosine.tolist() == [
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        1.0,
    ]
    assert not np.signbit(cosine[6:10]).any()


def _pairs(count=4096, seed=3):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2**32, (count, 2), dtype=np.uint64).astype(np.uint32)
    special = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0x3F800000,
        ],
        np.uint32,
    )
    grid = np.stack(np.meshgrid(special, special), axis=-1).reshape(-1, 2)
    near = np.stack([bits[:, 0], bits[:, 0] ^ np.uint32(1)], axis=-1)
    both = np.concatenate([grid, bits, near]).view(np.float32)
    return both[:, 0].copy(), both[:, 1].copy()


def _same_bits(got, want):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return bool(
        np.all(
            (got.view(np.uint32) == want.view(np.uint32))
            | (np.isnan(got) & np.isnan(want))
        )
    )


def test_nextafter_copysign_and_fdim_are_exact():
    x, y = _pairs()
    with np.errstate(over="ignore", invalid="ignore"):
        assert _same_bits(E.evaluate("nextafter", x, y), np.nextafter(x, y))
        assert _same_bits(E.evaluate("copysign", x, y), np.copysign(x, y))
        fdim = np.where(
            np.isnan(x) | np.isnan(y), np.nan, np.where(x > y, x - y, 0.0)
        ).astype(np.float32)
        assert _same_bits(E.evaluate("fdim", x, y), fdim)


def test_ilogb_is_exact():
    bits = np.concatenate(
        [E.stratified_bits(16, seed=7), np.arange(0, 1 << 16, dtype=np.uint32)]
    )
    x = bits.view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.frexp(x.astype(np.float64))[1] - 1
    want = np.where(
        np.isnan(x) | (x == 0), -(2**31), np.where(np.isinf(x), 2**31 - 1, exponent)
    ).astype(np.int32)
    assert np.array_equal(E.evaluate("ilogb", x), want)
