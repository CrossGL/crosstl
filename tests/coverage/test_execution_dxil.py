"""The HLSL probes and comparisons that account for what DXC's DXIL and Direct3D compute: DXC's lowering of fmod and
frexp, Direct3D's rounding of a float converted to 16 bits, the stride of a structured buffer of 3-component vectors,
and a compiler's freedom to fuse a multiply and an add."""

import re

import numpy as np
import pytest

from tools.coverage import execution, probes
from tools.coverage.model import Feature, Param, Signature, Type

from .construct_grammar import construct_feature

FLOAT2 = Type("float", (2,))
HALF2 = Type("half", (2,))


def function(name, result, *params, language="hlsl"):
    signature = Signature(
        result, tuple(p if isinstance(p, Param) else Param(p) for p in params)
    )
    return Feature(language, "function", name, signature, crossgl=name)


def frexp(kind=FLOAT2, exponent=None):
    return function("frexp", kind, kind, Param(exponent or kind, "out"))


def fmod(kind=FLOAT2):
    return function("fmod", kind, kind, kind)


def test_the_mantissa_reference_keeps_the_sign_of_the_argument_the_probe_reads():
    source = probes.build(frexp(), "hlsl").source
    assert "float2 a0 = float2(fin[i + 0u], fin[i + 1u]);" in source
    assert "float2 r = frexp(a0, a1);" in source
    data = execution.inputs("fin")
    mantissas = execution.mantissas(frexp()).view(np.float32)
    assert len(mantissas) == 2 * execution.THREADS
    for i in (0, 9, 63):
        for k in (0, 1):
            mantissa, exponent = np.frexp(data[i + k])
            assert mantissas[2 * i + k] == mantissa
            assert np.ldexp(mantissa, exponent) == data[i + k]
    assert (mantissas < 0).any()
    assert ((0.5 <= np.abs(mantissas)) & (np.abs(mantissas) < 1)).all()
    scalar = execution.mantissas(frexp(Type("float"))).view(np.float32)
    assert scalar[3] == np.frexp(data[3])[0]


@pytest.mark.parametrize(
    "feature",
    [
        fmod(),
        function("frexp", FLOAT2, FLOAT2, FLOAT2),
        frexp(Type("float", (2, 2))),
        function("frexp", Type("float"), Param(Type("float"), "out"), Type("float")),
    ],
)
def test_only_an_elementwise_float_frexp_has_a_mantissa_reference(feature):
    assert execution.mantissas(feature) is None


def test_dxil_s_unsigned_mantissa_matches_only_as_c_s_mantissa():
    mantissas = execution.mantissas(frexp())
    words = np.zeros(execution.THREADS * execution.WORDS, dtype=np.uint32)
    words[: len(mantissas)] = mantissas
    # DXC's DXIL keeps the exponent field 0x3f000000 and the significand, and drops the sign.
    native = words.copy()
    native[: len(mantissas)] = (mantissas & 0x007FFFFF) | 0x3F000000
    assert execution.difference(native, words).startswith(execution.DIVERGED)
    assert execution.difference(native, words, mantissas) == ""
    other = words.copy()
    other[0] ^= 1
    assert "1 output words differ" in execution.difference(native, other, mantissas)


@pytest.mark.parametrize(
    ("feature", "runs_as", "native", "reference"),
    [
        (fmod(), "hlsl", "float2 r = fmod(a0, a1);", execution.remainders),
        (frexp(), "hlsl", "float2 r = frexp(a0, a1);", execution.mantissas),
        (fmod(), "spirv", "%r = OpFRem %v2float %a0 %a1", execution.remainders),
    ],
)
def test_the_exact_words_follow_the_runtime_s_lowering(
    feature, runs_as, native, reference
):
    assert np.array_equal(
        execution.exact_words(feature, runs_as, native), reference(feature)
    )


@pytest.mark.parametrize(
    ("feature", "runs_as", "native"),
    [
        # DXC's SPIR-V keeps frexp's sign (FrexpStruct), and a SPIR-V fmod without OpFRem is exact.
        (frexp(), "spirv", "%r = OpExtInst %v2float %glsl FrexpStruct %a0"),
        (fmod(), "spirv", "%r = OpExtInst %v2float %glsl Fma %a0 %a1 %a1"),
        (fmod(), "hlsl", "float2 r = a0 - a1 * trunc(a0 / a1);"),
        (fmod(), "metal", "float2 r = fmod(a0, a1);"),
        (function("frac", FLOAT2, FLOAT2), "hlsl", "float2 r = frac(a0);"),
    ],
)
def test_other_runs_must_write_the_native_probe_s_words(feature, runs_as, native):
    assert execution.exact_words(feature, runs_as, native) is None


def test_hlsl_features_convert_16_bit_arguments_on_a_grid_binary16_holds():
    asint16 = function("asint16", Type("int16", (2,)), HALF2)
    hlsl = probes.build(asint16, "hlsl").source
    grid = "(half)((float)(int)(fin[i + 0u] * 256.0) * 0.00390625)"
    assert f"half2 a0 = half2({grid}, " in hlsl
    crossgl = probes.build(asint16, "crossgl").source
    assert "float16((float(int((fin[i + 1u] * 256.0))) * 0.00390625))" in crossgl
    # Elsewhere a conversion to 16 bits rounds to nearest, as CrossGL's does, so the probes read the input itself.
    metal = function("asint16", Type("int16", (2,)), HALF2, language="metal")
    assert "float16(fin[i + 0u])" in probes.build(metal, "crossgl").source
    data = execution.inputs("fin")
    on_grid = np.trunc(data * np.float32(256.0)).astype(np.float32) / np.float32(256)
    assert np.array_equal(on_grid.astype(np.float16).astype(np.float32), on_grid)
    assert np.abs(data).max() < 4


def test_hlsl_s_half_probe_is_binary16_on_every_tool_and_reads_a_value_it_holds():
    native = probes.build(construct_feature("hlsl", "half"), "hlsl").source
    assert "half v = half((float)(int)(fin[i] * 256.0) * 0.00390625);" in native
    assert re.search(r"\bfloat16_t w = v \* v;", native)
    crossgl = probes.build(construct_feature("hlsl", "half"), "crossgl").source
    assert "half v = half(float(int(fin[i] * 256.0)) * 0.00390625);" in crossgl
    assert "outp[i] = float(v * v);" in crossgl
    # Other languages' half probes, and their CrossGL probes, read the input itself.
    metal = construct_feature("metal", "half")
    assert "half v = half(fin[i]);" in probes.build(metal, "crossgl").source
    assert "half v = half(fin[i]);" in probes.build(metal, "metal").source


def test_a_3_component_output_of_an_hlsl_feature_keeps_the_structured_stride():
    cross = function(
        "cross", Type("float", (3,)), Type("float", (3,)), Type("float", (3,))
    )
    assert "RWStructuredBuffer<float3> outp" in probes.build(cross, "hlsl").source
    crossgl = probes.build(cross, "crossgl").source
    assert "        @buffer_layout(hlsl_structured) buffer vec3* outp;\n" in crossgl
    assert "        buffer float* fin;\n" in crossgl
    glsl = function("cross", *[Type("float", (3,))] * 3, language="glsl")
    assert "        buffer vec3* outp;\n" in probes.build(glsl, "crossgl").source
    two = function("lerp", FLOAT2, FLOAT2, FLOAT2, FLOAT2)
    assert "        buffer vec2* outp;\n" in probes.build(two, "crossgl").source


@pytest.mark.parametrize(
    "name", ["uniform-integer", "uniform-unsigned", "uniform-integer-vector"]
)
@pytest.mark.parametrize("language", ["crossgl", "hlsl", "glsl", "metal", "cuda"])
def test_uniform_integer_probes_multiply_last_so_no_compiler_fuses_an_add(
    name, language
):
    origin = "glsl" if language == "crossgl" else language
    source = probes.build(construct_feature(origin, name), language).source
    [store] = [line.strip() for line in source.splitlines() if "outp[i] =" in line]
    assert store.startswith("outp[i] = fin[i] * (") and store.endswith(");"), store
    # The product's right operand is the whole parenthesised sum.
    depth = 0
    for position, character in enumerate(store[len("outp[i] = fin[i] * ") :]):
        depth += {"(": 1, ")": -1}.get(character, 0)
        if depth == 0:
            assert store[len("outp[i] = fin[i] * ") + position + 1 :] == ";", store
            break
