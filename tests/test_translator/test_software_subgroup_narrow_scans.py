"""Software-subgroup scans and lane moves for narrow integer and binary16 values."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.GLSL_codegen import OpenGLSoftwareSubgroupError
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_votes import _codegen


def _source(body):
    return f"""shader NarrowScans {{
        StructuredBuffer<uint> inputs @register(t0);
        RWStructuredBuffer<uint> results @register(u1);
        compute {{
            @numthreads(32, 4, 1)
            void main(uint invocation @gl_LocalInvocationIndex) {{
                {body}
            }}
        }}
    }}"""


def _generate(target, body):
    return _codegen(target).generate_stage(parse(_source(body)), "compute")


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize(
    ("value_type", "bits", "signed"),
    (("int8_t", 8, True), ("uint8_t", 8, False), ("int16", 16, True)),
)
@pytest.mark.parametrize("operation", ("WavePrefixSum", "WavePrefixInclusiveProduct"))
def test_narrow_integer_scans_wrap_to_their_source_width(
    tmp_path, target, value_type, bits, signed, operation
):
    generated = _generate(
        target,
        f"""{value_type} value = {value_type}(inputs[invocation]);
        {value_type} total = {operation}(value);
        results[invocation] = uint(total);""",
    )
    if target == "opengl":
        # bitfieldExtract sign-extends int and zero-extends uint results.
        assert f", 0, {bits})" in generated
        assert "bitfieldExtract(crossglSoftwareSubgroupPrefix" in generated
    elif bits == 8:
        wrap = "<< 24) >> 24)" if signed else "& 255u)"
        assert wrap in generated
    else:
        # DirectX scans 16-bit integers in native 16-bit scratch.
        assert "groupshared int16_t __crossgl_software_subgroup_scratch_int16_t" in (
            generated
        )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize(
    "operation", ("WavePrefixSum", "WavePrefixInclusiveSum", "WavePrefixProduct")
)
def test_binary16_scans_round_each_partial(tmp_path, target, operation):
    generated = _generate(
        target,
        f"""float16 value = float16(float(inputs[invocation] % 5u) - 2.0);
        float16 total = {operation}(value);
        results[invocation] = asuint(float(total));""",
    )
    if target == "opengl":
        # Every tree node is rounded to binary16 before it is shared.
        assert "crossglSoftwareSubgroupPrefix" in generated
        assert "= crossgl_round_half1(" in generated
        if operation == "WavePrefixSum":
            assert "result = uintBitsToFloat(0x80000000u);" in generated
    else:
        assert "precise float16_t result = value;" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("value_type", ("int8_t", "uint16", "float16"))
def test_narrow_lane_moves_are_lossless(tmp_path, target, value_type):
    fill = "float16(0.0)" if value_type == "float16" else f"{value_type}(0)"
    generated = _generate(
        target,
        f"""{value_type} value = {value_type}(inputs[invocation]);
        {value_type} filled = WaveShuffleAndFillUp(value, {fill}, 1u);
        {value_type} read = WaveReadLaneAt(filled, 31u);
        results[invocation] = uint(read);""",
    )
    assert "WaveShuffleAndFillUp" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("opengl", "directx"))
def test_bfloat16_scans_remain_unsupported(target):
    error = (
        OpenGLSoftwareSubgroupError
        if target == "opengl"
        else DirectXSoftwareSubgroupError
    )
    with pytest.raises((error, ValueError)):
        _generate(
            target,
            """bfloat16 value = bfloat16(float(inputs[invocation]));
            bfloat16 total = WavePrefixInclusiveSum(value);
            results[invocation] = asuint(float(total));""",
        )
