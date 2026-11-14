"""bfloat16 vectors lower to per-lane uint payloads in HLSL.

DirectX has no bfloat16 type. A CrossGL ``bfloat16vecN`` (Metal ``bfloatN``)
travels as ``uintN`` whose lanes hold the bfloat16 bits; arithmetic widens the
lanes exactly, computes in binary32 and rounds each lane back once. Anything a
payload cannot express exactly must raise a typed diagnostic.
"""

import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.directx_codegen import (
    DirectXBFloat16UnsupportedError,
    HLSLCodeGen,
)


def _compute(body, declarations=""):
    return f"""
shader BFloatVectors {{
    StructuredBuffer<float> input;
    RWStructuredBuffer<float> output;
    {declarations}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            {body}
        }}
    }}
}}
"""


VECTOR_SHADER = _compute(
    """
            bfloat16vec3 a = bfloat16vec3(input[0], input[1], input[2]);
            bfloat16vec3 b = bfloat16vec3(1.5);
            bfloat16vec3 c = a * b + a;
            c += b;
            c.xy -= bfloat16vec2(0.25);
            bfloat16vec3 d = -c;
            bfloat16_t s = dot(a, b);
            bfloat16vec3 n = normalize(d);
            bfloat16vec3 k = cross(a, b);
            bfloat16vec3 m = max(a, b);
            bfloat16vec3 scaled = scale(a, s);
            bool3 less = a < b;
            ivec3 counts = ivec3(257, -3, 65535);
            bfloat16vec3 fromInts = bfloat16vec3(counts);
            uint16_t3 bits = as_type<uint16_t3>(fromInts);
            bfloat16vec3 back = as_type<bfloat16vec3>(bits);
            bfloat16vec2 packed = as_type<bfloat16vec2>(asuint(input[3]));
            float word = as_type<float>(packed);
            vec3 f = vec3(c);
            output[0] = f.x + float(s) + float(n.y) + float(d.z) + float(k.x);
            output[1] = float(m.z) + float(scaled.y) + float(back.x) + word;
            output[2] = less.x ? 1.0 : 0.0;
""",
    declarations="""
    bfloat16vec3 scale(bfloat16vec3 value, bfloat16_t factor) {
        return value * factor;
    }
""",
)


def _generate(source, **options):
    return HLSLCodeGen(target_profile="dx12", **options).generate(
        crosstl.translator.parse(source)
    )


def test_bfloat16_vectors_travel_as_uint_payloads():
    hlsl = _generate(VECTOR_SHADER)

    assert "bfloat16vec" not in hlsl
    assert "uint3 scale(uint3 value, uint factor)" in hlsl
    assert "float3 __crossgl_bfloat16_to_float(uint3 value)" in hlsl
    assert "uint3 __crossgl_bfloat16_from_float(float3 value)" in hlsl
    assert "uint3 __crossgl_bfloat16_from_int(int3 value)" in hlsl
    assert (
        "uint3 a = uint3(__crossgl_bfloat16_from_float(float(input[0])), "
        "__crossgl_bfloat16_from_float(float(input[1])), "
        "__crossgl_bfloat16_from_float(float(input[2])));"
    ) in hlsl
    assert "uint3 b = ((uint3)(__crossgl_bfloat16_from_float(float(1.5))));" in hlsl
    assert "uint3 fromInts = __crossgl_bfloat16_from_int(int3(counts));" in hlsl


def test_bfloat16_vector_arithmetic_rounds_every_operation_once():
    hlsl = _generate(VECTOR_SHADER)

    # a * b rounds before the addition, exactly like the source type would.
    assert (
        "uint3 c = __crossgl_bfloat16_from_float(float3(("
        "__crossgl_bfloat16_to_float(uint3(__crossgl_bfloat16_from_float(float3(("
        "__crossgl_bfloat16_to_float(uint3(a)) * "
        "__crossgl_bfloat16_to_float(uint3(b))))))) + "
        "__crossgl_bfloat16_to_float(uint3(a)))));"
    ) in hlsl
    assert (
        "c = __crossgl_bfloat16_from_float(float3(("
        "__crossgl_bfloat16_to_float(uint3(c)) + "
        "__crossgl_bfloat16_to_float(uint3(b)))));"
    ) in hlsl
    assert "c.xy = __crossgl_bfloat16_from_float(float2((" in hlsl
    # Negation flips the sign bit of every lane without rounding.
    assert "uint3 d = ((uint3(c) ^ 0x8000u) & 0xffffu);" in hlsl
    assert (
        "bool3 less = (__crossgl_bfloat16_to_float(uint3(a)) < "
        "__crossgl_bfloat16_to_float(uint3(b)));"
    ) in hlsl
    assert (
        "return __crossgl_bfloat16_from_float(float3(("
        "__crossgl_bfloat16_to_float(uint3(value)) * "
        "__crossgl_bfloat16_to_float(uint(factor)))));"
    ) in hlsl


def test_bfloat16_vector_builtins_use_binary32_and_round_back():
    hlsl = _generate(VECTOR_SHADER)

    assert (
        "uint s = __crossgl_bfloat16_from_float(float(dot("
        "__crossgl_bfloat16_to_float(uint3(a)), "
        "__crossgl_bfloat16_to_float(uint3(b)))));"
    ) in hlsl
    assert (
        "uint3 n = __crossgl_bfloat16_from_float(float3(normalize("
        "__crossgl_bfloat16_to_float(uint3(d)))));"
    ) in hlsl
    assert "uint3 k = __crossgl_bfloat16_from_float(float3(cross(" in hlsl
    assert "uint3 m = __crossgl_bfloat16_from_float(float3(max(" in hlsl


def test_bfloat16_vector_bitcasts_move_payload_bits():
    hlsl = _generate(VECTOR_SHADER)

    assert "uint16_t3 bits = uint16_t3((fromInts) & 0xffffu);" in hlsl
    assert "uint3 back = uint3(bits);" in hlsl
    assert (
        "uint2 packed = uint2(__crossgl_bitcast_uint_to_uint16x2(asuint(input[3])));"
    ) in hlsl
    assert (
        "float word = asfloat(__crossgl_bitcast_uint16x2_to_uint("
        "uint16_t2((packed) & 0xffffu)));"
    ) in hlsl


def test_bfloat16_vector_swizzles_keep_payload_lanes():
    hlsl = _generate(_compute("""
            bfloat16vec3 a = bfloat16vec3(input[0]);
            bfloat16vec2 b = a.xy;
            bfloat16_t c = a.z;
            a.x = c;
            output[0] = float(b.y) + float(a.x);
"""))

    assert "uint2 b = a.xy;" in hlsl
    assert "uint c = a.z;" in hlsl
    assert "a.x = c;" in hlsl
    assert (
        "output[0] = (__crossgl_bfloat16_to_float(uint(b.y)) + "
        "__crossgl_bfloat16_to_float(uint(a.x)));"
    ) in hlsl


@pytest.mark.parametrize(
    "spelling",
    ["bfloat16vec2", "vec2<bfloat16_t>", "vector<bfloat16_t, 2>"],
)
def test_bfloat16_vector_spellings_share_one_lowering(spelling):
    hlsl = _generate(_compute(f"""
            {spelling} a = {spelling}(input[0], input[1]);
            {spelling} b = a + a;
            output[0] = float(b.y);
"""))

    entry = hlsl.split("void CSMain")[-1]
    assert re.search(r"\bbfloat16(?:vec\d|_t\d?)\b|<", entry) is None
    assert "uint2 a = uint2(" in hlsl
    assert "uint2 b = __crossgl_bfloat16_from_float(float2((" in hlsl


REFUSALS = {
    "double-source": (
        "double3 x = double3(input[0]); bfloat16vec3 a = bfloat16vec3(x);",
        "unsupported-conversion-source",
        "bfloat16vec3 constructor",
    ),
    "binary-lane-mismatch": (
        "bfloat16vec3 a = bfloat16vec3(1.0); bfloat16vec2 b = bfloat16vec2(2.0);"
        " bfloat16vec3 c = a + b;",
        "bfloat16-vector-lane-mismatch",
        "+",
    ),
    "float-lane-mismatch": (
        "bfloat16vec3 a = bfloat16vec3(1.0); vec2 b = vec2(2.0); vec3 c = a + b;",
        "bfloat16-vector-lane-mismatch",
        "+",
    ),
    "conversion-lane-mismatch": (
        "vec2 x = vec2(input[0]); bfloat16vec3 a = x;",
        "bfloat16-vector-lane-mismatch",
        "bfloat16 vector conversion",
    ),
    "vector-to-scalar": (
        "bfloat16vec3 a = bfloat16vec3(1.0); bfloat16_t s = a;",
        "unsupported-conversion-target",
        "bfloat16 vector conversion",
    ),
    "increment": (
        "bfloat16vec3 a = bfloat16vec3(1.0); a++;",
        "unsupported-bfloat16-operator",
        "++",
    ),
    "logical-and": (
        "bfloat16vec3 a = bfloat16vec3(1.0); bool b = a && a;",
        "unsupported-bfloat16-operator",
        "&&",
    ),
    "bitwise-and": (
        "bfloat16vec3 a = bfloat16vec3(1.0); bfloat16vec3 b = a & a;",
        "unsupported-bfloat16-operator",
        "&",
    ),
    "remainder-assignment": (
        "bfloat16vec3 a = bfloat16vec3(1.0); a %= a;",
        "unsupported-bfloat16-compound-assignment",
        "%=",
    ),
    "scalar-target-vector-source": (
        "bfloat16vec3 a = bfloat16vec3(1.0); bfloat16_t s = bfloat16_t(1.0);"
        " s += a;",
        "unsupported-compound-assignment-source",
        "+=",
    ),
    "side-effecting-target": (
        "bfloat16vec3 values[2]; values[0] = bfloat16vec3(1.0); int k = 0;"
        " values[k++] += bfloat16vec3(1.0);",
        "side-effecting-assignment-target",
        "+=",
    ),
    "unproven-builtin": (
        "bfloat16vec3 a = bfloat16vec3(1.0); bfloat16vec3 b = cospi(a);",
        "unsupported-bfloat16-builtin",
        "cospi",
    ),
    "mixed-builtin": (
        "bfloat16vec3 a = bfloat16vec3(1.0); vec3 f = vec3(2.0);"
        " bfloat16vec3 b = max(a, f);",
        "unsupported-bfloat16-builtin",
        "max",
    ),
    "constructor-arity": (
        "bfloat16vec3 a = bfloat16vec3(1.0, 2.0);",
        "invalid-constructor-arity",
        "bfloat16vec3 constructor",
    ),
    "bitcast-width": (
        "uint3 u = uint3(16256u); bfloat16vec3 a = as_type<bfloat16vec3>(u);",
        "bitcast-width-or-type-mismatch",
        "as_type<bfloat16vec3>",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_inexpressible_bfloat16_vector_operations_fail_closed(case):
    body, reason, operation = REFUSALS[case]

    with pytest.raises(DirectXBFloat16UnsupportedError) as error:
        _generate(_compute(body + " output[0] = 1.0;"))

    assert error.value.reason == reason
    assert error.value.operation == operation


def test_bfloat16_vector_lowering_compiles_with_dxc_when_available(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    shader_path = tmp_path / "bfloat16_vectors.hlsl"
    output_path = tmp_path / "bfloat16_vectors.dxil"
    shader_path.write_text(_generate(VECTOR_SHADER), encoding="utf-8")
    validation = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_2",
            "-E",
            "CSMain",
            "-enable-16bit-types",
            str(shader_path),
            "-Fo",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert output_path.exists()
