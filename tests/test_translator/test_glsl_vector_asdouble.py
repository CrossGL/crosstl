"""GLSL output joins ``asdouble`` of two uint vectors component by component.

HLSL's ``double2 asdouble(uint2 lowbits, uint2 highbits)`` (and Slang's wider forms) join each component pair into
one double. GLSL's ``packDouble2x32`` takes a single ``uvec2`` pair, so the generator's
``packDouble2x32(uvec2(a0, a1))`` of two ``uvec2`` operands was a conversion glslang refuses. A vector form is now a
``dvecN`` of per-component ``packDouble2x32`` calls; an operand with side effects, which that form would repeat, is
refused.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic

GLSLANG = shutil.which("glslangValidator")


def kernel(body):
    return f"""
shader main {{
    RWStructuredBuffer<uint> uin;
    RWStructuredBuffer<vec4> outp;
    uint bump(uint value) {{
        uin[0] = value + 1u;
        return value;
    }}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(uvec3 id @ gl_GlobalInvocationID) {{
            uint i = id.x;
{body}
        }}
    }}
}}
"""


def translate(tmp_path, source):
    path = tmp_path / "kernel.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl")


def assert_compiles(code):
    if GLSLANG is None:
        return
    result = subprocess.run(
        [GLSLANG, "--stdin", "-S", "comp"],
        input=code,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_uint2_pairs_join_per_component(tmp_path):
    code = translate(
        tmp_path,
        kernel("""
            uvec2 a0 = uvec2(uin[i], uin[i + 1u]);
            uvec2 a1 = uvec2(uin[i + 2u], uin[i + 3u]);
            double2 r = asdouble(a0, a1);
            outp[i] = vec4(vec2(r), 0.0, 1.0);
"""),
    )

    assert "packDouble2x32(uvec2(a0, a1))" not in code
    assert "dvec2(packDouble2x32(uvec2(a0.x, a1.x))," in code
    assert "packDouble2x32(uvec2(a0.y, a1.y)))" in code
    assert_compiles(code)


def test_signed_and_compound_vector_operands_are_converted_once_each(tmp_path):
    code = translate(
        tmp_path,
        kernel("""
            uvec3 b = uvec3(uin[0], uin[1], uin[2]);
            ivec3 c = ivec3(b);
            double3 t = asdouble(b + uvec3(1u), c);
            outp[i] = vec4(float(t.x), float(t.y), float(t.z), 1.0);
"""),
    )

    assert "dvec3(packDouble2x32(uvec2((b + uvec3(1u)).x, uvec3(c).x))," in code
    assert "packDouble2x32(uvec2((b + uvec3(1u)).z, uvec3(c).z)))" in code
    assert_compiles(code)


def test_scalar_pair_keeps_one_pack(tmp_path):
    code = translate(
        tmp_path,
        kernel("""
            double s = asdouble(uin[i], uin[i + 1u]);
            outp[i] = vec4(float(s));
"""),
    )

    assert "double s = packDouble2x32(uvec2(uin[i], uin[(i + 1u)]));" in code
    assert_compiles(code)


def test_vector_operand_with_side_effects_is_refused(tmp_path):
    with pytest.raises(TranslationDiagnostic, match="asdouble of two uint vectors"):
        translate(
            tmp_path,
            kernel("""
            uvec2 a0 = uvec2(uin[i], uin[i + 1u]);
            double2 r = asdouble(a0, uvec2(bump(i), 0u));
            outp[i] = vec4(vec2(r), 0.0, 1.0);
"""),
        )
