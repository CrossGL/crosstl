"""CrossGL built-ins whose WGSL spelling differs are written with the WGSL name."""

import shutil
import subprocess

import pytest

import crosstl

SHADER = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* uin;
        buffer int* iout;
        buffer uint* uout;
        buffer vec4* fout;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            uint x = uin[i];
            ivec3 s = ivec3(int(x), 2, 3);
            uvec2 v = uvec2(x, x);
            int n = bitCount(x);
            ivec2 c = findLSB(v);
            ivec3 m = findMSB(s) + bitCount(s);
            iout[i] = n + c.x + m.y;
            uout[i] = bitfieldReverse(x) + bitfieldExtract(x, 1, 2)
                + bitfieldInsert(x, x, n, 3) + packHalf2x16(vec2(1.0))
                + packSnorm2x16(vec2(0.5)) + packUnorm2x16(vec2(0.5))
                + packSnorm4x8(vec4(0.5)) + packUnorm4x8(vec4(0.5));
            vec3 f = faceforward(vec3(1.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0));
            fout[i] = vec4(f, 1.0) + vec4(unpackHalf2x16(x), unpackSnorm2x16(x))
                + vec4(unpackUnorm2x16(x), 0.0, 0.0) + unpackSnorm4x8(x)
                + unpackUnorm4x8(x);
        }
    }
}
"""


def test_glsl_builtins_use_wgsl_names(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(SHADER)
    wgsl = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(wgsl)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    # WGSL returns the argument type; GLSL returns int.
    assert "var n: i32 = i32(countOneBits(x));" in wgsl
    assert "var c: vec2<i32> = vec2<i32>(firstTrailingBit(v));" in wgsl
    assert "(firstLeadingBit(s) + countOneBits(s))" in wgsl
    # WGSL takes the offset and count as u32.
    assert "extractBits(x, u32(1), u32(2))" in wgsl
    assert "insertBits(x, x, u32(n), u32(3))" in wgsl
    for name in (
        "reverseBits(x)",
        "faceForward(",
        "pack2x16float(",
        "pack2x16snorm(",
        "pack2x16unorm(",
        "pack4x8snorm(",
        "pack4x8unorm(",
        "unpack2x16float(x)",
        "unpack2x16snorm(x)",
        "unpack2x16unorm(x)",
        "unpack4x8snorm(x)",
        "unpack4x8unorm(x)",
    ):
        assert name in wgsl


REFUSAL_SHADER = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* uin;
        buffer int* iout;
        void main() {
            uint x = uin[0];
            STATEMENT
        }
    }
}
"""


@pytest.mark.parametrize(
    "statement",
    [
        "if (findMSB(x >> 1u) < 0) {\n                iout[0] = 1;\n            }",
        "iout[0] = findLSB(bitfieldReverse(x));",
        "iout[0] = bitCount(max(x, 1u)) - 1;",
    ],
    ids=["shift", "builtin-result", "bitcount"],
)
def test_bit_count_and_scan_refuse_an_argument_of_unknown_type(tmp_path, statement):
    # Left as u32, the result is still valid WGSL, but findMSB(0u) < 0 is false.
    path = tmp_path / "probe.cgl"
    path.write_text(REFUSAL_SHADER.replace("STATEMENT", statement))
    with pytest.raises(ValueError, match="cannot tell the type of"):
        crosstl.translate(str(path), backend="wgsl")


@pytest.mark.parametrize(
    "call, expected",
    [("bitCount()", "exactly 1 argument; got 0"), ("bitfieldExtract(x, 1)", "got 2")],
)
def test_bit_functions_refuse_the_wrong_argument_count(tmp_path, call, expected):
    path = tmp_path / "probe.cgl"
    path.write_text(REFUSAL_SHADER.replace("STATEMENT", f"iout[0] = int({call});"))
    with pytest.raises(ValueError, match=expected):
        crosstl.translate(str(path), backend="wgsl")


def test_scalar_faceforward_is_refused(tmp_path):
    # WGSL's faceForward takes vectors only; naga rejects faceForward(1.0, ...).
    path = tmp_path / "probe.cgl"
    path.write_text(
        REFUSAL_SHADER.replace(
            "STATEMENT", "iout[0] = int(faceforward(1.0, float(x), 3.0));"
        )
    )
    with pytest.raises(ValueError, match=r"faceforward\(\) on vectors only"):
        crosstl.translate(str(path), backend="wgsl")
