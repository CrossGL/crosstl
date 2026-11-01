"""CrossGL built-ins whose WGSL spelling differs are written with the WGSL name."""

import shutil
import subprocess

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
