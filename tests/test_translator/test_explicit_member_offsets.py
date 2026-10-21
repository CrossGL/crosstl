"""``@ offset(N)`` members of a CrossGL constant buffer keep their bytes on every target.

``cbuffer Params { float a; float b @offset(12); vec4 c; }`` places ``b`` at byte 12, past the byte 4 that both
HLSL packing and std140 give it, and ``c`` after it at byte 16. SPIR-V decorates the offsets, HLSL places the members
with ``packoffset`` and GLSL reads the block through its registers.
"""

import re
import shutil
import subprocess

import crosstl

SOURCE = """
shader main {
    cbuffer Params @binding(0) { float a; float b @offset(12); vec4 c; };
    RWStructuredBuffer<float> o @binding(1);
    compute {
        layout(local_size_x = 1) in;
        void main() { buffer_store(o, 0u, a + b + c.x); }
    }
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_hlsl_places_offset_members_with_packoffset(tmp_path):
    hlsl = _translate(tmp_path, "directx")

    assert "float a : packoffset(c0);" in hlsl
    assert "float b : packoffset(c0.w);" in hlsl
    assert "float4 c : packoffset(c1);" in hlsl


def test_spirv_decorates_offset_members(tmp_path):
    spirv = _translate(tmp_path, "vulkan")

    block = re.search(r"OpMemberDecorate (%\w+) 1 Offset 12\b", spirv)
    assert block is not None
    assert f"OpMemberDecorate {block.group(1)} 2 Offset 16" in spirv


def test_glsl_reads_offset_members_from_their_registers(tmp_path):
    glsl = _translate(tmp_path, "opengl")

    assert "b = uintBitsToFloat(crossgl_cbuffer_registers[0].w);" in glsl
    assert "c = uintBitsToFloat(crossgl_cbuffer_registers[1].xyzw);" in glsl
    if shutil.which("glslangValidator"):
        path = tmp_path / "out.comp"
        path.write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", str(path)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
