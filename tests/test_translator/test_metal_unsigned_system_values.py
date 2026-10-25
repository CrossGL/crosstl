"""MSL's uint [[vertex_id]], [[instance_id]] and [[primitive_id]] keep their unsigned arithmetic."""

import re
import shutil
import subprocess

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float flag; };
vertex VOut v(uint vid [[vertex_id]], uint iid [[instance_id]]) {
    VOut out;
    out.position = float4(float(vid), float(iid), 0.0, 1.0);
    // For vid == 0 the unsigned difference wraps and the comparison is false.
    out.flag = (vid - 1 < 3) ? 1.0 : 0.0;
    return out;
}
"""


def translate(tmp_path, backend):
    path = tmp_path / "shader.metal"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_indices_stay_uint(tmp_path):
    # The import used to read them as int, which made `vid - 1 < 3` true for vid == 0.
    cgl = translate(tmp_path, "cgl")
    assert "uint vid @gl_VertexID, uint iid @gl_InstanceID" in cgl
    glsl = translate(tmp_path, "opengl")
    assert "uint vid = uint(gl_VertexID);" in glsl
    essl = translate(tmp_path, "webgl")
    assert "((vid - 1u) < 3u)" in essl
    hlsl = translate(tmp_path, "directx")
    assert re.search(r"uint vid\s*:\s*SV_VertexID", hlsl)
    if shutil.which("glslangValidator"):
        (tmp_path / "out.vert").write_text(glsl)
        _run(["glslangValidator", "-S", "vert", "out.vert"], tmp_path)
        (tmp_path / "es.vert").write_text(essl)
        _run(["glslangValidator", "-S", "vert", "es.vert"], tmp_path)
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        _run(["dxc", "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"], tmp_path)
    spirv = translate(tmp_path, "vulkan")
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(spirv)
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "o.spv"],
            tmp_path,
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", "o.spv"], tmp_path)
