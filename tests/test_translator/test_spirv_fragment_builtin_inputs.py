"""Fragment built-in inputs in SPIR-V.

Vulkan requires ``Flat`` on every integer fragment input, built-ins included (``VUID-StandaloneSpirv-Flat-04744``;
glslang decorates ``gl_SampleID`` and ``gl_FragSizeEXT`` so). ``gl_SampleID`` and HLSL's ``SV_SampleIndex`` are the
``SampleId`` built-in (``SampleRateShading``). A pixel input struct shared with the vertex stage may carry
``[[vk::builtin("PointSize")]]``, which a fragment shader cannot declare: when nothing reads that member it takes an
undefined value instead of an interface variable, and reading it is refused.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from tests.spirv_friendly_names import friendly_spirv


def _translate(tmp_path, source, name, backend="vulkan", **options):
    path = tmp_path / name
    path.write_text(source)
    if backend == "vulkan":
        options["format_output"] = False
        return friendly_spirv(crosstl.translate(str(path), backend=backend, **options))
    return crosstl.translate(str(path), backend=backend, **options)


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _decorations(code, name):
    return sorted(
        line.split(f"%{name} ", 1)[1]
        for line in code.splitlines()
        if line.strip().startswith(f"OpDecorate %{name} ")
    )


def test_integer_fragment_builtins_are_flat(tmp_path):
    source = """#version 450
#extension GL_EXT_fragment_invocation_density : require
layout(location = 0) out vec4 color;
void main() { color = vec4(float(gl_FragSizeEXT.x + gl_FragInvocationCountEXT + gl_SampleID)); }
"""
    code = _translate(tmp_path, source, "density.frag")
    assert _decorations(code, "gl_FragSizeEXT") == ["BuiltIn FragSizeEXT", "Flat"]
    assert _decorations(code, "gl_SampleID") == ["BuiltIn SampleId", "Flat"]
    assert "OpCapability SampleRateShading" in code
    _spirv_val(tmp_path, code)


def test_hlsl_sample_index_is_the_sample_id_builtin(tmp_path):
    source = """
float4 main(float4 p : SV_Position, uint s : SV_SampleIndex) : SV_Target {
    return float4(p.xy, s, 1);
}
"""
    code = _translate(tmp_path, source, "sample.frag.hlsl", source_backend="directx")
    assert _decorations(code, "gl_SampleID") == ["BuiltIn SampleId", "Flat"]
    _spirv_val(tmp_path, code)


SHARED_STRUCT = """
struct VSOutput {
    float4 Pos : SV_POSITION;
    [[vk::location(0)]] float2 UV : TEXCOORD0;
    [[vk::builtin("PointSize")]] float PSize : PSIZE;
};
float4 main(VSOutput input) : SV_TARGET { return float4(input.UV, READ, 1.0); }
"""


def test_unread_point_size_member_of_a_pixel_input_is_not_an_interface(tmp_path):
    code = _translate(
        tmp_path,
        SHARED_STRUCT.replace("READ", "0.0"),
        "points.frag.hlsl",
        source_backend="directx",
    )
    assert "BuiltIn PointSize" not in code
    assert "OpUndef" in code
    _spirv_val(tmp_path, code)


def test_read_point_size_member_of_a_pixel_input_is_refused(tmp_path):
    with pytest.raises(TranslationDiagnostic, match="gl_PointSize"):
        _translate(
            tmp_path,
            SHARED_STRUCT.replace("READ", "input.PSize"),
            "points.frag.hlsl",
            source_backend="directx",
        )
