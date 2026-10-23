"""Vulkan input attachments are refused on OpenGL and kept on SPIR-V.

Two bugs affected HLSL ``[[vk::input_attachment_index(N)]] SubpassInput``:

* The HLSL importer mapped it to ``sampler2d``. SPIR-V output then refused
  ``subpassLoad`` for lacking a subpass input operand.
* GLSL and GLSL ES output declared a ``sampler2D`` and called
  ``subpassLoad`` on it, which no OpenGL compiler accepts. GLSL sources with
  ``subpassInput`` failed the same way.

OpenGL and OpenGL ES have no input attachments. ``subpassInput`` exists only
in Vulkan GLSL (GL_KHR_vulkan_glsl section 4.1.7), so those targets now
refuse it. SPIR-V keeps it as an input attachment.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl

HLSL = textwrap.dedent("""
    [[vk::input_attachment_index(0)]][[vk::binding(0)]] SubpassInput in_color;
    [[vk::input_attachment_index(1)]][[vk::binding(1)]] SubpassInput in_normal;

    float4 main() : SV_TARGET0
    {
        return in_color.SubpassLoad() + in_normal.SubpassLoad();
    }
    """)

GLSL = textwrap.dedent("""
    #version 450
    layout(input_attachment_index = 0, binding = 0) uniform subpassInput inputColor;
    layout(location = 0) out vec4 outColor;
    void main() {
        outColor = subpassLoad(inputColor);
    }
    """)


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"attachments{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path),
        backend=backend,
        source_backend="hlsl" if suffix == ".frag.hlsl" else None,
    )


def test_hlsl_import_keeps_the_subpass_input_type(tmp_path):
    code = _translate(tmp_path, HLSL, ".frag.hlsl", "cgl")
    assert re.search(
        r"input_attachment_index\(0\)\s*@ binding\(0\)\s*subpassInput in_color;", code
    )
    assert "sampler2d" not in code


def test_hlsl_input_attachments_lower_to_spirv(tmp_path):
    code = _translate(tmp_path, HLSL, ".frag.hlsl", "vulkan")
    assert "OpCapability InputAttachment" in code
    assert re.search(r"OpTypeImage %\w+ SubpassData", code)
    assert "InputAttachmentIndex 0" in code and "InputAttachmentIndex 1" in code
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    (tmp_path / "a.spvasm").write_text(code, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", "a.spvasm", "-o", "a.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "a.spv"],
    ):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, cwd=tmp_path
        )
        assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
@pytest.mark.parametrize(("source", "suffix"), [(HLSL, ".frag.hlsl"), (GLSL, ".frag")])
def test_opengl_targets_refuse_input_attachments(tmp_path, backend, source, suffix):
    with pytest.raises(ValueError, match="OpenGL has no subpass inputs"):
        _translate(tmp_path, source, suffix, backend)
