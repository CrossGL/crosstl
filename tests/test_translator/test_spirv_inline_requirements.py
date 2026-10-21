"""DXC's inline SPIR-V requirement attributes become module requirements.

``[[vk::ext_extension("SPV_...")]]`` and ``[[vk::ext_capability(N)]]`` on an HLSL entry point declare an
``OpExtension`` and an ``OpCapability`` (DXC's SPIR-V code generation documentation, "Inline SPIR-V"). The other
inline SPIR-V attributes change what instructions mean and are refused.
"""

import pytest

import crosstl
from crosstl.translator.codegen.spirv_inline_requirements import (
    SPIRV_CAPABILITY_NAMES,
)

# Source shape: Khronos Vulkan-Samples ray_tracing_position_fetch/hlsl/closesthit.rchit.hlsl, which names the
# capability through a #define.
REQUIREMENTS_HLSL = """
#define ShaderClockKHR 5055
RWStructuredBuffer<float> o : register(u0);
[[vk::ext_extension("SPV_KHR_shader_clock")]]
[[vk::ext_capability(ShaderClockKHR)]]
[numthreads(1, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) { o[t.x] = 1.0; }
"""


def _translate(tmp_path, source, backend="vulkan"):
    path = tmp_path / "shader.comp.hlsl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_capability_table_matches_the_spirv_grammar():
    assert SPIRV_CAPABILITY_NAMES[1] == "Shader"
    assert SPIRV_CAPABILITY_NAMES[5055] == "ShaderClockKHR"
    assert SPIRV_CAPABILITY_NAMES[5336] == "RayTracingPositionFetchKHR"
    assert SPIRV_CAPABILITY_NAMES[5284] == "FragmentBarycentricKHR"


def test_hlsl_imports_inline_requirements_as_attributes(tmp_path):
    crossgl = _translate(tmp_path, REQUIREMENTS_HLSL, backend="cgl")

    assert 'vk_ext_extension("SPV_KHR_shader_clock")' in crossgl
    assert "vk_ext_capability(5055)" in crossgl


def test_inline_requirements_declare_extension_and_capability(tmp_path):
    spirv = _translate(tmp_path, REQUIREMENTS_HLSL)
    lines = [line.strip() for line in spirv.splitlines()]

    assert "OpCapability ShaderClockKHR" in lines
    assert 'OpExtension "SPV_KHR_shader_clock"' in lines
    assert lines.count("OpCapability ShaderClockKHR") == 1


@pytest.mark.parametrize(
    "attribute,reason",
    [
        ("[[vk::ext_execution_mode(4446)]]", "has no SPIR-V lowering"),
        ("[[vk::ext_capability(99999)]]", "unknown SPIR-V capability '99999'"),
        ('[[vk::ext_extension("GL_EXT_foo")]]', "not a SPIR-V extension"),
    ],
)
def test_inline_spirv_attributes_without_a_lowering_are_refused(
    tmp_path, attribute, reason
):
    source = f"""
RWStructuredBuffer<float> o : register(u0);
{attribute}
[numthreads(1, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {{ o[t.x] = 1.0; }}
"""
    with pytest.raises(Exception, match=reason):
        _translate(tmp_path, source)
