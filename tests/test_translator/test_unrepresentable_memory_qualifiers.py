"""DirectX and SPIR-V output refuse resource memory qualifiers the target cannot express instead of dropping them.

HLSL has no volatile resources (DXC rejects ``volatile`` on a global) and no coherence wider than the device
(``globallycoherent``); the generator emitted such buffers as ordinary UAVs. Vulkan's ``Coherent`` reaches the device
at most, and the SPIR-V generator dropped Metal's ``coherent(system)``.
"""

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.hlsl_unrepresentable_memory import (
    hlsl_unrepresentable_memory_qualifier,
)
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

GLSL = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) volatile buffer Flags { uint flag[]; };
layout(std430, binding = 1) buffer Out { uint outp[]; };
void main() { outp[0] = flag[0]; }
"""

CROSSGL = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        volatile buffer float* outp;
        void main() {
            outp[gl_GlobalInvocationID.x] = 1.0;
        }
    }
}
"""

METAL = """#include <metal_stdlib>
using namespace metal;
kernel void probe({qualifier} device float* outp [[buffer(0)]],
                  uint3 id [[thread_position_in_grid]]) {{
    outp[id.x] = 1.0;
}}
"""


def _translate(tmp_path, source, suffix, backend="directx"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(
    "source, suffix, resource",
    [
        # The importer names an anonymous block's instance after the block.
        (GLSL, ".comp", "flags"),
        (CROSSGL, ".cgl", "outp"),
        (METAL.format(qualifier="volatile"), ".metal", "outp"),
    ],
    ids=["glsl-block", "crossgl-pointer", "metal-pointer"],
)
def test_volatile_resource_is_refused(tmp_path, source, suffix, resource):
    with pytest.raises(DirectXLoweringRefused, match=r"is volatile") as refused:
        _translate(tmp_path, source, suffix)
    assert refused.value.code == "project.translate.directx-lowering-unsupported"
    assert f"resource '{resource}' is volatile" in str(refused.value)


def test_system_scope_coherence_is_refused(tmp_path):
    with pytest.raises(DirectXLoweringRefused, match=r"system scope"):
        _translate(tmp_path, METAL.format(qualifier="coherent(system)"), ".metal")


def test_vulkan_refuses_system_scope_coherence(tmp_path):
    source = METAL.format(qualifier="coherent(system)")
    with pytest.raises(SpirvLoweringRefused, match=r"system scope") as refused:
        _translate(tmp_path, source, ".metal", backend="vulkan")
    assert refused.value.code == "project.translate.vulkan-lowering-unsupported"


def test_device_scope_coherence_is_globallycoherent(tmp_path):
    hlsl = _translate(tmp_path, METAL.format(qualifier="coherent(device)"), ".metal")
    assert "globallycoherent RWStructuredBuffer<float> outp" in hlsl


def test_plain_and_coherent_resources_are_not_refused():
    class Node:
        name = "b"
        qualifiers = ["coherent", "restrict", "readonly"]
        attributes = []

    assert hlsl_unrepresentable_memory_qualifier(Node()) is None
