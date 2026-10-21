"""HLSL ``register`` and ``vk::binding`` choose the Vulkan descriptor slots of the SPIR-V resources.

``[[vk::binding(X, Y)]]`` names the exact slot. ``register(Xn, spaceM)`` maps to descriptor set ``M`` and keeps
binding ``n`` when it is free; the register classes b, t, u and s share one binding namespace in Vulkan, so a
later class takes the next free binding. Resources are placed in class order, then register number, so the
slots do not depend on declaration order. Two resources that claim one register are rejected, as Direct3D
rejects them.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_hlsl_resources import HlslRegisterOverlapError

DECLARATIONS = [
    "cbuffer Params : register(b0) { float4 tint; };",
    "Texture2D<float4> albedo : register(t0);",
    "SamplerState linearSampler : register(s0);",
    "RWStructuredBuffer<float4> o : register(u0);",
    "Texture2D<float4> shadow : register(t2, space1);",
    "[[vk::binding(7, 2)]] StructuredBuffer<float> weights : register(t1);",
]

MAIN = """[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = tint * albedo.SampleLevel(linearSampler, float2(0.5, 0.5), 0)
        + shadow.Load(int3(0, 0, 0)) + weights[id.x];
}
"""

# b0 keeps binding 0; t0 and u0 find it taken and move to 1 and 2; s0 follows at 3. space1 is set 1, and
# vk::binding overrides the register.
EXPECTED_SLOTS = {
    "Params": (0, 0),
    "albedo": (0, 1),
    "o": (0, 2),
    "linearSampler": (0, 3),
    "shadow": (1, 2),
    "weights": (2, 7),
}


def _translate(tmp_path, source):
    path = tmp_path / "shader.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend="vulkan", source_backend="directx", format_output=False
    )


def _names(code):
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r'OpName (%\w+) "([^"]*)"', code)
    }


def _slots(code):
    names = _names(code)
    decorations = {}
    for match in re.finditer(r"OpDecorate (%\w+) (DescriptorSet|Binding) (\d+)", code):
        decorations.setdefault(match.group(1), {})[match.group(2)] = int(match.group(3))
    slots = {}
    for variable, decoration in decorations.items():
        name = names.get(variable)
        if name is None:
            raise AssertionError(f"Descriptor variable {variable} has no OpName")
        slots[name] = (decoration["DescriptorSet"], decoration["Binding"])
    return slots


def _validate(tmp_path, code):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    asm = tmp_path / "shader.spvasm"
    asm.write_text(code, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    subprocess.run(
        [assembler, "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        [validator, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_hlsl_registers_choose_descriptor_slots(tmp_path):
    code = _translate(tmp_path, "\n".join(DECLARATIONS) + "\n" + MAIN)
    assert _slots(code) == EXPECTED_SLOTS
    _validate(tmp_path, code)


def test_hlsl_register_slots_do_not_depend_on_declaration_order(tmp_path):
    forward = _slots(_translate(tmp_path, "\n".join(DECLARATIONS) + "\n" + MAIN))
    backward = _slots(
        _translate(tmp_path, "\n".join(reversed(DECLARATIONS)) + "\n" + MAIN)
    )
    assert forward == backward == EXPECTED_SLOTS


def test_resource_array_holds_consecutive_registers(tmp_path):
    source = """Texture2D<float4> layers[3] : register(t1);
Texture2D<float4> extra : register(t4);
SamplerState linearSampler : register(s1);
RWStructuredBuffer<float4> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = layers[2].SampleLevel(linearSampler, float2(0.5, 0.5), 0) + extra.Load(int3(0, 0, 0));
}
"""
    slots = _slots(_translate(tmp_path, source))
    # One binding holds the three descriptors of t1..t3; u0 keeps 0, and s1 finds 1 taken.
    assert slots == {
        "layers": (0, 1),
        "extra": (0, 4),
        "o": (0, 0),
        "linearSampler": (0, 2),
    }


@pytest.mark.parametrize(
    "declarations",
    [
        "Texture2D<float4> a : register(t0);\nTexture2D<float4> b : register(t0);",
        "Texture2D<float4> a[2] : register(t0);\nTexture2D<float4> b : register(t1);",
    ],
    ids=["same-register", "inside-array"],
)
def test_two_resources_on_one_register_are_rejected(tmp_path, declarations):
    source = f"""{declarations}
RWStructuredBuffer<float4> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    o[id.x] = a[0].Load(int3(0, 0, 0)) + b.Load(int3(0, 0, 0));
}}
"""
    if "a[2]" not in declarations:
        source = source.replace("a[0].Load", "a.Load")
    with pytest.raises(HlslRegisterOverlapError, match="both claim register t"):
        _translate(tmp_path, source)
