"""HLSL output numbers implicit ``t``, ``u`` and ``b`` registers in one slot sequence.

Every other CrossTL target binds buffers, textures and images in declaration order across classes (SPIR-V and
Vulkan GLSL bindings, Metal buffer indices), and DXC's SPIR-V back end maps register ``tN``/``uN``/``bN`` of a space
to binding N. Counting each HLSL class from 0 put ``readonly buffer fin; buffer outp;`` at ``t0`` and ``u0``, two
descriptors at binding 0. Samplers keep their own sequence.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.directx_values import (
    _DirectXResourceRegisterAllocator,
)
from crosstl.translator.codegen.hlsl_register_slots import (
    SharedResourceSlots,
    next_shared_resource_slot,
)

READ_ONLY_THEN_WRITABLE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        readonly buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = fin[i] * 2.0;
        }
    }
}
"""

MIXED = """
shader Mix {
    sampler2D colorMap;
    readonly buffer float* weights;
    image2D outImage;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            ivec2 p = ivec2(gl_GlobalInvocationID.xy);
            vec4 c = textureLod(colorMap, vec2(0.5, 0.5), 0.0) * weights[p.x];
            imageStore(outImage, p, c);
        }
    }
}
"""

EXPLICIT = """
shader Explicit {
    sampler2D autoTexture;
    RWStructuredBuffer<int> taken @register(u1);
    RWStructuredBuffer<int> autoValues;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec4 c = textureLod(autoTexture, vec2(0.5, 0.5), 0.0);
            buffer_store(autoValues, i, buffer_load(taken, i) + int(c.x));
        }
    }
}
"""

REGISTER = re.compile(r"^\S.*?\b(\w+)(?:\[[^\]]*\])? : register\((\w+)(?:, (space\d+))?\)", re.M)


def _registers(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    code = crosstl.translate(str(path), backend="directx")
    return code, {name: register for name, register, _ in REGISTER.findall(code)}


def _spirv_bindings(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "probe.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", "-spirv", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return {
        name: int(binding)
        for name, binding in re.findall(
            r"OpDecorate %(\w+) Binding (\d+)", result.stdout
        )
    }


def test_read_only_and_writable_buffers_take_consecutive_slots(tmp_path):
    code, registers = _registers(tmp_path, READ_ONLY_THEN_WRITABLE)
    assert registers == {"fin": "t0", "outp": "u1"}
    assert _spirv_bindings(tmp_path, code) == {"fin": 0, "outp": 1}


def test_textures_buffers_and_images_follow_declaration_order(tmp_path):
    code, registers = _registers(tmp_path, MIXED)
    assert registers == {
        "colorMap": "t0",
        "colorMapSampler": "s0",
        "weights": "t1",
        "outImage": "u2",
    }
    bindings = _spirv_bindings(tmp_path, code)
    assert [bindings[name] for name in ("colorMap", "weights", "outImage")] == [
        0,
        1,
        2,
    ]


def test_implicit_slot_skips_an_explicit_register_of_any_class(tmp_path):
    _, registers = _registers(tmp_path, EXPLICIT)
    assert registers["autoTexture"] == "t0"
    assert registers["taken"] == "u1"
    # Slot 1 is taken by u1, so the next implicit resource takes slot 2 in every class.
    assert registers["autoValues"] == "u2"


def test_shared_slot_must_be_free_in_every_shared_class():
    allocator = _DirectXResourceRegisterAllocator()
    allocator.add_allocation(("u", 0), 1, 2, "taken", "explicit")
    allocator.add_allocation(("b", 0), 3, 3, "block", "explicit")
    cursors = SharedResourceSlots()
    assert next_shared_resource_slot(allocator, "t", cursors, 0, 0, 1) == 0
    assert next_shared_resource_slot(allocator, "t", cursors, 0, 1, 1) == 4
    assert next_shared_resource_slot(allocator, "t", cursors, 0, 0, 2) == 4
    # Samplers and unshared cursors count only their own class.
    assert next_shared_resource_slot(allocator, "s", cursors, 0, 1, 1) == 1
    assert next_shared_resource_slot(allocator, "t", {}, 0, 1, 1) == 1
