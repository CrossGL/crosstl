"""HLSL's Shader Model 6.10 preview intrinsics ``DebugBreak`` and ``TriangleObjectPositions``.

DXC's preview intrinsic table (utils/hct/gen_intrin_main.txt) declares ``void DebugBreak()``, a breakpoint for any
stage, and ``BuiltInTrianglePositions TriangleObjectPositions()``, the object-space vertices of the hit triangle in
the any-hit and closest-hit stages. CrossGL carries both calls under their HLSL names, so HLSL imports them and HLSL
output writes them back. Every other target, Slang included (slangc 2026.20 declares neither name), refuses the call
with a typed diagnostic instead of writing it verbatim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics.builtin_function_names import (
    HLSL_FUNCTION_NAMES,
    HLSL_INTRINSIC_FUNCTION_NAMES,
    HLSL_PREVIEW_FUNCTION_NAMES,
)

INVENTORIES = Path(__file__).resolve().parents[1] / "conformance" / "inventory" / "data"

HLSL_DEBUG_BREAK = """RWStructuredBuffer<uint> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (o[id.x] == 7u) { DebugBreak(); }
    o[id.x] = 1u;
}
"""

CROSSGL_DEBUG_BREAK = """shader main {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* o;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (o[i] == 7u) { DebugBreak(); }
            o[i] = 1u;
        }
    }
}
"""

HLSL_TRIANGLE_POSITIONS = """struct [raypayload] Payload
{
    float4 color : read(caller) : write(anyhit, closesthit);
};
RWStructuredBuffer<float3> outp : register(u0);
[shader("closesthit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    BuiltInTrianglePositions t = TriangleObjectPositions();
    outp[0] = t.p0 + t.p1 + t.p2;
}
"""

# The targets that cannot write a breakpoint, each with the refusal it raises.
_DEBUG_BREAK_REFUSING_TARGETS = (
    "metal",
    "opengl",
    "wgsl",
    "cuda",
    "hip",
    "slang",
    "rust",
    "mojo",
)


def _translate(tmp_path, name, source, backend):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_preview_intrinsics_are_hlsl_function_names():
    assert HLSL_PREVIEW_FUNCTION_NAMES == frozenset(
        {"DebugBreak", "TriangleObjectPositions"}
    )
    assert HLSL_PREVIEW_FUNCTION_NAMES <= HLSL_FUNCTION_NAMES
    assert HLSL_PREVIEW_FUNCTION_NAMES <= HLSL_INTRINSIC_FUNCTION_NAMES


def test_preview_intrinsics_are_absent_from_the_stable_inventory():
    # The set holds only what the DXC 1.9 inventory lacks; fold a name into the main list once the inventory has it.
    inventory = json.loads((INVENTORIES / "hlsl.json").read_text(encoding="utf-8"))[
        "items"
    ]
    names = {
        re.split(r"::|\.", item["name"].split("(")[0])[-1].strip()
        for item in inventory
        if item.get("category") == "builtin-function"
    }
    assert not HLSL_PREVIEW_FUNCTION_NAMES & names


def test_hlsl_input_imports_debug_break(tmp_path):
    crossgl = _translate(tmp_path, "probe.hlsl", HLSL_DEBUG_BREAK, "cgl")
    assert "DebugBreak();" in crossgl


def test_hlsl_round_trip_keeps_debug_break(tmp_path):
    hlsl = _translate(tmp_path, "probe.hlsl", HLSL_DEBUG_BREAK, "directx")
    assert "DebugBreak();" in hlsl
    assert "[numthreads(1, 1, 1)]" in hlsl


def test_hlsl_output_writes_crossgl_debug_break(tmp_path):
    hlsl = _translate(tmp_path, "probe.cgl", CROSSGL_DEBUG_BREAK, "directx")
    assert "DebugBreak();" in hlsl


@pytest.mark.parametrize("backend", _DEBUG_BREAK_REFUSING_TARGETS)
def test_other_targets_refuse_debug_break(tmp_path, backend):
    with pytest.raises(TranslationDiagnostic, match="DebugBreak"):
        _translate(tmp_path, "probe.cgl", CROSSGL_DEBUG_BREAK, backend)


def test_spirv_refuses_debug_break(tmp_path):
    from crosstl.translator.codegen.spirv_values import SPIRVBuiltinLoweringError

    with pytest.raises(SPIRVBuiltinLoweringError, match="DebugBreak"):
        _translate(tmp_path, "probe.cgl", CROSSGL_DEBUG_BREAK, "vulkan")


def test_hlsl_round_trip_keeps_triangle_object_positions(tmp_path):
    crossgl = _translate(tmp_path, "probe.hlsl", HLSL_TRIANGLE_POSITIONS, "cgl")
    assert "TriangleObjectPositions()" in crossgl
    hlsl = _translate(tmp_path, "probe.hlsl", HLSL_TRIANGLE_POSITIONS, "directx")
    assert "BuiltInTrianglePositions t = TriangleObjectPositions();" in hlsl
    assert "outp[0] = ((t.p0 + t.p1) + t.p2);" in hlsl


@pytest.mark.parametrize("backend", ("metal", "opengl", "slang"))
def test_other_targets_refuse_triangle_object_positions(tmp_path, backend):
    with pytest.raises(TranslationDiagnostic, match="TriangleObjectPositions"):
        _translate(tmp_path, "probe.hlsl", HLSL_TRIANGLE_POSITIONS, backend)
