"""Each SPIR-V entry point has its own interface locations.

Two fragment functions of one Metal library that return the same colour struct both write ``Location 0``, which
SPIR-V allows because each ``OpEntryPoint`` lists its own variables. The generator tracked locations per execution
model across the module and refused the second function ("Duplicate SPIR-V output location 0").
"""

import re
import shutil
import subprocess

import pytest

import crosstl

LIBRARY = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    half3 color [[attribute(1)]];
};

struct Varyings {
    float4 position [[position]];
    half3 color;
};

struct FragData {
    half4 color [[color(0)]];
};

vertex Varyings plain_vertex(VertexIn in [[stage_in]]) {
    Varyings out;
    out.position = float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

vertex Varyings scaled_vertex(VertexIn in [[stage_in]]) {
    Varyings out;
    out.position = float4(in.position * 0.5, 1.0);
    out.color = in.color;
    return out;
}

fragment FragData plain(Varyings in [[stage_in]]) {
    return FragData{half4(in.color, 1.0h)};
}

fragment FragData dimmed(Varyings in [[stage_in]]) {
    return FragData{half4(in.color * 0.5h, 1.0h)};
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "library.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan")


def _locations(spirv):
    return {
        variable: int(location)
        for variable, location in re.findall(r"OpDecorate (%\w+) Location (\d+)", spirv)
    }


def _interfaces(spirv):
    return {
        name: interface.split()
        for name, interface in re.findall(
            r'OpEntryPoint \w+ %\w+ "(\w+)"((?: %\w+)*)', spirv
        )
    }


def test_entry_points_of_one_stage_reuse_their_locations(tmp_path):
    spirv = _translate(tmp_path, LIBRARY)
    locations = _locations(spirv)
    interfaces = _interfaces(spirv)
    assert set(interfaces) == {"plain_vertex", "scaled_vertex", "plain", "dimmed"}
    for entry, variables in interfaces.items():
        located = [locations[v] for v in variables if v in locations]
        # Within one interface the locations stay distinct per storage class.
        assert located, entry
    # The formatter may renumber ids, so the interface variables are told apart by their OpName.
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', spirv))
    for entry in ("plain", "dimmed"):
        outputs = [v for v in interfaces[entry] if "output" in names.get(v, v)]
        assert [locations[v] for v in outputs] == [0], (entry, outputs, names)
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        return
    assembly, module = tmp_path / "library.spvasm", tmp_path / "library.spv"
    assembly.write_text(spirv, encoding="utf-8")
    for command in (
        [spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(module)],
        [spirv_val, "--target-env", "vulkan1.1", str(module)],
    ):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_one_interface_still_refuses_a_shared_location(tmp_path):
    source = LIBRARY.replace(
        "struct FragData {\n    half4 color [[color(0)]];\n};",
        "struct FragData {\n    half4 color [[color(0)]];\n    half4 glow [[color(0)]];\n};",
    ).replace(
        "return FragData{half4(in.color, 1.0h)};",
        "return FragData{half4(in.color, 1.0h), half4(1.0h)};",
    )
    with pytest.raises(ValueError, match="Duplicate SPIR-V output location 0"):
        _translate(tmp_path, source)
