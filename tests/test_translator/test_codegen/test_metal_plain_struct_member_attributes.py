"""Metal attributes on members of structs that no entry passes as stage I/O.

A struct read through a device or constant pointer still carries its members' attributes. Two of them used to break:

- ``[[attribute(n)]]`` imports as a CrossGL ``@location(n)``. Outside a ``[[stage_in]]`` struct it was written back
  as ``[[location]]``, which the Metal compiler ignores as an unknown attribute, and the fail-closed path then refused
  the whole unit. A vertex input location is the member's attribute index (MSL 5.2.3.1), so it is written as
  ``[[attribute(n)]]`` wherever the struct is declared.
- ``[[id(n)]]`` on a value array member was written after the extent, where it appertains to the array type and the
  Metal compiler rejects it ("'id' attribute cannot be applied to types").
"""

import shutil
import subprocess

import pytest

import crosstl

LOCATION_SOURCE = """#include <metal_stdlib>
using namespace metal;
enum VertexAttribute { VertexAttributePosition = 4 };
struct VertexPosition { float3 position [[attribute(VertexAttributePosition)]]; };
struct VertexOut { float4 position [[position]]; };
vertex VertexOut vertex_main(const device VertexPosition *vertices [[buffer(0)]], uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = float4(vertices[vid].position, 1.0);
    return out;
}
"""

ID_SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Args {
    array<texture2d<float>, 2> textures [[id(0)]];
    array<device float *, 3> buffers [[id(2)]];
    array<uint, 3> constants [[id(5)]];
};
fragment float4 fragment_main(device Args &args [[buffer(0)]]) {
    return float4(args.buffers[1][0] + float(args.constants[2]));
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path), backend="metal", format_output=False, source_backend="metal"
    )


def _compile_if_available(tmp_path, source):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("the Metal compiler needs xcrun")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    path = tmp_path / "translated.metal"
    path.write_text(source)
    completed = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", str(path)]
        + ["-o", str(tmp_path / "translated.air")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_a_location_outside_stage_io_keeps_its_attribute_index(tmp_path):
    metal = _translate(tmp_path, "location.metal", LOCATION_SOURCE)
    assert "float3 position [[attribute(4)]];" in metal
    assert "[[location" not in metal


def test_a_location_outside_stage_io_compiles(tmp_path):
    _compile_if_available(
        tmp_path, _translate(tmp_path, "location.metal", LOCATION_SOURCE)
    )


def test_a_value_array_member_id_follows_the_declarator_name(tmp_path):
    metal = _translate(tmp_path, "ids.metal", ID_SOURCE)
    assert "uint constants [[id(5)]][3];" in metal
    assert "device float* buffers [[id(2)]][3];" in metal
    assert "] [[id(" not in metal


def test_a_value_array_member_id_compiles(tmp_path):
    _compile_if_available(tmp_path, _translate(tmp_path, "ids.metal", ID_SOURCE))
