"""GLSL uniform blocks keep their std140 offsets on Metal and HLSL.

A GLSL uniform block is laid out with the std140 rules (GLSL 4.60 section 7.6.2.2) and the host writes it that way.
Metal's natural struct layout and HLSL constant-buffer packing put most members elsewhere (a ``vec3`` after a
``float``, any member after an array, a matrix or a struct), so a translated shader must reproduce the std140 bytes:
Metal through explicit padding and packed storage, HLSL through ``packoffset``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.buffer_member_layout import packoffset_bytes
from crosstl.translator.codegen.glsl_block_layout import (
    block_struct_placements,
    crossgl_layout_type,
    hlsl_packoffsets_for_block,
)
from crosstl.translator.codegen.hlsl_buffer_layout import HlslLayoutError

BASIC = """#version 450
layout(std140, binding = 2) uniform Params {
    float a;
    vec3 b;
    vec2 c;
    float arr[2];
    float after;
    mat2 m2;
    mat3x4 m34;
    float tail;
} params;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(params.a, params.b.x, params.c.y,
                 params.arr[1] + params.after + params.m2[1][0] + params.m34[2][3] + params.tail);
}
"""

# glslangValidator -l -q reflection of BASIC.
BASIC_OFFSETS = {
    "a": 0,
    "b": 16,
    "c": 32,
    "arr": 48,
    "after": 80,
    "m2": 96,
    "m34": 128,
    "tail": 176,
}

NESTED = """#version 450
struct Light {
    vec3 position;
    float radius;
    vec2 uv;
};
layout(binding = 0) uniform Lights {
    float count;
    Light lights[2];
    float extra;
    vec3 tint;
    float strength;
};
layout(location = 0) out vec4 color;
void main() {
    color = vec4(lights[1].position * lights[0].radius + tint * strength, count + extra + lights[1].uv.y);
}
"""

# HLSL packing would put extra at 72 (after the last element's uv), tint at 80 and strength at 92.
NESTED_OFFSETS = {"count": 0, "lights": 16, "extra": 80, "tint": 96, "strength": 108}
LIGHT_OFFSETS = {"position": 0, "radius": 12, "uv": 16}

QUALIFIED = """#version 450
layout(std140, binding = 1) uniform Qualified {
    layout(row_major) mat3 rotation;
    layout(offset = 64) float scale;
    vec2 shift;
};
layout(location = 0) in vec3 p;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(rotation * p * scale, shift.x);
}
"""

QUALIFIED_OFFSETS = {"rotation": 0, "scale": 64, "shift": 72}

MISMATCHED = """#version 450
struct Pair {
    float first;
    vec3 second;
};
layout(binding = 0) uniform Mismatched {
    Pair pair;
};
layout(location = 0) out vec4 color;
void main() {
    color = vec4(pair.second, pair.first);
}
"""


def _translate(tmp_path, source, backend, name="shader.frag"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, source_backend="opengl")


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _no_structs(_name):
    return None


def _glslang_offsets(tmp_path, source, block):
    """Member offsets of ``block`` from glslang's reflection, or ``None`` without glslang."""
    validator = shutil.which("glslangValidator")
    if validator is None:
        return None
    path = tmp_path / "reflect.frag"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [validator, "-l", "-q", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    reflection = result.stdout.split("Uniform block reflection:", 1)[0]
    offsets = {}
    # A struct member is reflected through its leaves (``lights.position``); its offset is the smallest one.
    for match in re.finditer(
        rf"^(?:{block}\.)?(\w+)[\w.\[\]]*: offset (\d+)", reflection, re.M
    ):
        name, offset = match.group(1), int(match.group(2))
        offsets[name] = min(offset, offsets.get(name, offset))
    return offsets


def _hlsl_packoffsets(code, block):
    body = re.search(rf"cbuffer {block}\b[^{{]*\{{(.*?)\}};", code, re.S)
    assert body is not None, code
    offsets = {}
    for match in re.finditer(
        r"(\w+)(?:\[\d+\])?\s*:\s*packoffset\(([^)]*)\)", body.group(1)
    ):
        offsets[match.group(1)] = packoffset_bytes(match.group(2))
    return offsets


def _metal_offsets(tmp_path, code, checks):
    """Compile ``code`` with static assertions on member offsets and struct sizes."""
    if not shutil.which("xcrun"):
        pytest.skip("the Metal toolchain is not available")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not available")
    lines = []
    for struct, offsets in checks.items():
        for member, offset in offsets.items():
            if member == "sizeof":
                lines.append(
                    f'static_assert(sizeof({struct}) == {offset}, "{struct}");'
                )
            else:
                lines.append(
                    f"static_assert(__builtin_offsetof({struct}, {member}) == {offset}, "
                    f'"{struct}.{member}");'
                )
    path = tmp_path / "offsets.metal"
    path.write_text(code + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "offsets.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def test_std140_rules_match_glslang(tmp_path):
    members = [
        ("a", "float"),
        ("b", "vec3"),
        ("c", "vec2"),
        ("arr", "float[2]"),
        ("after", "float"),
        ("m2", "mat2"),
        ("m34", "mat3x4"),
        ("tail", "float"),
    ]
    block = crossgl_layout_type(
        "Params",
        lambda name: (
            [(n, t, False, None, None) for n, t in members]
            if name == "Params"
            else None
        ),
    )
    placements, size = block_struct_placements(block, "std140", root=True)
    assert {p.name: p.offset for p in placements} == BASIC_OFFSETS
    assert size == 180
    reflected = _glslang_offsets(tmp_path, BASIC, "Params")
    if reflected is not None:
        assert reflected == BASIC_OFFSETS


def test_std140_nested_struct_and_qualifiers_match_glslang(tmp_path):
    structs = {
        "Light": [
            ("position", "vec3", False, None, None),
            ("radius", "float", False, None, None),
            ("uv", "vec2", False, None, None),
        ],
        "Lights": [
            ("count", "float", False, None, None),
            ("lights", "Light[2]", False, None, None),
            ("extra", "float", False, None, None),
            ("tint", "vec3", False, None, None),
            ("strength", "float", False, None, None),
        ],
        "Qualified": [
            ("rotation", "mat3", True, None, None),
            ("scale", "float", False, 64, None),
            ("shift", "vec2", False, None, None),
        ],
    }
    for block, expected, source in (
        ("Lights", NESTED_OFFSETS, NESTED),
        ("Qualified", QUALIFIED_OFFSETS, QUALIFIED),
    ):
        layout_type = crossgl_layout_type(block, structs.get)
        placements, _size = block_struct_placements(layout_type, "std140", root=True)
        assert {p.name: p.offset for p in placements} == expected
        reflected = _glslang_offsets(tmp_path, source, block)
        if reflected is not None:
            assert {name: reflected[name] for name in expected} == expected


def test_std140_offset_qualifier_must_respect_alignment():
    block = crossgl_layout_type(
        "Bad",
        lambda name: (
            [("v", "vec4", False, None, None), ("w", "vec3", False, 20, None)]
            if name == "Bad"
            else None
        ),
    )
    with pytest.raises(
        HlslLayoutError, match="not a multiple of its std140 alignment 16"
    ):
        block_struct_placements(block, "std140", root=True)


def test_hlsl_packoffsets_are_skipped_when_packing_agrees():
    block = crossgl_layout_type(
        "Agree",
        lambda name: (
            [
                ("m", "mat4", False, None, None),
                ("v", "vec4", False, None, None),
                ("s", "float", False, None, None),
            ]
            if name == "Agree"
            else None
        ),
    )
    assert hlsl_packoffsets_for_block(block, "std140") is None


def test_glsl_import_marks_uniform_blocks_std140(tmp_path):
    crossgl = _flat(_translate(tmp_path, BASIC, "cgl"))
    assert re.search(
        r"buffer_layout\(std140\).*cbuffer Params|cbuffer Params[^{]*buffer_layout\(std140\)",
        crossgl,
    )
    qualified = _flat(_translate(tmp_path, QUALIFIED, "cgl"))
    assert re.search(
        r"@ ?row_major mat3 rotation|mat3 rotation @ ?row_major", qualified
    )
    assert re.search(
        r"float scale @ ?offset\(64\)|@ ?offset\(64\) float scale", qualified
    )


@pytest.mark.parametrize(
    ("source", "block", "expected"),
    (
        (BASIC, "Params", BASIC_OFFSETS),
        (NESTED, "Lights", NESTED_OFFSETS),
        (QUALIFIED, "Qualified", QUALIFIED_OFFSETS),
    ),
    ids=("basic", "nested", "qualified"),
)
def test_hlsl_places_members_at_std140_offsets(tmp_path, source, block, expected):
    code = _translate(tmp_path, source, "directx")
    assert _hlsl_packoffsets(code, block) == expected


@pytest.mark.parametrize(
    ("source", "block", "expected"),
    (
        (BASIC, "Params", BASIC_OFFSETS),
        (NESTED, "Lights", NESTED_OFFSETS),
        (QUALIFIED, "Qualified", QUALIFIED_OFFSETS),
    ),
    ids=("basic", "nested", "qualified"),
)
def test_dxc_reports_the_std140_offsets(tmp_path, source, block, expected):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not available")
    path = tmp_path / "shader.hlsl"
    path.write_text(_translate(tmp_path, source, "directx"), encoding="utf-8")
    listing = tmp_path / "shader.txt"
    result = subprocess.run(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(path), "-Fc", str(listing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    text = listing.read_text(encoding="utf-8")
    body = text[text.index(f"cbuffer {block}") :]
    body = body[: body.index(f"}} {block};")]
    # Top-level members are indented by seven columns in DXC's resource listing; struct members by eleven.
    offsets = {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r"^;       (?:\w+ )+?(\w+)(?:\[\d+\])?;+\s+; Offset:\s+(\d+)", body, re.M
        )
    }
    for match in re.finditer(
        r"^;       \}\s*(\w+)(?:\[\d+\])?;+\s+; Offset:\s+(\d+)", body, re.M
    ):
        offsets[match.group(1)] = int(match.group(2))
    assert offsets == expected


def test_hlsl_keeps_row_major_on_std140_matrices(tmp_path):
    code = _flat(_translate(tmp_path, QUALIFIED, "directx"))
    assert re.search(r"row_major float3x3 rotation : packoffset\(c0\)", code)


def test_hlsl_loads_structs_that_std140_lays_out_differently(tmp_path):
    """HLSL packs ``Pair.second`` at byte 4; std140 aligns it to 16, so the block is read from registers."""
    code = _flat(_translate(tmp_path, MISMATCHED, "directx"))
    assert re.search(
        r"cbuffer Mismatched : register\(b0\) \{ uint4 __crossgl_Mismatched_registers\[2\]; \};",
        code,
    ), code
    assert "pair.first = asfloat(__crossgl_Mismatched_registers[0].x);" in code
    assert "pair.second = asfloat(__crossgl_Mismatched_registers[1].xyz);" in code
    assert re.search(
        r"PSMain\(\) : SV_Target0 \{ __crossgl_load_Mismatched\(\);", code
    ), code


@pytest.mark.parametrize(
    ("source", "checks"),
    (
        (BASIC, {"Params": {**BASIC_OFFSETS, "sizeof": 192}}),
        (NESTED, {"Lights": NESTED_OFFSETS, "Light": {**LIGHT_OFFSETS, "sizeof": 32}}),
        (QUALIFIED, {"Qualified": QUALIFIED_OFFSETS}),
        (
            MISMATCHED,
            {
                "Mismatched": {"pair": 0},
                "Pair": {"first": 0, "second": 16, "sizeof": 32},
            },
        ),
    ),
    ids=("basic", "nested", "qualified", "mismatched-for-hlsl"),
)
def test_metal_places_members_at_std140_offsets(tmp_path, source, checks):
    code = _translate(tmp_path, source, "metal")
    _metal_offsets(tmp_path, code, checks)


def test_metal_reads_std140_arrays_and_matrices(tmp_path):
    flat = _flat(_translate(tmp_path, BASIC, "metal"))
    # float[2] has a 16-byte std140 stride: each element is the first word of a float4.
    assert re.search(r"params\.arr\[1\]\.x", flat)
    # mat2 columns sit 16 bytes apart: the register words are read back as columns.
    assert "crosstl_std140_float2x2_load(params.m2)" in flat or re.search(
        r"_load\(params\.m2\)", flat
    )


def test_metal_reads_std140_row_major_matrices_as_rows(tmp_path):
    flat = _flat(_translate(tmp_path, QUALIFIED, "metal"))
    assert re.search(r"transpose\((?:\w+\.)?rotation\)", flat)
