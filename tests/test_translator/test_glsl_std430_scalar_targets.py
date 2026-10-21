"""GLSL std430 and scalar blocks (Vulkan push constants, GL_EXT_scalar_block_layout) keep their offsets.

A push-constant block uses std430 unless it names std140 or scalar (GL_KHR_vulkan_glsl), and a block that names
``scalar`` packs every value at the alignment of its components, with no padding reserved after an array or a
struct. The expected offsets are glslang's reflection of each source. Metal offsets are checked with static
assertions under ``xcrun metal``, SPIR-V offsets with ``spirv-val``, and DXC places HLSL ``packoffset``s.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.buffer_member_layout import packoffset_bytes
from crosstl.translator.codegen.glsl_block_layout import (
    block_array_stride,
    block_struct_placements,
    crossgl_layout_type,
)

STD430_PUSH = """#version 450
struct Item { float x; vec3 y; };
layout(push_constant) uniform Push {
    vec3 tint;
    float scale;
    float weights[3];
    vec2 uv;
    mat3 rot;
    mat3x2 skew;
    vec3 dirs[2];
    Item item;
    int flags;
} pc;
layout(location = 0) in vec3 inPos;
layout(location = 0) out vec3 outColor;
void main() {
    outColor = pc.tint * pc.weights[2] + pc.dirs[1] + pc.item.y * pc.item.x
        + vec3(pc.uv, float(pc.flags)) + vec3(pc.skew[2], 0.0);
    gl_Position = vec4(pc.rot * inPos * pc.scale, 1.0);
}
"""
STD430_PUSH_OFFSETS = {
    "tint": 0,
    "scale": 12,
    "weights": 16,
    "uv": 32,
    "rot": 48,
    "skew": 96,
    "dirs": 128,
    "item": 160,
    "flags": 192,
}

SCALAR_PUSH = """#version 450
#extension GL_EXT_scalar_block_layout : require
struct Pair { vec2 a; float b; };
layout(push_constant, scalar) uniform Push {
    float f;
    vec3 v;
    mat3 m;
    mat2 m2;
    float arr[3];
    vec3 va[2];
    Pair p[2];
    float after;
} pc;
layout(location = 0) out vec4 outColor;
void main() {
    outColor = vec4(pc.m * pc.v * pc.f + pc.va[1] + vec3(pc.m2[1], pc.arr[2])
        + vec3(pc.p[1].a, pc.p[1].b + pc.after), 1.0);
}
"""
SCALAR_PUSH_OFFSETS = {
    "f": 0,
    "v": 4,
    "m": 16,
    "m2": 52,
    "arr": 68,
    "va": 80,
    "p": 104,
    "after": 128,
}

# The scalar layout reserves no padding after a struct or the last element of an array.
SCALAR_TAIL = """#version 450
#extension GL_EXT_scalar_block_layout : require
struct V { double d; float f; };
struct N { V v; float g; };
layout(scalar, binding = 0) uniform S { float x; V v; float after; N n[2]; float last; } s;
layout(location = 0) out vec4 o;
void main() { o = vec4(s.x + s.after + float(s.v.d) + s.n[1].g + s.last); }
"""
SCALAR_TAIL_OFFSETS = {"x": 0, "v": 8, "after": 20, "n": 24, "last": 56}

STD430_UNIFORM = """#version 450
#extension GL_EXT_scalar_block_layout : require
struct U { double d; float f; };
struct W { vec2 a; float b; };
layout(std430, binding = 1) uniform F {
    float x; U u[2]; float y; W w; float z; mat2x3 m; float q; mat3x2 n; float r; vec2 v2[3];
} f;
layout(location = 0) out vec4 o;
void main() {
    o = vec4(f.x + float(f.u[1].d) + f.y + f.w.b + f.z + f.m[1].z + f.q + f.n[2].y + f.r + f.v2[2].y);
}
"""
STD430_UNIFORM_OFFSETS = {
    "x": 0,
    "u": 8,
    "y": 40,
    "w": 48,
    "z": 64,
    "m": 80,
    "q": 112,
    "n": 120,
    "r": 144,
    "v2": 152,
}

# Every member of this std430 block sits where HLSL packing can put it with packoffset.
STD430_HLSL_PLACEABLE = """#version 450
layout(push_constant) uniform Push {
    float f;
    vec3 v;
    vec4 color;
    mat4 mvp;
    vec2 uv;
    float tail;
} pc;
layout(location = 0) out vec4 outColor;
void main() { outColor = pc.mvp * pc.color * pc.f + vec4(pc.v, pc.tail) + vec4(pc.uv, 0.0, 0.0); }
"""
STD430_HLSL_PLACEABLE_OFFSETS = {
    "f": 0,
    "v": 16,
    "color": 32,
    "mvp": 48,
    "uv": 112,
    "tail": 120,
}

# A std430 block with std140 offsets keeps a plain OpenGL uniform block.
STD430_AS_STD140 = """#version 450
layout(push_constant) uniform Push { mat4 mvp; vec4 color; vec3 dir; float scale; } pc;
layout(location = 0) out vec4 outColor;
void main() { outColor = pc.mvp * pc.color + vec4(pc.dir * pc.scale, 0.0); }
"""


def _translate(tmp_path, source, backend, name="shader.frag"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, source_backend="opengl")


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _stage_name(source):
    return "shader.vert" if "gl_Position" in source else "shader.frag"


def _glslang_offsets(tmp_path, source, block):
    """Member offsets of ``block`` in glslang's SPIR-V, in member order, or ``None`` without the tools."""
    validator, disassembler = shutil.which("glslangValidator"), shutil.which(
        "spirv-dis"
    )
    if validator is None or disassembler is None:
        return None
    path = tmp_path / ("reflect." + _stage_name(source).split(".")[1])
    path.write_text(source, encoding="utf-8")
    binary = tmp_path / "reflect.spv"
    result = subprocess.run(
        [validator, "-V", str(path), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    listing = subprocess.run(
        [disassembler, str(binary)], capture_output=True, text=True, check=True
    ).stdout
    # Reflection lists only the members a shader uses; the decorations place every member.
    return [
        int(match.group(2))
        for match in re.finditer(
            rf"OpMemberDecorate %{block} (\d+) Offset (\d+)", listing
        )
    ]


def _metal_compile(tmp_path, code, checks):
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
            str(tmp_path / "a"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def _crossgl_block(source, block_name, layout):
    """The layout type of ``block_name`` built from its GLSL declaration (no nested qualifiers)."""
    body = re.search(rf"{block_name}\s*\{{(.*?)\}}", source, re.S).group(1)
    structs = {
        match.group(1): [
            (member.split()[-1], " ".join(member.split()[:-1]), False, None, None)
            for member in (m.strip() for m in match.group(2).split(";"))
            if member
        ]
        for match in re.finditer(r"struct (\w+) \{(.*?)\};", source, re.S)
    }
    members = []
    for declaration in (m.strip() for m in body.split(";")):
        if not declaration:
            continue
        type_name, name = declaration.split()
        array = re.fullmatch(r"(\w+)\[(\d+)\]", name)
        if array is not None:
            name, type_name = array.group(1), f"{type_name}[{array.group(2)}]"
        members.append((name, type_name, False, None, None))
    from crosstl.translator.codegen.glsl_block_layout import crossgl_struct_layout_type

    return crossgl_struct_layout_type(block_name, members, structs.get)


@pytest.mark.parametrize(
    ("source", "block", "layout", "expected"),
    (
        (STD430_PUSH, "Push", "std430", STD430_PUSH_OFFSETS),
        (SCALAR_PUSH, "Push", "scalar", SCALAR_PUSH_OFFSETS),
        (SCALAR_TAIL, "S", "scalar", SCALAR_TAIL_OFFSETS),
        (STD430_UNIFORM, "F", "std430", STD430_UNIFORM_OFFSETS),
        (STD430_HLSL_PLACEABLE, "Push", "std430", STD430_HLSL_PLACEABLE_OFFSETS),
    ),
    ids=("std430-push", "scalar-push", "scalar-tail", "std430-uniform", "placeable"),
)
def test_block_rules_match_glslang(tmp_path, source, block, layout, expected):
    layout_type = _crossgl_block(source, block, layout)
    placements, _size = block_struct_placements(layout_type, layout, root=True)
    assert {p.name: p.offset for p in placements} == expected
    reflected = _glslang_offsets(tmp_path, source, block)
    if reflected is not None:
        assert reflected == list(expected.values())


def test_scalar_array_stride_rounds_the_struct_to_its_alignment():
    element = crossgl_layout_type(
        "V[2]",
        {
            "V": [("d", "double", False, None, None), ("f", "float", False, None, None)]
        }.get,
    )
    # struct V takes 12 bytes, but its 8-byte alignment makes the array stride 16 (glslang: ArrayStride 16).
    assert block_array_stride(element, "scalar") == 16
    assert block_array_stride(element, "std430") == 16
    assert block_array_stride(crossgl_layout_type("vec3[2]", {}.get), "scalar") == 12
    assert block_array_stride(crossgl_layout_type("vec3[2]", {}.get), "std430") == 16
    assert block_array_stride(crossgl_layout_type("float[2]", {}.get), "std430") == 4


@pytest.mark.parametrize(
    ("qualifier", "layout"),
    (("", "std430"), (", std140", "std140"), (", scalar", "scalar")),
    ids=("default", "std140", "scalar"),
)
def test_glsl_import_marks_push_constant_layout(tmp_path, qualifier, layout):
    source = (
        "#version 450\n"
        + (
            "#extension GL_EXT_scalar_block_layout : require\n"
            if layout == "scalar"
            else ""
        )
        + f"layout(push_constant{qualifier}) uniform Push {{ float a; float b[2]; }} pc;\n"
        + "layout(location = 0) out vec4 o;\n"
        + "void main() { o = vec4(pc.a + pc.b[1]); }\n"
    )
    crossgl = _translate(tmp_path, source, "cgl")
    assert f"cbuffer Push @push_constant @buffer_layout({layout}) {{" in crossgl


@pytest.mark.parametrize(
    ("source", "checks"),
    (
        (
            STD430_PUSH,
            {
                "Push": STD430_PUSH_OFFSETS,
                "Item": {"x": 0, "y": 16, "sizeof": 32},
            },
        ),
        (SCALAR_PUSH, {"Push": SCALAR_PUSH_OFFSETS, "Pair": {"b": 8, "sizeof": 12}}),
        (
            STD430_HLSL_PLACEABLE,
            {"Push": STD430_HLSL_PLACEABLE_OFFSETS},
        ),
    ),
    ids=("std430", "scalar", "placeable"),
)
def test_metal_places_members_at_block_offsets(tmp_path, source, checks):
    code = _translate(tmp_path, source, "metal", _stage_name(source))
    _metal_compile(tmp_path, code, checks)


def test_metal_reads_scalar_matrices_and_packed_vectors(tmp_path):
    flat = _flat(_translate(tmp_path, SCALAR_PUSH, "metal"))
    # mat3 columns are 12 bytes apart: the words are loaded into a native matrix.
    assert re.search(r"crosstl_scalar_float3x3_load\(\w+\.m\)", flat)
    # Packed vectors are read as natural vectors, which matrix products accept.
    assert re.search(r"float3\(\w+\.v\)", flat)
    assert re.search(r"float3\(\w+\.va\[1\]\)", flat)


def test_metal_keeps_std430_matrices_native(tmp_path):
    code = _translate(tmp_path, STD430_PUSH, "metal", "shader.vert")
    body = re.search(r"struct Push \{(.*?)\};", code, re.S).group(1)
    assert re.search(r"\bfloat3x3 rot;", body)
    assert re.search(r"\bfloat3x2 skew;", body)
    assert re.search(r"\bfloat3 dirs\[2\];", body)
    assert re.search(r"\bfloat weights\[3\];", body)


def _hlsl_packoffsets(code, block):
    body = re.search(rf"cbuffer {block}\b[^{{]*\{{(.*?)\}};", code, re.S)
    assert body is not None, code
    return {
        match.group(1): packoffset_bytes(match.group(2))
        for match in re.finditer(
            r"(\w+)(?:\[\d+\])?\s*:\s*packoffset\(([^)]*)\)", body.group(1)
        )
    }


def test_hlsl_places_std430_members_with_packoffset(tmp_path):
    code = _translate(tmp_path, STD430_HLSL_PLACEABLE, "directx")
    assert _hlsl_packoffsets(code, "Push") == STD430_HLSL_PLACEABLE_OFFSETS


def test_dxc_accepts_std430_packoffsets(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not available")
    path = tmp_path / "shader.hlsl"
    path.write_text(_translate(tmp_path, STD430_HLSL_PLACEABLE, "directx"))
    listing = tmp_path / "shader.txt"
    result = subprocess.run(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(path), "-Fc", str(listing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    text = listing.read_text(encoding="utf-8")
    body = text[text.index("cbuffer Push") :]
    body = body[: body.index("} Push;")]
    offsets = {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r"^;       (?:\w+ )+?(\w+)(?:\[\d+\])?;+\s+; Offset:\s+(\d+)", body, re.M
        )
    }
    assert offsets == STD430_HLSL_PLACEABLE_OFFSETS


def _hlsl_register_word_offsets(code, block, member):
    """Byte offset the HLSL loader reads ``member``'s first word from (register components and word reads)."""
    match = re.search(
        rf"\b{member}(?:\[\w+\])?(?:\._m00\w*)? = [^;]*?"
        rf"(?:__crossgl_{block}_registers\[(?:\w+(?: \* \d+)? \+ )?(\d+)\]\.([xyzw])"
        rf"|__crossgl_{block}_registers\[\w+\]\.([xyzw])"
        rf"|__crossgl_{block}_word\(\(\w+(?: \* \d+)?\) \+ (\d+)\))",
        code,
    )
    assert match is not None, (member, code)
    if match.group(4) is not None:
        return int(match.group(4)) * 4
    if match.group(3) is not None:
        return "xyzw".index(match.group(3)) * 4
    return int(match.group(1)) * 16 + "xyzw".index(match.group(2)) * 4


# The scalar holding each struct member's first word, by block.
HLSL_REGISTER_CASES = (
    (STD430_PUSH, "Push", STD430_PUSH_OFFSETS, {"item": r"item\.x"}),
    (SCALAR_PUSH, "Push", SCALAR_PUSH_OFFSETS, {"p": r"p\[\w+\]\.a"}),
    (SCALAR_TAIL, "S", SCALAR_TAIL_OFFSETS, {"v": r"v\.d", "n": r"n\[\w+\]\.v\.d"}),
    (STD430_UNIFORM, "F", STD430_UNIFORM_OFFSETS, {"u": r"u\[\w+\]\.d", "w": r"w\.a"}),
)
HLSL_REGISTER_IDS = ("std430-push", "scalar-push", "scalar-tail", "std430-uniform")


@pytest.mark.parametrize(
    ("source", "block", "expected", "first_scalars"),
    HLSL_REGISTER_CASES,
    ids=HLSL_REGISTER_IDS,
)
def test_hlsl_loads_blocks_packoffset_cannot_place(
    tmp_path, source, block, expected, first_scalars
):
    """A std430 ``float[]`` (4-byte stride) or a scalar ``mat3`` (12-byte columns) is read word by word."""
    code = _flat(_translate(tmp_path, source, "directx", _stage_name(source)))
    assert re.search(
        rf"cbuffer {block} : register\(b\d+\) \{{ uint4 __crossgl_{block}_registers\[\d+\]; \}};",
        code,
    ), code
    assert re.search(
        rf"(?:VS|PS)Main\([^)]*\)[^{{]*\{{ (?:[^{{}}]*; )?__crossgl_load_{block}\(\);",
        code,
    ), code
    for member, offset in expected.items():
        pattern = first_scalars.get(member, member)
        assert _hlsl_register_word_offsets(code, block, pattern) == offset, member


def test_hlsl_register_loads_follow_the_array_strides(tmp_path):
    code = _flat(_translate(tmp_path, SCALAR_PUSH, "directx"))
    # float arr[3] (stride 4), vec3 va[2] (stride 12), Pair p[2] (stride 12) under the scalar layout.
    assert re.search(
        r"arr\[(\w+)\] = asfloat\(__crossgl_Push_word\(\(\1\) \+ 17\)\);", code
    )
    assert re.search(
        r"va\[(\w+)\] = asfloat\(uint3\(__crossgl_Push_word\(\(\1 \* 3\) \+ 20\)", code
    )
    assert re.search(
        r"p\[(\w+)\]\.b = asfloat\(__crossgl_Push_word\(\(\1 \* 3\) \+ 28\)\);", code
    )
    # Columns of a matrix go through write masks: HLSL matrices are indexed by row.
    assert "m._m01_m11_m21 = asfloat(uint3(__crossgl_Push_registers[1].w, " in code
    assert "static float3x3 m;" in code


def test_hlsl_register_loads_reassemble_doubles(tmp_path):
    code = _flat(_translate(tmp_path, STD430_UNIFORM, "directx"))
    assert re.search(
        r"u\[(\w+)\]\.d = asdouble\(__crossgl_F_registers\[\1\]\.z, __crossgl_F_registers\[\1\]\.w\);",
        code,
    ), code


@pytest.mark.parametrize(
    ("source", "block", "expected", "first_scalars"),
    HLSL_REGISTER_CASES,
    ids=HLSL_REGISTER_IDS,
)
def test_dxc_compiles_register_loaded_blocks(
    tmp_path, source, block, expected, first_scalars
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not available")
    stage = _stage_name(source)
    path = tmp_path / "shader.hlsl"
    path.write_text(_translate(tmp_path, source, "directx", stage))
    profile, entry = (
        ("vs_6_2", "VSMain") if stage.endswith(".vert") else ("ps_6_2", "PSMain")
    )
    result = subprocess.run(
        [
            dxc,
            "-T",
            profile,
            "-E",
            entry,
            str(path),
            "-Fo",
            str(tmp_path / "shader.dxil"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    assert "warning" not in result.stderr.lower(), result.stderr[-3000:]


def _register_word_offsets(code, member):
    """Byte offsets the OpenGL loader reads ``member``'s first word from (static and per-element word reads)."""
    match = re.search(
        rf"\b{member}(?:\[\w+\])?(?:\[0\])? = [^;]*?"
        r"(?:crossgl_cbuffer_registers\[(?:\w+ \+ )?(\d+)\]\.([xyzw])"
        r"|\w+_word\(\(\w+(?: \* \d+)?\) \+ (\d+)\))",
        code,
    )
    assert match is not None, code
    if match.group(3) is not None:
        return int(match.group(3)) * 4
    return int(match.group(1)) * 16 + "xyzw".index(match.group(2)) * 4


@pytest.mark.parametrize(
    ("source", "expected"),
    ((STD430_PUSH, STD430_PUSH_OFFSETS), (SCALAR_PUSH, SCALAR_PUSH_OFFSETS)),
    ids=("std430", "scalar"),
)
def test_opengl_loads_block_members_from_their_offsets(tmp_path, source, expected):
    code = _flat(_translate(tmp_path, source, "opengl", _stage_name(source)))
    assert "uvec4 crossgl_cbuffer_registers[" in code
    assert re.search(r"void main\(\) \{ crossgl_load_Push\(\);", code)
    for member, offset in expected.items():
        if member == "item":
            member = r"item\.x"
        elif member == "p":
            member = r"p\[\w+\]\.a"
        assert _register_word_offsets(code, member) == offset, member


def test_opengl_output_compiles(tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not available")
    for source in (STD430_PUSH, SCALAR_PUSH, STD430_AS_STD140):
        stage = _stage_name(source)
        path = tmp_path / f"out.{stage.split('.')[1]}"
        path.write_text(_translate(tmp_path, source, "opengl", stage))
        result = subprocess.run(
            [validator, str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def test_opengl_keeps_a_std140_compatible_std430_block(tmp_path):
    code = _flat(_translate(tmp_path, STD430_AS_STD140, "opengl"))
    assert re.search(r"layout\(std140, binding = \d+\) uniform Push \{ mat4 mvp;", code)
    assert "crossgl_cbuffer_registers" not in code


def _spirv_member_offsets(code, block):
    return {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(rf"OpMemberDecorate %{block} (\d+) Offset (\d+)", code)
    }


@pytest.mark.parametrize(
    ("source", "block", "expected"),
    (
        (STD430_PUSH, "Push", STD430_PUSH_OFFSETS),
        (SCALAR_PUSH, "Push", SCALAR_PUSH_OFFSETS),
        (SCALAR_TAIL, "S", SCALAR_TAIL_OFFSETS),
        (STD430_UNIFORM, "F", STD430_UNIFORM_OFFSETS),
    ),
    ids=("std430-push", "scalar-push", "scalar-uniform", "std430-uniform"),
)
def test_spirv_decorates_block_offsets(tmp_path, source, block, expected):
    code = _translate(tmp_path, source, "vulkan", _stage_name(source))
    assert _spirv_member_offsets(code, block) == dict(enumerate(expected.values()))
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    asm = tmp_path / "shader.spvasm"
    asm.write_text(code)
    binary = tmp_path / "shader.spv"
    result = subprocess.run(
        [assembler, "--target-env", "vulkan1.0", str(asm), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    result = subprocess.run(
        [validator, "--target-env", "vulkan1.0", "--scalar-block-layout", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
