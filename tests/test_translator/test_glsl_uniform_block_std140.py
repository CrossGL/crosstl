"""GLSL uniform blocks keep their interface and their std140 layout on every target.

A uniform block is a host interface: the host binds it by its binding point (or, in OpenGL, finds it by its block
name) and writes its members at their std140 offsets (GLSL 4.60 sections 4.4.5 and 7.6.2.2). The GLSL importer used
to merge blocks without a descriptor set into one ``Uniforms`` constant buffer at binding 0, and the SPIR-V emitter
sized two- and three-component vectors as 16 bytes and ignored ``row_major``, so members after them moved.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

BLOCK = """#version 450
struct T { vec4 p; float q; };
layout(std140, binding = 3) uniform Params {
    float a; vec3 b; float c; float arr[2]; vec2 d; mat3 m;
    layout(row_major) mat4 r; T t; float e; mat2 n; bool f;
} params;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(params.b * params.a + params.c + params.arr[1], params.d.y) + vec4(params.m[1], params.e)
          + params.r * params.t.p + vec4(params.n[1], params.t.q, params.f ? 1.0 : 0.0);
}
"""

# std140 offsets of the members of Params, as glslang assigns them.
BLOCK_OFFSETS = [0, 16, 28, 32, 64, 80, 128, 192, 224, 240, 272]

LAYOUT_CASES = {
    "vectors_and_matrices": BLOCK,
    "non_square_matrices": (
        """#version 450
struct S { float x; vec2 y; };
layout(std140, binding = 1) uniform Shapes {
    vec2 a; vec2 b; float c; vec3 d[3]; S s[2]; mat2x3 m23; mat3x2 m32;
    layout(row_major) mat2x3 r23; layout(row_major) mat3x2 r32; float tail;
} shapes;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(shapes.a + shapes.b, shapes.c + shapes.d[2].z, shapes.s[1].y.x)
          + vec4(shapes.m23[1], shapes.m32[2].y) + vec4(shapes.r23[0], shapes.r32[1].x + shapes.tail);
}
"""
    ),
    "nested_structs": (
        """#version 450
struct Inner { vec3 p; float q; };
struct Outer { float x; Inner i; vec2 y; };
layout(std140, binding = 2) uniform Nested { Outer o; float z; Outer arr[2]; vec3 w; } nested;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(nested.o.i.p * nested.o.x, nested.z) + vec4(nested.arr[1].i.p, nested.arr[0].y.y + nested.w.x);
}
"""
    ),
}


def _translate(tmp_path, source, backend, suffix="frag"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _block_layout(assembly):
    """Each struct's member decorations by struct name (layout clones named like their struct), plus the array
    strides, from SPIR-V assembly."""
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', assembly))
    members = {}
    for target, index, decoration in re.findall(
        r"OpMemberDecorate (%\w+) (\d+) ((?:Offset|MatrixStride) \d+|RowMajor|ColMajor)",
        assembly,
    ):
        name = re.sub(r"_std140_\d+$", "", names.get(target, target.lstrip("%")))
        key = (name, int(index))
        members.setdefault(key, set()).add(decoration)
    strides = sorted(
        int(stride) for stride in re.findall(r"ArrayStride (\d+)", assembly)
    )
    return members, strides


def test_uniform_block_keeps_its_name_and_binding(tmp_path):
    crossgl = _translate(tmp_path, BLOCK, "cgl")
    assert "cbuffer Params @binding(3)" in crossgl
    assert "Uniforms" not in crossgl
    assert re.search(
        r"layout\(std140, binding = 3\) uniform Params\b",
        _translate(tmp_path, BLOCK, "opengl"),
    )
    assert "cbuffer Params : register(b3)" in _translate(tmp_path, BLOCK, "directx")
    assert re.search(
        r"constant Params\s*&\s*params \[\[buffer\(3\)\]\]",
        _translate(tmp_path, BLOCK, "metal"),
    )
    assembly = _translate(tmp_path, BLOCK, "vulkan")
    variable = re.search(
        r"(%\w+) = OpVariable %_ptr_Uniform_Params Uniform", assembly
    ).group(1)
    assert f"OpDecorate {variable} Binding 3" in assembly


def test_uniform_blocks_without_instances_keep_their_names(tmp_path):
    source = """#version 450
layout(std140, binding = 0) uniform Uniforms { uint index; };
layout(std140, binding = 1) uniform Block { int value; } block;
layout(location = 0) out vec4 color;
void main() { color = vec4(float(index) + float(block.value)); }
"""
    crossgl = _translate(tmp_path, source, "cgl")
    assert "cbuffer Uniforms @binding(0)" in crossgl
    assert "cbuffer Block @binding(1)" in crossgl
    hlsl = _translate(tmp_path, source, "directx")
    assert "cbuffer Uniforms : register(b0)" in hlsl
    assert "cbuffer Block : register(b1)" in hlsl


def test_spirv_uses_std140_offsets(tmp_path):
    members, _strides = _block_layout(_translate(tmp_path, BLOCK, "vulkan"))
    offsets = [
        int(
            next(
                d for d in members[("Params", index)] if d.startswith("Offset")
            ).split()[1]
        )
        for index in range(len(BLOCK_OFFSETS))
    ]
    assert offsets == BLOCK_OFFSETS
    assert "RowMajor" in members[("Params", 6)]
    assert "ColMajor" not in members[("Params", 6)]


@pytest.mark.parametrize("case", sorted(LAYOUT_CASES))
def test_spirv_layout_matches_glslang(tmp_path, case):
    if not (shutil.which("glslangValidator") and shutil.which("spirv-dis")):
        pytest.skip("glslangValidator and spirv-dis are not on PATH")
    source = LAYOUT_CASES[case]
    path = tmp_path / "reference.frag"
    path.write_text(source, encoding="utf-8")
    binary = tmp_path / "reference.spv"
    _run(["glslangValidator", "-V", str(path), "-o", str(binary)])
    expected = _block_layout(_run(["spirv-dis", str(binary)]))
    assembly = _translate(tmp_path, source, "vulkan")
    assert _block_layout(assembly) == expected
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(assembly, encoding="utf-8")
        _run(
            [
                "spirv-as",
                "--target-env",
                "vulkan1.1",
                str(tmp_path / "out.spvasm"),
                "-o",
                str(tmp_path / "o.spv"),
            ]
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", str(tmp_path / "o.spv")])


def test_row_major_members_stay_row_major(tmp_path):
    glsl = _translate(tmp_path, BLOCK, "opengl")
    assert "layout(row_major) mat4 r;" in glsl
    assert "  mat3 m;" in glsl
    assert re.search(r"row_major float4x4 r\b", _translate(tmp_path, BLOCK, "directx"))
    if shutil.which("glslangValidator"):
        path = tmp_path / "out.frag"
        path.write_text(glsl, encoding="utf-8")
        _run(["glslangValidator", str(path)])


@pytest.mark.parametrize(
    ("member", "reason"),
    (
        ("layout(row_major) mat2x3 r[2];", "array of row-major non-square matrices"),
        ("layout(row_major) float r;", "holds no matrix"),
    ),
)
def test_spirv_refuses_row_major_layouts_it_cannot_express(tmp_path, member, reason):
    source = f"""#version 450
layout(std140, binding = 0) uniform B {{ mat2x3 c[2]; {member} }} b;
layout(location = 0) out vec4 color;
void main() {{ color = vec4(b.c[0][0], 1.0); }}
"""
    with pytest.raises(Exception, match=re.escape(reason)):
        _translate(tmp_path, source, "vulkan")
