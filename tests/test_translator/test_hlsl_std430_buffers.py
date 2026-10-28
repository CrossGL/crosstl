"""CrossGL storage buffers without a ``@buffer_layout`` keep their std430 element layout in HLSL (ADR 0005).

An HLSL structured buffer aligns a vector to its scalar (D3D11.3 functional specification, section 5.3.7), so
``struct { float a; vec3 b; }`` puts ``b`` at byte 4 in a 16-byte element where std430 (GLSL 4.60 section 7.6.2.2)
puts it at byte 16 in a 32-byte one, and ``vec3 v[]`` has a 12-byte stride where std430 has 16. Such buffers become
buffers of 32-bit words read and written at their std430 offsets.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.default_buffer_layouts import (
    mark_default_buffer_layouts,
)
from crosstl.translator.semantics.structured_layouts import (
    legalize_structured_layouts,
)


def _program(body, structs="struct Item { float a; vec3 b; };", globals_=""):
    return f"""
shader Padded {{
    {structs}
    {globals_}
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        {body}
    }}
}}
"""


ITEMS = _program("""buffer Item* items;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            items[i].b = items[i].b * items[i].a;
        }""")


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(source, backend, tmp_path, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compact(code):
    return re.sub(r"\s+", "", code)


def _dxc(code, tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "out.hlsl"
    path.write_text(code, encoding="utf-8")
    entry = re.search(r"\]\s*void\s+(\w+)\s*\(", code).group(1)
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", entry, str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_padded_vector_member_is_read_at_its_std430_offset(tmp_path):
    code = _translate(ITEMS, "directx", tmp_path)
    compact = _compact(code)
    assert "RWStructuredBuffer<uint>items" in compact
    # std430: a at word 0, b at word 4 (16-byte aligned vec3), an 8-word (32-byte) element.
    assert "asfloat(items[(uint(i)*8u)])" in compact
    assert "__cgl_compute_items_store_vec3(((uint(i)*8u)+4u)" in compact
    _dxc(code, tmp_path)


def test_vector_buffer_has_the_std430_stride(tmp_path):
    source = _program("""buffer vec3* points;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            points[i] = points[i] * 2.0;
        }""")
    code = _translate(source, "directx", tmp_path)
    compact = _compact(code)
    assert "RWStructuredBuffer<uint>points" in compact
    assert "__cgl_compute_points_store_vec3((uint(i)*4u)" in compact
    _dxc(code, tmp_path)


def test_element_whose_layouts_agree_keeps_its_struct(tmp_path):
    source = _program(
        """buffer Item* items;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            items[i].b = items[i].a.y;
        }""",
        structs="struct Item { vec3 a; float b; };",
    )
    assert mark_default_buffer_layouts(_parse(source), "directx") == 0
    code = _translate(source, "directx", tmp_path)
    assert "RWStructuredBuffer<Item>items" in _compact(code)


def test_boolean_element_is_left_to_the_target():
    source = _program(
        """buffer Item* items;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            items[i].c = items[i].a > 0.0;
        }""",
        structs="struct Item { float a; vec3 b; bool c; };",
    )
    assert mark_default_buffer_layouts(_parse(source), "directx") == 0


@pytest.mark.parametrize("target", ["directx", "hlsl", "dx11", "dx12", "d3d11"])
def test_every_hlsl_target_marks_the_buffer(target):
    program = _parse(ITEMS)
    assert mark_default_buffer_layouts(program, target) == 1
    assert mark_default_buffer_layouts(program, target) == 0


def test_std430_targets_are_not_marked():
    for target in ("opengl", "vulkan", "spirv"):
        assert mark_default_buffer_layouts(_parse(ITEMS), target) == 0


def test_explicit_std430_mark_matches_the_default(tmp_path):
    marked = ITEMS.replace(
        "buffer Item* items;", "@ buffer_layout(std430)\n        buffer Item* items;"
    )
    assert _translate(marked, "directx", tmp_path, "marked.cgl") == _translate(
        ITEMS, "directx", tmp_path
    )
    # std430 is the GLSL rule set, so a GLSL target keeps the element type.
    glsl = _translate(marked, "opengl", tmp_path, "marked.cgl")
    assert "uint items" not in glsl and "vec3 b;" in glsl


def test_matrix_columns_follow_the_std430_stride(tmp_path):
    source = _program(
        """buffer Frame* frames;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            frames[i].scale = frames[i].basis[1].y;
        }""",
        structs="struct Frame { mat3 basis; float scale; };",
    )
    code = _translate(source, "directx", tmp_path)
    compact = _compact(code)
    # std430 mat3: three 16-byte columns, scale at byte 48, a 64-byte (16-word) element.
    assert "RWStructuredBuffer<uint>frames" in compact
    assert "(uint(i)*16u)+12u" in compact
    _dxc(code, tmp_path)


def test_buffer_passed_to_a_helper_keeps_the_same_words(tmp_path):
    source = _program(
        """buffer Item* items;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            touch(items, i);
        }""",
        globals_="""void touch(buffer Item* p, uint i) {
        p[i].b += vec3(p[i].a);
    }""",
    )
    code = _translate(source, "directx", tmp_path)
    compact = _compact(code)
    assert "RWStructuredBuffer<uint>items" in compact
    assert "RWStructuredBuffer<Item>" not in compact
    # HLSL has no pointers: the parameter is the word buffer plus a word offset, and it reads the argument's
    # words at the same std430 offsets.
    assert "uint*" not in compact
    assert "voidtouch(RWStructuredBuffer<uint>p,int64_tp_offset,uinti)" in compact
    assert "asfloat(p[uint((p_offset+(uint(i)*8u)))])" in compact
    assert "touch(items,int64_t(0),i);" in compact
    _dxc(code, tmp_path)


def test_stage_buffer_with_a_declared_layout_is_lowered_on_std430_targets(tmp_path):
    source = _program("""@ buffer_layout(hlsl_structured)
        buffer vec3* points;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            points[i] = points[i] + vec3(1.0);
        }""")
    glsl = _translate(source, "opengl", tmp_path)
    compact = _compact(glsl)
    # HLSL structured rules: a 12-byte (3-word) stride that std430 cannot declare for vec3.
    assert "vec3points[]" not in compact
    assert "(uint(i)*3u)" in compact


def test_stage_buffers_are_scoped_to_their_stage(tmp_path):
    source = """
shader Two {
    struct Item { float a; vec3 b; };
    compute {
        layout(local_size_x = 1) in;
        buffer Item* data;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            data[i].a = data[i].b.z;
        }
    }
    fragment {
        buffer float* data;
        vec4 main() @ gl_FragColor {
            return vec4(data[0]);
        }
    }
}
"""
    program = _parse(source)
    assert mark_default_buffer_layouts(program, "directx") == 1
    assert legalize_structured_layouts(program, "directx") == 1
    compute, fragment = program.stages.values()
    assert compute.local_variables[0].var_type.pointee_type.name == "uint"
    assert fragment.local_variables[0].var_type.pointee_type.name == "float"
    # The fragment stage still indexes its own float buffer.
    returned = fragment.entry_point.body.statements[0].value.arguments[0]
    assert isinstance(returned, A.ArrayAccessNode)
