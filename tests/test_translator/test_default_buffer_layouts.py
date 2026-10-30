"""CrossGL storage buffers without a ``@buffer_layout`` keep their std430 element layout on Metal (ADR 0005).

MSL sizes and aligns ``float3`` to 16 bytes (MSL 3.2 section 2.2, table 2.4), so ``struct { vec3 a; float b; }``
would put ``b`` at byte 16 in a 32-byte element where std430 (GLSL 4.60 section 7.6.2.2) puts it at byte 12 in a
16-byte one. The Metal generator declares such elements with packed vectors and padding at their std430 offsets.
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


def _program(body, structs="struct Item { vec3 a; float b; };", globals_=""):
    return f"""
shader Packed {{
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
            items[i].b = items[i].a.x + items[i].a.z;
        }""")


def _translate(source, backend, tmp_path, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _struct(code, name):
    match = re.search(rf"struct {name} \{{(.*?)\}};", code, re.S)
    assert match, code
    return [line.strip() for line in match.group(1).strip().splitlines()]


def _compile_metal(code, tmp_path):
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is unavailable")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    if probe.returncode:
        pytest.skip("the Metal compiler is unavailable")
    source = tmp_path / "shader.metal"
    source.write_text(code)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "shader.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_vec3_before_a_scalar_is_packed_at_its_std430_offset(tmp_path):
    code = _translate(ITEMS, "metal", tmp_path)
    assert _struct(code, "Item") == ["packed_float3 a;", "float b;"]
    assert re.search(r"device Item\s*\*\s*items\b", code)
    _compile_metal(code, tmp_path)


def test_other_targets_keep_the_plain_element(tmp_path):
    glsl = _translate(ITEMS, "opengl", tmp_path)
    assert "vec3 a;" in glsl and "packed" not in glsl
    hlsl = _translate(ITEMS, "directx", tmp_path)
    assert _struct(hlsl, "Item") == ["float3 a;", "float b;"]


def test_element_whose_layouts_agree_is_unchanged(tmp_path):
    source = _program(
        """buffer Item* items;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            items[i].b = items[i].a.y;
        }""",
        structs="struct Item { float b; vec3 a; };",
    )
    code = _translate(source, "metal", tmp_path)
    assert _struct(code, "Item") == ["float b;", "float3 a;"]
    program = _parse(source)
    assert mark_default_buffer_layouts(program, "metal") == 0


def test_boolean_element_is_left_to_the_target(tmp_path):
    source = _program(
        """buffer Item* items;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            items[i].b = items[i].a.x > 0.0;
        }""",
        structs="struct Item { vec3 a; bool b; };",
    )
    assert mark_default_buffer_layouts(_parse(source), "metal") == 0
    code = _translate(source, "metal", tmp_path)
    assert "packed_float3" not in code


def test_marker_only_marks_unmarked_metal_buffers():
    program = _parse(ITEMS)
    assert mark_default_buffer_layouts(program, "opengl") == 0
    assert mark_default_buffer_layouts(program, "metal") == 1
    # A second run finds the buffer marked already.
    assert mark_default_buffer_layouts(program, "metal") == 0
    stage = next(iter(program.stages.values()))
    marks = [
        (
            attribute.name,
            [str(getattr(arg, "name", arg)) for arg in attribute.arguments],
        )
        for variable in stage.local_variables
        for attribute in getattr(variable, "attributes", None) or []
        if isinstance(attribute, A.AttributeNode)
    ]
    assert ("buffer_layout", ["std430"]) in marks


def test_explicit_std430_mark_is_reproduced(tmp_path):
    source = ITEMS.replace(
        "buffer Item* items;", "@ buffer_layout(std430)\n        buffer Item* items;"
    )
    code = _translate(source, "metal", tmp_path)
    assert _struct(code, "Item") == ["packed_float3 a;", "float b;"]


def test_structured_buffer_nested_struct_and_helper_parameter(tmp_path):
    source = _program(
        """buffer Item* items;
        RWStructuredBuffer<Outer> outers;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            Item x = items[i];
            x.b += 1.0;
            items[i] = x;
            outers[i].item.a = x.a;
            outers[i].c = outers[i].item.b;
            touch(items, i);
        }""",
        structs="""struct Item { vec3 a; float b; };
    struct Outer { Item item; float c; };""",
        globals_="""void touch(buffer Item* p, uint i) {
        p[i].b += p[i].a.y;
    }""",
    )
    code = _translate(source, "metal", tmp_path)
    assert _struct(code, "Item") == ["packed_float3 a;", "float b;"]
    # std430 gives Outer the 16-byte alignment of Item's vec3: c at byte 16, 32 bytes in all.
    assert _struct(code, "Outer") == ["Item item;", "float c;", "char _pad2[12];"]
    assert re.search(r"device Item\s*\*\s*p\b", code)
    assert re.search(r"device Outer\s*\*\s*outers\b", code)
    _compile_metal(code, tmp_path)
