"""Matrix and array fields of structures nested in Metal stage-interface structures are split.

Metal rejects a vertex output or ``[[stage_in]]`` structure when a structure nested in it holds a matrix or an
array. The emitter splits such fields into one field per column or element, as it already does for the
stage-interface structure's own members, and rewrites every access to them.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader NestedVaryings {
    struct Inner {
        float weights[2];
        mat3x2 frame;
    };
    struct VertexData {
        vec2 uv;
        mat2 basis;
        Inner inner;
        flat int layer;
    };
    struct VertexInput {
        vec3 inPos @location(0);
    };
    struct VertexOutput {
        VertexData data @location(0);
        vec4 position @gl_Position;
    };
    struct FragmentInput {
        VertexData data @location(0);
    };
    float total(Inner value, int i) {
        return value.weights[i] + value.frame[1].y;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            Inner inner;
            inner.weights[0] = input.inPos.x;
            inner.weights[1] = input.inPos.y;
            inner.frame = mat3x2(1.0);
            output.data.uv = input.inPos.xy;
            output.data.basis = mat2(input.inPos.z);
            output.data.inner = inner;
            output.data.layer = 2;
            output.position = vec4(input.inPos, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(FragmentInput input) @location(0) {
            VertexData copy = input.data;
            float w = total(copy.inner, copy.layer) + input.data.inner.weights[copy.layer];
            return vec4(copy.uv + input.data.basis[1], w, input.data.inner.frame[2].x);
        }
    }
}
"""

GLSL_VERTEX = """#version 450
layout(location = 0) in vec3 inPos;
layout(location = 1) out Data {
    vec2 uv;
    mat2 basis;
    flat uint idx;
    float weights[2];
} data_out;
void main() {
    data_out.uv = inPos.xy;
    data_out.basis = mat2(inPos.x);
    data_out.idx = 3u;
    data_out.weights[0] = 1.0;
    data_out.weights[1] = inPos.z;
    gl_Position = vec4(inPos, 1.0);
}
"""

GLSL_FRAGMENT = """#version 450
layout(location = 1) in Data {
    vec2 uv;
    mat2 basis;
    flat uint idx;
    float weights[2];
} data_in;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(data_in.uv + data_in.basis[1], float(data_in.idx), data_in.weights[int(data_in.idx) & 1]);
}
"""


def _metal(source=PROGRAM):
    return MetalCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def _struct_body(code, name):
    match = re.search(rf"struct {name} \{{\n(.*?)\n\}};", code, re.S)
    assert match, f"struct {name} not emitted"
    return [line.strip() for line in match.group(1).splitlines()]


def _compile(tmp_path, code, stem):
    source = tmp_path / f"{stem}.metal"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.1",
            "-Werror",
            "-c",
            str(source),
            "-o",
            str(tmp_path / f"{stem}.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_nested_matrix_and_array_fields_are_split_into_columns_and_elements():
    code = _metal()
    assert _struct_body(code, "VertexData") == [
        "float2 uv;",
        "float2 basis_0;",
        "float2 basis_1;",
        "Inner inner;",
        "int layer [[flat]];",
    ]
    assert _struct_body(code, "Inner") == [
        "float weights_0;",
        "float weights_1;",
        "float2 frame_0;",
        "float2 frame_1;",
        "float2 frame_2;",
    ]
    assert "VertexData data [[user(locn0)]];" in _struct_body(code, "VertexOutput")


def test_accesses_to_split_nested_fields_are_rewritten_everywhere():
    code = _metal()
    flat = re.sub(r"\s+", " ", code)
    assert "float2x2(input.data.basis_0, input.data.basis_1)[1]" in flat
    assert (
        "float3x2(input.data.inner.frame_0, input.data.inner.frame_1, "
        "input.data.inner.frame_2)[2]" in flat
    )
    # A helper function taking the nested structure by value and a dynamic index go through the getter.
    assert "__crossgl_stage_io_get_Inner_weights(value, i)" in flat
    assert "__crossgl_stage_io_get_Inner_weights(input.data.inner, copy.layer)" in flat
    assert "inner.weights_0 = input.inPos.x;" in flat
    assert re.search(
        r"output\.data\.basis_1 = __crossgl_stage_io_matrix_\d+\[1\];", flat
    )
    assert ".weights[" not in flat and ".frame[" not in flat


def test_structures_not_reachable_from_stage_interfaces_keep_their_members():
    code = _metal("""
shader Plain {
    struct Pair {
        mat2 basis;
        float weights[2];
    };
    float sum(Pair p) {
        return p.basis[0].x + p.weights[1];
    }
    fragment {
        vec4 main() @location(0) {
            Pair p;
            p.basis = mat2(1.0);
            p.weights[0] = 0.0;
            p.weights[1] = 1.0;
            return vec4(sum(p));
        }
    }
}
""")
    assert _struct_body(code, "Pair") == ["float2x2 basis;", "float weights[2];"]


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="requires the Metal compiler")
def test_nested_stage_interface_structures_compile(tmp_path):
    _compile(tmp_path, _metal(), "nested")


@pytest.mark.parametrize(
    ("stem", "source"), [("block_vert", GLSL_VERTEX), ("block_frag", GLSL_FRAGMENT)]
)
def test_glsl_instance_block_with_matrix_and_array_translates_to_metal(
    tmp_path, stem, source
):
    path = tmp_path / f"{stem}.{'vert' if stem.endswith('vert') else 'frag'}"
    path.write_text(source, encoding="utf-8")
    code = crosstl.translate(str(path), backend="metal")
    assert _struct_body(code, "Data") == [
        "float2 uv;",
        "float2 basis_0;",
        "float2 basis_1;",
        "uint idx [[flat]];",
        "float weights_0;",
        "float weights_1;",
    ]
    if shutil.which("xcrun") is not None:
        _compile(tmp_path, code, stem)
