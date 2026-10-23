"""Stage varyings of structure type become one SPIR-V interface variable per field, keeping each field's decorations."""

import re
import shutil
import subprocess

import pytest

import crosstl

FLAT_FIELD = """
shader main {
    struct VertexData {
        vec2 uv;
        mat2 basis;
        flat uint texture_index;
    };
    struct VertexInput {
        vec3 inPos @location(0);
    };
    struct VertexOutput {
        VertexData vertex_out @location(1);
        vec3 after @location(6);
        vec4 position @gl_Position;
    };
    struct FragmentInput {
        VertexData vertex_in @location(1);
        vec3 after @location(6);
    };
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.vertex_out.uv = input.inPos.xy;
            output.vertex_out.basis = mat2(1.0);
            output.vertex_out.texture_index = 3u;
            output.after = input.inPos;
            output.position = vec4(input.inPos, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(FragmentInput input) @location(0) {
            return vec4(input.vertex_in.uv + input.vertex_in.basis[1],
                        float(input.vertex_in.texture_index), input.after.x);
        }
    }
}
"""

NESTED_FIELDS = """
shader main {
    struct Inner {
        vec3 normal;
        flat int id @location(9);
        bool lit;
    };
    struct VertexData {
        vec2 uv;
        Inner inner;
        vec4 arr[2];
        vec3 tail;
    };
    struct VertexInput {
        vec3 inPos @location(0);
    };
    struct VertexOutput {
        noperspective VertexData vertex_out @location(1);
        vec4 position @gl_Position;
    };
    struct FragmentInput {
        noperspective VertexData vertex_in @location(1);
    };
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.vertex_out.uv = input.inPos.xy;
            output.vertex_out.inner.normal = input.inPos;
            output.vertex_out.inner.id = 2;
            output.vertex_out.inner.lit = input.inPos.x > 0.0;
            output.vertex_out.arr[0] = vec4(1.0);
            output.vertex_out.arr[1] = vec4(2.0);
            output.vertex_out.tail = input.inPos;
            output.position = vec4(input.inPos, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(FragmentInput input) @location(0) {
            float l = input.vertex_in.inner.lit ? 1.0 : 0.0;
            return vec4(input.vertex_in.uv, float(input.vertex_in.inner.id) + l,
                        input.vertex_in.arr[1].x + input.vertex_in.tail.y);
        }
    }
}
"""

GLSL_INSTANCE_BLOCK = """#version 460
layout (location = 0) in VertexData {
    vec2 uv;
    flat uint texture_index;
} vertex_in;
layout (location = 0) out vec4 outColor;
void main() {
    outColor = vec4(vertex_in.uv, float(vertex_in.texture_index), 1.0);
}
"""


def _named(code):
    """Replace numeric ids with the debug or type name they stand for.

    Unformatted SPIR-V keeps numeric ids whether or not SPIRV-Tools is installed, so the checks do not depend on
    ``spirv-dis`` friendly names: an ``OpName`` id takes its name, and the 32-bit integer and bool types and their
    constants take ``uint``, ``int``, ``bool`` and ``uint_<value>`` spellings.
    """
    names = dict(re.findall(r'^OpName (%\d+) "(\w+)"$', code, re.M))
    scalar_types = {
        "OpTypeInt 32 0": "uint",
        "OpTypeInt 32 1": "int",
        "OpTypeBool": "bool",
    }
    for type_id, spelling in re.findall(
        r"^(%\d+) = (OpTypeInt 32 [01]|OpTypeBool)$", code, re.M
    ):
        names.setdefault(type_id, scalar_types[spelling])
    for constant_id, type_id, value in re.findall(
        r"^(%\d+) = OpConstant (%\d+) (\d+)$", code, re.M
    ):
        if names.get(type_id) in {"uint", "int"}:
            names.setdefault(constant_id, f"{names[type_id]}_{value}")
    return re.sub(
        r"%\d+\b",
        lambda match: "%" + names.get(match.group(0), match.group(0)[1:]),
        code,
    )


def _spirv(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    code = crosstl.translate(str(path), backend="vulkan", format_output=False)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm = tmp_path / "out.spvasm"
        asm.write_text(code)
        binary = tmp_path / "out.spv"
        for command in (
            ["spirv-as", "--target-env", "vulkan1.2", str(asm), "-o", str(binary)],
            ["spirv-val", "--target-env", "vulkan1.2", str(binary)],
        ):
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
    return _named(code)


def _decorations(code, prefix):
    """``{field: {decoration, ...}}`` for the interface variables named ``prefix<field>``."""
    found = {}
    for name, decoration in re.findall(
        rf"OpDecorate %{prefix}(\w+) (\w+(?: \d+)?)", code
    ):
        found.setdefault(name, set()).add(decoration)
    return found


def test_struct_varying_fields_get_their_own_locations_and_flat(tmp_path):
    code = _spirv(tmp_path, FLAT_FIELD)

    expected = {
        "uv": {"Location 1"},
        "basis": {"Location 2"},
        "texture_index": {"Location 4", "Flat"},
    }
    vertex = _decorations(code, "CrossGL_vertex_output_main_vertex_out_")
    fragment = _decorations(code, "CrossGL_fragment_input_input_vertex_in_")
    assert vertex == expected
    assert fragment == expected
    # No variable of the structure type remains on either side of the interface.
    assert not re.search(r"OpTypePointer (?:Input|Output) %VertexData\b", code)
    assert "OpDecorate %CrossGL_fragment_input_input_after Location 6" in code


def test_nested_struct_varying_fields_inherit_interpolation_and_own_locations(
    tmp_path,
):
    code = _spirv(tmp_path, NESTED_FIELDS)

    fragment = _decorations(code, "CrossGL_fragment_input_input_vertex_in_")
    assert fragment == {
        "uv": {"Location 1", "NoPerspective"},
        "inner_normal": {"Location 2", "NoPerspective"},
        # The field's own location restarts the count; an integer field stays flat only.
        "inner_id": {"Location 9", "Flat"},
        "inner_lit": {"Location 10", "Flat"},
        "arr": {"Location 11", "NoPerspective"},
        "tail": {"Location 13", "NoPerspective"},
    }
    vertex = _decorations(code, "CrossGL_vertex_output_main_vertex_out_")
    assert {
        name: sorted(d for d in decorations if d.startswith("Location"))
        for name, decorations in vertex.items()
    } == {
        name: sorted(d for d in decorations if d.startswith("Location"))
        for name, decorations in fragment.items()
    }
    # The bool field travels as uint and is converted back on load.
    assert re.search(r"OpSelect %uint %\w+ %uint_1 %uint_0", code)
    assert re.search(r"OpINotEqual %bool", code)


def test_glsl_instance_block_member_stays_flat_in_spirv(tmp_path):
    code = _spirv(tmp_path, GLSL_INSTANCE_BLOCK, "block.frag")

    fragment = _decorations(code, "CrossGL_fragment_input_input_vertex_in_")
    assert fragment["texture_index"] == {"Location 1", "Flat"}
    assert fragment["uv"] == {"Location 0"}


@pytest.mark.parametrize("member", ["VertexData vertex_out[2] @location(1);"])
def test_arrays_of_struct_varyings_keep_one_variable(tmp_path, member):
    source = FLAT_FIELD.replace("VertexData vertex_out @location(1);", member).replace(
        "output.vertex_out.", "output.vertex_out[0]."
    )
    source = source.split("    fragment {")[0] + "}\n"
    source = source.replace(
        "    struct FragmentInput {\n        VertexData vertex_in @location(1);\n        vec3 after @location(6);\n    };\n",
        "",
    )
    path = tmp_path / "arr.cgl"
    path.write_text(source)
    code = crosstl.translate(str(path), backend="vulkan")

    assert "CrossGL_vertex_output_main_vertex_out_uv" not in code
