"""WGSL entry-point IO always carries a binding and a legal interpolation.

WGSL section 12.3.1 requires every entry-point parameter and return value to
carry ``@builtin`` or ``@location``; section 12.3.1.4 requires integer user
vertex outputs and fragment inputs to be ``@interpolate(flat)``. System-value
semantics without a WGSL built-in value are refused instead of being demoted
to user locations.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.codegen.wgsl_stage_io import (
    WGSLStageIOUnsupportedError,
    is_integer_io_type,
    with_flat_integer_interpolation,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate_wgsl(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return WGSLCodeGen().generate(ast)


def validate_with_naga(tmp_path, code):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "shader.wgsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def entry_signature(code, stage):
    match = re.search(rf"fn {stage}_main\((.*?)\)(?: ->| \{{)", code)
    assert match, code
    return match.group(1)


def struct_body(code, name):
    match = re.search(rf"struct {name} \{{\n(.*?)\n\}};", code, re.S)
    assert match, code
    return match.group(1)


VERTEX_PARAMETERS = """
shader S {
    vertex {
        vec4 main(vec3 position @ POSITION, vec4 weights @ BLENDWEIGHT, vec2 uv)
            @ gl_Position {
            return vec4(position + weights.xyz + vec3(uv, 0.0), 1.0);
        }
    }
}
"""

FRAGMENT_PARAMETERS = """
shader S {
    fragment {
        vec4 main(vec4 color, vec2 uv @ TEXCOORD0, float fade @ BLENDWEIGHT)
            @ gl_FragColor {
            return color + vec4(uv, fade, 1.0);
        }
    }
}
"""


def test_unbound_vertex_parameters_take_free_locations(tmp_path):
    code = generate_wgsl(VERTEX_PARAMETERS)
    assert entry_signature(code, "vertex") == (
        "@location(0) position: vec3<f32>, @location(1) weights: vec4<f32>, "
        "@location(2) uv: vec2<f32>"
    )
    validate_with_naga(tmp_path, code)


def test_unbound_fragment_parameters_skip_semantic_locations(tmp_path):
    code = generate_wgsl(FRAGMENT_PARAMETERS)
    # TEXCOORD0 keeps its fixed location 5; the unbound parameters fill 0 and 1.
    assert entry_signature(code, "fragment") == (
        "@location(0) color: vec4<f32>, @location(5) uv: vec2<f32>, "
        "@location(1) fade: f32"
    )
    validate_with_naga(tmp_path, code)


def test_parameters_avoid_locations_of_interface_struct_members(tmp_path):
    code = generate_wgsl("""
shader S {
    struct VSIn {
        vec3 position;
        vec2 uv @ TEXCOORD1;
    }
    vertex {
        vec4 main(VSIn input, vec4 tint) @ gl_Position {
            return vec4(input.position, input.uv.x) + tint;
        }
    }
}
""")
    assert "@location(0) position: vec3<f32>" in struct_body(code, "VSIn")
    assert "@location(6) uv: vec2<f32>" in struct_body(code, "VSIn")
    assert (
        entry_signature(code, "vertex") == "input: VSIn, @location(1) tint: vec4<f32>"
    )
    validate_with_naga(tmp_path, code)


def test_resource_and_builtin_parameters_take_no_location(tmp_path):
    code = generate_wgsl("""
shader S {
    fragment {
        uniform sampler2D tex;
        vec4 main(vec2 uv, vec4 frag @ gl_FragCoord, bool front @ gl_FrontFacing,
                  uint sample @ SV_SampleIndex) @ gl_FragColor {
            return texture(tex, uv) + vec4(frag.x, float(sample), 0.0, 1.0);
        }
    }
}
""")
    assert entry_signature(code, "fragment") == (
        "@location(0) uv: vec2<f32>, @builtin(position) frag: vec4<f32>, "
        "@builtin(front_facing) front: bool, "
        "@builtin(sample_index) sample: u32"
    )
    validate_with_naga(tmp_path, code)


@pytest.mark.parametrize(
    "semantic, builtin",
    [
        ("gl_VertexIndex", "vertex_index"),
        ("gl_VertexID", "vertex_index"),
        ("SV_VertexID", "vertex_index"),
        ("gl_InstanceIndex", "instance_index"),
        ("gl_InstanceID", "instance_index"),
        ("SV_InstanceID", "instance_index"),
    ],
)
def test_vertex_system_semantics_map_to_builtins(tmp_path, semantic, builtin):
    code = generate_wgsl(f"""
shader S {{
    vertex {{
        vec4 main(vec3 position, uint index @ {semantic}) @ gl_Position {{
            return vec4(position, float(index));
        }}
    }}
}}
""")
    assert entry_signature(code, "vertex") == (
        f"@location(0) position: vec3<f32>, @builtin({builtin}) index: u32"
    )
    validate_with_naga(tmp_path, code)


def test_fragment_depth_and_sample_mask_outputs_map_to_builtins(tmp_path):
    code = generate_wgsl("""
shader S {
    struct FOut {
        vec4 color @ SV_Target0;
        float depth @ gl_FragDepth;
        uint mask @ SV_Coverage;
    }
    fragment {
        FOut main(vec4 color, uint coverage @ gl_SampleMaskIn) {
            FOut o;
            o.color = color;
            o.depth = 0.5;
            o.mask = coverage;
            return o;
        }
    }
}
""")
    body = struct_body(code, "FOut")
    assert "@location(0) color: vec4<f32>" in body
    assert "@builtin(frag_depth) depth: f32" in body
    assert "@builtin(sample_mask) mask: u32" in body
    assert "@builtin(sample_mask) coverage: u32" in entry_signature(code, "fragment")
    validate_with_naga(tmp_path, code)


def test_integer_varyings_are_flat(tmp_path):
    code = generate_wgsl("""
shader S {
    struct VSOut {
        vec4 position @ gl_Position;
        int id;
        uvec2 cell @ TEXCOORD3;
        vec2 uv;
    }
    vertex {
        VSOut main(vec3 p @ POSITION, int id @ TEXCOORD1) {
            VSOut o;
            o.position = vec4(p, 1.0);
            o.id = id;
            o.cell = uvec2(1u, 2u);
            o.uv = p.xy;
            return o;
        }
    }
    fragment {
        vec4 main(VSOut input) @ gl_FragColor {
            return vec4(float(input.id), float(input.cell.x), input.uv);
        }
    }
}
""")
    body = struct_body(code, "VSOut")
    assert "@location(0) @interpolate(flat) id: i32" in body
    assert "@location(8) @interpolate(flat) cell: vec2<u32>" in body
    assert "@location(1) uv: vec2<f32>" in body
    assert "@location(6) @interpolate(flat) id: i32" in entry_signature(code, "vertex")
    validate_with_naga(tmp_path, code)


def test_integer_array_varying_fields_are_flat(tmp_path):
    code = generate_wgsl("""
shader S {
    struct VSOut {
        vec4 position @ gl_Position;
        int ids[2];
    }
    vertex {
        VSOut main(vec3 p) {
            VSOut o;
            o.position = vec4(p, 1.0);
            o.ids[0] = 1;
            o.ids[1] = 2;
            return o;
        }
    }
    fragment {
        vec4 main(VSOut input) @ gl_FragColor {
            return vec4(float(input.ids[0] + input.ids[1]));
        }
    }
}
""")
    body = struct_body(code, "VSOut")
    flat_fields = re.findall(r"@location\((\d)\) @interpolate\(flat\) \w+: i32", body)
    assert flat_fields == ["0", "1"], body
    validate_with_naga(tmp_path, code)


def test_explicit_interpolation_is_kept(tmp_path):
    code = generate_wgsl("""
shader S {
    fragment {
        vec4 main(int id @ location(4) @ interpolate(flat), uvec2 cell @ interpolate(flat, either))
            @ gl_FragColor {
            return vec4(float(id), float(cell.x), 0.0, 1.0);
        }
    }
}
""")
    assert entry_signature(code, "fragment") == (
        "@location(4) @interpolate(flat) id: i32, "
        "@location(0) @interpolate(flat, either) cell: vec2<u32>"
    )
    validate_with_naga(tmp_path, code)


def test_unmapped_system_semantic_parameter_is_refused():
    with pytest.raises(WGSLStageIOUnsupportedError, match="SV_ClipDistance0"):
        generate_wgsl("""
shader S {
    fragment {
        vec4 main(vec4 color, float clip @ SV_ClipDistance0) @ gl_FragColor {
            return color;
        }
    }
}
""")


def test_primitive_index_builtin_is_u32():
    code = generate_wgsl("""
shader Probe {
    struct FragmentInput {
        vec2 uv;
        int pid @ gl_PrimitiveID;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, float(input.pid), 1.0);
            return output;
        }
    }
}
""")
    assert "@builtin(primitive_index) pid: u32" in code
    assert "f32(input.pid)" in code


def test_unmapped_system_semantic_member_is_refused():
    with pytest.raises(WGSLStageIOUnsupportedError, match="VSOut.clip"):
        generate_wgsl("""
shader S {
    struct VSOut {
        vec4 position @ gl_Position;
        float clip @ SV_ClipDistance0;
    }
    vertex {
        VSOut main(vec3 p) {
            VSOut o;
            o.position = vec4(p, 1.0);
            o.clip = p.x;
            return o;
        }
    }
}
""")


@pytest.mark.parametrize("param_type", ["mat4", "vec4 rows[2]"])
def test_matrix_and_array_parameters_are_refused(param_type):
    declaration = (
        f"{param_type.split()[0]} {param_type.split()[1]}"
        if " " in param_type
        else f"{param_type} value"
    )
    with pytest.raises(WGSLStageIOUnsupportedError, match="stage interface struct"):
        generate_wgsl(f"""
shader S {{
    vertex {{
        vec4 main(vec3 p, {declaration}) @ gl_Position {{
            return vec4(p, 1.0);
        }}
    }}
}}
""")


def test_compute_parameters_are_untouched():
    code = generate_wgsl("""
shader S {
    compute {
        layout(std430, binding = 0) buffer Data { float values[]; } data;
        layout(local_size_x = 1) in;
        void main(uvec3 id @ gl_GlobalInvocationID) {
            data.values[id.x] = 1.0;
        }
    }
}
""")
    assert "@builtin(global_invocation_id) id: vec3<u32>" in code
    assert "@location" not in code


@pytest.mark.parametrize(
    "type_name, expected",
    [
        ("i32", True),
        ("u32", True),
        ("vec3<i32>", True),
        ("vec4<u32>", True),
        ("f32", False),
        ("vec2<f32>", False),
        ("bool", False),
        ("mat2x2<f32>", False),
    ],
)
def test_integer_io_type_detection(type_name, expected):
    assert is_integer_io_type(type_name) is expected


def test_flat_interpolation_needs_a_location():
    assert with_flat_integer_interpolation("", "i32") == ""
    assert with_flat_integer_interpolation("@builtin(sample_index)", "u32") == (
        "@builtin(sample_index)"
    )
    assert with_flat_integer_interpolation("@location(1)", "i32") == (
        "@location(1) @interpolate(flat)"
    )
