"""Rust output writes rust-gpu (spirv_std) shaders, or refuses what rust-gpu cannot express.

Each translated program is checked for the spelling that carries its meaning and, when a rust-gpu build is
configured (``CROSSTL_COVERAGE_RUST_GPU``, see ``tools/coverage/rustgpu.py``), compiled with rust-gpu for Vulkan 1.1
the way the coverage harness compiles its Rust probes.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_rust_codegen import assert_generated_rust_smoke_compiles


def rust(source: str) -> str:
    """Rust output for CrossGL ``source``, through the whole pipeline the coverage harness uses."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "program.cgl")
        path.write_text(source, encoding="utf-8")
        return crosstl.translate(str(path), backend="rust", format_output=False)


def entry_signature(code: str, attribute: str) -> str:
    """The parameter list of the entry point after ``attribute``."""
    match = re.search(re.escape(attribute) + r"[^\n]*\npub fn \w+\((.*?)\) -> \(\) \{", code, re.S)
    assert match, code
    return match.group(1)


VERTEX_FRAGMENT = """
shader Lit {
    struct VertexInput {
        vec3 position @ POSITION;
        vec2 uv @ TEXCOORD0;
        int instance @ gl_InstanceID;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
        int layer @ TEXCOORD1;
    }
    struct FragmentInput {
        vec2 uv @ TEXCOORD0;
        int layer @ TEXCOORD1;
        vec4 coord @ gl_FragCoord;
    }
    struct FragmentOutput {
        vec4 color @ SV_Target0;
        float depth @ gl_FragDepth;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = vec4(input.position, 1.0);
            output.uv = input.uv;
            output.layer = input.instance;
            return output;
        }
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            if (input.uv.x < 0.0) {
                discard;
            }
            output.color = vec4(input.uv, float(input.layer), 1.0);
            output.depth = input.coord.z;
            return output;
        }
    }
}
"""


def test_vertex_and_fragment_interfaces_are_entry_point_parameters(tmp_path):
    code = rust(VERTEX_FRAGMENT)

    vertex = entry_signature(code, "#[spirv(vertex)]")
    # POSITION and TEXCOORDn take the locations every other target gives them (stage_io_semantic_locations).
    assert "#[spirv(location = 0)] input_position: Vec3" in vertex
    assert "#[spirv(location = 5)] input_uv: Vec2" in vertex
    # gl_InstanceID excludes the base instance.
    assert "#[spirv(instance_index)] input_instance: i32" in vertex
    assert "#[spirv(base_instance)] input_instance_base: i32" in vertex
    assert "#[spirv(position)] out_position: &mut Vec4" in vertex
    # Integer vertex outputs are flat, as the fragment inputs they feed must be.
    assert "#[spirv(location = 6, flat)] out_layer: &mut i32" in vertex
    assert "instance: input_instance.wrapping_sub(input_instance_base)" in code

    fragment = entry_signature(code, "#[spirv(fragment(depth_replacing))]")
    assert "#[spirv(location = 6, flat)] input_layer: i32" in fragment
    assert "#[spirv(frag_coord)] input_coord: Vec4" in fragment
    assert "#[spirv(location = 0)] out_color: &mut Vec4" in fragment
    assert "#[spirv(frag_depth)] out_depth: &mut f32" in fragment
    assert "spirv_std::arch::kill();" in code
    # Each return writes every output member through its parameter.
    assert re.search(r"\*out_color = (\w+)\.color;\s*\*out_depth = \1\.depth;\s*return;", code)

    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_fragment_value_output_and_early_fragment_tests(tmp_path):
    code = rust(
        """
        shader Flat {
            fragment {
                layout(early_fragment_tests) in;
                vec4 main(vec4 position @ gl_FragCoord, uint id @ gl_PrimitiveID) @ gl_FragColor {
                    return vec4(position.xy, float(id), 1.0);
                }
            }
        }
        """
    )
    fragment = entry_signature(code, "#[spirv(fragment(early_fragment_tests))]")
    assert "#[spirv(frag_coord)] position_in: Vec4" in fragment
    # Integer fragment inputs are flat, built-ins included (VUID-StandaloneSpirv-Flat-04744).
    assert "#[spirv(primitive_id, flat)] id_in: u32" in fragment
    assert "#[spirv(location = 0)] out_value: &mut Vec4" in fragment

    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_fragment_derivatives_are_spirv_std_derivative_calls(tmp_path):
    code = rust(
        """
        shader Edges {
            struct FragmentInput { vec2 uv; }
            struct FragmentOutput { vec4 color; }
            fragment {
                FragmentOutput main(FragmentInput input) {
                    FragmentOutput output;
                    float d = fwidth(input.uv.x) + dFdx(input.uv.y);
                    output.color = vec4(dFdy(input.uv), d, 1.0);
                    return output;
                }
            }
        }
        """
    )
    assert "spirv_std::arch::Derivative::fwidth(" in code
    assert "spirv_std::arch::Derivative::dfdx(" in code
    assert "spirv_std::arch::Derivative::dfdy(" in code

    assert_generated_rust_smoke_compiles(code, tmp_path)


@pytest.mark.parametrize(
    "member, reason",
    [
        ("vec2 uv @ noperspective;", "noperspective interpolation"),
        ("bool on;", "has type bool"),
        ("mat2x3 m;", "has type mat2x3"),
        ("vec4 uv @ gl_ClipDistance;", "rust-gpu fragment input"),
    ],
)
def test_fragment_inputs_rust_gpu_cannot_declare_are_refused(member, reason):
    source = f"""
    shader Bad {{
        struct FragmentInput {{ {member} }}
        fragment {{
            vec4 main(FragmentInput input) @ gl_FragColor {{ return vec4(1.0); }}
        }}
    }}
    """
    with pytest.raises(RustLoweringRefused, match=reason):
        rust(source)


def test_geometry_stage_is_refused():
    with pytest.raises(RustLoweringRefused, match="geometry entry points are not lowered"):
        rust(
            """
            shader G {
                struct VertexOutput { vec4 position @ gl_Position; }
                geometry {
                    layout(points) in;
                    layout(points, max_vertices = 1) out;
                    void main(VertexOutput input[1]) {
                        EmitVertex();
                    }
                }
            }
            """
        )


COMPUTE = """
shader Kernel {
    const int TILE = 4;
    const float SCALE = 0.5;
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer int* data;
        buffer uint* bits;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int k = 0;
            float tile[TILE * 2];
            for (;;) {
                tile[k++] = SCALE;
                if (k >= TILE) {
                    break;
                }
            }
            int total = 0;
            for (int j = 0, m = 2; j < TILE; j++, m += 2) {
                switch (j) {
                    case 0:
                    case 1:
                        total += m;
                        break;
                    case 2:
                        total -= 1;
                    default:
                        total += 10;
                        break;
                }
            }
            data[i] = total + k;
            bits[i] = -bits[i];
        }
    }
}
"""


def test_compute_statements_lower_to_rust_gpu(tmp_path):
    code = rust(COMPUTE)

    assert "#[spirv(compute(threads(64, 1, 1)))]" in code
    # Constants are const items; an integer-constant array extent is folded.
    assert re.search(r"const TILE: i32 = 4;", code)
    assert "[f32; 8]" in code
    # for(;;) loops forever; a C switch is a labeled block around a match whose arms fall through.
    assert "while true" in code or "loop {" in code
    assert re.search(r"'\w+: \{\s*match ", code)
    assert re.search(r"0 \| 1 =>", code)
    # Unsigned negation wraps, as C defines it.
    assert "wrapping_neg()" in code

    assert_generated_rust_smoke_compiles(code, tmp_path)
