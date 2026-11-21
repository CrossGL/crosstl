"""Metal output places CrossGL's rasterizer-ordered resources in raster order group 0.

``@rasterizer_ordered`` orders the accesses of overlapping fragments to a resource by primitive order. MSL states
the same with ``[[raster_order_group(0)]]`` on a fragment function's device buffer or texture argument. The
generator dropped the attribute, so the accesses of overlapping fragments ran unordered.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

FRAGMENT = textwrap.dedent("""
    shader S {
        DECLARATION
        struct FragmentInput {
            vec4 coord @ gl_FragCoord;
            vec2 uv;
        }
        struct FragmentOutput { vec4 color; }
        fragment {
            FragmentOutput main(FragmentInput input) {
                FragmentOutput output;
                uint k = uint(input.coord.y) * 64u + uint(input.coord.x);
                ivec2 p = ivec2(input.coord.xy);
                BODY
                output.color = vec4(input.uv, 0.0, 1.0);
                return output;
            }
        }
    }
    """)

DECLARATIONS = {
    "structured-buffer": (
        "RWStructuredBuffer<vec4> accum @rasterizer_ordered;",
        "accum[k] = accum[k] + vec4(input.uv, 0.0, 1.0);",
        "device float4* accum [[buffer(0)]] [[raster_order_group(0)]]",
    ),
    "read-only-structured-buffer": (
        "StructuredBuffer<vec4> accum @rasterizer_ordered;",
        "output.color = accum[k];",
        "const device float4* accum [[buffer(0)]] [[raster_order_group(0)]]",
    ),
    "buffer-pointer": (
        "buffer vec4* accum @rasterizer_ordered;",
        "accum[k] = accum[k] + vec4(input.uv, 0.0, 1.0);",
        "device float4* accum [[buffer(0)]] [[raster_order_group(0)]]",
    ),
    "storage-image": (
        "image2D accum @rasterizer_ordered @rgba32f;",
        "imageStore(accum, p, imageLoad(accum, p) + vec4(input.uv, 0.0, 1.0));",
        "texture2d<float, access::read_write> accum [[texture(0)]] [[raster_order_group(0)]]",
    ),
}


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _assert_compiles(tmp_path, metal):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(shader)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" not in result.stderr:
        assert result.returncode == 0, result.stderr


def _fragment(declaration, body):
    return FRAGMENT.replace("DECLARATION", declaration).replace("BODY", body)


@pytest.mark.parametrize("kind", sorted(DECLARATIONS))
def test_rasterizer_ordered_global_joins_raster_order_group_zero(tmp_path, kind):
    declaration, body, argument = DECLARATIONS[kind]

    metal = _translate(tmp_path, _fragment(declaration, body))

    assert argument in metal
    _assert_compiles(tmp_path, metal)


def test_unordered_resource_has_no_raster_order_group(tmp_path):
    declaration, body, _ = DECLARATIONS["structured-buffer"]
    source = _fragment(declaration.replace(" @rasterizer_ordered", ""), body)

    metal = _translate(tmp_path, source)

    assert "device float4* accum [[buffer(0)]]" in metal
    assert "raster_order_group" not in metal


def test_rasterizer_ordered_entry_parameter(tmp_path):
    source = textwrap.dedent("""
        shader S {
            struct FragmentIn {
                vec4 coord @gl_Position;
                vec2 uv;
            }
            fragment {
                @ stage_entry
                vec4 probe(FragmentIn in_, RWStructuredBuffer<vec4> accum @buffer(3) @rasterizer_ordered) {
                    uint k = uint(in_.coord.y) * 64u + uint(in_.coord.x);
                    buffer_store(accum, k, buffer_load(accum, k) + vec4(in_.uv, 0.0, 1.0));
                    return vec4(in_.uv, 0.0, 1.0);
                }
            }
        }
        """)

    metal = _translate(tmp_path, source)

    assert "device float4* accum [[buffer(3)]] [[raster_order_group(0)]]" in metal
    _assert_compiles(tmp_path, metal)


def test_array_of_ordered_buffers_orders_every_element(tmp_path):
    source = _fragment(
        "RWStructuredBuffer<vec4> accum[2] @rasterizer_ordered;",
        "accum[1][k] = accum[0][k] + vec4(input.uv, 0.0, 1.0);",
    )

    metal = _translate(tmp_path, source)

    assert metal.count("[[raster_order_group(0)]]") == 2
    _assert_compiles(tmp_path, metal)


def test_compute_stage_takes_no_raster_order_group(tmp_path):
    # Metal rejects the attribute on a kernel argument, and a kernel has no overlapping fragments to order.
    source = textwrap.dedent("""
        shader S {
            RWStructuredBuffer<vec4> accum @rasterizer_ordered;
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    accum[i] = accum[i] + vec4(1.0);
                }
            }
        }
        """)

    metal = _translate(tmp_path, source)

    assert "raster_order_group" not in metal
    _assert_compiles(tmp_path, metal)


def test_vertex_stage_of_the_same_program_takes_no_raster_order_group(tmp_path):
    source = textwrap.dedent("""
        shader S {
            RWStructuredBuffer<vec4> accum @rasterizer_ordered;
            struct VertexInput { vec4 position @ POSITION; }
            struct VertexOutput { vec4 position @ gl_Position; }
            struct FragmentInput { vec4 coord @ gl_FragCoord; }
            struct FragmentOutput { vec4 color; }
            vertex {
                VertexOutput main(VertexInput input) {
                    VertexOutput output;
                    output.position = input.position + accum[0];
                    return output;
                }
            }
            fragment {
                FragmentOutput main(FragmentInput input) {
                    FragmentOutput output;
                    uint k = uint(input.coord.x);
                    accum[k] = accum[k] + vec4(1.0);
                    output.color = accum[k];
                    return output;
                }
            }
        }
        """)

    metal = _translate(tmp_path, source)
    vertex, fragment = metal.split("fragment FragmentOutput", 1)

    assert "raster_order_group" not in vertex
    assert "[[buffer(0)]] [[raster_order_group(0)]]" in fragment
    _assert_compiles(tmp_path, metal)
