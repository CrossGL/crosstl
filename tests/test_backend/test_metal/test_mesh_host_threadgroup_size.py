"""A Metal mesh or object function keeps the host's threadgroup size through a Metal round trip.

Metal's ``[[mesh]]`` and ``[[object]]`` functions declare no threadgroup size: the host passes
``threadsPerObjectThreadgroup`` and ``threadsPerMeshThreadgroup`` when it draws. The importer marks such a stage
``@ host_workgroup_size``, and a stage whose result depends on the size (the importer stages 8-bit mesh indices and
the object payload in threadgroup memory behind a barrier) keeps the mark. Metal output then asked for a static
size to write ``max_total_threads_per_threadgroup`` and refused the round trip. It now writes no limit, as the
source does, and still writes the one a ``threadgroup_size`` source option or a CrossGL size gives.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

MESH = textwrap.dedent("""\
    #include <metal_stdlib>
    using namespace metal;

    struct Vertex {
        float4 position [[position]];
    };

    [[mesh]] void probe(mesh<Vertex, void, 2, 1, topology::line> output) {
        output.set_primitive_count(1);
        output.set_vertex(0, Vertex{float4(0.0, 0.0, 0.0, 1.0)});
        output.set_vertex(1, Vertex{float4(1.0, 0.0, 0.0, 1.0)});
        output.set_index(0, 0);
        output.set_index(1, 1);
    }
    """)

OBJECT = textwrap.dedent("""\
    #include <metal_stdlib>
    using namespace metal;

    struct Payload {
        uint values[2];
    };

    [[object]] void probe(object_data Payload& payload [[payload]], mesh_grid_properties grid,
                          uint index [[thread_index_in_threadgroup]]) {
        payload.values[index] = 1;
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
    """)

CROSSGL_MESH = textwrap.dedent("""\
    shader S {
        struct Vertex {
            vec4 position @gl_Position;
        };
        mesh {
            layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
            @max_vertices(3) @max_primitives(1) @outputtopology(triangle)
            void main(out Vertex verts[3] @vertices, out uvec3 tris[1] @indices) {
                SetMeshOutputCounts(3u, 1u);
                verts[0] = Vertex(vec4(0.0, 0.0, 0.0, 1.0));
                verts[1] = Vertex(vec4(1.0, 0.0, 0.0, 1.0));
                verts[2] = Vertex(vec4(0.0, 1.0, 0.0, 1.0));
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """)

SOURCES = {"mesh": MESH, "object": OBJECT}


def _translate(tmp_path, source, suffix=".metal", **options):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", **options)


def _entry_line(code, stage):
    return next(line for line in code.splitlines() if f"[[{stage}" in line)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "round_trip.metal").write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "round_trip.metal", "-o", "a.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage", sorted(SOURCES))
def test_round_trip_keeps_the_host_size(tmp_path, stage):
    code = _translate(tmp_path, SOURCES[stage])
    assert "threadgroup_barrier" in code
    assert "max_total_threads_per_threadgroup" not in _entry_line(code, stage)
    _compile(tmp_path, code)


@pytest.mark.parametrize("stage", sorted(SOURCES))
def test_threadgroup_size_option_sets_the_limit(tmp_path, stage):
    code = _translate(
        tmp_path, SOURCES[stage], source_options={"threadgroup_size": (32, 1, 1)}
    )
    assert "max_total_threads_per_threadgroup(32)" in code
    _compile(tmp_path, code)


def test_crossgl_mesh_size_sets_the_limit(tmp_path):
    code = _translate(tmp_path, CROSSGL_MESH, suffix=".cgl")
    assert "max_total_threads_per_threadgroup(4)" in code
    _compile(tmp_path, code)
