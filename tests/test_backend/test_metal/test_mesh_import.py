"""Metal ``[[mesh]]`` functions import as CrossGL mesh stages.

The importer used to print a mesh function verbatim as a plain helper that still called the ``metal::mesh``
methods, so no target got a mesh stage. It now rewrites the ``metal::mesh<V, P, NV, NP, topology>`` parameter into
``@vertices``, ``@primitives`` and ``@indices`` out arrays, ``set_primitive_count`` into ``SetMeshOutputCounts`` and
stages ``set_index`` writes so that every target receives whole index vectors.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused
from crosstl.translator.host_workgroup_size import HostDefinedWorkgroupSizeError

TRIANGLES = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float3 normal; };
struct POut { float3 color; };
struct Payload { uint count; float3 color; };
using TriMesh = metal::mesh<VOut, POut, 64, 32, metal::topology::triangle>;

[[mesh, max_total_threads_per_threadgroup(32)]]
void meshMain(TriMesh output,
              const object_data Payload& payload [[payload]],
              uint lid [[thread_index_in_threadgroup]])
{
    output.set_primitive_count(payload.count);
    if (lid < payload.count) {
        VOut v;
        v.position = float4(float(lid), 0.0, 0.0, 1.0);
        v.normal = float3(0.0, 0.0, 1.0);
        output.set_vertex(lid, v);
        POut p;
        p.color = payload.color;
        output.set_primitive(lid, p);
        output.set_index(3 * lid + 0, lid);
        output.set_index(3 * lid + 1, lid);
        output.set_index(3 * lid + 2, lid);
    }
}
"""

LINES = """#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; };
using LineMesh = metal::mesh<V, void, 4, 2, metal::topology::line>;
[[mesh]] void lines(LineMesh output, uint lid [[thread_index_in_threadgroup]]) {
    output.set_primitive_count(2);
    if (lid < 4) {
        V v;
        v.position = float4(float(lid), 0.0, 0.0, 1.0);
        output.set_vertex(lid, v);
        output.set_index(lid, lid);
    }
}
"""

POINTS = """#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; float size [[point_size]]; };
[[mesh]] void pts(metal::mesh<V, void, 8, 8, metal::topology::point> output,
                  uint lid [[thread_index_in_threadgroup]]) {
    output.set_primitive_count(8);
    V v;
    v.position = float4(float(lid), 0.0, 0.0, 1.0);
    v.size = 1.0;
    output.set_vertex(lid, v);
    output.set_index(lid, lid + 256);
}
"""

SIZE = {"threadgroup_size": (8, 1, 1)}


def _translate(tmp_path, source, backend, source_options=None):
    path = tmp_path / "mesh.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_options=source_options
    )


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_triangle_mesh_function_imports_as_a_mesh_stage(tmp_path):
    crossgl = _translate(tmp_path, TRIANGLES, "cgl")

    assert "set_primitive_count" not in crossgl
    assert "set_vertex" not in crossgl
    assert "set_index" not in crossgl
    assert "mesh {\n        @ host_workgroup_size\n" in crossgl
    assert (
        "void meshMain(out VOut output_vertices[64] @vertices, "
        "out POut output_primitives[32] @primitives, "
        "out uvec3 output_indices[32] @indices, "
        "in Payload payload @mesh_payload, uint lid @gl_LocalInvocationIndex) "
        "@max_total_threads_per_threadgroup(32) @outputtopology(triangle) {"
    ) in crossgl
    assert "groupshared uint output_index_values[96];" in crossgl
    assert "uint output_primitive_count = uint(payload.count);" in crossgl
    assert "SetMeshOutputCounts(64u, output_primitive_count);" in crossgl
    assert "output_vertices[lid] = v;" in crossgl
    assert "output_primitives[lid] = p;" in crossgl
    assert "output_index_values[3 * lid + 1] = uint(lid) & 255u;" in crossgl
    barrier = crossgl.index("workgroupBarrier();")
    copy = crossgl.index(
        "output_indices[primitive] = uvec3(output_index_values[3u * primitive + 0u], "
        "output_index_values[3u * primitive + 1u], output_index_values[3u * primitive + 2u]);"
    )
    assert crossgl.index("if (lid == 0u) {") < copy
    assert barrier < copy


def test_point_mesh_stores_8_bit_indices_directly(tmp_path):
    crossgl = _translate(tmp_path, POINTS, "cgl")

    assert "out uint output_indices[8] @indices" in crossgl
    assert "@outputtopology(point)" in crossgl
    assert "output_indices[lid] = uint(lid + 256) & 255u;" in crossgl
    assert "groupshared" not in crossgl
    assert "workgroupBarrier" not in crossgl


def test_mesh_without_a_thread_index_parameter_gains_one_for_the_index_copy(
    tmp_path,
):
    source = LINES.replace(", uint lid [[thread_index_in_threadgroup]]", "").replace(
        "lid", "0"
    )
    crossgl = _translate(tmp_path, source, "cgl")

    assert "uint output_invocation @gl_LocalInvocationIndex) @outputtopology(line)" in (
        crossgl
    )
    assert "if (output_invocation == 0u) {" in crossgl


def test_mesh_stage_with_host_chosen_size_needs_the_threadgroup_size(tmp_path):
    with pytest.raises(HostDefinedWorkgroupSizeError, match="Entry 'lines'"):
        _translate(tmp_path, LINES, "vulkan")

    crossgl = _translate(tmp_path, LINES, "cgl", SIZE)
    assert "@ host_workgroup_size(8, 1, 1)" in crossgl
    assert "[numthreads(8, 1, 1)]" in _translate(tmp_path, LINES, "directx", SIZE)


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            TRIANGLES.replace(
                "    output.set_primitive_count(payload.count);\n", ""
            ).replace(
                "if (lid < payload.count) {",
                "if (lid < payload.count) {\n"
                "        output.set_primitive_count(payload.count);",
            ),
            "must call set_primitive_count exactly once, as a top-level statement",
        ),
        (
            TRIANGLES.replace("    output.set_primitive_count(payload.count);\n", ""),
            "must call set_primitive_count exactly once",
        ),
        (
            TRIANGLES.replace(
                "    output.set_primitive_count(payload.count);\n",
                "    if (lid > 40) { return; }\n    output.set_primitive_count(payload.count);\n",
            ),
            "writes an output or returns before set_primitive_count",
        ),
        (
            TRIANGLES.replace(
                "        output.set_index(3 * lid + 2, lid);\n",
                "        output.set_index(3 * lid + 2, lid);\n        return;\n",
            ),
            "returns early, but its staged set_index writes",
        ),
        (
            TRIANGLES.replace(
                "        output.set_index(3 * lid + 0, lid);\n",
                "        output.set_indices(3 * lid, uchar2(lid));\n",
            ),
            "calls metal::mesh::set_indices",
        ),
        (
            TRIANGLES.replace(
                "        output.set_index(3 * lid + 0, lid);\n",
                "        output.set_primitive_count(payload.count);\n",
            ),
            "must call set_primitive_count exactly once",
        ),
        (
            TRIANGLES.replace(
                "using TriMesh = metal::mesh<VOut, POut, 64, 32, metal::topology::triangle>;",
                "using TriMesh = metal::mesh<VOut, POut, 64, 32, metal::topology::quad>;",
            ),
            "topology 'metal::topology::quad' is not point, line or triangle",
        ),
        (
            TRIANGLES.replace(
                "{\n    output.set_primitive_count",
                "{\n    TriMesh copy = output;\n    output.set_primitive_count",
            ),
            "uses its metal::mesh parameter 'output' other than in statement-level set_\\* calls",
        ),
        (
            TRIANGLES + """
[[object]] void objectMain(object_data Payload& payload [[payload]], mesh_grid_properties grid) {
    payload.count = 1;
    grid.set_threadgroups_per_grid(uint3(1, 1, 1));
}
""",
            "Metal object function 'objectMain'",
        ),
    ],
)
def test_mesh_constructs_without_a_faithful_form_are_refused(tmp_path, source, reason):
    with pytest.raises(MetalImportRefused, match=reason):
        _translate(tmp_path, source, "cgl")


@pytest.mark.parametrize("source", [LINES, POINTS, TRIANGLES])
def test_imported_mesh_stage_compiles_for_metal(tmp_path, source):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not available")
    if _run(["xcrun", "-sdk", "macosx", "--find", "metal"]).returncode != 0:
        pytest.skip("the Metal compiler is not available")
    metal = _translate(tmp_path, source, "metal", SIZE)
    assert "[[mesh" in metal
    shader = tmp_path / "roundtrip.metal"
    shader.write_text(metal, encoding="utf-8")
    result = _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "roundtrip.air"),
        ]
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", [LINES, POINTS])
def test_imported_mesh_stage_validates_as_glsl_and_spirv(tmp_path, source):
    glslang = shutil.which("glslangValidator")
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (glslang and spirv_as and spirv_val):
        pytest.skip("glslangValidator, spirv-as and spirv-val are required")

    glsl = tmp_path / "mesh.mesh"
    glsl.write_text(_translate(tmp_path, source, "opengl", SIZE), encoding="utf-8")
    result = _run(
        [
            glslang,
            "-S",
            "mesh",
            "--target-env",
            "vulkan1.3",
            "-o",
            str(tmp_path / "glsl.spv"),
            str(glsl),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assembly = _translate(tmp_path, source, "vulkan", SIZE)
    assert "OutputPrimitivesEXT 1\n" not in assembly
    spvasm = tmp_path / "mesh.spvasm"
    spvasm.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "mesh.spv"
    result = _run(
        [spirv_as, "--target-env", "vulkan1.3", str(spvasm), "-o", str(binary)]
    )
    assert result.returncode == 0, result.stderr
    result = _run([spirv_val, "--target-env", "vulkan1.3", str(binary)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_imported_triangle_mesh_and_fragment_compile_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not available")
    source = TRIANGLES + (
        "fragment float4 fragMain(VOut in [[stage_in]]) "
        "{ return float4(in.normal, 1.0); }\n"
    )
    hlsl = _translate(tmp_path, source, "directx", {"threadgroup_size": (32, 1, 1)})
    assert re.search(r"float3 normal\s*:\s*TEXCOORD0;", hlsl)
    assert re.search(r"float3 color\s*:\s*TEXCOORD1;", hlsl)
    shader = tmp_path / "mesh.hlsl"
    shader.write_text(hlsl, encoding="utf-8")
    for profile, entry in (("ms_6_5", "MSMain"), ("ps_6_5", "PSMain")):
        result = _run([dxc, "-T", profile, "-E", entry, str(shader)])
        assert result.returncode == 0, result.stdout + result.stderr
