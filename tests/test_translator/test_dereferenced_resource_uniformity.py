"""Reads through a dereferenced constant resource are uniform on software subgroups.

Metal kernels usually take their parameters as ``const constant Params*
params`` and leave early with ``if (params->tiles_n <= int(tid.x)) return;``
before a subgroup barrier. The importer spells the read ``(*params).tiles_n``,
which reads the resource's first element exactly as ``params[0].tiles_n``
does. Every invocation of a workgroup reads the same immutable value, so the
exit is workgroup-uniform and the software subgroup lowering may keep the
barrier. Writable and device-memory views, and reads at a lane-dependent
address, stay unproven.
"""

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.ast import (
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    UnaryOpNode,
)
from crosstl.translator.codegen.uniform_returns import UniformReturnAnalysis
from tests.test_translator.test_metal_builtin_ownership import _compile

WORKGROUP = (64, 1, 1)
UNPROVEN = {
    "directx": (
        "DirectX software subgroup barriers and early returns require "
        "statically uniform control flow"
    ),
    "opengl": (
        "OpenGL software subgroup barriers cannot follow an unproven "
        "divergent return, break, or continue"
    ),
}


def _source(parameter="const constant Params* params", exit="params->tiles_n"):
    return f"""#include <metal_stdlib>
using namespace metal;
struct Params {{
  int tiles_n;
  int tiles_m;
  int swizzle_log;
}};

kernel void tiles(
    {parameter} [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {{
  const int tid_x = int(tid.x) >> params->swizzle_log;
  if ({exit} <= tid_x || (*params).tiles_m <= static_cast<int>(tid.y)) {{
    return;
  }}
  float value = float(lane);
  simdgroup_barrier(mem_flags::mem_none);
  value = simd_sum(value);
  out[tid.x * 2 + simd_group] = value;
}}
"""


def _translate(root, target, source):
    (root / "tiles.metal").write_text(source, encoding="utf-8")
    options = {"software_subgroup_width": 32}
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("tiles.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=WORKGROUP,
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()
    (artifact,) = report["artifacts"]
    return artifact


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_constant_parameter_exits_keep_the_software_barrier(tmp_path, target):
    artifact = _translate(tmp_path, target, _source())

    assert artifact["status"] == "translated", artifact.get("error")
    generated = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    barrier = "GroupMemoryBarrierWithGroupSync" if target == "directx" else "barrier()"
    assert barrier in generated
    assert "WaveActiveSum" not in generated and "subgroupAdd" not in generated
    _compile(generated, target, tmp_path)


UNPROVEN_SOURCES = {
    # Other invocations may write a mutable view during the dispatch.
    "writable-view": {"parameter": "device Params* params"},
    "device-view": {"parameter": "const device Params* params"},
    # Each lane reads its own element.
    "lane-address": {"exit": "(params + lane)->tiles_n"},
}


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("name", sorted(UNPROVEN_SOURCES))
def test_unproven_reads_keep_the_exit_diagnostic(tmp_path, target, name):
    artifact = _translate(tmp_path, target, _source(**UNPROVEN_SOURCES[name]))

    assert artifact["status"] == "failed"
    assert UNPROVEN[target] in str(artifact.get("error"))


def test_storage_load_reads_the_first_element_of_a_dereferenced_name():
    analysis = UniformReturnAnalysis({}, set())
    params = IdentifierNode("params")

    resource, index = analysis.storage_load(UnaryOpNode("*", params))

    assert resource is params
    assert isinstance(index, LiteralNode) and index.value == 0
    assert analysis.storage_load(UnaryOpNode("-", params)) is None
    assert analysis.storage_load(UnaryOpNode("*", UnaryOpNode("*", params))) is None
    # A call is a load only when the target resolved it to `buffer_load`.
    call = FunctionCallNode(
        IdentifierNode("buffer_load"), [params, LiteralNode(0, PrimitiveType("int"))]
    )
    assert analysis.storage_load(call) is None
