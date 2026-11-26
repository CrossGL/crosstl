"""OpenGL software subgroups prove loops controlled by flow-uniform locals.

A loop whose collectives run in software subgroups must make every invocation
of the workgroup reach the same barriers. These kernels start or bound their
loops from locals that are uniform where the loop starts: assigned before it,
from uniform values, under uniform guards, including a helper's ``int2``
result that reads a function constant. Each refusal case breaks one of those
facts.
"""

import pytest

from crosstl.project import ProjectConfig, translate_project
from tests.test_translator.test_metal_builtin_ownership import _compile

SUBGROUP_INVALID = "project.translate.opengl-software-subgroup-invalid"

# 32 SIMD groups of 32 invocations; each walks keys lo + simd_gid, + 32, ...
STRIDED = """#include <metal_stdlib>
using namespace metal;

constant int span [[function_constant(0)]];
constant bool has_span = span > 0;

METAL_FUNC int2 visible(int pos, int n) {
  int lo = max(0, pos - span + 1);
  return int2(lo, min(n, pos + 1));
}

kernel void strided(
    device float* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int lo = 0;
  int hi = N;
  if (GUARD) {
    auto range = visible(int(tid.y), N);
    lo = LOW;
    hi = range.y;
  }
  float total = 0.0f;
  for (int i = START; i < hi; i += 32) {
    bool use = true;
    if ((i % 3) == 0) {
      use = false;
    }
    if (use) {
      float score = float(i);
      score = simd_sum(score);
      total += score;
    }
    BODY
  }
  if (simd_lid == 0) {
    out[tid.x * 32 + simd_gid] = total;
  }
}
"""

# Two SIMD groups, one per query row; every invocation of the workgroup
# shares the key counter i. The chunk loop reads four keys per trip, the tail
# loop the rest.
CHUNKED = """#include <metal_stdlib>
using namespace metal;

kernel void chunked(
    device float* out [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& skip [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  float total = 0.0f;
  for (int i = 0; i < 2; i++) {
    total += float(i);
  }
  int i = int(tid.z);
  if (GUARD) {
    int off = (skip / 128) * 128;
    ADVANCE
  }
  PRELUDE
  for (; i + 3 * 8 < N; i += 8 * 4) {
    for (int t = 0; t < 4; t++) {
      total += simd_sum(float(i + t * 8));
    }
  }
  for (; i < N; i += 8) {
    total += simd_sum(float(i));
  }
  out[(tid.z * 2 + tidtg.y) * 32 + simd_lid] = total;
}
"""


def _translate(root, source, entry, workgroup, *, maximum=None, constants=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.metal").write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("kernel.metal",),
            targets=("opengl",),
            output_dir="out",
            entry_points={"kernel.metal": entry},
            workgroup_size=workgroup,
            specialization_constants=constants or {},
            metadata_preconditions=(
                []
                if maximum is None
                else [
                    {
                        "source": "kernel.metal",
                        "entryPoint": entry,
                        "parameter": "N",
                        "minimum": 0,
                        "maximum": maximum,
                    }
                ]
            ),
            source_options={
                "metal": {
                    "preserve_resource_origins": True,
                    "target_options": {"opengl": {"software_subgroup_width": 32}},
                }
            },
        ),
        format_output=False,
    )
    payload = report.to_json()
    codes = [item["code"] for item in payload["diagnostics"]]
    generated = None
    if not codes:
        (artifact,) = payload["artifacts"]
        generated = (root / artifact["path"]).read_text(encoding="utf-8")
    return codes, generated


def _strided(**parts):
    source = STRIDED
    for name, default in (
        ("GUARD", "has_span"),
        ("LOW", "range.x"),
        ("START", "lo + int(simd_gid)"),
        ("BODY", ""),
    ):
        source = source.replace(name, parts.get(name.lower(), default))
    return source


def _chunked(**parts):
    source = CHUNKED
    for name, default in (
        ("GUARD", "skip > 128"),
        ("ADVANCE", "i += off;"),
        ("PRELUDE", ""),
    ):
        source = source.replace(name, parts.get(name.lower(), default))
    return source


@pytest.mark.parametrize(
    "start", ["lo + int(simd_gid)", "int(simd_gid) + lo", "lo + simd_gid"]
)
def test_strided_loop_starts_from_a_uniform_local(tmp_path, start):
    codes, generated = _translate(
        tmp_path, _strided(start=start), "strided", (1024, 1, 1), constants={"0": 4}
    )
    assert codes == []
    lines = [line.strip() for line in generated.splitlines()]
    loop = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("for (int crossglSoftwareSubgroupBase = lo;")
    )
    # Every invocation runs the base from lo to hi in steps of the subgroup
    # count; each adds its subgroup index and masks the trips past hi.
    assert lines[loop].endswith("crossglSoftwareSubgroupBase += 32) {")
    assert lines[loop + 1].startswith("int i = (crossglSoftwareSubgroupBase + ")
    assert "bool crossglSoftwareSubgroupLoopActive = (i < hi);" in lines
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "parts",
    [
        # The start differs between invocations of one subgroup.
        {"low": "range.x + int(simd_lid)"},
        # Only some invocations take the branch that moves the start.
        {"guard": "has_span && simd_lid == 0"},
        # The bound changes while the loop runs.
        {"body": "hi -= 1;"},
        # A stride other than the subgroup count skips keys in some subgroup.
        {"start": "lo + int(simd_gid) * 2"},
    ],
)
def test_strided_loop_refuses_nonuniform_controls(tmp_path, parts):
    codes, _ = _translate(
        tmp_path, _strided(**parts), "strided", (1024, 1, 1), constants={"0": 4}
    )
    assert SUBGROUP_INVALID in codes


def test_helper_reading_device_memory_is_not_uniform(tmp_path):
    source = _strided().replace(
        "METAL_FUNC int2 visible(int pos, int n) {\n  int lo = max(0, pos - span + 1);",
        "METAL_FUNC int2 visible(int pos, int n, device float* p) {\n"
        "  int lo = max(0, pos - int(p[0]));",
    )
    source = source.replace("visible(int(tid.y), N)", "visible(int(tid.y), N, out)")
    codes, _ = _translate(tmp_path, source, "strided", (1024, 1, 1), constants={"0": 4})
    assert SUBGROUP_INVALID in codes


def test_counter_advanced_under_a_uniform_guard_runs_chunk_and_tail_loops(tmp_path):
    codes, generated = _translate(
        tmp_path, _chunked(), "chunked", (32, 2, 1), maximum=2147482624
    )
    assert codes == []
    # Both loops keep their source form: every invocation counts alike.
    assert "for (; ((i + (3 * 8)) < N); i += (8 * 4)) {" in generated
    assert "for (; (i < N); i += 8) {" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "parts,maximum",
    [
        # Only some invocations advance the counter.
        ({"guard": "skip > 128 && simd_lid == 0"}, 2147482624),
        ({"advance": "i += off + int(simd_lid);"}, 2147482624),
        # A write inside another loop is outside the flow facts.
        ({"prelude": "for (int k = 0; k < 2; k++) { i += 1; }"}, 2147482624),
        # i + 24 < N admits a sum less than one chunk step (32) below
        # INT_MAX, so the step after it could overflow.
        ({}, 2147483640),
        # Nothing bounds N.
        ({}, None),
    ],
)
def test_chunk_loop_refuses_unproven_counters(tmp_path, parts, maximum):
    codes, _ = _translate(
        tmp_path, _chunked(**parts), "chunked", (32, 2, 1), maximum=maximum
    )
    assert SUBGROUP_INVALID in codes
