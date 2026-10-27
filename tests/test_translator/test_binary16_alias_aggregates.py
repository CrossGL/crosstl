"""Aggregates over ``float16_t`` storage match their ``half`` bindings.

Metal sources commonly spell binary16 through an alias, ``typedef half float16_t``.
Entry buffers of the alias are bound as CrossGL ``float16`` while aggregate members
and workgroup arrays keep the alias name, so resource-aggregate lowering must spell
the alias by its CrossGL type. Otherwise the member's pointer contract mismatches its
own binding and a ``float16_t`` workgroup array has no recognised layout.
"""

import os
import struct
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_BINARY16_ALIAS_AGGREGATES"
ENTRY = "stage_halves"
THREADS = 32
WORKGROUP = (THREADS, 1, 1)
# Exact binary16 values: small integers and halves.
VALUES = [(index - 13) * 0.5 for index in range(THREADS + 1)]
GUARD = [-7.0, 0.25, 1024.0, -3.5]

SOURCE = f"""#include <metal_stdlib>
using namespace metal;

typedef half float16_t;

template <typename T>
struct TileLoader {{
  const device T* src;
  threadgroup T* dst;
  TileLoader(const device T* src_, threadgroup T* dst_, uint lane)
      : src(src_ + lane + 1), dst(dst_ + lane) {{}}
  void load() const {{
    *dst = *src;
  }}
}};

[[kernel]] void {ENTRY}(
    const device float16_t* values [[buffer(0)]],
    device float* results [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]]) {{
  threadgroup float16_t tile[{THREADS}];
  TileLoader<float16_t> loader(values, tile, lane);
  loader.load();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  results[lane] = float(tile[{THREADS - 1} - lane]) * 2.0f;
}}
"""


def _expected():
    return [VALUES[THREADS - 1 - lane + 1] * 2.0 for lane in range(THREADS)] + GUARD


def _request(root, target):
    _, descriptor, package = _package(
        root, target, "float", WORKGROUP, source=SOURCE, software_subgroups=False
    )
    expected = _expected()
    halves = [struct.unpack("<e", struct.pack("<e", value))[0] for value in VALUES]
    assert halves == VALUES
    initial = [99.0] * THREADS + GUARD
    # OpenGL widens binary16 storage to float32; DirectX keeps its bits.
    dtypes = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding[
            "scalarLayout"
        ]["elementType"]
        for binding in descriptor["bindings"]
    }
    dtype = "float32" if dtypes["values"] == "float32" else "float16"
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": dtype, "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "float32", "shape": [len(initial)], "values": initial},
        },
    )
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "float32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, outputs


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_binary16_alias_aggregates_compile(tmp_path, target):
    request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "crosstl_resource_ref_float16" in generated
    assert "crosstl_workgroup_tile" in generated
    _compile(generated, target, tmp_path)


def test_binary16_alias_aggregates_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native binary16 aliases")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=SOURCE,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_binary16_alias_aggregates_are_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_binary16_alias_aggregates.py" in step
    assert "if:" not in step and "continue-on-error" not in step
