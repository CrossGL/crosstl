"""Index range assertions scoped to the storage a subscript indexes.

A helper that reads through a pointer can serve several threadgroup arrays of
different sizes. Resource lowering then emits one function that branches on
the pointer's allocation and indexes each array with the same 64-bit
expression. A function-scoped assertion gives every branch the same range, so
it either overstates the smaller array's bound or understates the larger one.
An ``indexed`` scope states the range per array.

The small array is filled and read up to a runtime ``count`` that the host keeps
at 16 or less. Only an assertion can state that bound, so the small array's
subscripts are proven by its scoped range or not at all. The view identity
already decides which array each subscript reaches.
"""

import shutil

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.codegen.index_range_contracts import (
    IndexRangeAssertion,
    IntegerRange,
    parse_index_range_assertions,
)
from tests.test_translator.test_metal_builtin_ownership import _compile

SOURCE = "shared.metal"
ENTRY = "shared_reader"
KERNEL = """#include <metal_stdlib>
using namespace metal;

struct Cursor {
    threadgroup float* tile;
    float read(long index) const { return tile[index]; }
    void write(long index, float value) { tile[index] = value; }
};

[[max_total_threads_per_threadgroup(64)]]
kernel void shared_reader(
    const device float* inputs [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float large[64];
    threadgroup float small[16];
    Cursor wide{large};
    Cursor narrow{small};
    wide.write(long(tid), inputs[tid]);
    if (tid < count) {
        narrow.write(long(tid), inputs[64u + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value = wide.read(long(63u - tid));
    if (tid < count) {
        value += narrow.read(long(count - 1u - tid));
    }
    outputs[tid] = value;
}
"""
OFFSET = "reference.offset + index"
# The device views index 80 inputs and 64 outputs.
DEVICE = (
    {"expression": OFFSET, "function": "crosstl_resource_load_float", "maximum": 79},
    {"expression": OFFSET, "function": "crosstl_resource_store_float", "maximum": 63},
)
LARGE = "crosstl_workgroup_large"
SMALL = "crosstl_workgroup_small"


def _translate(tmp_path, workgroup):
    (tmp_path / SOURCE).write_text(KERNEL, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=(SOURCE,),
        targets=("opengl",),
        output_dir="out",
        entry_points={SOURCE: (ENTRY,)},
        index_range_assertions=[
            {"minimum": 0, **item} for item in (*DEVICE, *workgroup)
        ],
        source_options={
            "metal": {"preserve_resource_origins": True, "threadgroup_size": 64}
        },
    )
    payload = translate_project(config, format_output=False).to_json()
    artifacts = sorted((tmp_path / "out").rglob("*.glsl"))
    return payload, artifacts


def _scoped(indexed, maximum, **extra):
    return {"expression": OFFSET, "indexed": indexed, "maximum": maximum, **extra}


def _index_errors(payload):
    return [
        (
            item["details"]["indexConversion"]["indexedValue"],
            item["details"]["indexConversion"]["reason"],
        )
        for item in payload["diagnostics"]
        if item["code"] == "project.translate.opengl-index-type-unsupported"
    ]


def test_indexed_scope_parses_serializes_and_matches():
    (assertion,) = parse_index_range_assertions(
        [{"expression": " a + b ", "indexed": " tile ", "minimum": 0, "maximum": 7}]
    )
    assert assertion.indexed == "tile"
    assert assertion.to_json() == {
        "source": "*",
        "expression": "a + b",
        "minimum": 0,
        "maximum": 7,
        "indexed": "tile",
    }
    assert assertion.applies_to("a+b", None, "tile")
    assert assertion.applies_to("a + b", "any_function", "tile")
    assert not assertion.applies_to("a + b", None, "other")
    assert not assertion.applies_to("a + b", None, None)
    (unscoped,) = parse_index_range_assertions(
        [{"expression": "a + b", "minimum": 0, "maximum": 7}]
    )
    assert "indexed" not in unscoped.to_json()
    assert unscoped.applies_to("a + b", None, "tile")
    assert unscoped.applies_to("a + b", None, None)


@pytest.mark.parametrize("indexed", ["", "   ", 3])
def test_indexed_scope_must_name_storage(indexed):
    with pytest.raises(ValueError, match="indexed must be non-empty"):
        IndexRangeAssertion("a", IntegerRange(0, 1), indexed=indexed)


def test_one_range_cannot_serve_arrays_of_different_sizes(tmp_path):
    # The larger array needs 63, which overstates the smaller array's bound.
    payload, _ = _translate(
        tmp_path,
        [
            {"expression": OFFSET, "function": name, "maximum": 63}
            for name in (
                "crosstl_workgroup_load_float",
                "crosstl_workgroup_store_float",
            )
        ],
    )
    assert payload["summary"]["translatedCount"] == 0
    assert _index_errors(payload) == [(SMALL, "index-range-out-of-bounds")]


def test_storage_scoped_ranges_narrow_each_array(tmp_path):
    payload, artifacts = _translate(tmp_path, [_scoped(LARGE, 63), _scoped(SMALL, 15)])
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert not payload["diagnostics"]
    assert [
        item.get("indexed") for item in payload["project"]["indexRangeAssertions"]
    ] == [None, None, LARGE, SMALL]
    (artifact,) = artifacts
    generated = artifact.read_text(encoding="utf-8")
    for storage in (LARGE, SMALL):
        assert f"{storage}[int(({OFFSET}))]" in generated
    _, module = _compile(generated, "opengl", tmp_path)
    if shutil.which("glslangValidator"):
        assert module.is_file() and module.stat().st_size


def test_function_and_storage_scopes_combine(tmp_path):
    # Load helpers are proved by the scoped pair; store helpers keep only the
    # large array's range. The small array's stores are bounded only by
    # ``tid < count``, a range far past its 16 elements, so they are refused.
    payload, _ = _translate(
        tmp_path,
        [
            _scoped(LARGE, 63),
            _scoped(SMALL, 15, function="crosstl_workgroup_load_float"),
        ],
    )
    assert payload["summary"]["translatedCount"] == 0
    assert _index_errors(payload) == [(SMALL, "index-range-out-of-bounds")]


def test_storage_scoped_range_still_checks_the_storage_bound(tmp_path):
    payload, _ = _translate(tmp_path, [_scoped(LARGE, 63), _scoped(SMALL, 16)])
    assert payload["summary"]["translatedCount"] == 0
    assert _index_errors(payload) == [(SMALL, "index-range-out-of-bounds")]


def test_storage_scope_does_not_apply_to_other_storage(tmp_path):
    payload, _ = _translate(tmp_path, [_scoped(LARGE, 63)])
    assert payload["summary"]["translatedCount"] == 0
    assert _index_errors(payload) == [(SMALL, "index-range-unproven")]
