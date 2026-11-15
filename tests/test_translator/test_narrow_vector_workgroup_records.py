"""Records may point at shared arrays of 16-bit floating vectors."""

import pytest

from crosstl._crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.resource_aggregates import (
    ResourceAggregateError,
    lower_resource_aggregates,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

TILE = 32
ELEMENT = {
    "half2": "f16vec2",
    "half4": "f16vec4",
    "bfloat2": "bfloat16vec2",
    "bfloat4": "bfloat16vec4",
}


def _source(kind, element=None):
    element = element or kind
    return f"""#include <metal_stdlib>
using namespace metal;

struct Rows {{
  const device {kind}* in;
  threadgroup {element}* buf;
  device {kind}* out;

  void load(int index) const {{ buf[index] = {element}(in[index]); }}

  void write(int index, int source) const {{ out[index] = {kind}(buf[source]); }}
}};

kernel void rotate(const device {kind}* in [[buffer(0)]],
                   device {kind}* out [[buffer(1)]],
                   uint lane [[thread_position_in_threadgroup]]) {{
  threadgroup {element} shared[{TILE}];
  Rows rows = {{in, shared, out}};
  rows.load(lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rows.write(lane, (lane + 1) % {TILE});
}}
"""


def _index_ranges(kind):
    return [
        {
            "source": "products.metal",
            "function": f"crosstl_{storage}_{operation}_{ELEMENT[kind]}{suffix}",
            "expression": "reference.offset + index",
            "minimum": 0,
            "maximum": TILE - 1,
        }
        for storage in ("workgroup", "resource")
        for operation in ("load", "store")
        for suffix in ("", "_2")
    ]


def _lower(tmp_path, kind, element=None):
    path = tmp_path / "rotate.metal"
    path.write_text(_source(kind, element))
    ast = parse(translate(str(path), backend="cgl", format_output=False))
    return lower_resource_aggregates(ast)


@pytest.mark.parametrize("kind", sorted(ELEMENT))
def test_shared_vector_allocation_is_lowered(tmp_path, kind):
    lowered = _lower(tmp_path, kind)
    shared = [
        node
        for node in lowered.global_variables
        if getattr(node, "name", "").startswith("crosstl_workgroup_shared")
    ]
    assert len(shared) == 1
    assert shared[0].var_type.element_type.name == ELEMENT[kind]


@pytest.mark.parametrize("element", ["half3", "short2", "ushort4"])
def test_other_narrow_shared_vectors_remain_rejected(tmp_path, element):
    with pytest.raises(ResourceAggregateError) as error:
        _lower(tmp_path, "float4" if element == "half3" else "int2", element)
    assert error.value.reason == "workgroup-allocation-layout"


@pytest.mark.parametrize(
    ("target", "kind"),
    [
        ("directx", "half2"),
        ("directx", "half4"),
        ("opengl", "half2"),
        ("opengl", "half4"),
        # DirectX refuses bfloat16 vector device buffers on their own.
        ("opengl", "bfloat2"),
        ("opengl", "bfloat4"),
    ],
)
def test_shared_narrow_vector_record_compiles(tmp_path, target, kind):
    _package(
        tmp_path,
        target,
        kind,
        (TILE, 1, 1),
        source=_source(kind),
        software_subgroups=False,
        index_range_assertions=_index_ranges(kind) if target == "opengl" else (),
    )
    suffix = {"directx": "hlsl", "opengl": "glsl"}[target]
    generated = (tmp_path / "out" / target / f"products.{suffix}").read_text()
    assert f"crosstl_workgroup_store_{ELEMENT[kind]}" in generated
    # DirectX keeps binary16 lanes in shared memory; OpenGL widens 16-bit
    # floats to binary32 as it does for scalar ``half`` and ``bfloat``.
    shared = {"directx": f"float16_t{kind[-1]}", "opengl": f"vec{kind[-1]}"}[target]
    prefix = {"directx": "groupshared", "opengl": "shared"}[target]
    assert f"{prefix} {shared} crosstl_workgroup_shared" in generated
    _compile(generated, target, tmp_path)
