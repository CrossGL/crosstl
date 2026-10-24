"""Private aggregates keep 16-bit shared allocations and record copies exact."""

import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import FLOAT16_BITS, FLOAT32_BITS
from crosstl.translator import parse
from crosstl.translator.ast import NamedType, PrimitiveType, VariableNode
from crosstl.translator.codegen.resource_aggregates import (
    ResourceAggregateError,
    _resolve_narrow_float_aliases,
    lower_resource_aggregates,
)
from tests.test_translator.test_bfloat_buffer_runtime import _storage
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_NARROW_WORKGROUP_RESOURCE_AGGREGATES"
NATIVE_TARGET = {"darwin": "metal", "win32": "directx", "linux": "opengl"}.get(
    sys.platform
)
GROUPS = 2
TILE = 10
# Even slots are copied bit for bit; odd slots are then doubled exactly.
WORDS = {
    "half": (
        (0x0001, 0x3555, 0x8000, 0x0200, 0x7BFF, 0xC000, 0x7C00, 0x5000),
        (0xFC00, 0x3C00, 0x0400, 0x8001, 0x8001, 0x3555, 0x3C00, 0xC000),
    ),
    "bfloat16_t": (
        # Doubling runs in binary32, which Metal flushes when subnormal, so
        # bfloat16 subnormals only occupy copied slots.
        (0x0001, 0x3EAB, 0x8000, 0x0080, 0x7F7F, 0xC000, 0x7F80, 0x4200),
        (0xFF80, 0x3F80, 0x0080, 0x8080, 0x8001, 0x3EAB, 0x3F80, 0xC000),
    ),
}
GUARD = {"half": 0xC300, "bfloat16_t": 0xC060}  # -3.5
ELEMENT = {"half": "float16", "bfloat16_t": "bfloat16"}


def _double(kind, word):
    if kind == "half":
        value = struct.unpack("<e", struct.pack("<H", word))[0]
        return struct.unpack("<H", struct.pack("<e", value * 2.0))[0]
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0] * 2.0
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    # Doubling a bfloat16 value is exact, so no rounding bits remain.
    assert bits & 0xFFFF == 0
    return bits >> 16


def _source(kind):
    alias = "typedef bfloat bfloat16_t;" if kind == "bfloat16_t" else ""
    return f"""#include <metal_stdlib>
using namespace metal;
{alias}
struct Loader {{
  struct alignas(sizeof({kind})) ReadVector {{ uint8_t bytes[sizeof({kind}) * 2]; }};
  const device {kind}* src;
  threadgroup {kind}* dst;
  Loader(const device {kind}* source, threadgroup {kind}* destination, uint lane)
      : src(source + 2u * lane), dst(destination + 2u * lane) {{}}
  void load_unsafe() const {{
    *((threadgroup ReadVector*)(&dst[0])) = *((const device ReadVector*)(&src[0]));
  }}
  void scale() const {{ dst[1] = {kind}(float(dst[1]) * 2.0f); }}
}};
kernel void narrow_shared(const device {kind}* inputs [[buffer(0)]],
                          device {kind}* results [[buffer(1)]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint3 group [[threadgroup_position_in_grid]]) {{
  threadgroup {kind} tile[{TILE}];
  for (uint i = lane; i < {TILE}u; i += 4u) {{ tile[i] = {kind}(-3.5f); }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Loader loader(inputs + 8u * group.x, tile + 1, lane);
  loader.load_unsafe();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  loader.scale();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < {TILE}u; i += 4u) {{ results[4u + {TILE}u * group.x + i] = tile[i]; }}
}}
"""


def _expected(kind):
    words = [GUARD[kind]] * 4
    for group in WORDS[kind]:
        words.append(GUARD[kind])
        words.extend(
            _double(kind, word) if index % 2 else word
            for index, word in enumerate(group)
        )
        words.append(GUARD[kind])
    return words + [GUARD[kind]] * 4


def _value(kind, target, words):
    if kind == "bfloat16_t":
        return _storage(target, words)
    if target == "opengl":
        # OpenGL widens binary16 buffers to binary32 words of the same value.
        return {
            "dtype": "float32",
            "shape": [len(words)],
            "encoding": FLOAT32_BITS,
            "values": [
                struct.unpack(
                    "<I",
                    struct.pack("<f", struct.unpack("<e", struct.pack("<H", word))[0]),
                )[0]
                for word in words
            ],
        }
    return {
        "dtype": "float16",
        "shape": [len(words)],
        "encoding": FLOAT16_BITS,
        "values": list(words),
    }


def _assertions(kind, target):
    workgroup = [
        {
            "source": "products.metal",
            "entryPoint": "narrow_shared",
            "function": "Loader__load_unsafe",
            "parameter": "self.dst",
            # Lanes write slots 1 through 8 of the ten-element tile.
            "minimum": 1,
            "maximum": 2 * 4,
        }
    ]
    ranges = [
        {
            "source": "products.metal",
            "function": f"crosstl_{storage}_{operation}_{ELEMENT[kind]}{suffix}",
            "expression": "reference.offset + index",
            "minimum": 0,
            "maximum": TILE - 1 if storage == "workgroup" else 8 + GROUPS * TILE - 1,
        }
        for storage in ("workgroup", "resource")
        for operation in ("load", "store")
        for suffix in ("", "_2")
    ] + [
        # The record copy reads two source elements past each lane's offset.
        {
            "source": "products.metal",
            "expression": f"source.offset + {element}",
            "minimum": 0,
            "maximum": 8 * GROUPS - 1,
        }
        for element in range(2)
    ]
    return workgroup, ranges if target == "opengl" else ()


def _request(root, target, kind):
    workgroup, ranges = _assertions(kind, target)
    source, descriptor, package = _package(
        root,
        target,
        kind,
        (4, 1, 1),
        source=_source(kind),
        software_subgroups=False,
        workgroup_access_assertions=workgroup if target != "metal" else (),
        index_range_assertions=ranges,
    )
    inputs_words = [word for group in WORDS[kind] for word in group]
    expected = _expected(kind)
    initial = [GUARD[kind]] * 4 + [0x5555] * (len(expected) - 8) + [GUARD[kind]] * 4
    inputs = _bound_values(
        descriptor,
        {
            "inputs": _value(kind, target, inputs_words),
            "results": _value(kind, target, initial),
        },
    )
    outputs = _bound_values(descriptor, {"results": _value(kind, target, expected)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": [4, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


def test_expected_words_cover_copies_scaling_and_guards():
    for kind in WORDS:
        expected = _expected(kind)
        assert len(expected) == 8 + GROUPS * TILE
        assert expected[4] == expected[13] == GUARD[kind]
        assert expected[5] == WORDS[kind][0][0]
        assert expected[6] == _double(kind, WORDS[kind][0][1])
    assert _double("half", 0x0200) == 0x0400
    assert _double("bfloat16_t", 0x8080) == 0x8100


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("kind", ["half", "bfloat16_t"])
def test_narrow_shared_aggregate_compiles(tmp_path, target, kind):
    _, request, _ = _request(tmp_path, target, kind)
    generated = request.artifact_path.read_text()
    if target != "metal":
        assert f"crosstl_workgroup_copy_{ELEMENT[kind]}" in generated
        assert f"crosstl_workgroup_store_{ELEMENT[kind]}" in generated
        assert not re.search(r"\bbfloat16_t\b", generated)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("kind", ["half", "bfloat16_t"])
def test_narrow_shared_aggregate_executes(tmp_path, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required narrow shared execution")
    source, request, expected = _request(tmp_path, NATIVE_TARGET, kind)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if NATIVE_TARGET == "metal" else None,
        original_entry="narrow_shared",
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


def test_narrow_shared_aggregate_execution_is_required():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_narrow_workgroup_resource_aggregates.py" in step
    assert "if:" not in step and "continue-on-error" not in step


def _lowered(kind, *, extent_bits=None):
    def run(tmp_path):
        path = tmp_path / "narrow.metal"
        path.write_text(_source(kind))
        ast = parse(translate(str(path), backend="cgl", format_output=False))
        for node in ast.walk():
            if extent_bits and isinstance(node, VariableNode) and node.name == "tile":
                node.var_type.element_type = PrimitiveType("uint", size_bits=16)
        return lower_resource_aggregates(
            ast,
            workgroup_access_assertions=_assertions(kind, "directx")[0],
        )

    return run


@pytest.mark.parametrize("kind", ["half", "bfloat16_t"])
def test_narrow_shared_lowering_names_the_storage_type(tmp_path, kind):
    lowered = _lowered(kind)(tmp_path)
    names = {
        node.name
        for node in lowered.walk()
        if isinstance(node, (PrimitiveType, NamedType))
    }
    assert ELEMENT[kind] in names
    assert "bfloat16_t" not in names


def test_narrow_integer_shared_allocations_remain_rejected(tmp_path):
    with pytest.raises(ResourceAggregateError) as error:
        _lowered("half", extent_bits=16)(tmp_path)
    assert error.value.reason == "workgroup-allocation-layout"


def test_aliases_of_wider_types_are_not_respelled(tmp_path):
    path = tmp_path / "narrow.metal"
    path.write_text(_source("bfloat16_t"))
    ast = parse(translate(str(path), backend="cgl", format_output=False))
    alias = next(
        node
        for node in ast.global_variables
        if isinstance(node, VariableNode) and node.is_type_alias
    )
    alias.var_type = PrimitiveType("float")

    def spellings():
        return Counter(
            node.name
            for node in ast.walk()
            if isinstance(node, (PrimitiveType, NamedType))
            and node is not alias.var_type
        )

    before = spellings()
    assert before["bfloat16_t"] == 5
    _resolve_narrow_float_aliases(ast)
    assert spellings() == before
    alias.var_type = PrimitiveType("bfloat16")
    _resolve_narrow_float_aliases(ast)
    after = spellings()
    assert after["bfloat16_t"] == 0
    assert after["bfloat16"] == before["bfloat16"] + 5
