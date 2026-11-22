"""Metal atomics that order memory keep their order through CrossGL's relaxed atomics and fences.

Metal's atomic functions without ``_explicit`` are sequentially consistent over all memory (MSL 4.1, section 6.15.4).
The importer used to copy them into CrossGL verbatim, where no generator knows them. It now writes the relaxed CrossGL
atomic between two sequentially consistent fences when the call is a whole statement, and refuses it elsewhere. An
``_explicit`` read-modify-write with an order other than ``memory_order_relaxed`` used to lose its order; it is
refused now.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.ordered_atomics import MetalAtomicOrderLoweringError

FENCE = (
    "atomicThreadFence(mem_device | mem_threadgroup | mem_threadgroup_imageblock | mem_texture | "
    "mem_object_data, memory_order_seq_cst, thread_scope_device);"
)

KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void k(device int* values [[buffer(0)]],
              device atomic_int* counters [[buffer(1)]],
              device atomic_uint* flags [[buffer(2)]],
              device atomic_ulong* wide [[buffer(3)]],
              uint id [[thread_position_in_grid]]) {
    int operand = values[id];
    atomic_fetch_add(&counters[id], operand);
    int previous = atomic_fetch_sub(&counters[id], 2);
    uint seen;
    seen = atomic_load(&flags[id]);
    atomic_store(&flags[id], seen + 1u);
    int expected = operand;
    bool swapped = atomic_compare_exchange_weak(&counters[id], &expected, previous);
    atomic_max(&wide[id], ulong(operand));
    atomic_min_explicit(&wide[id + 1], ulong(seen), memory_order_relaxed);
    values[id] = previous + expected + int(swapped);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _order_error(tmp_path, source):
    with pytest.raises(Exception) as raised:
        _translate(tmp_path, source, "cgl")
    error = raised.value
    while error is not None and not isinstance(error, MetalAtomicOrderLoweringError):
        error = error.__cause__ or error.__context__
    assert isinstance(error, MetalAtomicOrderLoweringError), raised.value
    return error


def _lines(crossgl):
    return [line.strip() for line in crossgl.splitlines() if line.strip()]


def _fenced(lines, statement):
    index = lines.index(statement)
    return lines[index - 1] == FENCE and lines[index + 1] == FENCE


def test_sequentially_consistent_atomics_import_between_fences(tmp_path):
    lines = _lines(_translate(tmp_path, KERNEL, "cgl"))

    assert _fenced(lines, "atomicAdd(counters[id], operand);")
    assert _fenced(lines, "int previous = atomicAdd(counters[id], -(2));")
    assert _fenced(lines, "seen = atomicLoad(flags[id]);")
    assert _fenced(lines, "atomicStore(flags[id], seen + 1u);")
    assert _fenced(
        lines,
        "bool swapped = atomicCompareExchangeWeak(counters[id], expected, previous);",
    )
    assert _fenced(lines, "atomicMax(wide[id], uint64(operand));")
    # A relaxed operation needs no fence; the one before it closes atomic_max.
    relaxed = "atomicMin(wide[id + 1], uint64(seen));"
    assert relaxed in lines and lines[lines.index(relaxed) + 1] != FENCE
    assert lines.count(FENCE) == 12
    assert not any("atomic_" in line for line in lines)


@pytest.mark.parametrize(
    ("statement", "reason"),
    [
        (
            "if (atomic_load(&flags[id]) == 0u) { values[id] = 1; }",
            "statement of its own",
        ),
        ("values[id] = atomic_fetch_add(&counters[id], 1);", "local variable"),
        ("atomic_fetch_add(&counters[id], helper(id));", "source function 'helper'"),
        ("atomic_fetch_add(&counters[id], values[id]++);", "increment or decrement"),
        (
            "atomic_fetch_add(&counters[id], atomic_load_explicit(&counters[0], memory_order_relaxed));",
            "the atomic 'atomic_load_explicit'",
        ),
    ],
)
def test_sequentially_consistent_atomic_outside_a_whole_statement_is_refused(
    tmp_path, statement, reason
):
    source = f"""#include <metal_stdlib>
using namespace metal;
int helper(uint id) {{ return int(id); }}
kernel void k(device int* values [[buffer(0)]],
              device atomic_int* counters [[buffer(1)]],
              device atomic_uint* flags [[buffer(2)]],
              uint id [[thread_position_in_grid]]) {{
    {statement}
}}
"""
    error = _order_error(tmp_path, source)
    assert reason in str(error)


@pytest.mark.parametrize(
    "call",
    [
        "atomic_fetch_add_explicit(&counters[id], 1, memory_order_seq_cst)",
        "atomic_exchange_explicit(&counters[id], 1, memory_order_acq_rel)",
        "atomic_fetch_sub_explicit(&counters[id], 1, memory_order_seq_cst, mem_flags::mem_device)",
    ],
)
def test_explicit_read_modify_write_keeps_only_a_relaxed_order(tmp_path, call):
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void k(device atomic_int* counters [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
    {call};
}}
"""
    error = _order_error(tmp_path, source)
    assert "only memory_order_relaxed" in str(error)


def test_explicit_read_modify_write_with_relaxed_order_and_memory_flags_imports(
    tmp_path,
):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void k(device atomic_int* counters [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&counters[id], 1, memory_order_relaxed, mem_flags::mem_device);
}
"""
    assert "atomicAdd(counters[id], 1);" in _translate(tmp_path, source, "cgl")


def test_ordered_atomics_round_trip_compiles(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not available")
    for name, source in (
        ("source", KERNEL),
        ("translated", _translate(tmp_path, KERNEL, "metal")),
    ):
        path = tmp_path / f"{name}.metal"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [xcrun, "-sdk", "macosx", "metal", "-std=metal4.1", "-c", str(path)]
            + ["-o", str(tmp_path / f"{name}.air")],
            capture_output=True,
            text=True,
            check=False,
        )
        if "unable to find utility" in result.stderr:
            pytest.skip("the Metal compiler is not installed")
        assert result.returncode == 0, result.stderr
