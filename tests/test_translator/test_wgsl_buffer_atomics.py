"""Atomic built-ins on buffers and workgroup memory become WGSL ``atomic<T>`` accesses.

WGSL types atomic memory ``atomic<i32>`` or ``atomic<u32>``, passes a pointer to every atomic built-in and allows no
other access to it (WGSL sections 6.2.8 and 17.8). The generator printed GLSL-style calls on plain ``u32`` memory,
``atomicAdd(g, 1u)``, which naga rejects. HLSL ``groupshared`` and Metal function-scope ``threadgroup`` variables
became private or function variables, giving every invocation its own copy. Strong compare-exchanges printed as
``atomicCompSwap``, which WGSL lacks; its ``atomicCompareExchangeWeak`` may fail spuriously, so the lowering retries.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_buffer_atomics import WGSLAtomicLoweringError
from crosstl.translator.codegen.wgsl_workgroup_locals import WGSLWorkgroupLocalError

PROGRAM = """shader main {
    struct B { uint head; int tail; uint data[]; };
    struct R { uint ro; };
    B b @glsl_buffer_block(std430) @binding(0);
    R r @glsl_buffer_block(std430) @binding(1) @readonly;
    shared uint g;
    shared int h[4];
    shared float gf;
    HELPERS
    compute {
    layout(local_size_x = 64) in;
    void main() {
        uint i = gl_LocalInvocationIndex;
        BODY
    }
    }
}
"""


def _translate(tmp_path, body, helpers="", suffix=".cgl", source=None):
    path = tmp_path / f"shader{suffix}"
    text = source if source is not None else PROGRAM
    path.write_text(
        text.replace("HELPERS", helpers).replace("BODY", body), encoding="utf-8"
    )
    return crosstl.translate(str(path), backend="wgsl")


def _validate(tmp_path, output):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "out.wgsl"
    path.write_text(output, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr + output


def _body(output):
    return re.sub(r"\s+", " ", output)


def test_atomic_targets_are_atomic_memory_addressed_by_pointer(tmp_path):
    output = _translate(
        tmp_path,
        """
        if (i == 0u) {
            atomicStore(g, 0u);
            atomicStore(h[1], 5);
        }
        barrier();
        uint a = atomicLoad(b.head);
        atomicStore(b.tail, int(a));
        int s = atomicLoad(h[1]);
        atomicExchange(g, a);
        b.data[i] = a + uint(s) + atomicAdd(g, 1u) + atomicLoad(g);
        """,
    )
    assert "head: atomic<u32>," in output
    assert "tail: atomic<i32>," in output
    # The data array is never an atomic target, so it stays plain memory.
    assert "data: array<u32>," in output
    assert "var<workgroup> g: atomic<u32>;" in output
    assert "var<workgroup> h: array<atomic<i32>, 4>;" in output
    assert "var<workgroup> gf: f32;" in output
    for call in (
        "atomicStore(&g, 0u)",
        "atomicStore(&h[1], 5)",
        "atomicLoad(&b.head)",
        "atomicStore(&b.tail, i32(a))",
        "atomicAdd(&g, 1u)",
    ):
        assert call in output, output
    _validate(tmp_path, output)


def test_plain_accesses_of_atomic_memory_load_and_store(tmp_path):
    output = _translate(
        tmp_path,
        """
        atomicAdd(g, 1u);
        atomicMin(h[i % 4u], 3);
        g = 7u;
        g += 2u;
        g++;
        h[i % 4u] -= 1;
        --h[0];
        b.head = g * 2u + uint(h[2]);
        """,
    )
    text = _body(output)
    assert "atomicStore(&g, 7u);" in text
    assert "atomicStore(&g, (atomicLoad(&g) + 2u));" in text
    assert "atomicStore(&g, (atomicLoad(&g) + 1u));" in text
    assert "atomicStore(&h[(i % 4u)], (atomicLoad(&h[(i % 4u)]) - 1));" in text
    assert "atomicStore(&h[0], (atomicLoad(&h[0]) - 1));" in text
    assert "b.head = ((atomicLoad(&g) * 2u) + u32(atomicLoad(&h[2])));" in text
    _validate(tmp_path, output)


def test_helpers_reach_atomic_memory_by_pointer(tmp_path):
    output = _translate(
        tmp_path,
        "add(i); b.head = twice(g) + uint(atomicAdd(g, 1u));",
        helpers="""
    uint twice(uint v) { return v * 2u; }
    void add(uint k) { atomicAdd(b.tail, int(k)); }
        """,
    )
    assert "atomicAdd(&b.tail, i32(k));" in output
    assert "b.head = (twice(atomicLoad(&g)) + u32(atomicAdd(&g, 1u)));" in output
    _validate(tmp_path, output)


@pytest.mark.parametrize(
    "body, helpers, count",
    [
        ("uint old = atomicCompSwap(b.data[i], 0u, i); b.head = old;", "", 1),
        ("atomicCompSwap(b.data[i], 0u, i);", "", 1),
        (
            "if (atomicCompSwap(g, 0u, 1u) == 0u) { b.head = 1u; }"
            " else if (atomicCompSwap(g, 1u, 2u) == 1u) { b.head = 2u; }",
            "",
            2,
        ),
        ("while (atomicCompSwap(g, 0u, 1u) != 0u) { } atomicExchange(g, 0u);", "", 1),
        (
            "for (uint k = 0u; atomicCompareExchange(g, k, k + 1u) != k; k++) { b.head = k; }",
            "",
            1,
        ),
        ("b.head = take();", "uint take() { return atomicCompSwap(g, 0u, 1u); }", 1),
    ],
)
def test_strong_compare_exchange_retries_the_weak_exchange(
    tmp_path, body, helpers, count
):
    output = _translate(tmp_path, body, helpers=helpers)
    assert "atomicCompSwap" not in output
    assert "atomicCompareExchange(" not in output
    retries = re.findall(
        r"if \((_crosstl_cas_\d+)\.exchanged \|\| \1\.old_value != \1_compare\) \{\s+break;",
        output,
    )
    assert len(retries) == count, output
    _validate(tmp_path, output)


def test_compare_exchange_in_a_loop_condition_is_tested_every_iteration(tmp_path):
    output = _translate(
        tmp_path, "while (atomicCompSwap(g, 0u, 1u) != 0u) { b.head += 1u; }"
    )
    text = _body(output)
    loop = text.index("loop {")
    exchange = text.index("var _crosstl_cas_0 = atomicCompareExchangeWeak", loop)
    exit_test = text.index("if (!(_crosstl_cas_0_old != 0u)) { break; }", exchange)
    assert text.index("b.head += 1u;", exit_test) > exit_test
    assert "while" not in text
    _validate(tmp_path, output)


def test_explicit_atomic_types_keep_scalar_value_typing(tmp_path):
    source = """shader main {
    struct C { atomic<uint> head; atomic<int> data[]; };
    C c @glsl_buffer_block(std430) @binding(0);
    shared atomic<uint> g;
    compute {
    layout(local_size_x = 64) in;
    void main() {
        atomicStore(g, 0u);
        uint a = atomicAdd(c.head, 1u);
        atomicMax(c.data[gl_GlobalInvocationID.x], int(a) + 1);
        uint d = atomicLoad(g) + 2u;
    }
    }
}
"""
    output = _translate(tmp_path, "", source=source)
    assert "head: atomic<u32>," in output
    assert "data: array<atomic<i32>>," in output
    assert "var<workgroup> g: atomic<u32>;" in output
    assert "var d: u32 = (atomicLoad(&g) + 2u);" in output
    _validate(tmp_path, output)


def test_hlsl_interlocked_and_groupshared_are_workgroup_atomics(tmp_path):
    source = """RWStructuredBuffer<uint> counts : register(u0);
StructuredBuffer<uint> weights : register(t0);
groupshared uint total;
groupshared float scratch[64];
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint old;
    scratch[gi] = 1.0;
    InterlockedAdd(counts[id.x], weights[id.x], old);
    InterlockedMax(total, old);
    uint prev;
    InterlockedCompareExchange(counts[0], 5, 7, prev);
    counts[1] = total + prev + uint(scratch[0]);
}
"""
    output = _translate(tmp_path, "", suffix=".hlsl", source=source)
    assert "var<storage, read_write> counts: array<atomic<u32>>;" in output
    assert "var<storage, read> weights: array<u32>;" in output
    assert "var<workgroup> total: atomic<u32>;" in output
    assert "var<workgroup> scratch: array<f32, 64>;" in output
    assert "old = atomicAdd(&counts[id.x], weights[id.x]);" in output
    assert (
        "atomicStore(&counts[1], ((atomicLoad(&total) + prev) + u32(scratch[0])));"
        in output
    )
    _validate(tmp_path, output)


def test_function_scope_workgroup_variables_move_to_module_scope(tmp_path):
    source = """shader main {
    compute {
    @ numthreads(64, 1, 1)
    void k(RWStructuredBuffer<uint> counter @buffer(0), RWStructuredBuffer<float> out_ @buffer(1),
           uint id @gl_GlobalInvocationID, uint lid @gl_LocalInvocationIndex) {
        threadgroup int count;
        threadgroup float tile[64];
        tile[lid] = float(id);
        if (lid == 0u) {
            atomicStore(count, 0);
        }
        workgroupBarrier();
        uint old = atomicAdd(counter[0], 1u);
        atomicMax(count, int(old));
        workgroupBarrier();
        buffer_store(out_, id, tile[63u - lid] + float(atomicLoad(count)));
    }
    }
}
"""
    output = _translate(tmp_path, "", source=source)
    assert "var<storage, read_write> counter: array<atomic<u32>>;" in output
    assert "var<workgroup> count: atomic<i32>;" in output
    assert "var<workgroup> tile: array<f32, 64>;" in output
    assert "var count" not in output and "var tile" not in output
    assert "tile[lid] = f32(id);" in output
    assert "atomicMax(&count, i32(old));" in output
    _validate(tmp_path, output)


@pytest.mark.parametrize(
    "body, helpers, reason",
    [
        ("atomicAdd(gf, 1.0);", "", "atomic f32 memory"),
        ("atomicAdd(r.ro, 1u);", "", "read address space"),
        ("atomicAdd(h[0], 1); int c[4] = h;", "", "a value of h holds atomic memory"),
        ("uint x = 0u; atomicAdd(x, 1u); b.head = x;", "", "first argument"),
        ("b.head = i > 0u ? atomicCompSwap(g, 0u, 1u) : 0u;", "", "conditionally"),
        (
            "if (i > 0u && atomicCompSwap(g, 0u, 1u) == 0u) { b.head = 1u; }",
            "",
            "conditionally",
        ),
        ("do { } while (atomicCompSwap(g, 0u, 1u) != 0u);", "", "do-while"),
        (
            "atomicAdd(g, 1u); bump(g);",
            "void bump(inout uint v) { v = v + 1u; }",
            "out or inout parameter of bump",
        ),
        (
            "atomicAdd(h[0], 1); h[idx()] += 1;",
            "uint idx() { return 1u; }",
            "subscripts twice",
        ),
        ("atomicAdd(g, 1u); uint y = (g = 3u);", "", "used as a value"),
        ("atomicAdd(g, 1u); b.head = g++;", "", "used with \\+\\+ as a value"),
        ("atomicAdd(b.head, 1u, 2u);", "", "takes 2 arguments"),
    ],
)
def test_atomic_accesses_wgsl_cannot_express_are_refused(
    tmp_path, body, helpers, reason
):
    with pytest.raises(WGSLAtomicLoweringError, match=reason):
        _translate(tmp_path, body, helpers=helpers)


def test_structure_with_atomic_members_in_read_only_storage_is_refused(tmp_path):
    source = """shader main {
    struct B { uint head; };
    B w @glsl_buffer_block(std430) @binding(0);
    B ro @glsl_buffer_block(std430) @binding(1) @readonly;
    compute {
    layout(local_size_x = 64) in;
    void main() {
        atomicAdd(w.head, ro.head);
    }
    }
}
"""
    with pytest.raises(WGSLAtomicLoweringError, match="ro, of structure B"):
        _translate(tmp_path, "", source=source)


def test_workgroup_local_with_an_initializer_is_refused(tmp_path):
    source = """shader main {
    compute {
    @ numthreads(64, 1, 1)
    void k(RWStructuredBuffer<uint> out_ @buffer(0), uint id @gl_GlobalInvocationID) {
        threadgroup uint seen = 0u;
        buffer_store(out_, id, seen);
    }
    }
}
"""
    with pytest.raises(WGSLWorkgroupLocalError, match="has no initializer"):
        _translate(tmp_path, "", source=source)


def test_programs_without_atomics_are_not_copied(tmp_path):
    output = _translate(tmp_path, "b.head = i;")
    assert "atomic" not in output
