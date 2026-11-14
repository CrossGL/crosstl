"""Coherence and ``volatile`` of buffers that Metal output passes to helpers or reads through byte offsets.

MSL makes ``coherent(scope)`` and ``volatile`` part of a pointer type, and a conversion may add them but never
drop them. Metal output used to declare helper parameters without them, so ``xcrun metal`` rejected the call
("would lose coherent(device) qualifier"). A GLSL buffer block lowered to ``device uchar*`` lost them through
``reinterpret_cast<device T*>``, which compiles to a plain load or store.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import (
    ArrayAccessNode,
    BinaryOpNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
)
from crosstl.translator.codegen.metal_buffer_coherence import (
    buffer_argument_root_name,
    coherent_array_parameter_declaration,
    memory_helper_token,
    merge_memory_contracts,
)


def translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def compact(code):
    """Join wrapped lines and drop the spaces clang-format puts around ``*`` and after ``(``."""
    code = re.sub(r"\s*\*\s*", "*", re.sub(r"\s+", " ", code))
    return re.sub(r"\(\s+", "(", code)


def metal_ir(tmp_path, code):
    """Compile with ``xcrun metal -std=metal3.2`` and return the textual AIR."""
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "out.metal").write_text(code)
    command = ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-S"]
    command += ["-emit-llvm", "-c", "out.metal", "-o", "-"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    if result.returncode != 0 and "missing Metal Toolchain" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_helper_parameter_inherits_coherence_of_its_argument(tmp_path):
    code = translate(
        tmp_path,
        """shader A {
    buffer uint data[] @binding(0) @coherent;
    void bump(buffer uint values[], uint i) {
        values[i] = values[i] + 1u;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            bump(data, 0u);
        }
    }
}""",
        "metal",
    )

    # MSL refuses ``coherent`` on a parameter written with array syntax, so the parameter is the pointer.
    assert "void bump(coherent device uint*values, uint i)" in compact(code)
    assert "coherent device uint*data [[buffer(0)]]" in compact(code)
    ir = metal_ir(tmp_path, code)
    assert "@air.load.device_coherent.i32" in ir
    assert "@air.store.device_coherent.i32" in ir


def test_chained_helpers_take_the_widest_contracts_of_all_callers(tmp_path):
    code = translate(
        tmp_path,
        """shader A {
    buffer uint data[] @binding(0) @coherent;
    buffer uint plain[] @binding(1);
    buffer uint vol[] @binding(2) @volatile;
    void inner(buffer uint values[], uint i) {
        values[i] = values[i] + 1u;
    }
    void outer(buffer uint values[], uint i) {
        inner(values, i);
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            outer(data, 0u);
            outer(plain, 1u);
            inner(vol, 2u);
        }
    }
}""",
        "metal",
    )

    flat = compact(code)
    assert "void inner(volatile coherent device uint*values, uint i)" in flat
    assert "void outer(coherent device uint*values, uint i)" in flat
    assert "device uint*plain [[buffer(1)]]" in flat
    assert "coherent device uint*plain" not in flat
    ir = metal_ir(tmp_path, code)
    assert "@air.store.device_coherent.volatile.i32" in ir


def test_threadgroup_coherence_widens_to_device_coherence(tmp_path):
    code = translate(
        tmp_path,
        """shader A {
    buffer uint near[] @binding(0) @coherent(threadgroup);
    buffer uint far[] @binding(1) @coherent(device);
    void put(buffer uint values[], uint i) {
        values[i] = 1u;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            put(near, 0u);
            put(far, 1u);
        }
    }
}""",
        "metal",
    )

    flat = compact(code)
    assert "void put(coherent(device) device uint*values, uint i)" in flat
    assert "coherent(threadgroup) device uint*near [[buffer(0)]]" in flat
    metal_ir(tmp_path, code)


def test_helper_called_only_with_plain_buffers_stays_plain(tmp_path):
    code = translate(
        tmp_path,
        """shader A {
    buffer uint data[] @binding(0) @coherent;
    buffer uint plain[] @binding(1);
    void put(buffer uint values[], uint i) {
        values[i] = 1u;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            put(plain, 0u);
            data[0] = 2u;
        }
    }
}""",
        "metal",
    )

    flat = compact(code)
    assert re.search(r"void put\(device uint(\*values| values\[\]), uint i\)", flat)
    assert "coherent device uint*data [[buffer(0)]]" in flat
    metal_ir(tmp_path, code)


def test_hlsl_globallycoherent_structured_buffer_reaches_helper(tmp_path):
    crossgl = translate(
        tmp_path,
        """globallycoherent RWStructuredBuffer<uint> data : register(u0);
RWStructuredBuffer<uint> plain : register(u1);
void put(RWStructuredBuffer<uint> b, uint i) { b[i] = b[i] + 1u; }
[numthreads(1, 1, 1)] void main() { put(data, 0u); put(plain, 1u); }
""",
        "cgl",
        suffix=".hlsl",
    )
    code = translate(tmp_path, crossgl, "metal")

    flat = compact(code)
    assert "void put(coherent(device) device uint*b, uint i)" in flat
    assert "coherent(device) device uint*data [[buffer(0)]]" in flat
    ir = metal_ir(tmp_path, code)
    assert "@air.store.device_coherent.i32" in ir


def test_glsl_buffer_block_accesses_keep_coherence_and_volatile(tmp_path):
    code = translate(
        tmp_path,
        """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) coherent buffer B { uint head; uint b[]; };
layout(std430, binding = 1) volatile buffer V { uint v[]; };
void main() {
    b[1] = b[0] + head;
    v[0] = v[1];
}
""",
        "metal",
        suffix=".comp",
    )

    flat = compact(code)
    assert "coherent device uchar*b_2 [[buffer(0)]]" in flat
    assert "volatile device uchar*v_2 [[buffer(1)]]" in flat
    assert "reinterpret_cast<coherent device uint*>(b_2 + 8)" in flat
    assert "reinterpret_cast<coherent const device uint*>(b_2 + 0)" in flat
    assert "reinterpret_cast<volatile device uint*>(v_2 + 0)" in flat
    assert "reinterpret_cast<device uint*>" not in flat
    assert "reinterpret_cast<const device uint*>" not in flat
    ir = metal_ir(tmp_path, code)
    assert "@air.load.device_coherent.i32" in ir
    assert "@air.store.device_coherent.i32" in ir
    assert "load volatile i32" in ir
    assert "store volatile i32" in ir


def test_glsl_buffer_block_helpers_and_atomics_keep_memory_qualifiers(tmp_path):
    code = translate(
        tmp_path,
        """#version 450
layout(local_size_x = 1) in;
struct P { vec4 a; uint n[2]; };
layout(std430, binding = 0) coherent buffer B { uint head; P items[]; } blk;
layout(std430, binding = 1) volatile buffer V { uint v[]; };
layout(std430, binding = 2) buffer Q { mat2 m; uint counter; };
void touch() { blk.head = atomicAdd(v[0], 1u); }
void main() {
    P p = blk.items[0];
    blk.items[1] = p;
    atomicCompSwap(v[1], 0u, 1u);
    atomicCompSwap(counter, 0u, 1u);
    m += mat2(1.0);
    touch();
}
""",
        "metal",
        suffix=".comp",
    )

    flat = compact(code)
    # Aggregate loads and compare-exchange helpers are specialised per memory contract.
    assert re.search(
        r"static P __crossgl_load_glsl_buffer_coherent_P_\w+\(coherent const device uchar\*buffer, uint offset\)",
        flat,
    )
    assert (
        "__crossgl_buffer_atomic_compare_exchange_volatile_uint(volatile device uchar*buffer"
        in flat
    )
    assert "__crossgl_buffer_atomic_compare_exchange_uint(" in flat
    assert "reinterpret_cast<volatile device atomic_uint*>(v_2 + 0)" in flat
    # The helper that GLSL globals reach through keeps both contracts on its forwarded parameters.
    assert re.search(
        r"void touch\((volatile device uchar\*v_2, coherent device uchar\*blk|"
        r"coherent device uchar\*blk, volatile device uchar\*v_2)\)",
        flat,
    )
    ir = metal_ir(tmp_path, code)
    assert "@air.load.device_coherent.f32" in ir
    assert "@air.store.device_coherent.i32" in ir


@pytest.mark.parametrize(
    ("own", "inherited", "expected"),
    [
        ([], [], []),
        ([("coherent", None)], [("coherent", None)], [("coherent", None)]),
        ([], [("coherent", "threadgroup")], [("coherent", "threadgroup")]),
        (
            [("coherent", "threadgroup")],
            [("coherent", "device")],
            [("coherent", "device")],
        ),
        ([("coherent", "system")], [("coherent", None)], [("coherent", "system")]),
        (
            [("coherent", None)],
            [("volatile", None)],
            [("volatile", None), ("coherent", None)],
        ),
        # An unknown scope is kept so that the declaration's diagnostic reports it.
        (
            [("coherent", "queuefamily")],
            [("coherent", "device")],
            [("coherent", "queuefamily"), ("coherent", "device")],
        ),
    ],
)
def test_merge_memory_contracts(own, inherited, expected):
    assert merge_memory_contracts(own, inherited) == expected


@pytest.mark.parametrize(
    ("type_name", "expected"),
    [
        ("uint[]", "uint *values"),
        ("uint[4]", "uint *values"),
        ("float[][4]", "float (*values)[4]"),
        ("uint", "uint values"),
    ],
)
def test_coherent_array_parameter_declaration(type_name, expected):
    assert coherent_array_parameter_declaration(type_name, "values") == expected


def test_memory_helper_token():
    assert memory_helper_token("") == ""
    assert memory_helper_token("coherent ") == "coherent_"
    assert memory_helper_token("volatile coherent(device) ") == (
        "volatile_coherent_device__"
    )


def test_buffer_argument_root_name():
    data = IdentifierNode("data")
    one = IdentifierNode("one")
    assert buffer_argument_root_name(data) == "data"
    assert buffer_argument_root_name(ArrayAccessNode(data, one)) == "data"
    assert buffer_argument_root_name(MemberAccessNode(data, "items")) == "data"
    assert buffer_argument_root_name(UnaryOpNode("&", data)) == "data"
    assert buffer_argument_root_name(BinaryOpNode(data, "+", one)) == "data"
    assert buffer_argument_root_name(BinaryOpNode(one, "*", data)) is None
    assert buffer_argument_root_name(UnaryOpNode("-", data)) is None
