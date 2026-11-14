"""Byte-address buffer atomics reach HLSL as RWByteAddressBuffer methods and SPIR-V as word atomics.

CrossGL spells them ``atomicAdd(buffer, offset, value)`` and ``atomicCompareExchange(buffer, offset, compare,
value)``, returning the previous word. The DirectX generator emitted ``InterlockedAdd(buffer, offset, value)`` as a
value and ``atomicMax``/``atomicCompareExchange`` verbatim, neither of which DXC accepts, and the SPIR-V generator
refused them. A standalone call is now the HLSL method statement and a call whose value is used goes through a
helper returning the method's ``out`` value; SPIR-V emits ``OpAtomic*`` on the word. ``InterlockedCompareStore``
imports as a standalone compare-exchange.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_byte_address_atomics import (
    ByteAddressAtomicUnsupportedError,
)
from crosstl.translator.diagnostics import TranslationDiagnostic

METHODS = {
    "InterlockedAdd": "atomicAdd",
    "InterlockedMin": "atomicMin",
    "InterlockedMax": "atomicMax",
    "InterlockedAnd": "atomicAnd",
    "InterlockedOr": "atomicOr",
    "InterlockedXor": "atomicXor",
    "InterlockedExchange": "atomicExchange",
}


def _hlsl(body, resources="RWByteAddressBuffer dst : register(u0);"):
    return f"""
{resources}
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
{body}
}}
"""


def _cgl(body, resource="RWByteAddressBuffer"):
    return f"""
shader main {{
    @ register(u0)
    {resource} dst;
    compute {{
    @ numthreads(64, 1, 1)
    void main(uvec3 tid @ gl_GlobalInvocationID) @ stage_entry {{
{body}
    }}
    }}
}}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _round_trip(tmp_path, source):
    cgl = _translate(tmp_path, source, ".hlsl", "cgl")
    return cgl, _translate(tmp_path, cgl, ".cgl", "directx")


def _dxc(tmp_path, code, profile="cs_6_0", entry=None):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "out.hlsl"
    path.write_text(code, encoding="utf-8")
    entry = entry or re.search(r"\]\s*void\s+(\w+)\s*\(", code).group(1)
    result = subprocess.run(
        [dxc, "-T", profile, "-E", entry, str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("method", sorted(set(METHODS) - {"InterlockedExchange"}))
def test_standalone_atomic_is_the_method_statement(tmp_path, method):
    cgl, code = _round_trip(tmp_path, _hlsl(f"    dst.{method}(tid.x * 4, 3u);"))
    assert f"{METHODS[method]}(dst, tid.x * 4, 3u);" in cgl
    assert f"dst.{method}((tid.x * 4), 3u);" in code, code
    assert "__crossgl_bab_atomic" not in code
    _dxc(tmp_path, code)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_atomic_with_original_returns_it_through_a_helper(tmp_path, method):
    body = f"    uint previous;\n    dst.{method}(8, 5u, previous);\n    dst.Store(0, previous);"
    cgl, code = _round_trip(tmp_path, _hlsl(body))
    assert f"previous = {METHODS[method]}(dst, 8, 5u);" in cgl
    helper = f"__crossgl_bab_atomic_{method[len('Interlocked'):].lower()}_uint"
    assert f"previous = {helper}(dst, 8, 5u);" in code, code
    assert f"buffer.{method}(offset, value, original);" in code
    _dxc(tmp_path, code)


def test_compare_exchange_and_compare_store(tmp_path):
    body = (
        "    uint seen;\n"
        "    dst.InterlockedCompareExchange(8, 1u, 2u, seen);\n"
        "    dst.InterlockedCompareStore(12, seen, 3u);"
    )
    cgl, code = _round_trip(tmp_path, _hlsl(body))
    assert "seen = atomicCompareExchange(dst, 8, 1u, 2u);" in cgl
    assert "atomicCompareExchange(dst, 12, seen, 3u);" in cgl
    assert "InterlockedCompareStore" not in cgl
    assert "seen = __crossgl_bab_atomic_compare_exchange_uint(dst, 8, 1u, 2u);" in code
    assert (
        "buffer.InterlockedCompareExchange(offset, compareValue, value, original);"
        in code
    )
    assert "dst.InterlockedCompareStore(12, seen, 3u);" in code, code
    _dxc(tmp_path, code)


@pytest.mark.parametrize("method", ["InterlockedMin", "InterlockedMax"])
def test_int_value_keeps_the_signed_min_and_max(tmp_path, method):
    body = (
        f"    int low = -2;\n    int previous;\n    dst.{method}(0, low, previous);\n"
        f"    dst.{method}(4, low);\n    dst.Store(8, asuint(previous));"
    )
    _cgl_code, code = _round_trip(tmp_path, _hlsl(body))
    helper = f"__crossgl_bab_atomic_{method[len('Interlocked'):].lower()}_int"
    assert f"previous = {helper}(dst, 0, low);" in code, code
    assert f"int {helper}(RWByteAddressBuffer buffer, uint offset, int value)" in code
    assert f"dst.{method}(4, low);" in code
    _dxc(tmp_path, code)


def test_int_value_of_a_bitwise_operation_passes_its_bits(tmp_path):
    body = "    int delta = -3;\n    uint previous;\n    dst.InterlockedAdd(4, delta, previous);\n    dst.InterlockedOr(8, delta);"
    _cgl_code, code = _round_trip(tmp_path, _hlsl(body))
    assert "previous = __crossgl_bab_atomic_add_uint(dst, 4, asuint(delta));" in code
    assert "dst.InterlockedOr(8, asuint(delta));" in code, code
    _dxc(tmp_path, code)


def test_atomic_values_used_inside_expressions(tmp_path):
    body = (
        "        uint total = atomicAdd(dst, 0u, 1u) + atomicOr(dst, 4u, 2u);\n"
        "        if (atomicCompSwap(dst, 8u, total, 0u) == 0u) {\n"
        "            atomicExchange(dst, 12u, total);\n"
        "        }"
    )
    code = _translate(tmp_path, _cgl(body), ".cgl", "directx")
    assert (
        "uint total = (__crossgl_bab_atomic_add_uint(dst, 0u, 1u) + "
        "__crossgl_bab_atomic_or_uint(dst, 4u, 2u));" in code
    ), code
    assert (
        "__crossgl_bab_atomic_compare_exchange_uint(dst, 8u, total, 0u) == 0u" in code
    )
    # InterlockedExchange always takes the out argument, so even a standalone exchange uses the helper.
    assert "__crossgl_bab_atomic_exchange_uint(dst, 12u, total);" in code
    assert "dst.InterlockedExchange(" not in code
    # Each helper is defined once, in the order the program first needs it.
    assert code.count("uint __crossgl_bab_atomic_add_uint(") == 1
    assert code.index("__crossgl_bab_atomic_add_uint(RW") < code.index(
        "__crossgl_bab_atomic_or_uint(RW"
    )
    _dxc(tmp_path, code)


def test_rasterizer_ordered_buffer_gets_its_own_helper(tmp_path):
    source = """
shader main {
    @ register(u1)
    RasterizerOrderedByteAddressBuffer counts;
    fragment {
    vec4 main() @ gl_FragColor {
        uint seen = atomicAdd(counts, 0u, 1u);
        atomicMax(counts, 4u, seen);
        return vec4(1.0);
    }
    }
}
"""
    code = _translate(tmp_path, source, ".cgl", "directx")
    assert (
        "uint __crossgl_rov_bab_atomic_add_uint("
        "RasterizerOrderedByteAddressBuffer buffer, uint offset, uint value)" in code
    ), code
    assert "uint seen = __crossgl_rov_bab_atomic_add_uint(counts, 0u, 1u);" in code
    assert "counts.InterlockedMax(4u, seen);" in code
    _dxc(tmp_path, code, profile="ps_6_0", entry="PSMain")


@pytest.mark.parametrize(
    "body, resource, reason",
    [
        ("        atomicAdd(dst, 0u, 1.5);", "RWByteAddressBuffer", "got float"),
        (
            "        uint v = atomicMax(dst, 0u, true);",
            "RWByteAddressBuffer",
            "got bool",
        ),
        ("        atomicAdd(dst, 0.5, 1u);", "RWByteAddressBuffer", "byte offset"),
        ("        atomicAdd(dst, 0u, 1u);", "ByteAddressBuffer", "read-only"),
        ("        atomicAdd(dst, 0u);", "RWByteAddressBuffer", "got 2 argument"),
        (
            "        atomicCompareExchange(dst, 0u, 1u);",
            "RWByteAddressBuffer",
            "compare value and value",
        ),
    ],
)
def test_atomics_hlsl_cannot_perform_are_refused(tmp_path, body, resource, reason):
    with pytest.raises(ByteAddressAtomicUnsupportedError, match=reason):
        _translate(tmp_path, _cgl(body, resource), ".cgl", "directx")


def _spirv(tmp_path, source):
    code = _translate(tmp_path, source, ".cgl", "vulkan")
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        binary = tmp_path / "out.spv"
        for command in (
            ["spirv-as", "--target-env", "vulkan1.1", "-o", str(binary), "-"],
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
        ):
            result = subprocess.run(
                command,
                input=code if command[0] == "spirv-as" else None,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
    return code


def _uint_ops(code, opcode):
    """Instructions ``opcode`` whose result type is the 32-bit unsigned integer.

    The emitter names IDs numerically; ``spirv-dis``, when installed, renames them (``%uint``), so resolve the type.
    """
    uint_ids = set(re.findall(r"(%\w+) = OpTypeInt 32 0\b", code))
    return [
        match for match in re.findall(rf"= {opcode} (%\w+) ", code) if match in uint_ids
    ]


@pytest.mark.parametrize(
    "call, opcode",
    [
        ("atomicAdd(dst, 4u, 1u)", "OpAtomicIAdd"),
        ("atomicMin(dst, 4u, 1u)", "OpAtomicUMin"),
        ("atomicMax(dst, 4u, 5)", "OpAtomicUMax"),
        ("atomicAnd(dst, 4u, 1u)", "OpAtomicAnd"),
        ("atomicOr(dst, 4u, 1u)", "OpAtomicOr"),
        ("atomicXor(dst, 4u, 1u)", "OpAtomicXor"),
        ("atomicExchange(dst, 4u, 1u)", "OpAtomicExchange"),
        ("atomicCompareExchange(dst, 4u, 1u, 2u)", "OpAtomicCompareExchange"),
        ("atomicCompSwap(dst, 4u, 1u, 2u)", "OpAtomicCompareExchange"),
    ],
)
def test_spirv_atomic_on_the_word(tmp_path, call, opcode):
    body = f"        uint previous = {call};\n        {call};\n        buffer_store(dst, 0u, previous);"
    code = _spirv(tmp_path, _cgl(body))
    assert len(_uint_ops(code, opcode)) == 2, code


@pytest.mark.parametrize(
    "name, opcode", [("atomicMin", "OpAtomicSMin"), ("atomicMax", "OpAtomicSMax")]
)
def test_spirv_int_value_is_the_signed_min_and_max(tmp_path, name, opcode):
    body = (
        "        int low = -2;\n"
        f"        int previous = {name}(dst, 4u, low);\n"
        "        buffer_store(dst, 0u, uint(previous));"
    )
    code = _spirv(tmp_path, _cgl(body))
    assert _uint_ops(code, opcode), code
    assert "OpAtomicU" not in code


def test_hlsl_atomics_round_trip_to_spirv(tmp_path):
    body = (
        "    uint seen;\n"
        "    dst.InterlockedAdd(0, 1u, seen);\n"
        "    dst.InterlockedCompareStore(4, seen, 3u);"
    )
    cgl = _translate(tmp_path, _hlsl(body), ".hlsl", "cgl")
    code = _spirv(tmp_path, cgl)
    assert "OpAtomicIAdd" in code and "OpAtomicCompareExchange" in code


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            "        atomicAdd(dst, 0u, 1u, 2u);",
            "takes the buffer, a byte offset and a value",
        ),
        ("        atomicMax(dst, 0u, 1.5);", "must be a scalar integer"),
    ],
)
def test_spirv_refuses_what_it_cannot_lower(tmp_path, body, reason):
    with pytest.raises(TranslationDiagnostic, match=reason):
        _translate(tmp_path, _cgl(body), ".cgl", "vulkan")


@pytest.mark.parametrize(
    "method, opcode",
    [("InterlockedMin", "OpAtomicSMin"), ("InterlockedMax", "OpAtomicSMax")],
)
def test_spirv_method_form_with_int_value_is_signed(tmp_path, method, opcode):
    body = (
        "        int low = -2;\n"
        "        int previous;\n"
        f"        dst.{method}(4u, low, previous);\n"
        f"        dst.{method}(8u, low);\n"
        "        uint seen;\n"
        f"        dst.{method}(12u, 3u, seen);\n"
        "        buffer_store(dst, 0u, uint(previous) + seen);"
    )
    code = _spirv(tmp_path, _cgl(body))
    assert len(_uint_ops(code, opcode)) == 2, code
    assert len(_uint_ops(code, f"OpAtomicU{opcode[-3:]}")) == 1, code


def test_spirv_atomic_on_a_buffer_element_evaluates_it_once(tmp_path):
    body = "        atomicAdd(counts[tid.x], 1u);"
    resources = "    @ register(u0)\n    RWStructuredBuffer<uint> counts;"
    source = _cgl(body).replace(
        "    @ register(u0)\n    RWByteAddressBuffer dst;", resources
    )
    code = _spirv(tmp_path, source)
    assert len(re.findall(r"= OpAccessChain ", code)) == 1, code
