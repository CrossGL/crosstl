"""Explicit Metal scalar bitcasts retain logical width through native execution."""

import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project.runtime_value_encoding import FLOAT16_BITS, FLOAT32_BITS
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SCALAR_BITCAST_RUNTIME"
CASES = {
    "signed64": ("long", "ulong", "uint64_t", "int64", "uint64", 64),
    "unsigned64": ("ulong", "long", "int64_t", "uint64", "int64", 64),
    "half": ("half", "ushort", "uint16_t", "float16", "uint16", 16),
    "half-signed": ("half", "short", "int16_t", "float16", "int16", 16),
    "unsigned16": ("ushort", "short", "int16_t", "uint16", "int16", 16),
    "signed16": ("short", "ushort", "uint16_t", "int16", "uint16", 16),
    "to-half": ("ushort", "half", "half", "uint16", "float16", 16),
    "float": ("float", "uint", "uint32_t", "float32", "uint32", 32),
    "to-float": ("uint", "float", "float", "uint32", "float32", 32),
}


def _source(case):
    input_type, output_type, target_type, _, _, _ = CASES[case]
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void scalar_bits(const device {input_type}* values [[buffer(0)]],
                        device {output_type}* results [[buffer(1)]],
                        device uint* counts [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {{
    uint cursor = i;
    results[i + 1u] = as_type<{target_type}>(values[cursor++]);
    counts[i + 1u] = cursor - i;
}}
"""


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("form", ("direct", "alias", "local-alias"))
def test_import_retains_explicit_scalar_bitcast_width(tmp_path, case, form):
    source_type, _, target_type, _, _, bits = CASES[case]
    declaration = f"using Result = {target_type};"
    target = target_type if form == "direct" else "Result"
    source = tmp_path / "source.metal"
    source.write_text(
        f"""#include <metal_stdlib>
using namespace metal;
{declaration if form == "alias" else ""}
{target_type} bits({source_type} value) {{
    {declaration if form == "local-alias" else ""}
    return as_type<{target}>(value);
}}
""",
        encoding="utf-8",
    )
    canonical = translate(str(source), backend="crossgl", format_output=False)
    if bits == 32:
        assert (
            "return asuint(value);" if case == "float" else "return asfloat(value);"
        ) in canonical
    else:
        assert "return as_type<" in canonical
        assert "return asuint(" not in canonical and "return asint(" not in canonical


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize(
    "source_type,target_type",
    (("half", "uint"), ("ushort", "uint"), ("uint", "uint16_t"), ("uint", "uint64_t")),
)
def test_unequal_scalar_widths_are_not_silently_cast(
    tmp_path, target, source_type, target_type
):
    source = tmp_path / "source.metal"
    source.write_text(
        f"""#include <metal_stdlib>
using namespace metal;
kernel void invalid(const device {source_type}* values [[buffer(0)]],
                    device {target_type}* results [[buffer(1)]]) {{
    results[0] = as_type<{target_type}>(values[0]);
}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="width|bit"):
        translate(str(source), backend=target, format_output=False)


def _words(bits):
    if bits == 16:
        # Every finite binary16 payload, including signed zero and subnormals.
        return [word for word in range(1 << 16) if word & 0x7C00 != 0x7C00]
    mask = (1 << bits) - 1
    words = {0, mask, mask >> 1, 1 << (bits - 1), 0x12345678, 0xFEDCBA98}
    for shift in range(bits):
        words.update((1 << shift, mask ^ (1 << shift)))
    if bits == 64:
        words.update((0x123456789ABCDEF0, 0xFEDCBA9876543210))
    else:
        # Avoid NaN payload portability in the separate 32-bit positive controls.
        words = {word for word in words if word & 0x7F800000 != 0x7F800000}
    return sorted(words)


@pytest.mark.parametrize(
    "operation", ("atomic_load_explicit", "atomic_exchange_explicit")
)
def test_atomic_result_width_is_inferred_before_bitcast_aliasing(tmp_path, operation):
    value = "values, memory_order_relaxed"
    if operation == "atomic_exchange_explicit":
        value = "values, 1.0f, memory_order_relaxed"
    source = tmp_path / "source.metal"
    source.write_text(
        f"""#include <metal_stdlib>
using namespace metal;
kernel void read_bits(device atomic_float* values [[buffer(0)]],
                      device uint* results [[buffer(1)]]) {{
    results[0] = as_type<uint>({operation}({value}));
}}
""",
        encoding="utf-8",
    )
    canonical = translate(str(source), backend="crossgl", format_output=False)
    assert "asuint(atomic" in canonical
    assert "as_type<uint>" not in canonical


def test_source_atomic_named_overload_retains_its_wide_result(tmp_path):
    source = tmp_path / "source.metal"
    source.write_text(
        """#include <metal_stdlib>
using namespace metal;
ulong atomic_load_explicit(uint value, uint order) {
    return ulong(value) + ulong(order);
}
ulong bits(uint value) {
    return as_type<uint64_t>(atomic_load_explicit(value, 1u));
}
""",
        encoding="utf-8",
    )
    canonical = translate(str(source), backend="crossgl", format_output=False)
    assert "as_type<uint64>(atomic_load_explicit(" in canonical
    assert "asuint(" not in canonical


def _payload(dtype, words, target):
    words = list(words)
    if dtype.startswith("float"):
        encoding = FLOAT16_BITS if dtype == "float16" else FLOAT32_BITS
        if target == "opengl" and dtype == "float16":
            dtype, encoding = "float32", FLOAT32_BITS
            words = [
                struct.unpack(
                    "<I",
                    struct.pack("<f", struct.unpack("<e", struct.pack("<H", word))[0]),
                )[0]
                for word in words
            ]
        return {
            "dtype": dtype,
            "shape": [len(words)],
            "encoding": encoding,
            "values": words,
        }
    bits = int(dtype.lstrip("uint"))
    signed = dtype.startswith("int")
    values = [
        word - (1 << bits) if signed and word >= 1 << (bits - 1) else word
        for word in words
    ]
    if target == "opengl" and bits == 16:
        dtype = "int32" if signed else "uint32"
    return {"dtype": dtype, "shape": [len(values)], "values": values}


def _case(root, target, case):
    _, _, _, input_dtype, output_dtype, bits = CASES[case]
    source, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    words = _words(bits)
    if bits == 16 and input_dtype != "float16" and output_dtype != "float16":
        words = sorted(set(words) | {0x7FFF, 0xFFFF, 0x7C00, 0xFC00})
    guard = 0x3555
    initial = [guard] * (len(words) + 2)
    inputs = {
        "values": _payload(input_dtype, words, target),
        "results": _payload(output_dtype, initial, target),
        "counts": _payload("uint32", initial, target),
    }
    outputs = {
        "results": _payload(output_dtype, [guard, *words, guard], target),
        "counts": _payload("uint32", [guard, *([1] * len(words)), guard], target),
    }
    request = _request(descriptor, package, inputs, outputs, len(words))
    assert not request.execution_plan.diagnostics
    return source, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("case", CASES)
def test_scalar_bitcasts_compile_from_source_and_saved_intermediate(
    tmp_path, target, case
):
    path = tmp_path / "source.metal"
    path.write_text(_source(case), encoding="utf-8")
    generated = translate(str(path), backend=target, format_output=False)
    canonical = tmp_path / "source.cgl"
    canonical.write_text(
        translate(str(path), backend="crossgl", format_output=False), encoding="utf-8"
    )
    replay = translate(str(canonical), backend=target, format_output=False)
    assert generated == replay
    assert generated.count("cursor++") == 1
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_scalar_bitcasts_execute_with_exact_payloads(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required scalar bitcast execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="scalar_bits",
    )


def test_scalar_bitcast_native_gate_uses_existing_runners():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_scalar_bitcast_widths.py" in step
    assert "--timeout-seconds 360" in step
    assert "continue-on-error" not in step
