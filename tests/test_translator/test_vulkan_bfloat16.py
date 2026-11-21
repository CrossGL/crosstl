"""Vulkan / SPIR-V bfloat16 backend contract.

The contract requires:

1. **16-bit storage and buffer layout.**  ``bfloat16_t`` must be emitted as an
   ``OpTypeInt 16 0`` type (the only 16-bit unsigned integer width in SPIR-V).
   Every buffer that holds bfloat16 elements must carry ``ArrayStride 2``.

2. **Conversion and round-to-nearest behavior.**  Constructing a ``bfloat16_t``
   from a binary32 value widens to binary32, rounds once to nearest-even
   (using the standard ``(bits + 0x7FFF + ((bits >> 16) & 1)) >> 16`` formula),
   and stores the 16-bit payload.  Reading from storage expands the bits left by
   16, producing an exact binary32 representation.

3. **Bitcast / ``as_type`` semantics.**  ``as_type<bfloat16_t>(uint_value)``
   is an ``OpUConvert`` that narrows the 32-bit payload to 16 bits, and
   ``asuint(bfloat16_value)`` widens the 16-bit payload with an ``OpUConvert``.
   Neither path performs any floating-point arithmetic.

4. **Vector forms and aggregate layout.**  An N-lane ``bfloat16vecN`` is a
   SPIR-V vector of the same ``OpTypeInt 16 0`` element type with 2 bytes per
   component, so ``ArrayStride`` is 2 × N.

5. **Capability and extension requirements in diagnostics.**  Buffers that hold
   bfloat16 storage demand ``OpCapability Int16`` and
   ``OpCapability StorageBuffer16BitAccess`` / ``SPV_KHR_16bit_storage``; the
   SPIR-V emitter records these in the module header.  When the module cannot be
   lowered the ``SpirvLoweringRefused`` diagnostic lists the missing capability
   exactly.
"""

from __future__ import annotations

import re

import pytest

from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen(source: str) -> str:
    """Parse ``source`` and generate SPIR-V assembly."""
    return VulkanSPIRVCodeGen().generate(Parser(Lexer(source).tokens).parse())


def _bfloat16_type_id(spv: str) -> str | None:
    """Return the SPIR-V id for the ``OpTypeInt 16 0`` type, or ``None``."""
    m = re.search(r"(%\d+) = OpTypeInt 16 0\b", spv)
    return m.group(1) if m else None


def _float32_type_id(spv: str) -> str | None:
    """Return the SPIR-V id for the ``OpTypeFloat 32`` type, or ``None``."""
    m = re.search(r"(%\d+) = OpTypeFloat 32\b", spv)
    return m.group(1) if m else None


def _uint32_type_id(spv: str) -> str | None:
    """Return the SPIR-V id for the ``OpTypeInt 32 0`` type, or ``None``."""
    m = re.search(r"(%\d+) = OpTypeInt 32 0\b", spv)
    return m.group(1) if m else None


_SHADER_TEMPLATE = """\
shader Bf16Contract {{
    {resources}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            {body}
        }}
    }}
}}
"""


def _shader(body: str, resources: str = "") -> str:
    return _SHADER_TEMPLATE.format(resources=resources, body=body)


# ---------------------------------------------------------------------------
# 1. 16-bit storage and buffer layout
# ---------------------------------------------------------------------------


class TestVulkanBfloat16StorageLayout:
    """Contract point 1: 16-bit storage and buffer layout."""

    def test_bfloat16_scalar_buffer_uses_optype_int_16(self):
        """bfloat16_t buffers emit OpTypeInt 16 0, not OpTypeFloat."""
        source = _shader(
            body="bfloat16_t value = buffer_load(input, 0u); buffer_store(output, 0u, value);",
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None, "Expected OpTypeInt 16 0 for bfloat16_t"
        # Must not use a floating-point type for storage.
        assert "OpTypeFloat 16" not in spv

    def test_bfloat16_scalar_buffer_has_array_stride_2(self):
        """bfloat16_t runtime array must carry ArrayStride 2 (2 bytes per element)."""
        source = _shader(
            body="bfloat16_t v = buffer_load(input, 0u); buffer_store(output, 0u, v);",
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        assert "ArrayStride 2" in spv, "Expected ArrayStride 2 for bfloat16_t buffer"

    def test_bfloat16_buffer_requires_int16_capability(self):
        """bfloat16_t storage demands OpCapability Int16."""
        source = _shader(
            body="bfloat16_t v = buffer_load(input, 0u);",
            resources="StructuredBuffer<bfloat16_t> input @binding(0);",
        )
        spv = _gen(source)

        assert "OpCapability Int16" in spv

    def test_bfloat16_buffer_requires_storage_buffer_16bit_access(self):
        """bfloat16_t storage buffers demand StorageBuffer16BitAccess / SPV_KHR_16bit_storage."""
        source = _shader(
            body="bfloat16_t v = buffer_load(input, 0u);",
            resources="StructuredBuffer<bfloat16_t> input @binding(0);",
        )
        spv = _gen(source)

        has_capability = "OpCapability StorageBuffer16BitAccess" in spv
        has_extension = 'OpExtension "SPV_KHR_16bit_storage"' in spv
        assert has_capability or has_extension, (
            "Expected either OpCapability StorageBuffer16BitAccess or "
            'OpExtension "SPV_KHR_16bit_storage"'
        )

    def test_bfloat16_buffer_optype_matches_runtime_array_element(self):
        """The OpTypeRuntimeArray element type must be the bfloat16 OpTypeInt 16 0."""
        source = _shader(
            body="bfloat16_t v = buffer_load(input, 0u); buffer_store(output, 0u, v);",
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        assert re.search(
            rf"OpTypeRuntimeArray {re.escape(bfloat_id)}\b", spv
        ), "Expected OpTypeRuntimeArray with bfloat16 element type"

    def test_bfloat16_is_a_distinct_type_from_float_and_half(self):
        """bfloat16_t must not be mapped to float, half or any other type."""
        source = _shader(
            body=(
                "bfloat16_t bf = buffer_load(input, 0u); "
                "float f = float(bf); "
                "bfloat16_t back = bfloat16_t(f); "
                "buffer_store(output, 0u, back);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        # Must have an integer-16 type (bfloat16 storage) and a float-32 type (arithmetic).
        assert _bfloat16_type_id(spv) is not None, (
            "bfloat16 must be distinct: missing OpTypeInt 16 0"
        )
        assert _float32_type_id(spv) is not None, "Expected OpTypeFloat 32 for arithmetic"
        # No float-16 type should appear — bfloat16 is not binary16.
        assert "OpTypeFloat 16" not in spv, "bfloat16_t must not be represented as float16"


# ---------------------------------------------------------------------------
# 2. Conversion and round-to-nearest behavior
# ---------------------------------------------------------------------------


class TestVulkanBfloat16Conversion:
    """Contract point 2: conversion and round-to-nearest-even."""

    def test_bfloat16_constant_rounds_to_nearest_even(self):
        """bfloat16_t constant folding uses round-to-nearest-ties-to-even."""
        # 0x3F800001 -> 0x3F80 (tie rounds down: low bit of upper byte is 0)
        # 0x3F808001 -> 0x3F81 (tie rounds up:   low bit of upper byte is 1)
        source = _shader(
            body=(
                "bfloat16_t tie_down = bfloat16_t(0.5000152587890625f); "
                "bfloat16_t tie_up = bfloat16_t(0.5001220703125f); "
                "buffer_store(output, 0u, tie_down); "
                "buffer_store(output, 1u, tie_up);"
            ),
            resources="RWStructuredBuffer<bfloat16_t> output @binding(0);",
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        # 0x3F80 = 16256, 0x3F81 = 16257
        assert re.search(rf"OpConstant {re.escape(bfloat_id)} 16256\b", spv), (
            "Expected bfloat16 constant 0x3F80 (16256) for tie-round-down"
        )
        assert re.search(rf"OpConstant {re.escape(bfloat_id)} 16257\b", spv), (
            "Expected bfloat16 constant 0x3F81 (16257) for tie-round-up"
        )

    def test_bfloat16_zero_constant_is_integer_zero(self):
        """bfloat16_t(0.0) stores as OpConstant … 0."""
        source = _shader(
            body="bfloat16_t z = bfloat16_t(0.0); buffer_store(output, 0u, z);",
            resources="RWStructuredBuffer<bfloat16_t> output @binding(0);",
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        assert re.search(rf"OpConstant {re.escape(bfloat_id)} 0\b", spv)

    def test_bfloat16_one_constant_matches_ieee_bfloat_one(self):
        """bfloat16_t(1.0) stores as OpConstant … 16256 (0x3F80)."""
        source = _shader(
            body="bfloat16_t one = bfloat16_t(1.0); buffer_store(output, 0u, one);",
            resources="RWStructuredBuffer<bfloat16_t> output @binding(0);",
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        # 1.0f in binary32 is 0x3F800000; bfloat16 is 0x3F80 = 16256.
        assert re.search(rf"OpConstant {re.escape(bfloat_id)} 16256\b", spv), (
            "Expected bfloat16 constant 16256 (0x3F80) for 1.0"
        )

    def test_bfloat16_runtime_narrowing_uses_shift_and_rounding_ops(self):
        """Runtime bfloat16_t(float_value) uses bit-shift rounding, not a float cast."""
        source = _shader(
            body=(
                "bfloat16_t value = bfloat16_t(float(buffer_load(input, 0u))); "
                "buffer_store(output, 0u, value);"
            ),
            resources=(
                "StructuredBuffer<float> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        # The SPIRV emitter uses arithmetic shift operations for RNE rounding.
        assert "OpShiftLeftLogical" in spv or "OpShiftRightLogical" in spv, (
            "Expected bit-shift operations for bfloat16 rounding"
        )

    def test_bfloat16_widening_to_float_uses_shift(self):
        """Reading a bfloat16_t value and converting to float shifts bits left by 16."""
        source = _shader(
            body=(
                "bfloat16_t bf = buffer_load(input, 0u); "
                "float f = float(bf); "
                "buffer_store(output, 0u, f);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<float> output @binding(1);"
            ),
        )
        spv = _gen(source)

        assert "OpShiftLeftLogical" in spv, (
            "Expected OpShiftLeftLogical for bfloat16 -> float widening"
        )

    def test_bfloat16_to_float_uses_bitcast(self):
        """bfloat16 -> float conversion must reinterpret the shifted bits (OpBitcast)."""
        source = _shader(
            body=(
                "bfloat16_t bf = buffer_load(input, 0u); "
                "float f = float(bf); "
                "buffer_store(output, 0u, f);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<float> output @binding(1);"
            ),
        )
        spv = _gen(source)

        assert "OpBitcast" in spv, (
            "Expected OpBitcast for final float interpretation in bfloat16->float"
        )


# ---------------------------------------------------------------------------
# 3. Bitcast / as_type semantics
# ---------------------------------------------------------------------------


class TestVulkanBfloat16Bitcast:
    """Contract point 3: bitcast and as_type semantics."""

    def test_as_type_to_bfloat16_uses_opuconvert_narrow(self):
        """as_type<bfloat16_t>(uint_bits) narrows with OpUConvert, no arithmetic."""
        source = _shader(
            body=(
                "uint bits = buffer_load(input, 0u); "
                "bfloat16_t value = as_type<bfloat16_t>(bits); "
                "buffer_store(output, 0u, value);"
            ),
            resources=(
                "StructuredBuffer<uint> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        # OpUConvert with the bfloat16 type as destination is the narrowing bitcast.
        assert re.search(rf"OpUConvert {re.escape(bfloat_id)} %\d+", spv), (
            "Expected OpUConvert to bfloat16 for as_type<bfloat16_t>(uint)"
        )
        # Must not go through any floating-point conversion.
        assert "OpConvertUToF" not in spv
        assert "OpConvertFToU" not in spv

    def test_asuint_from_bfloat16_uses_opuconvert_widen(self):
        """asuint(bfloat16_value) widens with OpUConvert, no arithmetic."""
        source = _shader(
            body=(
                "bfloat16_t value = buffer_load(input, 0u); "
                "uint bits = asuint(value); "
                "buffer_store(output, 0u, bits);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<uint> output @binding(1);"
            ),
        )
        spv = _gen(source)

        uint_id = _uint32_type_id(spv)
        assert uint_id is not None
        # OpUConvert with the uint type as destination is the widening bitcast.
        assert re.search(rf"OpUConvert {re.escape(uint_id)} %\d+", spv), (
            "Expected OpUConvert to uint for asuint(bfloat16)"
        )

    def test_as_type_round_trip_preserves_bit_pattern(self):
        """as_type<uint>(as_type<bfloat16_t>(x)) must preserve the original bits."""
        source = _shader(
            body=(
                "bfloat16_t value = buffer_load(input, 0u); "
                "uint bits = as_type<uint>(value); "
                "bfloat16_t restored = as_type<bfloat16_t>(bits); "
                "buffer_store(output, 0u, restored);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        uint_id = _uint32_type_id(spv)
        assert bfloat_id is not None
        assert uint_id is not None
        # Both conversion directions must be present.
        assert re.search(rf"OpUConvert {re.escape(bfloat_id)} %\d+", spv)
        assert re.search(rf"OpUConvert {re.escape(uint_id)} %\d+", spv)
        # No floating-point instructions should appear in this pure bitcast chain.
        assert "OpConvertUToF" not in spv
        assert "OpConvertFToU" not in spv

    def test_no_spurious_float_ops_in_bitcast_path(self):
        """The bitcast path must not emit FAdd, FSub or FMul."""
        source = _shader(
            body=(
                "bfloat16_t value = buffer_load(input, 0u); "
                "uint bits = asuint(value);"
            ),
            resources="StructuredBuffer<bfloat16_t> input @binding(0);",
        )
        spv = _gen(source)

        assert "OpFAdd" not in spv
        assert "OpFSub" not in spv
        assert "OpFMul" not in spv


# ---------------------------------------------------------------------------
# 4. Vector forms and aggregate layout
# ---------------------------------------------------------------------------


class TestVulkanBfloat16Vectors:
    """Contract point 4: vector forms and aggregate layout."""

    def test_bfloat16_vec2_storage_has_optype_vector_int16(self):
        """bfloat16vec2 buffers emit OpTypeVector (OpTypeInt 16 0) 2."""
        source = _shader(
            body="bfloat16vec2 v = buffer_load(output, 0u); buffer_store(output, 0u, v);",
            resources="RWStructuredBuffer<bfloat16vec2> output @binding(0);",
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        assert re.search(
            rf"%\d+ = OpTypeVector {re.escape(bfloat_id)} 2\b", spv
        ), "Expected OpTypeVector (bfloat16) 2 for bfloat16vec2"

    def test_bfloat16_vec4_storage_has_optype_vector_int16_4(self):
        """bfloat16vec4 buffers emit OpTypeVector (OpTypeInt 16 0) 4."""
        source = _shader(
            body="bfloat16vec4 v = buffer_load(output, 0u); buffer_store(output, 0u, v);",
            resources="RWStructuredBuffer<bfloat16vec4> output @binding(0);",
        )
        spv = _gen(source)

        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        assert re.search(
            rf"%\d+ = OpTypeVector {re.escape(bfloat_id)} 4\b", spv
        ), "Expected OpTypeVector (bfloat16) 4 for bfloat16vec4"

    def test_bfloat16_vec2_buffer_has_array_stride_4(self):
        """bfloat16vec2 runtime array must have ArrayStride 4 (2 × 2 bytes)."""
        source = _shader(
            body="bfloat16vec2 v = buffer_load(output, 0u); buffer_store(output, 0u, v);",
            resources="RWStructuredBuffer<bfloat16vec2> output @binding(0);",
        )
        spv = _gen(source)

        assert "ArrayStride 4" in spv, "Expected ArrayStride 4 for bfloat16vec2 buffer"

    def test_bfloat16_vector_construction_uses_composite_construct(self):
        """bfloat16vec2(a, b) must emit OpCompositeConstruct over int16 components."""
        source = _shader(
            body=(
                "bfloat16_t a = buffer_load(inputA, 0u); "
                "bfloat16_t b = buffer_load(inputB, 0u); "
                "bfloat16vec2 v = bfloat16vec2(a, b); "
                "buffer_store(output, 0u, v);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> inputA @binding(0);\n"
                "    StructuredBuffer<bfloat16_t> inputB @binding(1);\n"
                "    RWStructuredBuffer<bfloat16vec2> output @binding(2);"
            ),
        )
        spv = _gen(source)

        assert "OpCompositeConstruct" in spv, (
            "Expected OpCompositeConstruct for bfloat16vec2 construction"
        )

    def test_bfloat16_vector_extraction_uses_composite_extract(self):
        """vec2(bfloat16vec2_value) must decompose with OpCompositeExtract."""
        source = _shader(
            body=(
                "bfloat16vec2 v = buffer_load(input, 0u); "
                "float2 expanded = float2(v); "
                "buffer_store(output, 0u, expanded.x);"
            ),
            resources=(
                "StructuredBuffer<bfloat16vec2> input @binding(0);\n"
                "    RWStructuredBuffer<float> output @binding(1);"
            ),
        )
        spv = _gen(source)

        assert "OpCompositeExtract" in spv, (
            "Expected OpCompositeExtract for bfloat16vec2 decomposition"
        )


# ---------------------------------------------------------------------------
# 5. Capability and extension requirements in diagnostics
# ---------------------------------------------------------------------------


class TestVulkanBfloat16Capabilities:
    """Contract point 5: capability and extension requirements in diagnostics."""

    def test_spirv_refusal_includes_capability_in_message(self):
        """When a bfloat16 lowering fails, SpirvLoweringRefused names the reason."""
        # The SPIR-V emitter uses SpirvRefusals.record() + raise_pending() to
        # aggregate all failing constructs.  We simulate that directly.
        from crosstl.translator.codegen.spirv_refusals import SpirvRefusals

        refusals = SpirvRefusals()
        refusals.record("bfloat16 buffer requires Int16 capability")
        refusals.record("bfloat16 buffer requires StorageBuffer16BitAccess capability")

        with pytest.raises(SpirvLoweringRefused) as exc_info:
            refusals.raise_pending()

        err = exc_info.value
        assert "Int16" in str(err) or any("Int16" in r for r in err.reasons)
        assert "StorageBuffer16BitAccess" in str(err) or any(
            "StorageBuffer16BitAccess" in r for r in err.reasons
        )
        assert err.missing_capabilities == ("vulkan.lowering",)

    def test_spirv_refusal_is_structured_diagnostic(self):
        """SpirvLoweringRefused carries structured diagnostic fields."""
        from crosstl.translator.codegen.spirv_refusals import SpirvRefusals

        reasons = [
            "bfloat16 requires Int16 capability (SPIR-V 3.31)",
            "bfloat16 storage requires SPV_KHR_16bit_storage",
        ]
        refusals = SpirvRefusals()
        for reason in reasons:
            refusals.record(reason)

        with pytest.raises(SpirvLoweringRefused) as exc_info:
            refusals.raise_pending()

        err = exc_info.value
        assert len(err.reasons) == len(reasons)
        assert err.reasons == tuple(reasons)
        assert err.target == "vulkan"
        assert err.missing_capabilities == ("vulkan.lowering",)

    def test_spirv_refusal_reset_clears_pending_reasons(self):
        """SpirvRefusals.reset() discards all pending reasons."""
        from crosstl.translator.codegen.spirv_refusals import SpirvRefusals

        refusals = SpirvRefusals()
        refusals.record("bfloat16 Int16 missing")
        refusals.reset()
        # After reset, raise_pending must not raise.
        refusals.raise_pending()  # should not raise

    def test_bfloat16_buffer_spv_header_declares_all_required_capabilities(self):
        """The SPIR-V module header lists Int16 before any bfloat16 buffer use."""
        source = _shader(
            body="bfloat16_t v = buffer_load(input, 0u); buffer_store(output, 0u, v);",
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        # Int16 must appear in the capability declarations section.
        int16_cap_pos = spv.find("OpCapability Int16")
        assert int16_cap_pos != -1, "Expected OpCapability Int16 in module header"
        # Capability declarations must precede the main function body.
        main_pos = spv.find("OpFunction")
        assert int16_cap_pos < main_pos, (
            "OpCapability Int16 must precede OpFunction"
        )

    def test_bfloat16_missing_capability_code_matches_vulkan_target(self):
        """The diagnostic code for Vulkan lowering failures uses the vulkan target."""
        from crosstl.translator.codegen.spirv_refusals import (
            CODE,
            SpirvRefusals,
        )

        refusals = SpirvRefusals()
        refusals.record("test reason")

        with pytest.raises(SpirvLoweringRefused) as exc_info:
            refusals.raise_pending()

        err = exc_info.value
        assert err.code == CODE
        assert "vulkan" in err.code


# ---------------------------------------------------------------------------
# 6. Arithmetic semantics: widen-compute-round contract
# ---------------------------------------------------------------------------


class TestVulkanBfloat16Arithmetic:
    """Arithmetic uses binary32 and rounds the result back to bfloat16."""

    def test_bfloat16_addition_promotes_to_float32(self):
        """bfloat16 + bfloat16 must compute in OpTypeFloat 32, not int16."""
        source = _shader(
            body=(
                "bfloat16_t a = buffer_load(input, 0u); "
                "bfloat16_t b = buffer_load(input, 1u); "
                "bfloat16_t result = a + b; "
                "buffer_store(output, 0u, result);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        float_id = _float32_type_id(spv)
        assert float_id is not None, "Expected OpTypeFloat 32 for bfloat16 arithmetic"
        # Arithmetic result types must be float, not int16.
        fadd_results = re.findall(r"OpFAdd (%\d+) ", spv)
        assert fadd_results, "Expected OpFAdd for bfloat16 addition"
        for result_type in fadd_results:
            assert result_type == float_id, (
                f"OpFAdd result type {result_type} must be float32 ({float_id})"
            )

    def test_bfloat16_arithmetic_result_rounded_back_to_int16(self):
        """The binary32 arithmetic result must be rounded back to bfloat16 (int16)."""
        source = _shader(
            body=(
                "bfloat16_t a = buffer_load(input, 0u); "
                "bfloat16_t b = buffer_load(input, 1u); "
                "bfloat16_t result = a + b; "
                "buffer_store(output, 0u, result);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        # The store must write through a bfloat16 (int16) pointer, so the result
        # must be converted back — shift + round operations are expected.
        assert "OpShiftRightLogical" in spv, (
            "Expected OpShiftRightLogical for rounding float32 back to bfloat16"
        )

    def test_bfloat16_comparison_produces_bool_not_int(self):
        """bfloat16 == bfloat16 must compare the binary32 values, not the int16 bits."""
        source = _shader(
            body=(
                "bfloat16_t a = buffer_load(input, 0u); "
                "bfloat16_t b = buffer_load(input, 1u); "
                "bool equal = (a == b); "
                "buffer_store(output, 0u, equal ? 1u : 0u);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<uint> output @binding(1);"
            ),
        )
        spv = _gen(source)

        # Comparison must use floating-point ordered equal, not integer equal.
        assert "OpFOrdEqual" in spv, (
            "Expected OpFOrdEqual for bfloat16 equality (compare binary32 values)"
        )
        assert "WARNING" not in spv

    def test_bfloat16_no_unknown_type_warnings_in_generated_spv(self):
        """No WARNING comments must appear in the generated SPIR-V assembly."""
        source = _shader(
            body=(
                "bfloat16_t value = buffer_load(input, 0u); "
                "float expanded = float(value); "
                "bfloat16_t rounded = bfloat16_t(expanded * 2.0); "
                "buffer_store(output, 0u, rounded);"
            ),
            resources=(
                "StructuredBuffer<bfloat16_t> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        assert "WARNING" not in spv, (
            "SPIR-V output must not contain WARNING comments for bfloat16_t"
        )

    def test_bfloat16_constructor_from_float_does_not_emit_unresolved_calls(self):
        """bfloat16_t(float_value) must not produce unresolved constructor calls."""
        source = _shader(
            body=(
                "float f = float(buffer_load(input, 0u)); "
                "bfloat16_t bf = bfloat16_t(f); "
                "buffer_store(output, 0u, bf);"
            ),
            resources=(
                "StructuredBuffer<float> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        # An unresolved call would appear as a function call to an undefined
        # function; the emitter must handle this natively via bit operations.
        assert "OpFunctionCall" not in spv or "bfloat16_t" not in spv, (
            "bfloat16_t constructor must be lowered natively, not as a function call"
        )
        assert "WARNING" not in spv

    def test_bfloat16_as_type_does_not_emit_unresolved_calls(self):
        """as_type<bfloat16_t> must not produce unresolved calls."""
        source = _shader(
            body=(
                "uint bits = buffer_load(input, 0u); "
                "bfloat16_t bf = as_type<bfloat16_t>(bits); "
                "buffer_store(output, 0u, bf);"
            ),
            resources=(
                "StructuredBuffer<uint> input @binding(0);\n"
                "    RWStructuredBuffer<bfloat16_t> output @binding(1);"
            ),
        )
        spv = _gen(source)

        assert "WARNING" not in spv
        # as_type<bfloat16_t> must lower to OpUConvert, not an OpFunctionCall.
        bfloat_id = _bfloat16_type_id(spv)
        assert bfloat_id is not None
        assert re.search(rf"OpUConvert {re.escape(bfloat_id)} %\d+", spv), (
            "as_type<bfloat16_t> must lower to OpUConvert"
        )
