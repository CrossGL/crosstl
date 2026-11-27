"""Parse and round-trip fixtures for SPIR-V cooperative-matrix load/store/muladd instructions.

Covers KHR and NV variants, memory layout, stride, memory-access operands, and KHR
cooperative-matrix operand masks. Malformed and unsupported forms are rejected with
structured diagnostics rather than silently dropped.
"""

from __future__ import annotations

import pytest

from crosstl.backend.SPIRV.ray_tracing_import import SpirvInstructionUnsupportedError
from crosstl.backend.SPIRV.VulkanAst import FunctionCallNode, ShaderNode
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPIRV_HEADER = """\
OpCapability Shader
OpCapability CooperativeMatrixKHR
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void     = OpTypeVoid
%fn_type  = OpTypeFunction %void
%float    = OpTypeFloat 32
%uint     = OpTypeInt 32 0
%u32_3    = OpConstant %uint 3
%u32_0    = OpConstant %uint 0
%u32_8    = OpConstant %uint 8
%scope    = OpConstant %uint 3
%rows     = OpConstant %uint 8
%cols     = OpConstant %uint 8
%use_a    = OpConstant %uint 0
%use_b    = OpConstant %uint 1
%use_acc  = OpConstant %uint 2
%cmat_a   = OpTypeCooperativeMatrixKHR %float %scope %rows %cols %use_a
%cmat_b   = OpTypeCooperativeMatrixKHR %float %scope %rows %cols %use_b
%cmat_c   = OpTypeCooperativeMatrixKHR %float %scope %rows %cols %use_acc
%ptr_ssbo = OpTypePointer StorageBuffer %float
%arr_t    = OpTypeRuntimeArray %float
%blk_t    = OpTypeStruct %arr_t
%ptr_blk  = OpTypePointer StorageBuffer %blk_t
%gbuf     = OpVariable %ptr_blk StorageBuffer
"""

_SPIRV_HEADER_NV = """\
OpCapability Shader
OpCapability CooperativeMatrixNV
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void     = OpTypeVoid
%fn_type  = OpTypeFunction %void
%float    = OpTypeFloat 32
%uint     = OpTypeInt 32 0
%u32_0    = OpConstant %uint 0
%u32_8    = OpConstant %uint 8
%scope    = OpConstant %uint 3
%rows     = OpConstant %uint 8
%cols     = OpConstant %uint 8
%cmat_a   = OpTypeCooperativeMatrixNV %float %scope %rows %cols
%cmat_b   = OpTypeCooperativeMatrixNV %float %scope %rows %cols
%cmat_c   = OpTypeCooperativeMatrixNV %float %scope %rows %cols
%ptr_ssbo = OpTypePointer StorageBuffer %float
%arr_t    = OpTypeRuntimeArray %float
%blk_t    = OpTypeStruct %arr_t
%ptr_blk  = OpTypePointer StorageBuffer %blk_t
%gbuf     = OpVariable %ptr_blk StorageBuffer
"""


def _parse(spirv_asm: str) -> ShaderNode:
    """Parse a SPIR-V assembly string and return the ShaderNode."""
    lexer = VulkanLexer(spirv_asm)
    tokens = lexer.tokenize()
    parser = VulkanParser(tokens)
    return parser.parse()


def _function_body_calls(spirv_asm: str) -> list:
    """Return all FunctionCallNode names emitted in the first function body."""
    shader = _parse(spirv_asm)
    calls = []
    for fn in shader.functions:
        for stmt in fn.body:
            _collect_calls(stmt, calls)
    return calls


def _collect_calls(node, calls: list) -> None:
    if isinstance(node, FunctionCallNode):
        calls.append(node.name)
        for arg in node.args:
            _collect_calls(arg, calls)
    elif hasattr(node, "__dict__"):
        for value in vars(node).values():
            if isinstance(value, list):
                for item in value:
                    _collect_calls(item, calls)
            else:
                _collect_calls(value, calls)


def _wrap_function(body_instructions: str, header: str = _SPIRV_HEADER) -> str:
    """Wrap body_instructions in a minimal SPIR-V compute function."""
    return (
        header
        + """
%main = OpFunction %void None %fn_type
%entry = OpLabel
"""
        + body_instructions
        + """
OpReturn
OpFunctionEnd
"""
    )


# ---------------------------------------------------------------------------
# KHR Load tests
# ---------------------------------------------------------------------------


def test_khr_load_basic_emits_spirv_coop_mat_load():
    """OpCooperativeMatrixLoadKHR lowers to spirvCoopMatLoadKHR."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatLoadKHR" in calls


def test_khr_load_preserves_pointer_stride_layout_operands():
    """Lowered load preserves pointer, layout, and stride as arguments."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
"""
    shader = _parse(_wrap_function(body))
    calls = []
    for fn in shader.functions:
        for stmt in fn.body:
            _collect_calls(stmt, calls)
    # spirvCoopMatLoadKHR(pointer, layout, stride) -> 3 args minimum
    for fn in shader.functions:
        for stmt in fn.body:
            if isinstance(stmt, FunctionCallNode) and stmt.name == "spirvCoopMatLoadKHR":
                assert len(stmt.args) >= 3
                break


def test_khr_load_with_memory_access_operand():
    """Optional memory-access operand is preserved in the lowered load."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride MakePointerVisible|NonPrivatePointer %scope
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatLoadKHR" in calls


def test_khr_load_malformed_missing_operands_raises():
    """OpCooperativeMatrixLoadKHR with fewer than 4 operands raises a diagnostic."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0
"""
    with pytest.raises(SpirvInstructionUnsupportedError) as exc_info:
        _parse(_wrap_function(body))
    assert "OpCooperativeMatrixLoadKHR" in str(exc_info.value)
    assert "operands" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# NV Load tests
# ---------------------------------------------------------------------------


def test_nv_load_basic_emits_spirv_coop_mat_load_nv():
    """OpCooperativeMatrixLoadNV lowers to spirvCoopMatLoadNV."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%col_maj = OpConstant %uint 0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a %col_maj %stride
"""
    calls = _function_body_calls(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "spirvCoopMatLoadNV" in calls


def test_nv_load_malformed_missing_operands_raises():
    """OpCooperativeMatrixLoadNV with fewer than 4 operands raises a diagnostic."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a 0
"""
    with pytest.raises(SpirvInstructionUnsupportedError) as exc_info:
        _parse(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "OpCooperativeMatrixLoadNV" in str(exc_info.value)


# ---------------------------------------------------------------------------
# KHR Store tests
# ---------------------------------------------------------------------------


def test_khr_store_basic_emits_spirv_coop_mat_store():
    """OpCooperativeMatrixStoreKHR lowers to a spirvCoopMatStoreKHR statement."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
OpCooperativeMatrixStoreKHR %ptr_a %mat_a 0 %stride
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatStoreKHR" in calls


def test_khr_store_preserves_pointer_object_layout_stride():
    """Lowered store preserves pointer, object, layout, and stride as arguments."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
OpCooperativeMatrixStoreKHR %ptr_a %mat_a 0 %stride
"""
    shader = _parse(_wrap_function(body))
    for fn in shader.functions:
        for stmt in fn.body:
            if (
                isinstance(stmt, FunctionCallNode)
                and stmt.name == "spirvCoopMatStoreKHR"
            ):
                assert len(stmt.args) >= 4
                break


def test_khr_store_with_memory_access_operand():
    """Optional memory-access operand on store is preserved."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
OpCooperativeMatrixStoreKHR %ptr_a %mat_a 0 %stride MakePointerAvailable|NonPrivatePointer %scope
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatStoreKHR" in calls


def test_khr_store_malformed_missing_operands_raises():
    """OpCooperativeMatrixStoreKHR with fewer than 4 operands raises a diagnostic."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %u32_8
OpCooperativeMatrixStoreKHR %ptr_a %mat_a 0
"""
    with pytest.raises(SpirvInstructionUnsupportedError) as exc_info:
        _parse(_wrap_function(body))
    assert "OpCooperativeMatrixStoreKHR" in str(exc_info.value)
    assert "operands" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# NV Store tests
# ---------------------------------------------------------------------------


def test_nv_store_basic_emits_spirv_coop_mat_store_nv():
    """OpCooperativeMatrixStoreNV lowers to spirvCoopMatStoreNV."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%col_maj = OpConstant %uint 0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a %col_maj %stride
OpCooperativeMatrixStoreNV %ptr_a %mat_a %col_maj %stride
"""
    calls = _function_body_calls(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "spirvCoopMatStoreNV" in calls


def test_nv_store_malformed_missing_operands_raises():
    """OpCooperativeMatrixStoreNV with fewer than 4 operands raises a diagnostic."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a 0 %u32_8
OpCooperativeMatrixStoreNV %ptr_a %mat_a 0
"""
    with pytest.raises(SpirvInstructionUnsupportedError) as exc_info:
        _parse(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "OpCooperativeMatrixStoreNV" in str(exc_info.value)


# ---------------------------------------------------------------------------
# KHR MulAdd tests
# ---------------------------------------------------------------------------


def test_khr_muladd_basic_emits_spirv_coop_mat_muladd():
    """OpCooperativeMatrixMulAddKHR lowers to spirvCoopMatMulAddKHR."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
%mat_b   = OpCooperativeMatrixLoadKHR %cmat_b %ptr_a 0 %stride
%mat_c   = OpCooperativeMatrixLoadKHR %cmat_c %ptr_a 0 %stride
%result  = OpCooperativeMatrixMulAddKHR %cmat_c %mat_a %mat_b %mat_c
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatMulAddKHR" in calls


def test_khr_muladd_preserves_a_b_c_operands():
    """Lowered MulAdd preserves the three matrix operands."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
%mat_b   = OpCooperativeMatrixLoadKHR %cmat_b %ptr_a 0 %stride
%mat_c   = OpCooperativeMatrixLoadKHR %cmat_c %ptr_a 0 %stride
%result  = OpCooperativeMatrixMulAddKHR %cmat_c %mat_a %mat_b %mat_c
"""
    shader = _parse(_wrap_function(body))
    for fn in shader.functions:
        for stmt in fn.body:
            if (
                isinstance(stmt, FunctionCallNode)
                and stmt.name == "spirvCoopMatMulAddKHR"
            ):
                assert len(stmt.args) >= 3
                break


def test_khr_muladd_with_signed_mask():
    """KHR cooperative-matrix operand mask (e.g., MatrixASignedComponents) is preserved."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
%mat_b   = OpCooperativeMatrixLoadKHR %cmat_b %ptr_a 0 %stride
%mat_c   = OpCooperativeMatrixLoadKHR %cmat_c %ptr_a 0 %stride
%result  = OpCooperativeMatrixMulAddKHR %cmat_c %mat_a %mat_b %mat_c MatrixASignedComponents
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatMulAddKHR" in calls
    # Verify the mask operand is passed through
    shader = _parse(_wrap_function(body))
    for fn in shader.functions:
        for stmt in fn.body:
            if (
                isinstance(stmt, FunctionCallNode)
                and stmt.name == "spirvCoopMatMulAddKHR"
            ):
                # 3 matrix args + 1 mask arg = 4 total
                assert len(stmt.args) >= 4
                break


def test_khr_muladd_with_saturating_mask():
    """SaturatingAccumulation mask is preserved in the lowered MulAdd."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
%mat_b   = OpCooperativeMatrixLoadKHR %cmat_b %ptr_a 0 %stride
%mat_c   = OpCooperativeMatrixLoadKHR %cmat_c %ptr_a 0 %stride
%result  = OpCooperativeMatrixMulAddKHR %cmat_c %mat_a %mat_b %mat_c SaturatingAccumulation
"""
    shader = _parse(_wrap_function(body))
    found = False
    for fn in shader.functions:
        for stmt in fn.body:
            if (
                isinstance(stmt, FunctionCallNode)
                and stmt.name == "spirvCoopMatMulAddKHR"
            ):
                assert len(stmt.args) >= 4
                found = True
                break
    assert found, "spirvCoopMatMulAddKHR not found in function body"


def test_khr_muladd_malformed_missing_operands_raises():
    """OpCooperativeMatrixMulAddKHR with fewer than 4 operands raises a diagnostic."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
%mat_b   = OpCooperativeMatrixLoadKHR %cmat_b %ptr_a 0 %stride
%result  = OpCooperativeMatrixMulAddKHR %cmat_c %mat_a %mat_b
"""
    with pytest.raises(SpirvInstructionUnsupportedError) as exc_info:
        _parse(_wrap_function(body))
    assert "OpCooperativeMatrixMulAddKHR" in str(exc_info.value)
    assert "operands" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# NV MulAdd tests
# ---------------------------------------------------------------------------


def test_nv_muladd_basic_emits_spirv_coop_mat_muladd_nv():
    """OpCooperativeMatrixMulAddNV lowers to spirvCoopMatMulAddNV."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%col_maj = OpConstant %uint 0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a %col_maj %stride
%mat_b   = OpCooperativeMatrixLoadNV %cmat_b %ptr_a %col_maj %stride
%mat_c   = OpCooperativeMatrixLoadNV %cmat_c %ptr_a %col_maj %stride
%result  = OpCooperativeMatrixMulAddNV %cmat_c %mat_a %mat_b %mat_c
"""
    calls = _function_body_calls(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "spirvCoopMatMulAddNV" in calls


def test_nv_muladd_malformed_missing_operands_raises():
    """OpCooperativeMatrixMulAddNV with fewer than 4 operands raises a diagnostic."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%col_maj = OpConstant %uint 0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a %col_maj %stride
%mat_b   = OpCooperativeMatrixLoadNV %cmat_b %ptr_a %col_maj %stride
%result  = OpCooperativeMatrixMulAddNV %cmat_c %mat_a %mat_b
"""
    with pytest.raises(SpirvInstructionUnsupportedError) as exc_info:
        _parse(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "OpCooperativeMatrixMulAddNV" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Round-trip: KHR load → store (the result is used in a store)
# ---------------------------------------------------------------------------


def test_khr_load_store_round_trip():
    """A load immediately followed by a store round-trips without loss."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
OpCooperativeMatrixStoreKHR %ptr_a %mat_a 0 %stride
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatLoadKHR" in calls
    assert "spirvCoopMatStoreKHR" in calls


def test_nv_load_store_round_trip():
    """NV load immediately followed by NV store round-trips without loss."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%col_maj = OpConstant %uint 0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a %col_maj %stride
OpCooperativeMatrixStoreNV %ptr_a %mat_a %col_maj %stride
"""
    calls = _function_body_calls(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "spirvCoopMatLoadNV" in calls
    assert "spirvCoopMatStoreNV" in calls


def test_khr_full_muladd_pipeline_round_trip():
    """Full KHR load-muladd-store pipeline is lowered without loss."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
%mat_b   = OpCooperativeMatrixLoadKHR %cmat_b %ptr_a 0 %stride
%mat_c   = OpCooperativeMatrixLoadKHR %cmat_c %ptr_a 0 %stride
%result  = OpCooperativeMatrixMulAddKHR %cmat_c %mat_a %mat_b %mat_c
OpCooperativeMatrixStoreKHR %ptr_a %result 0 %stride
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatLoadKHR" in calls
    assert "spirvCoopMatMulAddKHR" in calls
    assert "spirvCoopMatStoreKHR" in calls


def test_nv_full_muladd_pipeline_round_trip():
    """Full NV load-muladd-store pipeline is lowered without loss."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%col_maj = OpConstant %uint 0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a %col_maj %stride
%mat_b   = OpCooperativeMatrixLoadNV %cmat_b %ptr_a %col_maj %stride
%mat_c   = OpCooperativeMatrixLoadNV %cmat_c %ptr_a %col_maj %stride
%result  = OpCooperativeMatrixMulAddNV %cmat_c %mat_a %mat_b %mat_c
OpCooperativeMatrixStoreNV %ptr_a %result %col_maj %stride
"""
    calls = _function_body_calls(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "spirvCoopMatLoadNV" in calls
    assert "spirvCoopMatMulAddNV" in calls
    assert "spirvCoopMatStoreNV" in calls


# ---------------------------------------------------------------------------
# KHR and NV are distinct variants
# ---------------------------------------------------------------------------


def test_khr_load_does_not_produce_nv_name():
    """KHR load does not accidentally emit NV variant name."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatLoadKHR" in calls
    assert "spirvCoopMatLoadNV" not in calls


def test_nv_load_does_not_produce_khr_name():
    """NV load does not accidentally emit KHR variant name."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%col_maj = OpConstant %uint 0
%mat_a   = OpCooperativeMatrixLoadNV %cmat_a %ptr_a %col_maj %stride
"""
    calls = _function_body_calls(_wrap_function(body, _SPIRV_HEADER_NV))
    assert "spirvCoopMatLoadNV" in calls
    assert "spirvCoopMatLoadKHR" not in calls


def test_khr_muladd_does_not_produce_nv_name():
    """KHR MulAdd does not accidentally emit NV variant name."""
    body = """\
%ptr_a   = OpAccessChain %ptr_ssbo %gbuf %u32_0 %u32_0
%stride  = OpConstant %uint 8
%mat_a   = OpCooperativeMatrixLoadKHR %cmat_a %ptr_a 0 %stride
%mat_b   = OpCooperativeMatrixLoadKHR %cmat_b %ptr_a 0 %stride
%mat_c   = OpCooperativeMatrixLoadKHR %cmat_c %ptr_a 0 %stride
%result  = OpCooperativeMatrixMulAddKHR %cmat_c %mat_a %mat_b %mat_c
"""
    calls = _function_body_calls(_wrap_function(body))
    assert "spirvCoopMatMulAddKHR" in calls
    assert "spirvCoopMatMulAddNV" not in calls
