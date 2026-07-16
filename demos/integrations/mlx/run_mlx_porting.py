#!/usr/bin/env python3
"""Run pinned MLX project-porting checks through the public CrossTL CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from crosstl.project import (
    VulkanComputeRuntime,
    build_project_test_runner_plan,
    build_runtime_artifact_manifest,
    execute_project_test_runner_plan,
    native_runtime_parity_adapters,
)
from crosstl.project.runtime_verification import (
    RuntimeParityAdapter,
    build_runtime_test_manifest,
    plan_runtime_test_manifest,
)

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_METAL_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_ARANGE_SOURCE = "mlx/backend/metal/kernels/arange.metal"
MLX_ARG_REDUCE_SOURCE = "mlx/backend/metal/kernels/arg_reduce.metal"
MLX_BINARY_TWO_SOURCE = "mlx/backend/metal/kernels/binary_two.metal"
MLX_FENCE_SOURCE = "mlx/backend/metal/kernels/fence.metal"
MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT = 3
MLX_FFT_SOURCE = "mlx/backend/metal/kernels/fft.metal"
MLX_FFT_SHA256 = "3a1fbb38ed64f50a49a20d0c5adb1748d9d06ea20e5931e99aa26be543cb7825"
MLX_FFT_SOURCE_SIZE_BYTES = 3278
MLX_GEMV_SOURCE = "mlx/backend/metal/kernels/gemv.metal"
MLX_GEMV_SHA256 = "c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f"
MLX_GEMV_SOURCE_SIZE_BYTES = 5383
MLX_LAYER_NORM_SOURCE = "mlx/backend/metal/kernels/layer_norm.metal"
MLX_LOGSUMEXP_SOURCE = "mlx/backend/metal/kernels/logsumexp.metal"
MLX_METAL_ROUNDTRIP_SOURCE = MLX_FENCE_SOURCE
MLX_RANDOM_SOURCE = "mlx/backend/metal/kernels/random.metal"
MLX_RMS_NORM_SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
MLX_ROPE_SOURCE = "mlx/backend/metal/kernels/rope.metal"
MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE = (
    "mlx/backend/metal/kernels/scaled_dot_product_attention.metal"
)
MLX_SOFTMAX_SOURCE = "mlx/backend/metal/kernels/softmax.metal"
MLX_TERNARY_SOURCE = "mlx/backend/metal/kernels/ternary.metal"
REFERENCE_ACCESSOR_FIXTURE_NAME = "reference_accessor_lvalue.metal"
REFERENCE_ACCESSOR_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / REFERENCE_ACCESSOR_FIXTURE_NAME
)
REFERENCE_ACCESSOR_TARGETS = ("directx", "opengl")
REFERENCE_ACCESSOR_SENTINEL = "73.25"
REFERENCE_ACCESSOR_DXC_ENTRY_POINT = "CSMain"
TEMPLATE_MEMBER_POINTER_FIXTURE_NAME = "template_member_buffer_pointer.metal"
TEMPLATE_MEMBER_POINTER_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / TEMPLATE_MEMBER_POINTER_FIXTURE_NAME
)
TEMPLATE_MEMBER_POINTER_TARGETS = ("directx", "opengl")
MLX_OPENGL_FRONTIER_SOURCES = (
    MLX_ARG_REDUCE_SOURCE,
    MLX_BINARY_TWO_SOURCE,
    MLX_LOGSUMEXP_SOURCE,
    MLX_RMS_NORM_SOURCE,
    MLX_ROPE_SOURCE,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
    MLX_SOFTMAX_SOURCE,
    MLX_TERNARY_SOURCE,
)
MLX_DIRECTX_VULKAN_FRONTIER_SOURCES = (
    MLX_ARANGE_SOURCE,
    MLX_ARG_REDUCE_SOURCE,
    MLX_BINARY_TWO_SOURCE,
    MLX_LAYER_NORM_SOURCE,
    MLX_LOGSUMEXP_SOURCE,
    MLX_RANDOM_SOURCE,
    MLX_RMS_NORM_SOURCE,
    MLX_ROPE_SOURCE,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
    MLX_SOFTMAX_SOURCE,
    MLX_TERNARY_SOURCE,
)
MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES = (
    MLX_ARG_REDUCE_SOURCE,
    MLX_LAYER_NORM_SOURCE,
    MLX_LOGSUMEXP_SOURCE,
    MLX_RMS_NORM_SOURCE,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
    MLX_SOFTMAX_SOURCE,
)
MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES = tuple(
    source
    for source in MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    if source not in MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
)
MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES = tuple(
    source
    for source in MLX_OPENGL_FRONTIER_SOURCES
    if source in MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
)
MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES = tuple(
    source
    for source in MLX_OPENGL_FRONTIER_SOURCES
    if source not in MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
)
MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES = MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES = tuple(
    dict.fromkeys(
        (
            *MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
            *MLX_OPENGL_FRONTIER_SOURCES,
        )
    )
)
MLX_BLOCKED_REDUCED_FRONTIER_SOURCES = (MLX_FENCE_SOURCE,)
MLX_REDUCED_FRONTIER_SOURCES = tuple(
    dict.fromkeys(
        sorted(
            (
                *MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES,
                *MLX_BLOCKED_REDUCED_FRONTIER_SOURCES,
            )
        )
    )
)
# Pinned generated compute entries in the complete DirectX source frontier.
MLX_DIRECTX_FRONTIER_ENTRY_POINT_COUNTS = {
    MLX_ARANGE_SOURCE: 11,
    MLX_ARG_REDUCE_SOURCE: 24,
    MLX_BINARY_TWO_SOURCE: 225,
    MLX_LAYER_NORM_SOURCE: 12,
    MLX_LOGSUMEXP_SOURCE: 6,
    MLX_RANDOM_SOURCE: 2,
    MLX_RMS_NORM_SOURCE: 12,
    MLX_ROPE_SOURCE: 18,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: 42,
    MLX_SOFTMAX_SOURCE: 10,
    MLX_TERNARY_SOURCE: 212,
}
# DXC compiles only sources whose pinned host dispatch does not require a runtime
# workgroup-size variant. Dynamic sources remain explicit failed artifact records.
MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES = MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES
MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS = {
    source: MLX_DIRECTX_FRONTIER_ENTRY_POINT_COUNTS[source]
    for source in MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
}
MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT = sum(
    MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS.values()
)
MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS = {
    source: MLX_DIRECTX_FRONTIER_ENTRY_POINT_COUNTS[source]
    for source in MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
}
MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE = (
    "project.translate.workgroup-size-entry-ambiguous"
)
MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_MESSAGE = (
    "A multi-entry compute artifact cannot apply one workgroup size when the "
    "emitted entry points do not declare a shared source size. Select one entry "
    "point or emit independently specialized artifacts."
)
MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE = "https://github.com/CrossGL/crosstl/issues/1750"
MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE = {
    MLX_ARG_REDUCE_SOURCE: {
        "specializationCount": 51,
        "hostSource": "mlx/backend/metal/primitives.cpp",
        "hostLines": "117-132",
        "dispatchFormulas": [
            "[32 * ceil_div(min(ceil_div(axis_size, 4), "
            "maxTotalThreadsPerThreadgroup), 32), 1, 1]"
        ],
        "dispatchSelection": ["single host dispatch path"],
        "runtimeOperands": ["axis_size", "maxTotalThreadsPerThreadgroup"],
        "materializationParameters": ["N_READS", "Op", "T"],
    },
    MLX_LAYER_NORM_SOURCE: {
        "specializationCount": 16,
        "hostSource": "mlx/backend/metal/normalization.cpp",
        "hostLines": "248-297,346-422",
        "dispatchFormulas": [
            "forward block: [32 * ceil_div(ceil_div(axis_size, 8), 32), 1, 1]",
            "forward looped: [maxTotalThreadsPerThreadgroup, 1, 1]",
            "VJP block: [32 * ceil_div(ceil_div(axis_size, 8), 32), 1, 1]",
            "VJP looped: [maxTotalThreadsPerThreadgroup, 1, 1]",
        ],
        "dispatchSelection": [
            "forward block when axis_size <= 6656; looped when axis_size > 6656",
            "VJP block when axis_size <= 8192; looped when axis_size > 8192",
        ],
        "runtimeOperands": [
            "axis_size",
            "forward_or_vjp_entry",
            "maxTotalThreadsPerThreadgroup",
        ],
        "materializationParameters": ["N_READS", "T"],
    },
    MLX_LOGSUMEXP_SOURCE: {
        "specializationCount": 7,
        "hostSource": "mlx/backend/metal/logsumexp.cpp",
        "hostLines": "58-91",
        "dispatchFormulas": [
            "block: [32 * ceil_div(ceil_div(axis_size, 4), 32), 1, 1]",
            "looped: [maxTotalThreadsPerThreadgroup, 1, 1]",
        ],
        "dispatchSelection": [
            "block when axis_size <= 4096; looped when axis_size > 4096"
        ],
        "runtimeOperands": ["axis_size", "maxTotalThreadsPerThreadgroup"],
        "materializationParameters": ["AccT", "N_READS", "T"],
    },
    MLX_RMS_NORM_SOURCE: {
        "specializationCount": 12,
        "hostSource": "mlx/backend/metal/normalization.cpp",
        "hostLines": "52-91,137-197",
        "dispatchFormulas": [
            "block: [32 * ceil_div(ceil_div(axis_size, 4), 32), 1, 1]",
            "looped: [maxTotalThreadsPerThreadgroup, 1, 1]",
        ],
        "dispatchSelection": [
            "block when axis_size <= 4096; looped when axis_size > 4096"
        ],
        "runtimeOperands": ["axis_size", "maxTotalThreadsPerThreadgroup"],
        "materializationParameters": ["N_READS", "T"],
    },
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: {
        "specializationCount": 42,
        "hostSource": "mlx/backend/metal/scaled_dot_product_attention.cpp",
        "hostLines": (
            "31-32,160-163,194-195,323-326,350-415,440-484,561-586," "613-640,685-753"
        ),
        "dispatchFormulas": [
            "full attention (NAX and Metal): [32, 4, 1]",
            "vector: [1024, 1, 1]",
            "two-pass first pass: [32, q_heads / kv_heads, q_sequence_length]",
            "two-pass final pass: [1024, 1, 1]",
        ],
        "dispatchSelection": [
            "full attention when q_sequence_length > 8",
            "vector mode when q_sequence_length <= 8",
            "two-pass vector when ((device architecture is d or s) and "
            "k_sequence_length >= 1024) or (kv_heads < q_heads and "
            "k_sequence_length >= 4096); otherwise one-pass vector",
        ],
        "runtimeOperands": [
            "selected_pass",
            "device_architecture",
            "q_heads",
            "kv_heads",
            "q_sequence_length",
            "k_sequence_length",
        ],
        "materializationParameters": ["D", "T", "V"],
    },
    MLX_SOFTMAX_SOURCE: {
        "specializationCount": 17,
        "hostSource": "mlx/backend/metal/softmax.cpp",
        "hostLines": "46-83",
        "dispatchFormulas": [
            "block: [32 * ceil_div(ceil_div(axis_size, 4), 32), 1, 1]",
            "looped: [maxTotalThreadsPerThreadgroup, 1, 1]",
        ],
        "dispatchSelection": [
            "block when axis_size <= 4096; looped when axis_size > 4096"
        ],
        "runtimeOperands": ["axis_size", "maxTotalThreadsPerThreadgroup"],
        "materializationParameters": ["AccT", "N_READS", "T"],
    },
}
MLX_FRONTIER_SPECIALIZATION_CONSTANTS = {
    "1": False,
    "2": False,
    "3": False,
    "20": False,
    "21": False,
    "22": False,
    "23": False,
    "24": False,
    "25": False,
    "26": 1,
}
MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS = {
    MLX_RMS_NORM_SOURCE: {"has_w": 20},
    MLX_ROPE_SOURCE: {"forward": 1, "traditional": 2, "hs_transpose": 3},
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: {
        "has_mask": 20,
        "query_transposed": 21,
        "do_causal": 22,
        "bool_mask": 23,
        "float_mask": 24,
        "has_sinks": 25,
        "blocks": 26,
    },
}
MLX_FENCE_REQUESTED_CONTRACT = {
    "memoryFlags": ["mem_device"],
    "memoryOrder": "memory_order_seq_cst",
    "threadScope": "thread_scope_system",
}
MLX_FENCE_TARGET_CONTRACTS = {
    "directx": {
        "diagnosticCode": "project.translate.directx-atomic-fence-unsupported",
        "missingCapability": "directx.atomic-thread-fence-contract-lowering",
        "targetDescription": "HLSL",
    },
    "opengl": {
        "diagnosticCode": "project.translate.opengl-atomic-fence-unsupported",
        "missingCapability": "opengl.atomic-thread-fence-contract-lowering",
        "targetDescription": "OpenGL GLSL",
    },
    "vulkan": {
        "diagnosticCode": "project.translate.vulkan-atomic-fence-unsupported",
        "missingCapability": "spirv.atomic-thread-fence-contract-lowering",
        "targetDescription": "Vulkan SPIR-V",
    },
}
EXPECTED_METAL_KERNEL_COUNT = 40
FULL_CORPUS_TARGETS = ("directx", "opengl", "vulkan")
FULL_CORPUS_EXPECTED_ARTIFACT_COUNT = EXPECTED_METAL_KERNEL_COUNT * len(
    FULL_CORPUS_TARGETS
)
FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT = len(MLX_FENCE_TARGET_CONTRACTS)
FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT = (
    FULL_CORPUS_EXPECTED_ARTIFACT_COUNT - FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
)
FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS = 4096
FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK = 131072
FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS = 900
GEMV_MAX_TEMPLATE_SPECIALIZATIONS = 4096
GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK = 2097152
GEMV_EXPECTED_SPECIALIZATION_COUNT = 225
GEMV_EXPECTED_ENTRY_POINT_COUNT = 224
GEMV_DIRECTX_TRANSLATION_TIMEOUT_SECONDS = 900
GEMV_DIRECTX_ENTRY_PROFILE = "cs_6_0"
GEMV_DIRECTX_LIBRARY_PROFILE = "lib_6_6"
GEMV_DIRECTX_EXPECTED_ENTRY_POINTS = (
    "CSMain",
    *(f"CSMain_{index}" for index in range(2, GEMV_EXPECTED_ENTRY_POINT_COUNT + 1)),
)
GEMV_DIRECTX_COMPILER_ENTRY_POINTS = ("CSMain", "CSMain_85", "CSMain_113")
GEMV_WORKGROUP_SIZE_RULE = (32, "BN", "BM")
GEMV_REPORT_WORKGROUP_SIZE_RULE = tuple(
    str(component) for component in GEMV_WORKGROUP_SIZE_RULE
)
GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES = (
    (32, 1, 1),
    (32, 1, 4),
    (32, 1, 8),
    (32, 2, 1),
    (32, 4, 1),
    (32, 8, 1),
    (32, 16, 1),
)
GEMV_DIRECTX_LIBRARY_NUMTHREADS_WARNING_MESSAGE = (
    "attribute 'numthreads' ignored without accompanying shader attribute "
    "[-Wmisplaced-attributes]"
)
GEMV_DIRECTX_EXPECTED_LIBRARY_WARNING_COUNT = GEMV_EXPECTED_ENTRY_POINT_COUNT
GEMV_OPENGL_TRANSLATION_TIMEOUT_SECONDS = 900
GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE = (
    "project.translate.opengl-workgroup-pointer-unsupported"
)
GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY = "opengl.workgroup-pointer-lowering"
GEMV_OPENGL_EXPECTED_MESSAGE = (
    "OpenGL cannot prove the workgroup pointer access range for "
    "'GEMVKernel_float_1_8_1_32_4_4_0__run.tgp_memory' into shared backing "
    "'tgp_memory'"
)
GEMV_OPENGL_EXPECTED_POINTER_EVIDENCE = {
    "backingName": "tgp_memory",
    "function": "GEMVKernel_float_1_8_1_32_4_4_0__run",
    "materializationName": (
        "GEMVKernel_float_1_8_1_32_4_4_0_run__glsl_tgp_memory_"
        "gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0_"
        "tgp_memory_float_64"
    ),
    "offsetExpression": "0",
    "parameter": "tgp_memory",
    "reason": "unprovable-view-access",
}
GEMV_OPENGL_BACKING_RANGE_ISSUE = "https://github.com/CrossGL/crosstl/issues/1671"
GEMV_OPENGL_SUBGROUP_WIDTH_ISSUE = "https://github.com/CrossGL/crosstl/issues/1786"
GEMV_OPENGL_EXECUTION_TRACKED_ISSUES = (GEMV_OPENGL_SUBGROUP_WIDTH_ISSUE,)
GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES = GEMV_OPENGL_EXECUTION_TRACKED_ISSUES
GEMV_OPENGL_FRONTIER_TRACKED_ISSUES = (
    GEMV_OPENGL_BACKING_RANGE_ISSUE,
    *GEMV_OPENGL_EXECUTION_TRACKED_ISSUES,
)
FFT_OPENGL_MAX_TEMPLATE_SPECIALIZATIONS = 4096
FFT_OPENGL_MAX_TEMPLATE_MATERIALIZATION_WORK = 2097152
FFT_OPENGL_TRANSLATION_TIMEOUT_SECONDS = 900
FFT_OPENGL_EXPECTED_SPECIALIZATION_COUNT = 117
FFT_OPENGL_EXPECTED_FUNCTION_CONSTANT_COUNT = 22
FFT_OPENGL_WORKGROUP_POINTER_ISSUE = "https://github.com/CrossGL/crosstl/issues/1671"
FFT_OPENGL_EXPECTED_DIAGNOSTIC_CODE = (
    "project.translate.opengl-workgroup-pointer-unsupported"
)
FFT_OPENGL_EXPECTED_MISSING_CAPABILITY = "opengl.workgroup-pointer-lowering"
FFT_OPENGL_EXPECTED_MESSAGE = (
    "OpenGL cannot prove the workgroup pointer access range for "
    "'ReadWriter_float2_float2__load.crosstl_ptr_buf' into shared backing "
    "'shared_in'"
)
FFT_OPENGL_EXPECTED_POINTER_EVIDENCE = {
    "backingName": "shared_in",
    "function": "ReadWriter_float2_float2__load",
    "materializationName": (
        "ReadWriter_float2_float2_load__glsl_crosstl_ptr_buf_"
        "fft_mem_256_float2_float2_shared_in_vec2_256"
    ),
    "offsetExpression": "0",
    "parameter": "crosstl_ptr_buf",
    "reason": "unprovable-view-access",
}
REDUCED_FRONTIER_MODE = "reduced-frontier"
FULL_CORPUS_MODE = "full-corpus"
FRONTIER_VALIDATION_TRACKED_ISSUES: tuple[str, ...] = ()
FULL_CORPUS_TRANSLATION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1376",
    "https://github.com/CrossGL/crosstl/issues/1676",
    "https://github.com/CrossGL/crosstl/issues/1476",
    "https://github.com/CrossGL/crosstl/issues/1479",
    "https://github.com/CrossGL/crosstl/issues/1490",
    "https://github.com/CrossGL/crosstl/issues/1544",
    "https://github.com/CrossGL/crosstl/issues/1546",
    "https://github.com/CrossGL/crosstl/issues/1554",
    "https://github.com/CrossGL/crosstl/issues/1559",
    "https://github.com/CrossGL/crosstl/issues/1562",
    "https://github.com/CrossGL/crosstl/issues/1659",
    "https://github.com/CrossGL/crosstl/issues/1669",
    "https://github.com/CrossGL/crosstl/issues/1671",
    "https://github.com/CrossGL/crosstl/issues/1672",
)
RUNTIME_READINESS_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1388",
    "https://github.com/CrossGL/crosstl/issues/1471",
)
VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES: tuple[str, ...] = ()
FENCE_CONTRACT_TRACKED_ISSUES = ("https://github.com/CrossGL/crosstl/issues/1537",)
VULKAN_GEMV_REPORTING_TRACKED_ISSUE = "https://github.com/CrossGL/crosstl/issues/1517"
FULL_CORPUS_SEMANTIC_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1491",
)
METAL_ROUNDTRIP_SEMANTIC_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1660",
)
OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES: tuple[str, ...] = ()
OPENGL_SCALED_DOT_PRODUCT_ATTENTION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1538",
)
RUNTIME_READINESS_ENTRY_POINTS = {
    "directx": "CSMain",
    "opengl": "main",
    "vulkan": "arangeuint32",
}
RUNTIME_READINESS_DEFAULT_VARIANTS = {
    "directx": "uint8",
    "opengl": "uint32",
    "vulkan": "uint32",
}
ARANGE_RUNTIME_VARIANTS = {
    "uint8": {
        "start": 3,
        "step": 2,
        "expected": [3, 5, 7, 9],
    },
    "uint32": {
        "start": 300,
        "step": 17,
        "expected": [300, 317, 334, 351],
    },
    "int32": {
        "start": -3,
        "step": 2,
        "expected": [-3, -1, 1, 3],
    },
    "float32": {
        "start": 1.5,
        "step": 0.25,
        "expected": [1.5, 1.75, 2.0, 2.25],
    },
}
VULKAN_ARANGE_RUNTIME_VARIANTS = ("uint32", "int32", "float32")
RUNTIME_FIXTURE_EXECUTION_ADAPTER_KIND = "mlx-arange-reference-runtime"
NATIVE_RUNTIME_EXECUTION_SCOPE = "native-runtime-execution-readiness"
RUNTIME_READINESS_DIAGNOSTIC_CODES = frozenset(
    (
        "project.runtime-test-manifest.entry-points-unavailable",
        "project.runtime-test-manifest.resource-bindings-unavailable",
        "project.runtime-test-manifest.dispatch-unavailable",
    )
)
RUNTIME_READINESS_PLAN_DIAGNOSTIC_CODES = frozenset(
    ("project.runtime-verification.resource-unbound",)
)
FULL_CORPUS_TRACKED_ISSUES = (
    *FRONTIER_VALIDATION_TRACKED_ISSUES,
    MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE,
    "https://github.com/CrossGL/crosstl/issues/1312",
    "https://github.com/CrossGL/crosstl/issues/1670",
    *FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
    *OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES,
    *OPENGL_SCALED_DOT_PRODUCT_ATTENTION_TRACKED_ISSUES,
    *GEMV_OPENGL_EXECUTION_TRACKED_ISSUES,
    *RUNTIME_READINESS_TRACKED_ISSUES,
    *FENCE_CONTRACT_TRACKED_ISSUES,
    *VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES,
    VULKAN_GEMV_REPORTING_TRACKED_ISSUE,
    *FULL_CORPUS_SEMANTIC_TRACKED_ISSUES,
    *METAL_ROUNDTRIP_SEMANTIC_TRACKED_ISSUES,
)
RESOLVED_FRONTIER_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1728",
    "https://github.com/CrossGL/crosstl/issues/1701",
    "https://github.com/CrossGL/crosstl/issues/1694",
    "https://github.com/CrossGL/crosstl/issues/1695",
    "https://github.com/CrossGL/crosstl/issues/1667",
    "https://github.com/CrossGL/crosstl/issues/1668",
    "https://github.com/CrossGL/crosstl/issues/1661",
    "https://github.com/CrossGL/crosstl/issues/1573",
    "https://github.com/CrossGL/crosstl/issues/1555",
    "https://github.com/CrossGL/crosstl/issues/1561",
    "https://github.com/CrossGL/crosstl/issues/1551",
    "https://github.com/CrossGL/crosstl/issues/1498",
    "https://github.com/CrossGL/crosstl/issues/1535",
    "https://github.com/CrossGL/crosstl/issues/1489",
    "https://github.com/CrossGL/crosstl/issues/1504",
    "https://github.com/CrossGL/crosstl/issues/1503",
    "https://github.com/CrossGL/crosstl/issues/1502",
    "https://github.com/CrossGL/crosstl/issues/1500",
    "https://github.com/CrossGL/crosstl/issues/1454",
    "https://github.com/CrossGL/crosstl/issues/1453",
    "https://github.com/CrossGL/crosstl/issues/1452",
    "https://github.com/CrossGL/crosstl/issues/1354",
    "https://github.com/CrossGL/crosstl/issues/1362",
    "https://github.com/CrossGL/crosstl/issues/1394",
    "https://github.com/CrossGL/crosstl/issues/1396",
    "https://github.com/CrossGL/crosstl/issues/1392",
    "https://github.com/CrossGL/crosstl/issues/1317",
    "https://github.com/CrossGL/crosstl/issues/1300",
    "https://github.com/CrossGL/crosstl/issues/939",
    "https://github.com/CrossGL/crosstl/issues/940",
    "https://github.com/CrossGL/crosstl/issues/941",
    "https://github.com/CrossGL/crosstl/issues/943",
    "https://github.com/CrossGL/crosstl/issues/944",
    "https://github.com/CrossGL/crosstl/issues/945",
    "https://github.com/CrossGL/crosstl/issues/946",
    "https://github.com/CrossGL/crosstl/issues/979",
    "https://github.com/CrossGL/crosstl/issues/980",
    "https://github.com/CrossGL/crosstl/issues/981",
    "https://github.com/CrossGL/crosstl/issues/982",
    "https://github.com/CrossGL/crosstl/issues/983",
    "https://github.com/CrossGL/crosstl/issues/984",
    "https://github.com/CrossGL/crosstl/issues/985",
    "https://github.com/CrossGL/crosstl/issues/1001",
    "https://github.com/CrossGL/crosstl/issues/1002",
    "https://github.com/CrossGL/crosstl/issues/1003",
    "https://github.com/CrossGL/crosstl/issues/1004",
    "https://github.com/CrossGL/crosstl/issues/1006",
    "https://github.com/CrossGL/crosstl/issues/1007",
    "https://github.com/CrossGL/crosstl/issues/1012",
    "https://github.com/CrossGL/crosstl/issues/1013",
    "https://github.com/CrossGL/crosstl/issues/1019",
    "https://github.com/CrossGL/crosstl/issues/1026",
    "https://github.com/CrossGL/crosstl/issues/1027",
    "https://github.com/CrossGL/crosstl/issues/1028",
    "https://github.com/CrossGL/crosstl/issues/1029",
    "https://github.com/CrossGL/crosstl/issues/1030",
    "https://github.com/CrossGL/crosstl/issues/1031",
    "https://github.com/CrossGL/crosstl/issues/1033",
    "https://github.com/CrossGL/crosstl/issues/1034",
    "https://github.com/CrossGL/crosstl/issues/1035",
    "https://github.com/CrossGL/crosstl/issues/1036",
    "https://github.com/CrossGL/crosstl/issues/1032",
    "https://github.com/CrossGL/crosstl/issues/1037",
    "https://github.com/CrossGL/crosstl/issues/1038",
    "https://github.com/CrossGL/crosstl/issues/1039",
    "https://github.com/CrossGL/crosstl/issues/1068",
    "https://github.com/CrossGL/crosstl/issues/1104",
    "https://github.com/CrossGL/crosstl/issues/1105",
    "https://github.com/CrossGL/crosstl/issues/1106",
    "https://github.com/CrossGL/crosstl/issues/1107",
    "https://github.com/CrossGL/crosstl/issues/1110",
    "https://github.com/CrossGL/crosstl/issues/1111",
    "https://github.com/CrossGL/crosstl/issues/1122",
    "https://github.com/CrossGL/crosstl/issues/1124",
    "https://github.com/CrossGL/crosstl/issues/1126",
    "https://github.com/CrossGL/crosstl/issues/1127",
    "https://github.com/CrossGL/crosstl/issues/852",
    "https://github.com/CrossGL/crosstl/issues/1146",
    "https://github.com/CrossGL/crosstl/issues/1155",
    "https://github.com/CrossGL/crosstl/issues/1160",
    "https://github.com/CrossGL/crosstl/issues/1184",
    "https://github.com/CrossGL/crosstl/issues/1203",
    "https://github.com/CrossGL/crosstl/issues/1204",
    "https://github.com/CrossGL/crosstl/issues/1206",
    "https://github.com/CrossGL/crosstl/issues/1205",
    "https://github.com/CrossGL/crosstl/issues/1207",
    "https://github.com/CrossGL/crosstl/issues/1218",
    "https://github.com/CrossGL/crosstl/issues/1222",
    "https://github.com/CrossGL/crosstl/issues/1238",
    "https://github.com/CrossGL/crosstl/issues/1239",
    "https://github.com/CrossGL/crosstl/issues/1240",
    "https://github.com/CrossGL/crosstl/issues/1246",
    "https://github.com/CrossGL/crosstl/issues/1248",
    "https://github.com/CrossGL/crosstl/issues/1249",
    "https://github.com/CrossGL/crosstl/issues/1250",
    "https://github.com/CrossGL/crosstl/issues/1259",
    "https://github.com/CrossGL/crosstl/issues/1260",
    "https://github.com/CrossGL/crosstl/issues/1261",
    "https://github.com/CrossGL/crosstl/issues/1274",
    "https://github.com/CrossGL/crosstl/issues/1287",
    "https://github.com/CrossGL/crosstl/issues/1329",
    "https://github.com/CrossGL/crosstl/issues/1338",
    "https://github.com/CrossGL/crosstl/issues/1340",
    "https://github.com/CrossGL/crosstl/issues/1346",
    "https://github.com/CrossGL/crosstl/issues/1355",
)


class PortingCheckError(RuntimeError):
    """Raised when an MLX project-porting check fails."""


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PortingCheckError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PortingCheckError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PortingCheckError(message)


def _resolve_work_dir(mlx_root: Path, work_dir: str | None) -> Path:
    if work_dir:
        candidate = Path(work_dir)
        if not candidate.is_absolute():
            candidate = mlx_root / candidate
    else:
        candidate = mlx_root / ".crosstl-mlx-porting"
    resolved = candidate.resolve()
    root = mlx_root.resolve()
    _require(
        _is_relative_to(resolved, root) and resolved != root,
        f"work directory must be inside the MLX checkout: {resolved}",
    )
    return resolved


def _run_command(
    name: str,
    command: Sequence[str],
    *,
    log_dir: Path,
    check: bool = True,
    timeout_seconds: int | None = None,
) -> CommandResult:
    stdout_path = log_dir / f"{name}.stdout"
    stderr_path = log_dir / f"{name}.stderr"
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = stderr + f"\n{name} timed out after {timeout_seconds} seconds.\n"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result = CommandResult(
        name=name,
        command=list(command),
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if check and returncode != 0:
        raise PortingCheckError(
            "{} failed with exit code {}. See {} and {}.".format(
                name,
                returncode,
                stdout_path,
                stderr_path,
            )
        )
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_native_metal_toolchain(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {
            "status": "not-applicable",
            "platform": sys.platform,
            "reason": "native Metal validation requires macOS",
        }

    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return {
            "status": "toolchain-unavailable",
            "platform": sys.platform,
            "reason": "xcrun is not installed",
        }

    native_dir = work_dir / "native-metal"
    native_dir.mkdir(parents=True, exist_ok=True)
    source_path = native_dir / "toolchain-probe.metal"
    output_path = native_dir / "toolchain-probe.air"
    output_path.unlink(missing_ok=True)
    source_path.write_text(
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void crosstl_mlx_probe() {}\n",
        encoding="utf-8",
    )
    result = _run_command(
        "probe-native-metal-toolchain",
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    available = result.returncode == 0 and output_path.is_file()
    return {
        "status": "available" if available else "toolchain-unavailable",
        "platform": sys.platform,
        "xcrun": xcrun,
        "probeSource": _relpath(source_path, mlx_root),
        "probeArtifact": (
            _relpath(output_path, mlx_root) if output_path.is_file() else None
        ),
        "returncode": result.returncode,
        "stdout": _relpath(result.stdout_path, mlx_root),
        "stderr": _relpath(result.stderr_path, mlx_root),
        "reason": (
            None
            if available
            else "a minimal Metal compilation did not produce an AIR artifact"
        ),
    }


def _validate_native_metal_artifact(
    *,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    artifact_path: Path,
    required: bool,
) -> dict[str, Any]:
    probe = _probe_native_metal_toolchain(mlx_root, work_dir, log_dir)
    if probe["status"] != "available":
        _require(
            not required,
            "native Metal validation was required, but a usable macOS Metal "
            f"toolchain was not available ({probe['reason']})",
        )
        return {
            **probe,
            "required": required,
            "artifactCompiled": False,
        }

    output_path = work_dir / "native-metal" / "mlx-fence-roundtrip.air"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    result = _run_command(
        "validate-metal-roundtrip-native",
        [
            str(probe["xcrun"]),
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(artifact_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        result.returncode == 0,
        "native Metal validation failed for the generated MLX fence round-trip "
        f"artifact; see {result.stdout_path} and {result.stderr_path}",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "native Metal validation did not produce the expected AIR artifact",
    )
    return {
        **probe,
        "status": "validated",
        "required": required,
        "artifactCompiled": True,
        "sourceArtifact": _relpath(artifact_path, mlx_root),
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compileStdout": _relpath(result.stdout_path, mlx_root),
        "compileStderr": _relpath(result.stderr_path, mlx_root),
    }


def _write_project_config(
    path: Path,
    *,
    include: str | Sequence[str],
    targets: Sequence[str],
    output_dir: str,
    specialization_constants: Mapping[str, bool | int | float] | None = None,
    metal_source_options: Mapping[str, int] | None = None,
    metal_target_options: Mapping[str, Mapping[str, int]] | None = None,
    entry_points: Mapping[str, str] | None = None,
    workgroup_size_rules: Mapping[str, Sequence[str | int]] | None = None,
) -> None:
    include_values = [include] if isinstance(include, str) else list(include)
    include_list = ", ".join(json.dumps(value) for value in include_values)
    target_list = ", ".join(json.dumps(target) for target in targets)
    lines = [
        "[project]",
        f'source_roots = ["{MLX_METAL_KERNEL_ROOT}"]',
        f"include = [{include_list}]",
        'include_dirs = ["."]',
        f"targets = [{target_list}]",
        f'output_dir = "{output_dir}"',
        "",
        "[project.sources]",
        '"**/*.metal" = "metal"',
        "",
    ]
    if entry_points:
        lines.append("[project.entry_points]")
        for source, entry_point in entry_points.items():
            lines.append(f"{json.dumps(source)} = {json.dumps(entry_point)}")
        lines.append("")
    if specialization_constants:
        lines.append("[project.specialization_constants]")
        for selector, value in specialization_constants.items():
            lines.append(f"{json.dumps(str(selector))} = {json.dumps(value)}")
        lines.append("")
    if workgroup_size_rules:
        lines.append("[project.workgroup_size_rules]")
        for source, components in workgroup_size_rules.items():
            rule = ", ".join(json.dumps(component) for component in components)
            lines.append(f"{json.dumps(source)} = [{rule}]")
        lines.append("")
    if metal_source_options or metal_target_options:
        lines.append("[project.source_options.metal]")
        for key, value in (metal_source_options or {}).items():
            lines.append(f"{key} = {value}")
        lines.append("")
        for target, options in (metal_target_options or {}).items():
            lines.append(f"[project.source_options.metal.target_options.{target}]")
            for key, value in options.items():
                lines.append(f"{key} = {value}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reference_accessor_project_config(path: Path, output_dir: str) -> None:
    target_list = ", ".join(json.dumps(target) for target in REFERENCE_ACCESSOR_TARGETS)
    lines = [
        "[project]",
        'source_roots = ["."]',
        f'include = ["{REFERENCE_ACCESSOR_FIXTURE_NAME}"]',
        f"targets = [{target_list}]",
        f'output_dir = "{output_dir}"',
        "",
        "[project.sources]",
        '"*.metal" = "metal"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_template_member_pointer_project_config(path: Path, output_dir: str) -> None:
    target_list = ", ".join(
        json.dumps(target) for target in TEMPLATE_MEMBER_POINTER_TARGETS
    )
    lines = [
        "[project]",
        'source_roots = ["."]',
        f'include = ["{TEMPLATE_MEMBER_POINTER_FIXTURE_NAME}"]',
        f"targets = [{target_list}]",
        f'output_dir = "{output_dir}"',
        "",
        "[project.sources]",
        '"*.metal" = "metal"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _verify_mlx_checkout(mlx_root: Path, python: str, log_dir: Path) -> dict[str, Any]:
    _require(mlx_root.is_dir(), f"MLX checkout does not exist: {mlx_root}")
    _require(
        (mlx_root / MLX_ARANGE_SOURCE).is_file(),
        f"MLX Metal frontier source is missing: {MLX_ARANGE_SOURCE}",
    )
    for source in MLX_REDUCED_FRONTIER_SOURCES:
        _require(
            (mlx_root / source).is_file(),
            f"MLX Metal frontier source is missing: {source}",
        )
    _require(
        (mlx_root / MLX_GEMV_SOURCE).is_file(),
        f"MLX GEMV source is missing: {MLX_GEMV_SOURCE}",
    )
    result = _run_command(
        "mlx-revision",
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"],
        log_dir=log_dir,
    )
    revision = result.stdout_path.read_text(encoding="utf-8").strip()
    _require(
        revision == MLX_COMMIT,
        f"MLX checkout must be pinned to {MLX_COMMIT}; found {revision}",
    )
    return {
        "name": "mlx-checkout",
        "status": "passed",
        "repository": MLX_REPOSITORY,
        "commit": revision,
        "python": python,
    }


def _scan_metal_kernels(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "scan-metal-kernels.toml"
    report_path = report_dir / "scan-metal-kernels.json"
    _write_project_config(
        config_path,
        include=f"{MLX_METAL_KERNEL_ROOT}/**/*.metal",
        targets=("directx", "opengl", "vulkan"),
        output_dir=_relpath(work_dir / "out-scan", mlx_root),
    )
    _run_command(
        "scan-metal-kernels",
        [
            python,
            "-m",
            "crosstl",
            "scan",
            str(mlx_root),
            "--config",
            str(config_path),
            "--output",
            str(report_path),
        ],
        log_dir=log_dir,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    units = payload.get("units", [])
    _require(isinstance(summary, dict), "scan report summary must be an object")
    _require(isinstance(units, list), "scan report units must be a list")
    _require(
        summary.get("unitCount") == EXPECTED_METAL_KERNEL_COUNT,
        "expected {} MLX Metal kernels, found {}".format(
            EXPECTED_METAL_KERNEL_COUNT,
            summary.get("unitCount"),
        ),
    )
    _require(
        summary.get("diagnosticCounts", {}).get("error", 0) == 0,
        "MLX Metal scan reported errors",
    )
    unit_paths = {unit.get("path") for unit in units if isinstance(unit, dict)}
    for source in MLX_REDUCED_FRONTIER_SOURCES:
        _require(source in unit_paths, f"{source} was not scanned")
    return {
        "name": "metal-kernel-scan",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "unitCount": summary.get("unitCount"),
        "includeDependencyCount": summary.get("includeDependencyCount"),
        "targets": ["directx", "opengl", "vulkan"],
    }


def _check_metal_roundtrip(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_metal_toolchain: bool,
) -> dict[str, Any]:
    config_path = config_dir / "metal-roundtrip.toml"
    report_path = report_dir / "metal-roundtrip.json"
    output_dir = work_dir / "out-metal-roundtrip"
    _write_project_config(
        config_path,
        include=MLX_METAL_ROUNDTRIP_SOURCE,
        targets=("metal",),
        output_dir=_relpath(output_dir, mlx_root),
    )
    _run_command(
        "translate-metal-roundtrip",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--validate",
        ],
        log_dir=log_dir,
    )

    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, dict), "Metal round-trip summary must be an object")
    _require(
        summary.get("unitCount") == 1,
        "Metal round-trip translation must scan exactly one pinned MLX source",
    )
    _require(
        summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0,
        "Metal round-trip translation must emit one clean artifact",
    )
    diagnostic_counts = summary.get("diagnosticCounts", {})
    diagnostics = payload.get("diagnostics", [])
    _require(
        isinstance(diagnostics, list), "Metal round-trip diagnostics must be a list"
    )
    allowed_unavailable = [
        diagnostic
        for diagnostic in diagnostics
        if isinstance(diagnostic, dict)
        and diagnostic.get("severity") == "warning"
        and diagnostic.get("code") == "project.validate.toolchain-unavailable"
        and diagnostic.get("target") == "metal"
        and not require_metal_toolchain
    ]
    unexpected_diagnostics = [
        diagnostic
        for diagnostic in diagnostics
        if isinstance(diagnostic, dict)
        and diagnostic.get("severity") in {"error", "warning"}
        and diagnostic not in allowed_unavailable
    ]
    _require(
        isinstance(diagnostic_counts, dict)
        and diagnostic_counts.get("error", 0) == 0
        and diagnostic_counts.get("warning", 0) == len(allowed_unavailable)
        and not unexpected_diagnostics,
        "Metal round-trip translation reported unexpected diagnostics",
    )

    artifacts = payload.get("artifacts", [])
    _require(isinstance(artifacts, list), "Metal round-trip artifacts must be a list")
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and item.get("source") == MLX_METAL_ROUNDTRIP_SOURCE
            and item.get("sourceBackend") == "metal"
            and item.get("target") == "metal"
        ),
        None,
    )
    _require(isinstance(artifact, dict), "Metal round-trip artifact is missing")
    _require(
        artifact.get("status") == "translated",
        "Metal round-trip artifact was not translated",
    )
    expected_path = output_dir / "metal" / MLX_METAL_ROUNDTRIP_SOURCE
    expected_report_path = _relpath(expected_path, mlx_root)
    _require(
        artifact.get("path") == expected_report_path,
        "Metal round-trip artifact path does not match the bounded project output",
    )
    _require(
        artifact.get("provenance")
        == {
            "pipeline": "single-file-translate",
            "intermediate": "crossgl",
        },
        "Metal round-trip artifact did not traverse the CrossGL project pipeline",
    )
    _require(
        expected_path.is_file() and expected_path.stat().st_size > 0,
        f"Metal round-trip artifact is missing or empty: {expected_report_path}",
    )

    source_path = mlx_root / MLX_METAL_ROUNDTRIP_SOURCE
    source_hash = artifact.get("sourceHash", {})
    generated_hash = artifact.get("generatedHash", {})
    _require(
        source_hash.get("algorithm") == "sha256"
        and source_hash.get("value") == _sha256(source_path),
        "Metal round-trip source hash does not match the pinned MLX source",
    )
    _require(
        generated_hash.get("algorithm") == "sha256"
        and generated_hash.get("value") == _sha256(expected_path),
        "Metal round-trip generated hash does not match the emitted artifact",
    )
    _require(
        artifact.get("sourceSizeBytes") == source_path.stat().st_size
        and artifact.get("generatedSizeBytes") == expected_path.stat().st_size,
        "Metal round-trip source or generated byte-size metadata is inconsistent",
    )
    _require(
        source_hash.get("value") != generated_hash.get("value"),
        "Metal round-trip unexpectedly copied the source without translation",
    )

    generated = expected_path.read_text(encoding="utf-8")
    for fragment in (
        "#include <metal_stdlib>",
        "kernel void input_coherent",
        "kernel void fence_update",
        "kernel void fence_wait",
        "[[buffer(0)]]",
        "[[thread_position_in_grid]]",
        "metal::mem_flags::mem_device",
        "metal::memory_order_seq_cst",
        "metal::thread_scope_system",
    ):
        _require(
            fragment in generated,
            f"Metal round-trip artifact is missing expected generated form: {fragment}",
        )
    _require(
        generated.count("metal::atomic_thread_fence(")
        == MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT,
        "Metal round-trip artifact did not preserve every source atomic fence",
    )
    for forbidden in ("threadgroup_barrier(", "unsupported", "fallback"):
        _require(
            forbidden not in generated,
            f"Metal round-trip artifact contains forbidden generated form: {forbidden}",
        )

    validation = payload.get("validation", {})
    _require(isinstance(validation, dict), "Metal round-trip validation is missing")
    validation_summary = validation.get("summary", {})
    _require(
        isinstance(validation_summary, dict)
        and validation_summary.get("artifactCount") == 1
        and validation_summary.get("okCount") == 1
        and validation_summary.get("failedCount") == 0,
        "Metal round-trip generated-artifact validation was not clean",
    )
    validation_artifact = next(
        (
            item
            for item in validation.get("artifacts", [])
            if isinstance(item, dict) and item.get("path") == expected_report_path
        ),
        None,
    )
    _require(
        isinstance(validation_artifact, dict)
        and validation_artifact.get("status") == "ok"
        and validation_artifact.get("sourceHashStatus") == "ok"
        and validation_artifact.get("generatedHashStatus") == "ok"
        and validation_artifact.get("sourceSizeStatus") == "ok"
        and validation_artifact.get("generatedSizeStatus") == "ok"
        and validation_artifact.get("sourceMapStatus") == "ok"
        and validation_artifact.get("sourceRemapStatus") == "ok",
        "Metal round-trip artifact metadata did not pass project validation",
    )

    native_validation = _validate_native_metal_artifact(
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        artifact_path=expected_path,
        required=require_metal_toolchain,
    )
    return {
        "name": "metal-roundtrip",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_METAL_ROUNDTRIP_SOURCE,
        "target": "metal",
        "roundTripStages": ["metal", "crossgl", "metal"],
        "unitCount": 1,
        "artifactCount": 1,
        "artifact": expected_report_path,
        "generatedHash": generated_hash,
        "generatedSizeBytes": artifact.get("generatedSizeBytes"),
        "diagnosticCounts": diagnostic_counts,
        "artifactValidationStatus": "validated",
        "nativeMetalValidation": native_validation,
        "fenceContract": {
            "memoryFlags": ["mem_device"],
            "memoryOrder": "memory_order_seq_cst",
            "threadScope": "thread_scope_system",
            "occurrences": MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT,
            "preserved": True,
        },
        "semanticReadinessStatus": "blocked",
        "semanticTrackedIssues": list(METAL_ROUNDTRIP_SEMANTIC_TRACKED_ISSUES),
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
    }


def _atomic_fence_expected_message(target_contract: Mapping[str, str]) -> str:
    memory_flags = " | ".join(MLX_FENCE_REQUESTED_CONTRACT["memoryFlags"])
    return (
        "Cannot lower CrossGL atomicThreadFence to "
        f"{target_contract['targetDescription']} without changing its semantics "
        f"(flags={memory_flags}, "
        f"order={MLX_FENCE_REQUESTED_CONTRACT['memoryOrder']}, "
        f"scope={MLX_FENCE_REQUESTED_CONTRACT['threadScope']}): "
        "unsupported system thread scope"
    )


def _validate_atomic_fence_contract_report(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    exact_report: bool,
) -> dict[str, dict[str, Any]]:
    targets = tuple(MLX_FENCE_TARGET_CONTRACTS)
    summary = payload.get("summary", {})
    _require(isinstance(summary, Mapping), "fence contract summary must be an object")
    expected_diagnostic_codes = {
        contract["diagnosticCode"]: 1
        for contract in MLX_FENCE_TARGET_CONTRACTS.values()
    }
    expected_missing_capabilities = {
        contract["missingCapability"]: 1
        for contract in MLX_FENCE_TARGET_CONTRACTS.values()
    }
    diagnostics_by_code = summary.get("diagnosticsByCode", {})
    missing_capability_counts = summary.get("missingCapabilityCounts", {})
    _require(
        isinstance(diagnostics_by_code, Mapping),
        "fence contract diagnostic code counts must be an object",
    )
    _require(
        isinstance(missing_capability_counts, Mapping),
        "fence contract missing capability counts must be an object",
    )
    if exact_report:
        _require(
            summary.get("unitCount") == 1
            and summary.get("artifactCount") == len(targets)
            and summary.get("translatedCount") == 0
            and summary.get("failedCount") == len(targets),
            "fence contract translation must report one failed artifact per target",
        )
        _require(
            summary.get("diagnosticCounts")
            == {"error": len(targets), "note": 0, "warning": 0},
            "fence contract translation reported unexpected diagnostic severities",
        )
        _require(
            diagnostics_by_code == expected_diagnostic_codes,
            "fence contract translation diagnostic codes changed",
        )
        _require(
            missing_capability_counts == expected_missing_capabilities,
            "fence contract translation missing capabilities changed",
        )
    else:
        _require(
            all(
                diagnostics_by_code.get(code) == count
                for code, count in expected_diagnostic_codes.items()
            ),
            "full-corpus fence contract diagnostic codes changed",
        )
        _require(
            all(
                missing_capability_counts.get(capability) == count
                for capability, count in expected_missing_capabilities.items()
            ),
            "full-corpus fence contract missing capabilities changed",
        )

    diagnostics = payload.get("diagnostics", [])
    artifacts = payload.get("artifacts", [])
    _require(isinstance(diagnostics, list), "fence contract diagnostics must be a list")
    _require(isinstance(artifacts, list), "fence contract artifacts must be a list")
    if exact_report:
        _require(
            len(diagnostics) == len(targets) and len(artifacts) == len(targets),
            "fence contract report must contain one diagnostic and artifact per target",
        )

    target_results: dict[str, dict[str, Any]] = {}
    output_root = output_dir.resolve()
    for target, contract in MLX_FENCE_TARGET_CONTRACTS.items():
        target_diagnostics = [
            diagnostic
            for diagnostic in diagnostics
            if isinstance(diagnostic, Mapping)
            and diagnostic.get("target") == target
            and str(diagnostic.get("message", "")).startswith(
                "Cannot lower CrossGL atomicThreadFence"
            )
        ]
        _require(
            len(target_diagnostics) == 1,
            f"fence contract report must contain one {target} diagnostic",
        )
        diagnostic = target_diagnostics[0]
        expected_message = _atomic_fence_expected_message(contract)
        _require(
            {
                "severity": diagnostic.get("severity"),
                "code": diagnostic.get("code"),
                "message": diagnostic.get("message"),
                "target": diagnostic.get("target"),
                "sourceBackend": diagnostic.get("sourceBackend"),
                "missingCapabilities": diagnostic.get("missingCapabilities"),
            }
            == {
                "severity": "error",
                "code": contract["diagnosticCode"],
                "message": expected_message,
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": [contract["missingCapability"]],
            },
            f"fence contract {target} structured diagnostic changed",
        )
        location = diagnostic.get("location", {})
        _require(
            isinstance(location, Mapping) and location.get("file") == MLX_FENCE_SOURCE,
            f"fence contract {target} diagnostic source changed",
        )
        if exact_report:
            target_summary = summary.get("artifactsByTarget", {}).get(target, {})
            _require(
                target_summary.get("artifactCount") == 1
                and target_summary.get("translatedCount") == 0
                and target_summary.get("failedCount") == 1,
                f"fence contract {target} artifact summary changed",
            )

        target_artifacts = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, Mapping)
            and artifact.get("source") == MLX_FENCE_SOURCE
            and artifact.get("target") == target
        ]
        _require(
            len(target_artifacts) == 1,
            f"fence contract report must contain one {target} artifact record",
        )
        artifact = target_artifacts[0]
        _require(
            artifact.get("sourceBackend") == "metal"
            and artifact.get("status") == "failed"
            and artifact.get("error") == expected_message,
            f"fence contract {target} failed artifact record changed",
        )
        artifact_path = artifact.get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"fence contract {target} artifact path is missing",
        )
        generated_path = (mlx_root / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_root),
            f"fence contract {target} artifact path escaped its output directory",
        )
        _require(
            not generated_path.exists(),
            f"fence contract {target} unexpectedly emitted {artifact_path}",
        )
        _require(
            "generatedHash" not in artifact and "generatedSizeBytes" not in artifact,
            f"fence contract {target} recorded generated artifact metadata",
        )
        target_results[target] = {
            "diagnosticCode": contract["diagnosticCode"],
            "missingCapability": contract["missingCapability"],
            "requestedContract": dict(MLX_FENCE_REQUESTED_CONTRACT),
            "artifactStatus": "failed",
            "artifactEmitted": False,
        }
    return target_results


def _check_atomic_fence_contract(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "fence-contract.toml"
    report_path = report_dir / "fence-contract.json"
    output_dir = work_dir / "out-fence-contract"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    targets = tuple(MLX_FENCE_TARGET_CONTRACTS)
    _write_project_config(
        config_path,
        include=MLX_FENCE_SOURCE,
        targets=targets,
        output_dir=_relpath(output_dir, mlx_root),
    )
    result = _run_command(
        "translate-fence-contract",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        result.returncode == 1,
        "atomic fence contract translation must fail with exit code 1",
    )

    payload = _load_json(report_path)
    target_results = _validate_atomic_fence_contract_report(
        mlx_root,
        output_dir,
        payload,
        exact_report=True,
    )

    return {
        "name": "atomic-fence-contract",
        "status": "blocked-as-expected",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_FENCE_SOURCE,
        "targets": list(targets),
        "artifactRecordCount": len(targets),
        "failedArtifactCount": len(targets),
        "emittedArtifactCount": 0,
        "requestedContract": dict(MLX_FENCE_REQUESTED_CONTRACT),
        "targetContracts": target_results,
        "semanticReadinessStatus": "blocked",
        "semanticTrackedIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
    }


def _strip_shader_comments(source: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)


def _shader_function_definition(
    source: str,
    function_pattern: str,
    *,
    return_pattern: str = "void",
) -> tuple[re.Match[str], str] | None:
    header = re.search(
        rf"\b(?:{return_pattern})\s+(?P<helper>{function_pattern})\s*"
        rf"\((?P<parameters>[^)]*)\)\s*\{{",
        source,
        flags=re.DOTALL,
    )
    if header is None:
        return None

    depth = 1
    body_start = header.end()
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return header, source[body_start:index]
    return None


def _reference_accessor_write_evidence(
    generated: str,
    *,
    target: str,
) -> dict[str, Any]:
    source = _strip_shader_comments(generated)
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}.get(target, target)
    helper_write = re.search(
        r"\b(?:frag_at[A-Za-z_0-9]*|[A-Za-z_]\w*frag_at[A-Za-z_0-9]*)"
        r"\s*\([^;{}]*\)\s*=(?!=)",
        source,
    )
    _require(
        helper_write is None,
        f"{target_name} reference accessor write still targets the accessor "
        "or a value-return helper",
    )

    sentinel = rf"{re.escape(REFERENCE_ACCESSOR_SENTINEL)}0*[fF]?"
    written_value = (
        rf"(?:{sentinel}|float\s*\(\s*{sentinel}\s*\)|"
        rf"\(\s*float\s*\)\s*{sentinel})"
    )
    write = re.search(
        rf"(?P<storage>(?P<owner>\b[A-Za-z_]\w*"
        rf"(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\.\s*val_frags\s*"
        rf"\[[^\]\n;]+\])\s*=(?!=)\s*{written_value}\s*;",
        source,
    )
    _require(
        write is not None,
        f"{target_name} reference accessor did not write the sentinel directly "
        "to original val_frags storage",
    )

    owner = re.sub(r"\s+", "", write.group("owner"))
    _require(
        owner == "tile",
        f"{target_name} reference accessor wrote a copied receiver instead of "
        "the original tile storage",
    )
    owner_parts = re.findall(r"[A-Za-z_]\w*", write.group("owner"))
    owner_pattern = r"\s*\.\s*".join(re.escape(part) for part in owner_parts)
    storage_lvalue = re.sub(r"\s+", "", write.group("storage"))
    readback_lvalues = [
        re.sub(r"\s+", "", match.group("storage"))
        for match in re.finditer(
            rf"(?<![=!<>])=(?!=)\s*(?P<storage>\b{owner_pattern}\s*\.\s*"
            rf"val_frags\s*\[[^\]\n;]+\])\s*;",
            source[write.end() :],
        )
    ]
    _require(
        storage_lvalue in readback_lvalues,
        f"{target_name} reference accessor fixture did not read back the exact "
        "val_frags lvalue written through the accessor",
    )

    return {
        "status": "verified-original-storage-write",
        "storageMember": "val_frags",
        "storageLvalue": storage_lvalue,
        "sentinel": REFERENCE_ACCESSOR_SENTINEL,
        "readBackFromSameStorage": True,
        "readBackFromWrittenLvalue": True,
        "readBackLvalue": storage_lvalue,
        "valueReturningHelperUsedForWrite": False,
    }


def _reference_accessor_const_read_evidence(
    generated: str,
    *,
    target: str,
) -> dict[str, Any]:
    source = _strip_shader_comments(generated)
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}.get(target, target)
    accessor_helper = re.search(
        r"\b(?:frag_at[A-Za-z_0-9]*|[A-Za-z_]\w*frag_at[A-Za-z_0-9]*)" r"\s*\(",
        source,
    )
    _require(
        accessor_helper is None,
        f"{target_name} reference accessor artifact still contains a frag_at "
        "helper or call",
    )

    helper_patterns = {
        "directx": (
            r"ReferenceAccessorTile__store",
            r"ReferenceAccessorOps__store",
        ),
        "opengl": (
            r"ReferenceAccessorTile_store[A-Za-z_0-9]*",
            r"ReferenceAccessorOps_store[A-Za-z_0-9]*",
        ),
    }
    _require(
        target in helper_patterns,
        f"unsupported reference accessor evidence target: {target}",
    )
    tile_helper_pattern, read_helper_pattern = helper_patterns[target]
    tile_store = re.search(
        rf"\bvoid\s+(?P<helper>{tile_helper_pattern})\s*"
        rf"\((?P<parameters>[^)]*)\)\s*\{{(?P<body>[^{{}}]*)\}}",
        source,
        flags=re.DOTALL,
    )
    _require(
        tile_store is not None,
        f"{target_name} reference accessor artifact is missing the lowered "
        "const store helper",
    )
    _require(
        re.search(
            r"\bReferenceAccessorTile\s+self\b",
            tile_store.group("parameters"),
        )
        is not None,
        f"{target_name} lowered const store helper is missing its tile receiver",
    )
    direct_read = re.search(
        rf"\b(?P<helper>{read_helper_pattern})\s*\(\s*"
        rf"(?P<storage>self\s*\.\s*val_frags\s*\[[^\]\n;]+\])\s*,",
        tile_store.group("body"),
    )
    _require(
        direct_read is not None,
        f"{target_name} implicit const accessor was not lowered to original "
        "self.val_frags storage passed directly to the read-only helper",
    )
    kernel_call = re.search(
        rf"\b{re.escape(tile_store.group('helper'))}\s*\(\s*tile\s*,",
        source,
    )
    _require(
        kernel_call is not None,
        f"{target_name} reference accessor kernel does not invoke the lowered "
        "const store path",
    )

    return {
        "status": "verified-original-storage-const-read",
        "storageMember": "val_frags",
        "storageLvalue": re.sub(r"\s+", "", direct_read.group("storage")),
        "implicitReceiver": "self",
        "passedDirectlyToHelper": True,
        "accessorCallEliminated": True,
        "kernelPathInvoked": True,
        "loweredTileHelper": tile_store.group("helper"),
        "loweredReadHelper": direct_read.group("helper"),
    }


def _reference_accessor_nested_const_alias_evidence(
    generated: str,
    *,
    target: str,
) -> dict[str, Any]:
    source = _strip_shader_comments(generated)
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}.get(target, target)
    accessor_helper = re.search(
        r"\b(?:frag_at[A-Za-z_0-9]*|[A-Za-z_]\w*frag_at[A-Za-z_0-9]*)" r"\s*\(",
        source,
    )
    _require(
        accessor_helper is None,
        f"{target_name} nested reference accessor artifact still contains a "
        "frag_at helper or call",
    )
    _require(
        re.search(r"\baccum\b", source) is None,
        f"{target_name} nested reference accessor artifact still contains the "
        "accum reference alias",
    )

    helper_patterns = {
        "directx": r"ReferenceAccessorStoreLoop__store",
        "opengl": r"ReferenceAccessorStoreLoop_+store[A-Za-z_0-9]*",
    }
    _require(
        target in helper_patterns,
        f"unsupported nested reference accessor evidence target: {target}",
    )
    store_definition = _shader_function_definition(
        source,
        helper_patterns[target],
    )
    _require(
        store_definition is not None,
        f"{target_name} nested reference accessor artifact is missing the "
        "lowered const store helper",
    )
    store_header, store_body = store_definition
    _require(
        re.search(
            r"\bReferenceAccessorStoreLoop\s+self\b",
            store_header.group("parameters"),
        )
        is not None,
        f"{target_name} lowered nested const store helper is missing its "
        "outer value receiver",
    )

    direct_read = re.search(
        r"(?<![=!<>])=(?!=)\s*"
        r"(?P<storage>self\s*\.\s*nestedTile\s*\.\s*val_frags\s*"
        r"\[[^\]\n;]+\]\s*\[\s*k\s*\])\s*;",
        store_body,
    )
    lane = r"(?:uint\s*\(\s*k\s*\)|k)"
    helper_read = None
    if direct_read is None:
        helper_read = re.search(
            r"\bCrossGLMetalVectorIndex_[A-Za-z0-9_]+_set\s*\(\s*stored\s*,\s*"
            rf"{lane}\s*,\s*"
            r"\bCrossGLMetalVectorIndex_[A-Za-z0-9_]+_get\s*\(\s*"
            r"(?P<storage>self\s*\.\s*nestedTile\s*\.\s*val_frags\s*"
            r"\[[^\]\n;]+\])\s*,\s*"
            rf"{lane}\s*\)\s*\)",
            store_body,
        )
    storage_read = direct_read or helper_read
    _require(
        storage_read is not None,
        f"{target_name} nested const reference alias was not eliminated to "
        "self.nestedTile.val_frags storage indexed by k",
    )
    kernel_call = re.search(
        rf"\b{re.escape(store_header.group('helper'))}\s*\(\s*nestedStore\s*,",
        source,
    )
    _require(
        kernel_call is not None,
        f"{target_name} reference accessor kernel does not invoke the lowered "
        "nested const store path",
    )

    storage_lvalue = storage_read.group("storage")
    component_read_lowering = "direct-index"
    if helper_read is not None:
        storage_lvalue = f"{storage_lvalue}[k]"
        component_read_lowering = "lane-helper"

    return {
        "status": "verified-original-nested-storage-const-alias-read",
        "storageMember": "val_frags",
        "storagePath": "self.nestedTile.val_frags",
        "storageLvalue": re.sub(r"\s+", "", storage_lvalue),
        "componentReadLowering": component_read_lowering,
        "outerReceiver": "self",
        "tileMember": "nestedTile",
        "fragmentType": "float2",
        "aliasName": "accum",
        "aliasEliminated": True,
        "accessorCallEliminated": True,
        "indexedAliasRead": "accum[k]",
        "readFromOriginalStorage": True,
        "kernelPathInvoked": True,
        "loweredStoreHelper": store_header.group("helper"),
    }


def _template_member_pointer_evidence(
    generated: str,
    *,
    target: str,
) -> dict[str, Any]:
    source = _strip_shader_comments(generated)
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}.get(target, target)
    helper_patterns = {
        "directx": r"ReducedMMAFrag__load__[A-Za-z_0-9]+",
        "opengl": r"ReducedMMAFrag_+load[A-Za-z_0-9]+",
    }
    outer_helper_patterns = {
        "directx": r"ReducedMMATile__load__[A-Za-z_0-9]+",
        "opengl": r"ReducedMMATile_+load[A-Za-z_0-9]+",
    }
    _require(
        target in helper_patterns,
        f"unsupported template member pointer evidence target: {target}",
    )

    unresolved_call = re.search(
        r"(?:\bReducedMMAFrag\s*::\s*(?:template\s+)?load\b|"
        r"\b[A-Za-z_]\w*\s*(?:\.|->)\s*(?:template\s+)?load\b)",
        source,
    )
    _require(
        unresolved_call is None,
        f"{target_name} template member pointer artifact retained an unresolved "
        "source member call",
    )

    helper_definition = _shader_function_definition(
        source,
        helper_patterns[target],
        return_pattern="float",
    )
    _require(
        helper_definition is not None,
        f"{target_name} template member pointer artifact is missing the "
        "materialized fragment load helper",
    )
    helper_header, helper_body = helper_definition
    helper_name = helper_header.group("helper")
    helper_parameters = helper_header.group("parameters")
    scalarized_parameter = re.search(
        r"(?:^|,)\s*(?:(?:in|out|inout|const)\s+)*float\s+src\b",
        helper_parameters,
    )
    _require(
        scalarized_parameter is None,
        f"{target_name} materialized fragment load helper has a scalarized "
        "float src parameter instead of a pointer-backed source view",
    )

    return_statement = re.search(
        r"\breturn\s+(?P<expression>[^;]+);",
        helper_body,
        flags=re.DOTALL,
    )
    _require(
        return_statement is not None,
        f"{target_name} materialized fragment load helper is missing its "
        "source read return",
    )
    return_expression = return_statement.group("expression")
    indexed_read = re.search(
        r"(?P<read>\bsrc\s*\[\s*(?P<index>[^\]\n;]+)\s*\])",
        return_expression,
    )
    if indexed_read is None:
        indexed_read = re.search(
            r"(?P<read>\bsrc\s*\.\s*Load\s*\(\s*(?P<index>[^)\n;]+)\s*\))",
            return_expression,
        )
    _require(
        indexed_read is not None,
        f"{target_name} materialized fragment load helper does not perform "
        "an indexed read from its source view",
    )
    read_index = re.sub(r"\s+", "", indexed_read.group("index"))
    _require(
        re.search(r"\bstride\b", indexed_read.group("index")) is not None,
        f"{target_name} materialized fragment load helper lost the stride in "
        "its indexed source read",
    )

    outer_definition = _shader_function_definition(
        source,
        outer_helper_patterns[target],
    )
    _require(
        outer_definition is not None,
        f"{target_name} template member pointer artifact is missing the "
        "materialized outer tile load helper",
    )
    outer_header, outer_body = outer_definition
    outer_name = outer_header.group("helper")
    outer_parameters = outer_header.group("parameters")
    helper_call = re.search(
        rf"\bself\s*\.\s*value\s*=(?!=)\s*"
        rf"(?P<call>{re.escape(helper_name)}\s*\((?P<arguments>[^;]+)\))\s*;",
        outer_body,
        flags=re.DOTALL,
    )
    _require(
        helper_call is not None,
        f"{target_name} outer tile helper does not assign the materialized "
        "fragment load result to self.value",
    )
    helper_call_arguments = helper_call.group("arguments")

    if target == "directx":
        helper_resource = re.search(
            r"(?P<type>StructuredBuffer\s*<\s*float\s*>)\s+src\b",
            helper_parameters,
        )
        _require(
            helper_resource is not None,
            "DirectX materialized fragment load helper must retain src as a "
            "StructuredBuffer<float> parameter",
        )
        outer_resource = re.search(
            r"StructuredBuffer\s*<\s*float\s*>\s+src\b",
            outer_parameters,
        )
        _require(
            outer_resource is not None,
            "DirectX outer tile helper must retain src as a "
            "StructuredBuffer<float> parameter",
        )
        offset_type = r"u?int(?:32_t|64_t)?"
        offset_parameter = re.search(
            rf"\b{offset_type}\s+src_offset\b", helper_parameters
        )
        outer_offset_parameter = re.search(
            rf"\b{offset_type}\s+src_offset\b", outer_parameters
        )
        addressed_argument = re.search(
            r"(?P<index>&\s*\(?\s*src\s*\[\s*index\s*\]\s*\)?)",
            helper_call_arguments,
        )
        expected_offset = (
            r"src_offset\s*\+\s*index|index\s*\+\s*src_offset"
            if outer_offset_parameter is not None
            else r"index"
        )
        resource_offset_argument = re.search(
            rf"\bsrc\s*,\s*(?:{offset_type}\s*\(\s*)?\(*\s*"
            rf"(?P<offset>{expected_offset})\s*\)*",
            helper_call_arguments,
        )
        _require(
            addressed_argument is not None
            or (offset_parameter is not None and resource_offset_argument is not None),
            "DirectX materialized fragment call does not preserve the addressed "
            "src[index] source position",
        )
        if offset_parameter is not None:
            _require(
                "src_offset" in read_index,
                "DirectX offset-backed fragment helper does not apply "
                "src_offset in its indexed source read",
            )
        source_view = {
            "representation": (
                "structured-buffer-plus-offset"
                if offset_parameter is not None
                else "structured-buffer-addressed-element"
            ),
            "resourceName": "src",
            "parameterName": "src",
            "parameterType": re.sub(r"\s+", "", helper_resource.group("type")),
            "offsetParameter": "src_offset" if offset_parameter is not None else None,
            "scalarizedPointerParameter": False,
        }
        source_index_expression = re.sub(
            r"\s+",
            "",
            (
                addressed_argument.group("index")
                if addressed_argument is not None
                else resource_offset_argument.group("offset")
            ),
        )
    else:
        global_source = re.search(
            r"\blayout\s*\([^)]*\)\s*readonly\s+buffer\s+\w+\s*"
            r"\{\s*float\s+src\s*\[\s*\]\s*;\s*\}\s*;",
            source,
            flags=re.DOTALL,
        )
        _require(
            global_source is not None,
            "OpenGL template member pointer artifact is missing the readonly "
            "float src storage buffer",
        )
        _require(
            re.search(r"\bint\s+src_offset\b", helper_parameters) is not None,
            "OpenGL materialized fragment load helper must retain an int "
            "src_offset buffer-view parameter",
        )
        _require(
            re.search(r"\bint\s+src_offset\b", outer_parameters) is not None,
            "OpenGL outer tile helper must retain an int src_offset "
            "buffer-view parameter",
        )
        _require(
            "src_offset" in read_index,
            "OpenGL materialized fragment load helper does not apply "
            "src_offset in its indexed source read",
        )
        source_index = re.search(
            r"(?P<index>src_offset\s*\+\s*index|index\s*\+\s*src_offset)",
            helper_call_arguments,
        )
        _require(
            source_index is not None,
            "OpenGL materialized fragment call does not preserve index in the "
            "src_offset buffer view",
        )
        source_view = {
            "representation": "global-storage-buffer-plus-offset",
            "resourceName": "src",
            "parameterName": "src_offset",
            "parameterType": "int",
            "offsetParameter": "src_offset",
            "scalarizedPointerParameter": False,
        }
        source_index_expression = re.sub(r"\s+", "", source_index.group("index"))

    kernel_call = re.search(
        rf"\b{re.escape(outer_name)}\s*\(\s*tile\s*,"
        rf"(?P<arguments>[^;]*\bgid\b[^;]*)\)\s*;",
        source,
        flags=re.DOTALL,
    )
    _require(
        kernel_call is not None,
        f"{target_name} kernel does not invoke the materialized outer tile "
        "load path with gid",
    )
    if target == "directx":
        _require(
            re.search(r"\bsrc\b", kernel_call.group("arguments")) is not None,
            "DirectX kernel does not pass the source buffer to the outer tile "
            "load helper",
        )
    output_data_flow = re.search(
        r"(?P<write>\bout_?\s*\[\s*gid\s*\]\s*=(?!=)\s*tile\s*\.\s*value\s*;)",
        source,
    )
    _require(
        output_data_flow is not None,
        f"{target_name} kernel does not write the loaded tile value to out[gid]",
    )

    return {
        "status": "verified-materialized-pointer-indexed-read",
        "materializedHelper": helper_name,
        "materializedOuterHelper": outer_name,
        "helperParameters": re.sub(r"\s+", " ", helper_parameters).strip(),
        "sourceView": source_view,
        "sourceIndexExpression": source_index_expression,
        "indexedReadExpression": re.sub(r"\s+", "", indexed_read.group("read")),
        "indexedReadIndex": read_index,
        "materializedHelperCall": re.sub(r"\s+", "", helper_call.group("call")),
        "outputDataFlowExpression": re.sub(r"\s+", "", output_data_flow.group("write")),
        "sourceIndexPreserved": True,
        "sourceViewParameterRetained": True,
        "indexedReadFromSourceView": True,
        "scalarizedPointerParameter": False,
        "unresolvedSourceCallRetained": False,
        "kernelPathInvoked": True,
    }


def _validate_reference_accessor_directx(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    artifact_path: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "status": "not-required",
            "required": False,
            "nativeCompiler": "dxc",
        }

    dxc = shutil.which("dxc")
    _require(dxc is not None, "reference accessor DirectX validation requires dxc")
    output_path = work_dir / "validation" / "reference-accessor.dxil"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    result = _run_command(
        "validate-reference-accessor-directx",
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            REFERENCE_ACCESSOR_DXC_ENTRY_POINT,
            str(artifact_path),
            "-Fo",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        result.returncode == 0,
        "reference accessor DirectX compilation failed; inspect "
        "validate-reference-accessor-directx logs",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "reference accessor DirectX compilation did not produce DXIL",
    )
    return {
        "status": "validated",
        "required": True,
        "nativeCompiler": "dxc",
        "entryPoint": REFERENCE_ACCESSOR_DXC_ENTRY_POINT,
        "profile": "cs_6_0",
        "compiledArtifact": _relpath(output_path, mlx_root),
        "stdout": _relpath(result.stdout_path, mlx_root),
        "stderr": _relpath(result.stderr_path, mlx_root),
    }


def _validate_reference_accessor_opengl(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    artifact_path: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "status": "not-required",
            "required": False,
            "nativeCompiler": "glslangValidator",
            "spirvValidator": "spirv-val",
        }

    tools = {
        "glslangValidator": shutil.which("glslangValidator"),
        "spirv-val": shutil.which("spirv-val"),
    }
    missing_tools = sorted(name for name, value in tools.items() if value is None)
    _require(
        not missing_tools,
        "reference accessor OpenGL validation requires: " + ", ".join(missing_tools),
    )
    output_path = work_dir / "validation" / "reference-accessor-opengl.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    compile_result = _run_command(
        "validate-reference-accessor-opengl",
        [
            str(tools["glslangValidator"]),
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(artifact_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        compile_result.returncode == 0,
        "reference accessor OpenGL compilation failed; inspect "
        "validate-reference-accessor-opengl logs",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "reference accessor OpenGL compilation did not produce SPIR-V",
    )
    validation_result = _run_command(
        "validate-reference-accessor-opengl-spirv",
        [
            str(tools["spirv-val"]),
            "--target-env",
            "opengl4.5",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        validation_result.returncode == 0,
        "reference accessor OpenGL SPIR-V validation failed; inspect "
        "validate-reference-accessor-opengl-spirv logs",
    )
    return {
        "status": "validated",
        "required": True,
        "nativeCompiler": "glslangValidator",
        "spirvValidator": "spirv-val",
        "targetEnvironments": ["opengl", "opengl4.5"],
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compileStdout": _relpath(compile_result.stdout_path, mlx_root),
        "compileStderr": _relpath(compile_result.stderr_path, mlx_root),
        "validationStdout": _relpath(validation_result.stdout_path, mlx_root),
        "validationStderr": _relpath(validation_result.stderr_path, mlx_root),
    }


def _check_reference_accessor_lvalue_identity(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_directx_toolchain: bool,
    require_opengl_toolchain: bool,
) -> dict[str, Any]:
    _require(
        REFERENCE_ACCESSOR_FIXTURE_PATH.is_file(),
        f"reference accessor fixture is missing: {REFERENCE_ACCESSOR_FIXTURE_PATH}",
    )
    project_dir = work_dir / "reference-accessor-project"
    output_dir = project_dir / "generated"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    staged_source = project_dir / REFERENCE_ACCESSOR_FIXTURE_NAME
    shutil.copyfile(REFERENCE_ACCESSOR_FIXTURE_PATH, staged_source)

    config_path = config_dir / "reference-accessor.toml"
    report_path = report_dir / "reference-accessor.json"
    report_path.unlink(missing_ok=True)
    _write_reference_accessor_project_config(config_path, "generated")
    result = _run_command(
        "translate-reference-accessor",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(project_dir),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        report_path.is_file(),
        "reference accessor project translation did not produce a report",
    )
    payload = _load_json(report_path)
    if result.returncode != 0:
        messages = [
            str(item.get("message"))
            for item in payload.get("diagnostics", [])
            if isinstance(item, Mapping) and isinstance(item.get("message"), str)
        ]
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"reference accessor translation failed{detail}")

    summary = payload.get("summary", {})
    diagnostics = payload.get("diagnostics", [])
    artifacts = payload.get("artifacts", [])
    diagnostic_counts = (
        summary.get("diagnosticCounts", {}) if isinstance(summary, Mapping) else {}
    )
    _require(
        isinstance(summary, Mapping)
        and isinstance(diagnostics, list)
        and isinstance(artifacts, list)
        and isinstance(diagnostic_counts, Mapping),
        "reference accessor report must contain structured summary collections",
    )
    _require(
        summary.get("unitCount") == 1
        and summary.get("artifactCount") == len(REFERENCE_ACCESSOR_TARGETS)
        and summary.get("translatedCount") == len(REFERENCE_ACCESSOR_TARGETS)
        and summary.get("failedCount") == 0,
        "reference accessor project translation did not emit both clean artifacts",
    )
    _require(
        len(artifacts) == len(REFERENCE_ACCESSOR_TARGETS)
        and all(isinstance(artifact, Mapping) for artifact in artifacts),
        "reference accessor report must contain exactly one artifact per target",
    )
    _require(
        not diagnostics
        and all(
            diagnostic_counts.get(severity) == 0
            for severity in ("note", "warning", "error")
        ),
        "reference accessor project translation must have zero diagnostics",
    )

    artifacts_by_target = {
        artifact.get("target"): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("source") == REFERENCE_ACCESSOR_FIXTURE_NAME
        and artifact.get("status") == "translated"
        and artifact.get("target") in REFERENCE_ACCESSOR_TARGETS
    }
    _require(
        set(artifacts_by_target) == set(REFERENCE_ACCESSOR_TARGETS),
        "reference accessor report does not contain the expected target artifacts",
    )

    target_proofs: dict[str, Any] = {}
    for target in REFERENCE_ACCESSOR_TARGETS:
        artifact_path = artifacts_by_target[target].get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"reference accessor {target} artifact path is missing",
        )
        generated_path = (project_dir / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_dir.resolve()),
            f"reference accessor {target} artifact escaped its output directory",
        )
        _require(
            generated_path.is_file(),
            f"reference accessor {target} artifact is missing: {artifact_path}",
        )
        generated = generated_path.read_text(encoding="utf-8")
        write_evidence = _reference_accessor_write_evidence(
            generated,
            target=target,
        )
        const_read_evidence = _reference_accessor_const_read_evidence(
            generated,
            target=target,
        )
        nested_const_alias_evidence = _reference_accessor_nested_const_alias_evidence(
            generated,
            target=target,
        )
        if target == "directx":
            native_validation = _validate_reference_accessor_directx(
                mlx_root,
                work_dir,
                log_dir,
                generated_path,
                required=require_directx_toolchain,
            )
        else:
            native_validation = _validate_reference_accessor_opengl(
                mlx_root,
                work_dir,
                log_dir,
                generated_path,
                required=require_opengl_toolchain,
            )
        target_proofs[target] = {
            "artifact": _relpath(generated_path, mlx_root),
            "artifactSha256": _sha256(generated_path),
            "writeEvidence": write_evidence,
            "constReadEvidence": const_read_evidence,
            "nestedConstAliasEvidence": nested_const_alias_evidence,
            "nativeValidation": native_validation,
        }

    return {
        "name": "reference-accessor-lvalue-identity",
        "status": "passed",
        "proofStatus": "verified-original-storage-write",
        "constReadProofStatus": "verified-original-storage-const-read",
        "nestedConstAliasProofStatus": (
            "verified-original-nested-storage-const-alias-read"
        ),
        "scope": "reduced-mlx-shaped-fixture",
        "translationSurface": "crosstl translate-project",
        "report": _relpath(report_path, mlx_root),
        "sourceFixture": (
            "demos/integrations/mlx/fixtures/" + REFERENCE_ACCESSOR_FIXTURE_NAME
        ),
        "stagedSource": _relpath(staged_source, mlx_root),
        "sourceSha256": _sha256(staged_source),
        "accessorContract": {
            "method": "frag_at",
            "returnType": "thread float&",
            "storageExpression": "val_frags[i * width + j]",
            "writeExpression": "tile.frag_at(1, 1) = 73.25f",
            "constRead": {
                "returnType": "const thread float&",
                "enclosingMethod": "store(...) const",
                "implicitCall": "frag_at(i, j)",
                "helperParameterType": "const thread float&",
            },
            "nestedConstAliasRead": {
                "outerType": "ReferenceAccessorStoreLoop",
                "tileMember": "nestedTile",
                "fragmentReturnType": "const thread float2&",
                "enclosingMethod": "store(...) const",
                "aliasDeclaration": (
                    "thread const auto& accum = nestedTile.frag_at(i, j)"
                ),
                "readExpression": "accum[k]",
                "storageExpression": "nestedTile.val_frags[i * width + j][k]",
            },
        },
        "targets": list(REFERENCE_ACCESSOR_TARGETS),
        "artifactCount": len(REFERENCE_ACCESSOR_TARGETS),
        "projectDiagnosticCount": 0,
        "targetProofs": target_proofs,
        "nativeToolchainRequiredByTarget": {
            "directx": require_directx_toolchain,
            "opengl": require_opengl_toolchain,
        },
        "upstreamMlxRuntimeExecuted": False,
        "runtimeParityClaimed": False,
    }


def _check_template_member_buffer_pointer(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    _require(
        TEMPLATE_MEMBER_POINTER_FIXTURE_PATH.is_file(),
        "template member pointer fixture is missing: "
        f"{TEMPLATE_MEMBER_POINTER_FIXTURE_PATH}",
    )
    project_dir = work_dir / "template-member-pointer-project"
    output_dir = project_dir / "generated"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    staged_source = project_dir / TEMPLATE_MEMBER_POINTER_FIXTURE_NAME
    shutil.copyfile(TEMPLATE_MEMBER_POINTER_FIXTURE_PATH, staged_source)

    config_path = config_dir / "template-member-pointer.toml"
    report_path = report_dir / "template-member-pointer.json"
    report_path.unlink(missing_ok=True)
    _write_template_member_pointer_project_config(config_path, "generated")
    result = _run_command(
        "translate-template-member-pointer",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(project_dir),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        report_path.is_file(),
        "template member pointer project translation did not produce a report",
    )
    payload = _load_json(report_path)
    if result.returncode != 0:
        messages = [
            str(item.get("message"))
            for item in payload.get("diagnostics", [])
            if isinstance(item, Mapping) and isinstance(item.get("message"), str)
        ]
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"template member pointer translation failed{detail}")

    summary = payload.get("summary", {})
    diagnostics = payload.get("diagnostics", [])
    artifacts = payload.get("artifacts", [])
    diagnostic_counts = (
        summary.get("diagnosticCounts", {}) if isinstance(summary, Mapping) else {}
    )
    _require(
        isinstance(summary, Mapping)
        and isinstance(diagnostics, list)
        and isinstance(artifacts, list)
        and isinstance(diagnostic_counts, Mapping),
        "template member pointer report must contain structured summary collections",
    )
    _require(
        summary.get("unitCount") == 1
        and summary.get("artifactCount") == len(TEMPLATE_MEMBER_POINTER_TARGETS)
        and summary.get("translatedCount") == len(TEMPLATE_MEMBER_POINTER_TARGETS)
        and summary.get("failedCount") == 0,
        "template member pointer project translation did not emit both clean "
        "artifacts",
    )
    _require(
        len(artifacts) == len(TEMPLATE_MEMBER_POINTER_TARGETS)
        and all(isinstance(artifact, Mapping) for artifact in artifacts),
        "template member pointer report must contain exactly one artifact per "
        "target",
    )
    _require(
        not diagnostics
        and all(
            diagnostic_counts.get(severity) == 0
            for severity in ("note", "warning", "error")
        ),
        "template member pointer project translation must have zero diagnostics",
    )

    artifacts_by_target = {
        artifact.get("target"): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("source") == TEMPLATE_MEMBER_POINTER_FIXTURE_NAME
        and artifact.get("status") == "translated"
        and artifact.get("target") in TEMPLATE_MEMBER_POINTER_TARGETS
    }
    _require(
        set(artifacts_by_target) == set(TEMPLATE_MEMBER_POINTER_TARGETS),
        "template member pointer report does not contain both DirectX and "
        "OpenGL artifacts",
    )

    target_proofs: dict[str, Any] = {}
    for target in TEMPLATE_MEMBER_POINTER_TARGETS:
        artifact_path = artifacts_by_target[target].get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"template member pointer {target} artifact path is missing",
        )
        generated_path = (project_dir / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_dir.resolve()),
            f"template member pointer {target} artifact escaped its output "
            "directory",
        )
        _require(
            generated_path.is_file(),
            f"template member pointer {target} artifact is missing: {artifact_path}",
        )
        generated = generated_path.read_text(encoding="utf-8")
        target_proofs[target] = {
            "artifact": _relpath(generated_path, mlx_root),
            "artifactSha256": _sha256(generated_path),
            "structuralEvidence": _template_member_pointer_evidence(
                generated,
                target=target,
            ),
        }

    return {
        "name": "template-member-buffer-pointer",
        "status": "passed",
        "proofStatus": "verified-materialized-pointer-indexed-read",
        "scope": "reduced-mlx-shaped-fixture",
        "translationSurface": "crosstl translate-project",
        "report": _relpath(report_path, mlx_root),
        "sourceFixture": (
            "demos/integrations/mlx/fixtures/" + TEMPLATE_MEMBER_POINTER_FIXTURE_NAME
        ),
        "stagedSource": _relpath(staged_source, mlx_root),
        "sourceSha256": _sha256(staged_source),
        "sourceContract": {
            "outerType": "ReducedMMATile",
            "outerMethod": "load",
            "outerTemplateParameter": "U",
            "sourcePointerType": "const device U*",
            "fragmentType": "ReducedMMAFrag",
            "fragmentMethod": "load",
            "fragmentTemplateParameter": "SrcPtrType",
            "pointerArgument": "&(src[index])",
            "indexedReadExpression": "src[stride]",
        },
        "targets": list(TEMPLATE_MEMBER_POINTER_TARGETS),
        "artifactCount": len(TEMPLATE_MEMBER_POINTER_TARGETS),
        "projectDiagnosticCount": 0,
        "targetProofs": target_proofs,
        "generatedArtifactEvidenceOnly": True,
        "nativeToolchainValidationIncluded": False,
        "upstreamMlxRuntimeExecuted": False,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
    }


def _scaled_attention_local_alias_evidence(
    mlx_root: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in payload.get("artifacts", [])
        if isinstance(artifact, Mapping)
        and artifact.get("source") == MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
        and artifact.get("target") == "vulkan"
        and artifact.get("status") == "translated"
    ]
    _require(
        len(artifacts) == 1,
        "scaled-attention local-alias evidence requires one Vulkan artifact",
    )

    artifact_path = artifacts[0].get("path")
    _require(
        isinstance(artifact_path, str),
        "scaled-attention Vulkan artifact path is missing",
    )
    generated_path = mlx_root / artifact_path
    _require(
        generated_path.is_file(),
        f"scaled-attention Vulkan artifact is missing: {artifact_path}",
    )
    generated = generated_path.read_text(encoding="utf-8")

    forbidden_alias_residue = re.compile(
        r"\bLimits_u3cU_u3e\b|\bUnknown type U\b|unknown function ['\"]U['\"]"
    )
    residue = forbidden_alias_residue.search(generated)
    _require(
        residue is None,
        "scaled-attention Vulkan artifact retained local alias residue: "
        f"{residue.group(0) if residue else ''}",
    )
    vulkan_warnings = [
        line
        for line in generated.splitlines()
        if line.lstrip().startswith("; WARNING:")
    ]
    _require(
        not vulkan_warnings,
        "scaled-attention Vulkan project artifact emitted a semantic warning: "
        + (vulkan_warnings[0] if vulkan_warnings else ""),
    )

    vulkan_entry_count = len(
        re.findall(
            r"(?m)^[ \t]*OpEntryPoint[ \t]+GLCompute\b",
            generated,
        )
    )
    expected_entry_count = MLX_DIRECTX_FRONTIER_ENTRY_POINT_COUNTS[
        MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    ]
    _require(
        vulkan_entry_count == expected_entry_count,
        "scaled-attention Vulkan artifact did not retain all "
        f"{expected_entry_count} materialized entries",
    )

    return {
        "source": MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
        "target": "vulkan",
        "artifact": artifact_path,
        "entryCount": vulkan_entry_count,
        "resolvedDeclarationTypeCount": 402,
        "resolvedCastCount": 87,
        "resolvedStaticMemberCount": 42,
        "vulkanProjectWarningCount": 0,
        "remainingAliasShapesTrackedBy": (
            "https://github.com/CrossGL/crosstl/issues/1567"
        ),
        "unreachableGenericWarningsTrackedBy": (
            "https://github.com/CrossGL/crosstl/issues/1568"
        ),
    }


def _directx_toolchain_entry_point(run: Mapping[str, Any]) -> str | None:
    command = run.get("command")
    if not isinstance(command, list) or any(
        not isinstance(argument, str) for argument in command
    ):
        return None
    try:
        profile = command[command.index("-T") + 1]
        entry_point = command[command.index("-E") + 1]
    except (ValueError, IndexError):
        return None
    if not profile.startswith("cs_") or not entry_point:
        return None
    return entry_point


def _require_frontier_project_join(
    payload: Mapping[str, Any],
    *,
    target: str,
    sources: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    expected_sources = set(sources)
    project = payload.get("project")
    include_patterns = (
        project.get("includePatterns") if isinstance(project, Mapping) else None
    )
    _require(
        isinstance(project, Mapping)
        and isinstance(include_patterns, list)
        and len(include_patterns) == len(sources)
        and set(include_patterns) == expected_sources
        and project.get("targets") == [target]
        and project.get("workgroupSizeRules") == {}
        and project.get("workgroupSizeRuleCount") == 0,
        f"{target.title()} frontier config/report join changed or gained an "
        "unproved workgroup-size rule",
    )
    units = payload.get("units")
    _require(
        isinstance(units, list) and len(units) == len(sources),
        f"{target.title()} frontier source units are incomplete",
    )
    units_by_source: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        _require(
            isinstance(unit, Mapping)
            and unit.get("id") == unit.get("path")
            and unit.get("sourceBackend") == "metal"
            and unit.get("path") in expected_sources,
            f"{target.title()} frontier source-unit provenance changed",
        )
        source = str(unit["path"])
        _require(
            source not in units_by_source,
            f"{target.title()} frontier duplicated source unit {source}",
        )
        units_by_source[source] = unit
    _require(
        set(units_by_source) == expected_sources,
        f"{target.title()} frontier source units do not match the config",
    )
    return units_by_source


def _require_clean_frontier_report(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    target: str,
    sources: Sequence[str],
    validated: bool = True,
) -> dict[str, Mapping[str, Any]]:
    units_by_source = _require_frontier_project_join(
        payload,
        target=target,
        sources=sources,
    )
    summary = payload.get("summary")
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == len(sources)
        and summary.get("artifactCount") == len(sources)
        and summary.get("translatedCount") == len(sources)
        and summary.get("failedCount") == 0
        and isinstance(summary.get("diagnosticCounts"), Mapping)
        and summary["diagnosticCounts"].get("error", 0) == 0,
        f"{target.title()} frontier did not retain every clean source artifact",
    )
    target_summary = summary.get("artifactsByTarget", {}).get(target, {})
    _require(
        isinstance(target_summary, Mapping)
        and target_summary.get("artifactCount") == len(sources)
        and target_summary.get("translatedCount") == len(sources)
        and target_summary.get("failedCount") == 0,
        f"{target.title()} frontier target artifact accounting changed",
    )
    if validated:
        validation = payload.get("validation")
        _require(
            isinstance(validation, Mapping)
            and isinstance(validation.get("summary"), Mapping)
            and validation["summary"].get("failedCount") == 0,
            f"{target.title()} frontier artifact validation reported failures",
        )
    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == len(sources),
        f"{target.title()} frontier artifact records are incomplete",
    )
    artifacts_by_source: dict[str, Mapping[str, Any]] = {}
    output_root = output_dir.resolve()
    for artifact in artifacts:
        _require(
            isinstance(artifact, Mapping)
            and artifact.get("target") == target
            and artifact.get("status") == "translated"
            and artifact.get("source") in units_by_source,
            f"{target.title()} frontier translated artifact contract changed",
        )
        source = str(artifact["source"])
        _require(
            source not in artifacts_by_source,
            f"{target.title()} frontier duplicated artifact source {source}",
        )
        artifact_path = artifact.get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"{target.title()} frontier artifact path is missing for {source}",
        )
        generated_path = (mlx_root / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_root) and generated_path.is_file(),
            f"{target.title()} frontier artifact is missing or escaped output: "
            f"{artifact_path}",
        )
        _require(
            artifact.get("sourceHash") == units_by_source[source].get("sourceHash")
            and artifact.get("sourceSizeBytes")
            == units_by_source[source].get("sourceSizeBytes"),
            f"{target.title()} frontier artifact source provenance changed for {source}",
        )
        artifacts_by_source[source] = artifact
    _require(
        set(artifacts_by_source) == set(sources),
        f"{target.title()} frontier artifacts do not match the config source set",
    )
    return artifacts_by_source


def _require_dynamic_workgroup_blocker_report(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    target: str,
    sources: Sequence[str],
    validated: bool,
) -> dict[str, dict[str, Any]]:
    units_by_source = _require_frontier_project_join(
        payload,
        target=target,
        sources=sources,
    )
    expected_sources = set(sources)
    summary = payload.get("summary")
    expected_diagnostics = {
        MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE: len(sources),
    }
    if validated:
        expected_diagnostics["project.validate.failed-artifact"] = len(sources)
    expected_error_count = sum(expected_diagnostics.values())
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == len(sources)
        and summary.get("artifactCount") == len(sources)
        and summary.get("translatedCount") == 0
        and summary.get("failedCount") == len(sources)
        and summary.get("diagnosticCounts")
        == {"error": expected_error_count, "note": 0, "warning": 0}
        and summary.get("diagnosticsByCode") == expected_diagnostics
        and summary.get("artifactsByTarget")
        == {
            target: {
                "artifactCount": len(sources),
                "translatedCount": 0,
                "failedCount": len(sources),
            }
        },
        f"{target.title()} dynamic-workgroup frontier accounting changed",
    )
    error_diagnostics = [
        diagnostic
        for diagnostic in payload.get("diagnostics", [])
        if isinstance(diagnostic, Mapping) and diagnostic.get("severity") == "error"
    ]
    _require(
        len(error_diagnostics) == expected_error_count
        and Counter(str(diagnostic.get("code")) for diagnostic in error_diagnostics)
        == Counter(expected_diagnostics),
        f"{target.title()} dynamic-workgroup diagnostics changed",
    )
    diagnostics_by_source: dict[str, Mapping[str, Any]] = {}
    source_entries_by_source: dict[str, set[str]] = {}
    for diagnostic in error_diagnostics:
        if diagnostic.get("code") != MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE:
            continue
        details = diagnostic.get("details")
        source = details.get("sourcePath") if isinstance(details, Mapping) else None
        _require(
            isinstance(source, str)
            and source in expected_sources
            and source not in diagnostics_by_source
            and diagnostic.get("target") == target
            and diagnostic.get("checkKind") == "execution-specialization"
            and diagnostic.get("message") == MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_MESSAGE
            and diagnostic.get("missingCapabilities")
            == ["execution.workgroup-size-specialization"],
            f"{target.title()} dynamic-workgroup diagnostic/source join changed",
        )
        execution = details.get("executionSpecialization")
        source_entries = (
            execution.get("sourceEntryPoints")
            if isinstance(execution, Mapping)
            else None
        )
        _require(
            isinstance(source_entries, list)
            and execution.get("reason") == "aggregate-entry-size-unproven"
            and len(source_entries) == MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS[source]
            and all(isinstance(entry, str) and entry for entry in source_entries)
            and len(set(source_entries)) == len(source_entries),
            f"{target.title()} dynamic-workgroup entry evidence changed for {source}",
        )
        diagnostics_by_source[source] = diagnostic
        source_entries_by_source[source] = set(source_entries)
    _require(
        set(diagnostics_by_source) == expected_sources,
        f"{target.title()} dynamic-workgroup diagnostics do not cover the config",
    )

    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == len(sources),
        f"{target.title()} dynamic-workgroup artifact records are incomplete",
    )
    artifacts_by_source: dict[str, Mapping[str, Any]] = {}
    output_root = output_dir.resolve()
    for artifact in artifacts:
        source = artifact.get("source") if isinstance(artifact, Mapping) else None
        _require(
            isinstance(source, str)
            and source in expected_sources
            and source not in artifacts_by_source
            and artifact.get("target") == target
            and artifact.get("status") == "failed"
            and artifact.get("error") == MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_MESSAGE
            and artifact.get("sourceHash") == units_by_source[source].get("sourceHash")
            and artifact.get("sourceSizeBytes")
            == units_by_source[source].get("sourceSizeBytes")
            and "generatedHash" not in artifact
            and "generatedSizeBytes" not in artifact,
            f"{target.title()} dynamic-workgroup artifact/source join changed",
        )
        artifact_path = artifact.get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"{target.title()} dynamic-workgroup artifact path is missing for {source}",
        )
        generated_path = (mlx_root / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_root)
            and not generated_path.exists(),
            f"{target.title()} dynamic-workgroup source unexpectedly emitted "
            f"{artifact_path}",
        )
        materialization = artifact.get("templateMaterialization")
        specializations = (
            materialization.get("specializations")
            if isinstance(materialization, Mapping)
            else None
        )
        host_names = [
            record.get("hostName")
            for record in specializations or []
            if isinstance(record, Mapping) and record.get("hostName") is not None
        ]
        expected_specialization_count = MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[source][
            "specializationCount"
        ]
        _require(
            isinstance(materialization, Mapping)
            and materialization.get("status") == "materialized"
            and isinstance(specializations, list)
            and materialization.get("specializationCount")
            == expected_specialization_count
            and len(specializations) == expected_specialization_count
            and all(isinstance(record, Mapping) for record in specializations)
            and materialization.get("unsupported") == []
            and len(host_names) == MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS[source]
            and all(
                isinstance(host_name, str) and host_name for host_name in host_names
            )
            and len(set(host_names)) == len(host_names)
            and set(host_names) == source_entries_by_source[source],
            f"{target.title()} dynamic-workgroup materialization changed for {source}",
        )
        artifacts_by_source[source] = artifact
    _require(
        set(artifacts_by_source) == expected_sources,
        f"{target.title()} dynamic-workgroup artifacts do not cover the config",
    )
    if validated:
        validation = payload.get("validation")
        _require(
            isinstance(validation, Mapping)
            and isinstance(validation.get("summary"), Mapping)
            and validation["summary"].get("failedCount") == len(sources),
            f"{target.title()} dynamic-workgroup validation accounting changed",
        )
    return {
        source: {
            **MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[source],
            "entryPointCount": MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS[source],
            "diagnosticCode": MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE,
            "artifactStatus": "failed",
            "artifactEmitted": False,
            "sourceEntryPointIdentityStatus": "matched-materialized-host-names",
        }
        for source in sources
    }


def _run_frontier_project(
    *,
    mlx_root: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    command_name: str,
    target: str,
    sources: Sequence[str],
    output_dir: Path,
    validate: bool = False,
    run_toolchains: bool = False,
    check: bool = True,
    specialization_constants: Mapping[str, bool | int | float] | None = None,
) -> tuple[CommandResult, dict[str, Any], Path, Path]:
    config_path = config_dir / f"{command_name}.toml"
    report_path = report_dir / f"{command_name}.json"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    report_path.unlink(missing_ok=True)
    _write_project_config(
        config_path,
        include=sources,
        targets=(target,),
        output_dir=_relpath(output_dir, mlx_root),
        specialization_constants=specialization_constants,
    )
    command = [
        python,
        "-m",
        "crosstl",
        "translate-project",
        str(mlx_root),
        "--config",
        str(config_path),
        "--report",
        str(report_path),
    ]
    if validate:
        command.append("--validate")
    if run_toolchains:
        command.append("--run-toolchains")
    result = _run_command(command_name, command, log_dir=log_dir, check=check)
    _require(report_path.is_file(), f"{command_name} did not produce a project report")
    return result, _load_json(report_path), config_path, report_path


def _translate_directx_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_directx_toolchain: bool,
) -> dict[str, Any]:
    run_toolchains = not FRONTIER_VALIDATION_TRACKED_ISSUES
    clean_output_dir = work_dir / "out-directx-frontier"
    _result, clean_payload, _config_path, report_path = _run_frontier_project(
        mlx_root=mlx_root,
        config_dir=config_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        python=python,
        command_name="directx-frontier",
        target="directx",
        sources=MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES,
        output_dir=clean_output_dir,
        validate=True,
        specialization_constants=MLX_FRONTIER_SPECIALIZATION_CONSTANTS,
    )
    _require_clean_frontier_report(
        mlx_root,
        clean_output_dir,
        clean_payload,
        target="directx",
        sources=MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES,
    )

    blocked_output_dir = work_dir / "out-directx-workgroup-frontier"
    blocked_result, blocked_payload, _blocked_config, blocked_report_path = (
        _run_frontier_project(
            mlx_root=mlx_root,
            config_dir=config_dir,
            report_dir=report_dir,
            log_dir=log_dir,
            python=python,
            command_name="directx-workgroup-frontier",
            target="directx",
            sources=MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
            output_dir=blocked_output_dir,
            validate=True,
            check=False,
            specialization_constants=MLX_FRONTIER_SPECIALIZATION_CONSTANTS,
        )
    )
    _require(
        blocked_result.returncode == 1,
        "DirectX dynamic-workgroup frontier must fail with exit code 1",
    )
    dispatch_evidence = _require_dynamic_workgroup_blocker_report(
        mlx_root,
        blocked_output_dir,
        blocked_payload,
        target="directx",
        sources=MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
        validated=True,
    )

    directx_runs: list[Mapping[str, Any]] = []
    if run_toolchains and require_directx_toolchain:
        toolchain_output_dir = work_dir / "out-directx-frontier-toolchain"
        _toolchain_result, toolchain_payload, _toolchain_config, _toolchain_report = (
            _run_frontier_project(
                mlx_root=mlx_root,
                config_dir=config_dir,
                report_dir=report_dir,
                log_dir=log_dir,
                python=python,
                command_name="validate-directx-frontier-toolchain",
                target="directx",
                sources=MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES,
                output_dir=toolchain_output_dir,
                run_toolchains=True,
                specialization_constants=MLX_FRONTIER_SPECIALIZATION_CONSTANTS,
            )
        )
        artifacts = _require_clean_frontier_report(
            mlx_root,
            toolchain_output_dir,
            toolchain_payload,
            target="directx",
            sources=MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES,
        )
        validation = toolchain_payload.get("validation")
        runs = (
            validation.get("toolchainRuns") if isinstance(validation, Mapping) else None
        )
        _require(
            isinstance(runs, list), "DirectX frontier toolchainRuns must be a list"
        )
        directx_runs = [
            run
            for run in runs
            if isinstance(run, Mapping) and run.get("target") == "directx"
        ]
        artifact_paths = {artifact["path"] for artifact in artifacts.values()}
        validated_paths = {
            run.get("path") for run in directx_runs if run.get("status") == "ok"
        }
        _require(
            artifact_paths <= validated_paths,
            "DirectX toolchain did not validate every translated source artifact",
        )

    directx_entry_points_by_source: dict[str, list[str]] = {
        source: [] for source in MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    }
    for run in directx_runs:
        if run.get("status") != "ok":
            continue
        source = run.get("source")
        _require(
            source in directx_entry_points_by_source,
            f"DirectX toolchain validation reported an unexpected source: {source}",
        )
        entry_point = _directx_toolchain_entry_point(run)
        _require(
            entry_point is not None,
            "DirectX toolchain validation did not record a compute entry command",
        )
        validated_entries = directx_entry_points_by_source[source]
        _require(
            entry_point not in validated_entries,
            f"DirectX toolchain validation duplicated {source} entry {entry_point}",
        )
        validated_entries.append(entry_point)
    directx_validated_entry_point_counts = {
        source: len(entry_points)
        for source, entry_points in directx_entry_points_by_source.items()
        if entry_points
    }
    directx_validated_sources = [
        source
        for source in MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
        if source in directx_validated_entry_point_counts
    ]
    for run in directx_runs:
        _require(run.get("status") == "ok", "DirectX toolchain validation failed")
    if require_directx_toolchain and run_toolchains:
        _require(
            directx_validated_sources == list(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
            "DirectX toolchain did not validate every configured source",
        )
        _require(
            directx_validated_entry_point_counts
            == MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS,
            "DirectX toolchain did not validate every generated compute entry point",
        )
    return {
        "name": "directx-frontier",
        "status": "passed-with-expected-workgroup-blockers",
        "scope": "target-split-frontier",
        "report": _relpath(report_path, mlx_root),
        "workgroupBlockedReport": _relpath(blocked_report_path, mlx_root),
        "sources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "unitCount": len(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "artifactCount": len(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "translatedSources": list(MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES),
        "translatedArtifactCount": len(MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES),
        "workgroupBlockedSources": list(MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES),
        "workgroupBlockedArtifactCount": len(MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES),
        "target": "directx",
        "toolchainRuns": len(directx_runs),
        "directxToolchainRequired": require_directx_toolchain,
        "directxToolchainSources": list(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
        "directxToolchainArtifactCount": len(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
        "directxToolchainExpectedEntryPointCounts": dict(
            MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
        ),
        "directxToolchainExpectedEntryPointCount": (
            MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT
        ),
        "directxToolchainValidatedSources": directx_validated_sources,
        "directxToolchainValidatedArtifactCount": len(directx_validated_sources),
        "directxToolchainValidatedEntryPointCounts": (
            directx_validated_entry_point_counts
        ),
        "directxToolchainValidatedEntryPointCount": sum(
            directx_validated_entry_point_counts.values()
        ),
        "directxValidationStatus": (
            "validated" if run_toolchains and directx_runs else "not-required"
        ),
        "dynamicWorkgroupDispatchEvidence": dispatch_evidence,
        "semanticReadinessStatus": "not-established",
        "trackedIssues": [
            MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE,
            *FRONTIER_VALIDATION_TRACKED_ISSUES,
        ],
        "runtimeParityClaimed": False,
    }


def _translate_vulkan_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_toolchain: bool,
    run_optional_toolchain: bool,
) -> dict[str, Any]:
    output_dir = work_dir / "out-vulkan-frontier"
    _result, payload, _config_path, report_path = _run_frontier_project(
        mlx_root=mlx_root,
        config_dir=config_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        python=python,
        command_name="vulkan-frontier",
        target="vulkan",
        sources=MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
        output_dir=output_dir,
        validate=True,
        specialization_constants=MLX_FRONTIER_SPECIALIZATION_CONSTANTS,
    )
    _require_clean_frontier_report(
        mlx_root,
        output_dir,
        payload,
        target="vulkan",
        sources=MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
    )
    scaled_attention_alias_evidence = _scaled_attention_local_alias_evidence(
        mlx_root, payload
    )

    vulkan_runs: list[Mapping[str, Any]] = []
    run_toolchains = not FRONTIER_VALIDATION_TRACKED_ISSUES
    if run_toolchains and (require_toolchain or run_optional_toolchain):
        toolchain_output_dir = work_dir / "out-vulkan-frontier-toolchain"
        _toolchain_result, toolchain_payload, _toolchain_config, _toolchain_report = (
            _run_frontier_project(
                mlx_root=mlx_root,
                config_dir=config_dir,
                report_dir=report_dir,
                log_dir=log_dir,
                python=python,
                command_name="validate-vulkan-frontier-toolchain",
                target="vulkan",
                sources=MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
                output_dir=toolchain_output_dir,
                run_toolchains=True,
                specialization_constants=MLX_FRONTIER_SPECIALIZATION_CONSTANTS,
            )
        )
        artifacts = _require_clean_frontier_report(
            mlx_root,
            toolchain_output_dir,
            toolchain_payload,
            target="vulkan",
            sources=MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
        )
        validation = toolchain_payload.get("validation")
        runs = (
            validation.get("toolchainRuns") if isinstance(validation, Mapping) else None
        )
        _require(isinstance(runs, list), "Vulkan frontier toolchainRuns must be a list")
        vulkan_runs = [
            run
            for run in runs
            if isinstance(run, Mapping) and run.get("target") == "vulkan"
        ]
        for run in vulkan_runs:
            _require(run.get("status") == "ok", "Vulkan toolchain validation failed")
        if require_toolchain:
            artifact_paths = {artifact["path"] for artifact in artifacts.values()}
            validated_paths = {
                run.get("path") for run in vulkan_runs if run.get("status") == "ok"
            }
            _require(
                artifact_paths <= validated_paths,
                "Vulkan toolchain did not validate every configured source artifact",
            )

    return {
        "name": "vulkan-frontier",
        "status": "passed",
        "scope": "target-split-frontier",
        "report": _relpath(report_path, mlx_root),
        "sources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "unitCount": len(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "artifactCount": len(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "translatedArtifactCount": len(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "target": "vulkan",
        "toolchainRuns": len(vulkan_runs),
        "vulkanToolchainRequired": require_toolchain,
        "vulkanValidationStatus": "validated" if vulkan_runs else "not-run",
        "semanticReadinessStatus": "not-established",
        "regressionEvidence": [scaled_attention_alias_evidence],
        "trackedIssues": list(FRONTIER_VALIDATION_TRACKED_ISSUES),
        "runtimeParityClaimed": False,
    }


def _check_arange_opengl(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "arange-opengl.toml"
    report_path = report_dir / "arange-opengl.json"
    _write_project_config(
        config_path,
        include=MLX_ARANGE_SOURCE,
        targets=("opengl",),
        output_dir=_relpath(work_dir / "out-arange-opengl", mlx_root),
        entry_points={MLX_ARANGE_SOURCE: "arangeuint32"},
    )
    result = _run_command(
        "translate-arange-opengl",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, dict), "OpenGL report summary must be an object")
    if result.returncode != 0:
        messages = []
        for diagnostic in payload.get("diagnostics", []):
            if isinstance(diagnostic, dict):
                message = diagnostic.get("message")
                if isinstance(message, str):
                    messages.append(message)
        for artifact in payload.get("artifacts", []):
            if isinstance(artifact, dict):
                error = artifact.get("error")
                if isinstance(error, str):
                    messages.append(error)
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"OpenGL arange translation failed{detail}")

    _require(
        summary.get("translatedCount") == 1 and summary.get("failedCount") == 0,
        "OpenGL arange translation succeeded but the report did not show one clean artifact",
    )
    artifacts = payload.get("artifacts", [])
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and item.get("source") == MLX_ARANGE_SOURCE
            and item.get("target") == "opengl"
        ),
        None,
    )
    _require(isinstance(artifact, dict), "OpenGL arange artifact is missing")
    artifact_path = artifact.get("path")
    _require(isinstance(artifact_path, str), "OpenGL arange artifact path is missing")
    generated_path = mlx_root / artifact_path
    _require(
        generated_path.is_file(), f"OpenGL arange artifact is missing: {artifact_path}"
    )
    generated = generated_path.read_text(encoding="utf-8")
    generated_lower = generated.lower()
    _require(
        "#include <metal" not in generated_lower
        and "#pragma metal" not in generated_lower,
        "OpenGL arange artifact retained a Metal system preprocessor line",
    )
    _require(
        artifact.get("entryPoint")
        == {"source": "arangeuint32", "target": "main", "stage": "compute"},
        "OpenGL arange artifact did not record the selected compute entry",
    )
    _require(
        generated.count("void main()") == 1 and "compute_main" not in generated,
        "OpenGL arange artifact is not independently loadable through one main entry",
    )
    resource_bindings = re.findall(
        r"layout\s*\(\s*std(?:140|430)\s*,\s*binding\s*=\s*(\d+)\s*\)\s*"
        r"(?:uniform|buffer)\b",
        generated,
    )
    _require(
        sorted(int(binding) for binding in resource_bindings) == [0, 1, 2],
        "OpenGL arange artifact must expose only start, step, and output resources",
    )
    normalized_source = re.sub(r"\s+", "", generated)
    _require(
        "out_[index]=(start+(index*step));" in normalized_source,
        "OpenGL arange artifact did not preserve uint32 arange data flow",
    )
    _require(
        "arangeuint8" not in generated and "arangefloat" not in generated,
        "OpenGL arange artifact retained unrelated materialized entries",
    )
    _require(
        "log1p__metal_overload_" not in generated
        and "subgroupShuffle" not in generated
        and "complex64_t probe(" not in generated,
        "OpenGL arange artifact retained unrelated helper dependencies",
    )
    native_validation = _validate_arange_opengl(
        mlx_root,
        work_dir,
        log_dir,
        generated_path,
    )
    return {
        "name": "arange-opengl",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_ARANGE_SOURCE,
        "target": "opengl",
        "metalIncludesFiltered": True,
        "selectedEntryPoint": "arangeuint32",
        "targetEntryPoint": "main",
        "interfaceResourceCount": 3,
        "standaloneArtifact": True,
        "arangeDataFlowPreserved": True,
        **native_validation,
        "trackedIssues": list(OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES),
    }


def _validate_arange_opengl(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    generated_path: Path,
) -> dict[str, Any]:
    validator = shutil.which("glslangValidator")
    if validator is None:
        return {
            "nativeValidationAttempted": False,
            "nativeValidationBlockerConfirmed": False,
            "nativeValidationStatus": "not-run-tool-unavailable",
            "nativeValidator": "glslangValidator",
            "nativeValidatorStatus": "unavailable",
        }

    output_path = work_dir / "validation" / "arange-opengl.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command(
        "validate-arange-opengl",
        [
            validator,
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(generated_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    if result.returncode == 0:
        _require(
            not OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES,
            "OpenGL arange validation passed while tracked validation issues remain",
        )
        return {
            "nativeValidationAttempted": True,
            "nativeValidationBlockerConfirmed": False,
            "nativeValidationStatus": "validated",
            "nativeValidator": "glslangValidator",
            "nativeValidatorStatus": "available",
            "nativeValidationExitCode": 0,
            "nativeValidationOutput": _relpath(output_path, mlx_root),
        }

    raise PortingCheckError(
        "OpenGL arange validation failed without a tracked validation issue"
    )


def _check_opengl_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_toolchain: bool,
) -> dict[str, Any]:
    clean_output_dir = work_dir / "out-opengl-frontier"
    _result, payload, _config_path, report_path = _run_frontier_project(
        mlx_root=mlx_root,
        config_dir=config_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        python=python,
        command_name="opengl-frontier",
        target="opengl",
        sources=MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES,
        output_dir=clean_output_dir,
    )
    artifacts_by_source = _require_clean_frontier_report(
        mlx_root,
        clean_output_dir,
        payload,
        target="opengl",
        sources=MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES,
        validated=False,
    )
    summary = payload.get("summary", {})
    diagnostic_counts = summary.get("diagnosticCounts", {})
    diagnostics = payload.get("diagnostics", [])
    _require(
        isinstance(diagnostic_counts, Mapping)
        and diagnostics == []
        and all(
            diagnostic_counts.get(severity) == 0
            for severity in ("note", "warning", "error")
        ),
        "OpenGL clean frontier translation must have zero project diagnostics",
    )

    blocked_output_dir = work_dir / "out-opengl-workgroup-frontier"
    blocked_result, blocked_payload, _blocked_config, blocked_report_path = (
        _run_frontier_project(
            mlx_root=mlx_root,
            config_dir=config_dir,
            report_dir=report_dir,
            log_dir=log_dir,
            python=python,
            command_name="opengl-workgroup-frontier",
            target="opengl",
            sources=MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
            output_dir=blocked_output_dir,
            check=False,
        )
    )
    _require(
        blocked_result.returncode == 1,
        "OpenGL dynamic-workgroup frontier must fail with exit code 1",
    )
    dispatch_evidence = _require_dynamic_workgroup_blocker_report(
        mlx_root,
        blocked_output_dir,
        blocked_payload,
        target="opengl",
        sources=MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
        validated=False,
    )

    generated_paths: dict[str, Path] = {}
    for source in MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES:
        artifact_path = artifacts_by_source[source].get("path")
        _require(
            isinstance(artifact_path, str),
            f"OpenGL frontier artifact path is missing for {source}",
        )
        generated_path = mlx_root / artifact_path
        _require(
            generated_path.is_file(),
            f"OpenGL frontier artifact is missing: {artifact_path}",
        )
        generated_paths[source] = generated_path

    specialization_evidence: dict[str, dict[str, int]] = {}
    for source, expected_constants in MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS.items():
        if source not in artifacts_by_source:
            continue
        artifact = artifacts_by_source[source]
        reflected_constants = artifact.get("specializationConstants", [])
        _require(
            isinstance(reflected_constants, list),
            f"OpenGL specialization metadata is missing for {source}",
        )
        reflected_ids = {
            constant.get("name"): constant.get("id")
            for constant in reflected_constants
            if isinstance(constant, Mapping)
        }
        _require(
            reflected_ids == expected_constants,
            f"OpenGL specialization metadata does not match {source}",
        )
        materialization = artifact.get("specializationMaterialization")
        _require(
            isinstance(materialization, Mapping)
            and materialization.get("mode") == "deferred"
            and materialization.get("targetSupportsDeferredSpecialization") is True,
            f"OpenGL specialization deferral is not recorded for {source}",
        )
        generated = generated_paths[source].read_text(encoding="utf-8")
        for name, constant_id in expected_constants.items():
            _require(
                re.search(
                    rf"layout\s*\(\s*constant_id\s*=\s*{constant_id}\s*\)"
                    rf"\s*const\s+\w+\s+{re.escape(name)}\s*=",
                    generated,
                )
                is not None,
                f"OpenGL artifact did not preserve specialization id {constant_id} "
                f"for {name}",
            )
            _require(
                re.search(rf"\buniform\s+\w+\s+{re.escape(name)}\b", generated) is None,
                f"OpenGL artifact lowered specialization input {name} as a uniform",
            )
        specialization_evidence[source] = dict(expected_constants)

    validation_status = "not-required"
    validation_outputs: dict[str, str] = {}
    toolchain_validated_sources: list[str] = []
    if require_toolchain:
        required_tools = {
            "glslangValidator": shutil.which("glslangValidator"),
            "spirv-val": shutil.which("spirv-val"),
        }
        missing_tools = sorted(
            name for name, resolved in required_tools.items() if resolved is None
        )
        _require(
            not missing_tools,
            "OpenGL frontier validation requires: " + ", ".join(missing_tools),
        )

        for source, generated_path in generated_paths.items():
            stem = Path(source).stem
            command_name = stem.replace("_", "-")
            output_path = work_dir / "validation" / f"{stem}-opengl.spv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            compile_result = _run_command(
                f"validate-{command_name}-opengl",
                [
                    str(required_tools["glslangValidator"]),
                    "--target-env",
                    "opengl",
                    "--target-env",
                    "spirv1.3",
                    "-S",
                    "comp",
                    str(generated_path),
                    "-o",
                    str(output_path),
                ],
                log_dir=log_dir,
                check=False,
            )
            _require(
                compile_result.returncode == 0,
                (
                    f"OpenGL {stem} native compilation failed; inspect "
                    f"validate-{command_name}-opengl logs"
                ),
            )
            _require(
                output_path.is_file(),
                f"OpenGL {stem} compilation succeeded without producing SPIR-V",
            )
            validation_result = _run_command(
                f"validate-{command_name}-opengl-spirv",
                [
                    str(required_tools["spirv-val"]),
                    "--target-env",
                    "spv1.3",
                    str(output_path),
                ],
                log_dir=log_dir,
                check=False,
            )
            _require(
                validation_result.returncode == 0,
                (
                    f"OpenGL {stem} SPIR-V validation failed; inspect "
                    f"validate-{command_name}-opengl-spirv logs"
                ),
            )
            validation_outputs[source] = _relpath(output_path, mlx_root)
            toolchain_validated_sources.append(source)
        _require(
            toolchain_validated_sources == list(MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES),
            "OpenGL frontier toolchain did not validate every translated source",
        )
        validation_status = "validated"

    return {
        "name": "opengl-frontier",
        "status": "passed-with-expected-workgroup-blockers",
        "scope": "target-split-frontier",
        "report": _relpath(report_path, mlx_root),
        "workgroupBlockedReport": _relpath(blocked_report_path, mlx_root),
        "sources": list(MLX_OPENGL_FRONTIER_SOURCES),
        "sourceCount": len(MLX_OPENGL_FRONTIER_SOURCES),
        "target": "opengl",
        "artifactCount": len(MLX_OPENGL_FRONTIER_SOURCES),
        "translatedSources": list(MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES),
        "translatedArtifactCount": len(MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES),
        "workgroupBlockedSources": list(MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES),
        "workgroupBlockedArtifactCount": len(
            MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
        ),
        "projectDiagnosticCount": 0,
        "expectedProjectErrorCount": len(MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES),
        "toolchainRequired": require_toolchain,
        "toolchainSources": list(MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES),
        "toolchainValidatedSources": toolchain_validated_sources,
        "toolchainValidatedArtifactCount": len(toolchain_validated_sources),
        "nativeValidationStatus": validation_status,
        "nativeValidator": "glslangValidator",
        "spirvValidator": "spirv-val",
        "nativeValidationOutputs": validation_outputs,
        "specializationConstants": specialization_evidence,
        "dynamicWorkgroupDispatchEvidence": dispatch_evidence,
        "trackedIssues": [MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE],
        "runtimeIntegrationIncluded": False,
    }


def _check_fft_opengl_workgroup_pointer_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "fft-opengl-workgroup-pointer.toml"
    report_path = report_dir / "fft-opengl-workgroup-pointer.json"
    output_dir = work_dir / "out-fft-opengl-workgroup-pointer"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_project_config(
        config_path,
        include=MLX_FFT_SOURCE,
        targets=("opengl",),
        output_dir=_relpath(output_dir, mlx_root),
        metal_source_options={
            "max_template_specializations": FFT_OPENGL_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": (
                FFT_OPENGL_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        },
    )
    result = _run_command(
        "translate-fft-opengl-workgroup-pointer",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--no-format",
        ],
        log_dir=log_dir,
        check=False,
        timeout_seconds=FFT_OPENGL_TRANSLATION_TIMEOUT_SECONDS,
    )
    _require(
        result.returncode == 1,
        "OpenGL FFT translation must remain fail-closed at its tracked "
        "workgroup-pointer frontier",
    )
    _require(
        report_path.is_file(),
        "OpenGL FFT translation did not produce a project report",
    )

    payload = _load_json(report_path)
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        "OpenGL FFT translation report kind changed",
    )
    summary = payload.get("summary", {})
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 0
        and summary.get("failedCount") == 1,
        "OpenGL FFT translation report did not retain one failed artifact",
    )
    _require(
        summary.get("diagnosticCounts") == {"error": 1, "note": 0, "warning": 0}
        and summary.get("diagnosticsByCode") == {FFT_OPENGL_EXPECTED_DIAGNOSTIC_CODE: 1}
        and summary.get("missingCapabilityCounts")
        == {FFT_OPENGL_EXPECTED_MISSING_CAPABILITY: 1},
        "OpenGL FFT workgroup-pointer diagnostic summary changed",
    )

    diagnostics = payload.get("diagnostics", [])
    _require(
        isinstance(diagnostics, list) and len(diagnostics) == 1,
        "OpenGL FFT report must contain one workgroup-pointer diagnostic",
    )
    diagnostic = diagnostics[0]
    _require(
        isinstance(diagnostic, Mapping)
        and diagnostic.get("severity") == "error"
        and diagnostic.get("code") == FFT_OPENGL_EXPECTED_DIAGNOSTIC_CODE
        and diagnostic.get("message") == FFT_OPENGL_EXPECTED_MESSAGE
        and diagnostic.get("sourceBackend") == "metal"
        and diagnostic.get("target") == "opengl"
        and diagnostic.get("missingCapabilities")
        == [FFT_OPENGL_EXPECTED_MISSING_CAPABILITY],
        "OpenGL FFT workgroup-pointer diagnostic contract changed",
    )
    location = diagnostic.get("location", {})
    _require(
        isinstance(location, Mapping) and location.get("file") == MLX_FFT_SOURCE,
        "OpenGL FFT diagnostic source changed",
    )
    details = diagnostic.get("details", {})
    _require(
        isinstance(details, Mapping)
        and details.get("sourcePath") == MLX_FFT_SOURCE
        and details.get("workgroupPointer") == FFT_OPENGL_EXPECTED_POINTER_EVIDENCE,
        "OpenGL FFT diagnostic lost concrete workgroup-pointer provenance",
    )

    artifacts = payload.get("artifacts", [])
    _require(
        isinstance(artifacts, list) and len(artifacts) == 1,
        "OpenGL FFT report must contain one failed artifact record",
    )
    artifact = artifacts[0]
    artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
    _require(
        isinstance(artifact, Mapping)
        and artifact.get("source") == MLX_FFT_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "opengl"
        and artifact.get("status") == "failed"
        and artifact.get("error") == FFT_OPENGL_EXPECTED_MESSAGE
        and isinstance(artifact_path, str),
        "OpenGL FFT failed artifact contract changed",
    )
    _require(
        details.get("targetArtifact") == artifact_path,
        "OpenGL FFT diagnostic and artifact paths disagree",
    )
    _require(
        artifact.get("sourceHash") == {"algorithm": "sha256", "value": MLX_FFT_SHA256}
        and artifact.get("sourceSizeBytes") == MLX_FFT_SOURCE_SIZE_BYTES,
        "OpenGL FFT source identity changed at the pinned MLX commit",
    )
    specialization_constants = artifact.get("specializationConstants", [])
    materialization = artifact.get("templateMaterialization", {})
    _require(
        isinstance(specialization_constants, list)
        and len(specialization_constants)
        == FFT_OPENGL_EXPECTED_FUNCTION_CONSTANT_COUNT,
        "OpenGL FFT function-constant inventory changed",
    )
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and len(materialization.get("specializations", []))
        == FFT_OPENGL_EXPECTED_SPECIALIZATION_COUNT
        and materialization.get("unsupported") == [],
        "OpenGL FFT template materialization evidence changed",
    )

    generated_path = (mlx_root / artifact_path).resolve()
    _require(
        _is_relative_to(generated_path, output_dir.resolve()),
        "OpenGL FFT artifact path escaped its output directory",
    )
    _require(
        not generated_path.exists(),
        "OpenGL FFT emitted GLSL despite its fail-closed project diagnostic",
    )

    return {
        "name": "fft-opengl-workgroup-pointer-frontier",
        "status": "blocked-as-expected",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_FFT_SOURCE,
        "sourceHash": MLX_FFT_SHA256,
        "target": "opengl",
        "artifactStatus": "failed",
        "artifactEmitted": False,
        "nativeValidationAttempted": False,
        "nativeValidationStatus": "not-run-no-artifact",
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": FFT_OPENGL_EXPECTED_SPECIALIZATION_COUNT,
        "functionConstantCount": FFT_OPENGL_EXPECTED_FUNCTION_CONSTANT_COUNT,
        "workgroupPointer": dict(FFT_OPENGL_EXPECTED_POINTER_EVIDENCE),
        "provenanceStatus": "concrete-backing-preserved",
        "accessRangeStatus": "unprovable",
        "trackedIssue": FFT_OPENGL_WORKGROUP_POINTER_ISSUE,
        "maxTemplateSpecializations": FFT_OPENGL_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": FFT_OPENGL_MAX_TEMPLATE_MATERIALIZATION_WORK,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
    }


def _dxc_diagnostics(result: CommandResult) -> list[dict[str, Any]]:
    header_pattern = re.compile(
        r"^(?P<path>.+):(?P<line>\d+):(?P<column>\d+): "
        r"(?P<severity>warning|error|note): (?P<message>.+)$",
        re.IGNORECASE,
    )
    diagnostics: list[dict[str, Any]] = []
    for stream_path in (result.stdout_path, result.stderr_path):
        lines = stream_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if re.search(r"\b(?:warning|error|note):", line, re.IGNORECASE) is None:
                continue
            match = header_pattern.match(line)
            _require(
                match is not None,
                f"DXC emitted an unrecognized diagnostic: {line}",
            )
            diagnostics.append(
                {
                    "severity": match.group("severity").lower(),
                    "message": match.group("message"),
                    "sourceLine": (
                        lines[index + 1].strip() if index + 1 < len(lines) else ""
                    ),
                }
            )
    return diagnostics


def _bare_value_discard_statements(source: str) -> list[str]:
    pattern = re.compile(
        r"(?m)^[ \t]*(?P<expression>"
        r"(?:[A-Za-z_]\w*(?:\s*(?:\.\s*[A-Za-z_]\w*|\[[^\]\r\n]+\]))*)"
        r"|(?:true|false)|(?:\d+(?:\.\d+)?[fFuUlL]*))"
        r"[ \t]*;[ \t]*(?://[^\r\n]*)?$"
    )
    control_flow_statements = {"break", "continue", "discard", "return"}
    return [
        expression
        for match in pattern.finditer(source)
        if (expression := match.group("expression").replace(" ", ""))
        not in control_flow_statements
    ]


def _require_gemv_project_workgroup_size_rule(
    payload: Mapping[str, Any], *, target: str
) -> None:
    project = payload.get("project")
    expected_rules = {
        MLX_GEMV_SOURCE: list(GEMV_REPORT_WORKGROUP_SIZE_RULE),
    }
    _require(
        isinstance(project, Mapping)
        and project.get("workgroupSizeRules") == expected_rules
        and project.get("workgroupSizeRuleCount") == 1,
        f"{target} GEMV report did not retain the exact workgroup-size rule",
    )


def _is_sha256_contract_identity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("algorithm") == "sha256"
        and isinstance(value.get("value"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["value"]) is not None
    )


def _gemv_directx_execution_evidence(
    artifact: Mapping[str, Any],
    specializations: Sequence[Any],
    compute_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    host_materializations: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in specializations:
        _require(
            isinstance(record, Mapping),
            "DirectX GEMV materialization records must be objects",
        )
        if "hostName" not in record:
            continue
        host_name = record.get("hostName")
        materialized_name = record.get("materializedName")
        _require(
            isinstance(host_name, str)
            and bool(host_name)
            and isinstance(materialized_name, str)
            and bool(materialized_name),
            "DirectX GEMV host materialization identity is incomplete",
        )
        identity = (host_name, materialized_name)
        _require(
            identity not in host_materializations,
            "DirectX GEMV host materialization identities must be unique",
        )
        host_materializations[identity] = record
    _require(
        len(host_materializations) == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "DirectX GEMV must retain exactly 224 host-named materializations",
    )

    execution = artifact.get("execution")
    execution_entries = (
        execution.get("entryPoints") if isinstance(execution, Mapping) else None
    )
    _require(
        isinstance(execution_entries, list)
        and len(execution_entries) == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "DirectX GEMV report must contain exactly 224 execution entries",
    )
    rule_path = f"project.workgroup_size_rules[{json.dumps(MLX_GEMV_SOURCE)}]"
    _require(
        execution.get("provenance")
        == {
            "kind": "materialized-template-rule",
            "path": rule_path,
        }
        and _is_sha256_contract_identity(execution.get("identity")),
        "DirectX GEMV execution provenance or identity changed",
    )

    report_entries_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    report_entries_by_target: dict[str, Mapping[str, Any]] = {}
    report_workgroup_size_counts: Counter[tuple[int, int, int]] = Counter()
    for entry in execution_entries:
        _require(
            isinstance(entry, Mapping),
            "DirectX GEMV execution entries must be objects",
        )
        source_entry = entry.get("sourceEntryPoint")
        materialized_entry = entry.get("materializedEntryPoint")
        target_entry = entry.get("targetEntryPoint")
        _require(
            isinstance(source_entry, str)
            and bool(source_entry)
            and isinstance(materialized_entry, str)
            and bool(materialized_entry)
            and isinstance(target_entry, str)
            and bool(target_entry),
            "DirectX GEMV execution entry identity is incomplete",
        )
        materialization_identity = (source_entry, materialized_entry)
        _require(
            materialization_identity not in report_entries_by_identity,
            "DirectX GEMV execution materialization identities must be unique",
        )
        _require(
            target_entry not in report_entries_by_target,
            "DirectX GEMV execution target identities must be unique",
        )
        host_record = host_materializations.get(materialization_identity)
        _require(
            host_record is not None,
            "DirectX GEMV execution entry has no matching host/materialized identity",
        )
        entry_materialization = entry.get("materialization")
        _require(
            isinstance(entry_materialization, Mapping)
            and entry_materialization.get("hostName") == source_entry
            and entry_materialization.get("materializedName") == materialized_entry
            and entry_materialization.get("name") == host_record.get("name"),
            "DirectX GEMV execution entry materialization identity changed",
        )
        parameters = entry.get("parameters")
        parameter_sources = entry.get("parameterSources")
        _require(
            isinstance(parameters, Mapping)
            and parameters == host_record.get("parameters")
            and isinstance(parameter_sources, Mapping)
            and parameter_sources == host_record.get("parameterSources"),
            "DirectX GEMV execution parameters lost materialization provenance",
        )
        rule = entry.get("rule")
        _require(
            isinstance(rule, Mapping)
            and rule.get("components") == list(GEMV_REPORT_WORKGROUP_SIZE_RULE)
            and rule.get("sourcePattern") == MLX_GEMV_SOURCE
            and rule.get("path") == rule_path,
            "DirectX GEMV execution entry workgroup-size rule changed",
        )
        try:
            resolved_size = (
                32,
                int(str(parameters["BN"])),
                int(str(parameters["BM"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PortingCheckError(
                "DirectX GEMV execution entry lacks integral BN/BM parameters"
            ) from exc
        _require(
            entry.get("workgroupSize") == list(resolved_size)
            and _is_sha256_contract_identity(entry.get("identity")),
            "DirectX GEMV execution entry workgroup-size contract changed",
        )
        report_entries_by_identity[materialization_identity] = entry
        report_entries_by_target[target_entry] = entry
        report_workgroup_size_counts[resolved_size] += 1

    _require(
        set(report_entries_by_identity) == set(host_materializations),
        "DirectX GEMV execution entries must join every host materialization by identity",
    )
    source_entry_points = execution.get("sourceEntryPoints")
    _require(
        isinstance(source_entry_points, list)
        and len(source_entry_points) == GEMV_EXPECTED_ENTRY_POINT_COUNT
        and len(set(source_entry_points)) == GEMV_EXPECTED_ENTRY_POINT_COUNT
        and set(source_entry_points)
        == {identity[0] for identity in report_entries_by_identity},
        "DirectX GEMV execution source identities changed",
    )

    generated_entries_by_target = {
        str(entry["targetEntryPoint"]): entry for entry in compute_entries
    }
    _require(
        len(generated_entries_by_target) == GEMV_EXPECTED_ENTRY_POINT_COUNT
        and set(generated_entries_by_target) == set(report_entries_by_target),
        "DirectX GEMV generated target entries do not match report identities",
    )
    for target_entry, report_entry in report_entries_by_target.items():
        generated_size = generated_entries_by_target[target_entry]["workgroupSize"]
        _require(
            generated_size == tuple(report_entry["workgroupSize"]),
            "DirectX GEMV generated numthreads declaration does not match its "
            f"report entry contract: {target_entry}",
        )

    resolved_sizes = tuple(sorted(report_workgroup_size_counts))
    _require(
        resolved_sizes == GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES,
        "DirectX GEMV resolved workgroup-size set changed",
    )
    return {
        "hostNamedMaterializationCount": len(host_materializations),
        "reportExecutionEntryCount": len(report_entries_by_identity),
        "executionIdentityJoinCount": len(report_entries_by_identity),
        "generatedTargetEntryIdentityCount": len(generated_entries_by_target),
        "generatedNumthreadsContractCount": len(generated_entries_by_target),
        "resolvedWorkgroupSizes": [list(size) for size in resolved_sizes],
        "resolvedWorkgroupSizeCounts": [
            {
                "workgroupSize": list(size),
                "entryCount": report_workgroup_size_counts[size],
            }
            for size in resolved_sizes
        ],
    }


def _check_gemv_directx_compiler_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    dxc = shutil.which("dxc")
    _require(dxc is not None, "DirectX GEMV compiler validation requires dxc")

    source_path = mlx_root / MLX_GEMV_SOURCE
    _require(source_path.is_file(), f"pinned MLX GEMV source is missing: {source_path}")
    _require(
        source_path.stat().st_size == MLX_GEMV_SOURCE_SIZE_BYTES
        and _sha256(source_path) == MLX_GEMV_SHA256,
        "DirectX GEMV source identity changed at the pinned MLX commit",
    )

    config_path = config_dir / "gemv-directx-compiler-frontier.toml"
    report_path = report_dir / "gemv-directx-compiler-frontier.json"
    output_dir = work_dir / "out-gemv-directx-compiler-frontier"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    report_path.unlink(missing_ok=True)
    _write_project_config(
        config_path,
        include=MLX_GEMV_SOURCE,
        targets=("directx",),
        output_dir=_relpath(output_dir, mlx_root),
        workgroup_size_rules={
            MLX_GEMV_SOURCE: GEMV_WORKGROUP_SIZE_RULE,
        },
        metal_source_options={
            "max_template_specializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
    )
    translation = _run_command(
        "translate-gemv-directx-compiler-frontier",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--no-format",
        ],
        log_dir=log_dir,
        check=False,
        timeout_seconds=GEMV_DIRECTX_TRANSLATION_TIMEOUT_SECONDS,
    )
    _require(
        translation.returncode == 0,
        "DirectX GEMV project translation failed; inspect the translation logs",
    )
    _require(
        report_path.is_file(),
        "DirectX GEMV translation did not produce a project report",
    )

    payload = _load_json(report_path)
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        "DirectX GEMV translation report kind changed",
    )
    _require_gemv_project_workgroup_size_rule(payload, target="DirectX")
    summary = payload.get("summary", {})
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("skippedCount") == 0
        and summary.get("targetCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0,
        "DirectX GEMV report did not retain one clean translated artifact",
    )
    _require(
        summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0}
        and summary.get("diagnosticsByCode") == {}
        and summary.get("missingCapabilityCounts") == {}
        and payload.get("diagnostics") == [],
        "DirectX GEMV project translation emitted an unexpected diagnostic",
    )
    _require(
        summary.get("artifactProvenanceByPipeline") == {"single-file-translate": 1}
        and summary.get("artifactProvenanceByIntermediate") == {"crossgl": 1},
        "DirectX GEMV report provenance summary changed",
    )

    expected_source_hash = {"algorithm": "sha256", "value": MLX_GEMV_SHA256}
    units = payload.get("units", [])
    _require(
        isinstance(units, list)
        and len(units) == 1
        and isinstance(units[0], Mapping)
        and units[0].get("id") == MLX_GEMV_SOURCE
        and units[0].get("path") == MLX_GEMV_SOURCE
        and units[0].get("sourceBackend") == "metal"
        and units[0].get("extension") == ".metal"
        and units[0].get("sourceHash") == expected_source_hash
        and units[0].get("sourceSizeBytes") == MLX_GEMV_SOURCE_SIZE_BYTES,
        "DirectX GEMV source-unit provenance changed at the pinned MLX commit",
    )

    artifacts = payload.get("artifacts", [])
    _require(
        isinstance(artifacts, list)
        and len(artifacts) == 1
        and isinstance(artifacts[0], Mapping),
        "DirectX GEMV report must contain one artifact record",
    )
    artifact = artifacts[0]
    artifact_path = artifact.get("path")
    _require(
        artifact.get("source") == MLX_GEMV_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "directx"
        and artifact.get("status") == "translated"
        and isinstance(artifact_path, str),
        "DirectX GEMV translated artifact contract changed",
    )
    _require(
        artifact.get("sourceHash") == expected_source_hash
        and artifact.get("sourceSizeBytes") == MLX_GEMV_SOURCE_SIZE_BYTES
        and artifact.get("provenance")
        == {"intermediate": "crossgl", "pipeline": "single-file-translate"},
        "DirectX GEMV artifact provenance changed",
    )

    generated_path = (mlx_root / artifact_path).resolve()
    _require(
        _is_relative_to(generated_path, output_dir.resolve()),
        "DirectX GEMV artifact path escaped its output directory",
    )
    _require(
        generated_path.is_file(),
        f"DirectX GEMV artifact is missing: {artifact_path}",
    )
    generated_hash = _sha256(generated_path)
    generated_size = generated_path.stat().st_size
    _require(
        artifact.get("generatedHash")
        == {"algorithm": "sha256", "value": generated_hash}
        and artifact.get("generatedSizeBytes") == generated_size
        and generated_size > 0,
        "DirectX GEMV artifact hash or size does not match the emitted file",
    )

    materialization = artifact.get("templateMaterialization", {})
    specializations = (
        materialization.get("specializations", [])
        if isinstance(materialization, Mapping)
        else []
    )
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("specializationCount")
        == GEMV_EXPECTED_SPECIALIZATION_COUNT
        and isinstance(specializations, list)
        and len(specializations) == GEMV_EXPECTED_SPECIALIZATION_COUNT
        and materialization.get("unsupported") == [],
        "DirectX GEMV artifact did not retain complete materialization evidence",
    )

    generated = generated_path.read_text(encoding="utf-8")
    compute_entry_pattern = re.compile(
        r"(?m)^[ \t]*(?P<source_expression>"
        r"\[numthreads\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*,\s*"
        r"(?P<z>\d+)\s*\)\])[ \t]*\r?\n"
        r"[ \t]*void[ \t]+(?P<target_entry>CSMain(?:_\d+)?)[ \t]*\("
    )
    compute_entries = [
        {
            "sourceExpression": match.group("source_expression"),
            "workgroupSize": tuple(
                int(match.group(component)) for component in ("x", "y", "z")
            ),
            "targetEntryPoint": match.group("target_entry"),
        }
        for match in compute_entry_pattern.finditer(generated)
    ]
    entry_points = [str(entry["targetEntryPoint"]) for entry in compute_entries]
    _require(
        len(compute_entries) == GEMV_EXPECTED_ENTRY_POINT_COUNT
        and len(set(entry_points)) == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "DirectX GEMV artifact compute-entry count changed",
    )
    _require(
        all(
            entry_point in entry_points
            for entry_point in GEMV_DIRECTX_COMPILER_ENTRY_POINTS
        ),
        "DirectX GEMV compiler-frontier entry selection changed",
    )
    _require(
        entry_points == list(GEMV_DIRECTX_EXPECTED_ENTRY_POINTS),
        "DirectX GEMV artifact export set changed",
    )
    execution_evidence = _gemv_directx_execution_evidence(
        artifact,
        specializations,
        compute_entries,
    )
    residue = re.search(
        r"BinaryOpNode|IdentifierNode|LiteralNode|PrimitiveType|"
        r"\b(?:acc_type|nullptr)\b|(?:^|\s)WARNING:",
        generated,
        re.MULTILINE,
    )
    _require(
        residue is None,
        "DirectX GEMV artifact retained unresolved materialization text: "
        f"{residue.group(0) if residue else ''}",
    )
    bare_value_discards = _bare_value_discard_statements(generated)
    _require(
        not bare_value_discards,
        "DirectX GEMV artifact retained a bare value-discard statement: "
        f"{bare_value_discards[0] if bare_value_discards else ''};",
    )

    compile_dir = work_dir / "validation" / "gemv-directx"
    compile_dir.mkdir(parents=True, exist_ok=True)
    entry_profile_runs = []
    for entry_point in GEMV_DIRECTX_COMPILER_ENTRY_POINTS:
        output_path = compile_dir / f"{entry_point}.dxil"
        output_path.unlink(missing_ok=True)
        command_name = entry_point.lower().replace("_", "-")
        compile_result = _run_command(
            f"compile-gemv-directx-{command_name}",
            [
                dxc,
                "-T",
                GEMV_DIRECTX_ENTRY_PROFILE,
                "-E",
                entry_point,
                str(generated_path),
                "-Fo",
                str(output_path),
            ],
            log_dir=log_dir,
            check=False,
        )
        _require(
            compile_result.returncode == 0,
            f"DXC failed to compile DirectX GEMV entry {entry_point}",
        )
        _require(
            output_path.is_file() and output_path.stat().st_size > 0,
            f"DXC did not emit a binary for DirectX GEMV entry {entry_point}",
        )
        diagnostics = _dxc_diagnostics(compile_result)
        _require(
            not diagnostics,
            f"DXC diagnostics changed for DirectX GEMV entry {entry_point}",
        )
        entry_profile_runs.append(
            {
                "entryPoint": entry_point,
                "profile": GEMV_DIRECTX_ENTRY_PROFILE,
                "status": "compiled",
                "output": _relpath(output_path, mlx_root),
                "outputHash": {
                    "algorithm": "sha256",
                    "value": _sha256(output_path),
                },
                "outputSizeBytes": output_path.stat().st_size,
                "diagnosticCount": 0,
                "unusedValueWarningCount": 0,
                "stdout": _relpath(compile_result.stdout_path, mlx_root),
                "stderr": _relpath(compile_result.stderr_path, mlx_root),
            }
        )

    library_exports = ";".join(GEMV_DIRECTX_EXPECTED_ENTRY_POINTS)
    library_output_path = compile_dir / "all-entries.dxil"
    library_output_path.unlink(missing_ok=True)
    library_result = _run_command(
        "compile-gemv-directx-all-entries",
        [
            dxc,
            "-T",
            GEMV_DIRECTX_LIBRARY_PROFILE,
            "-exports",
            library_exports,
            str(generated_path),
            "-Fo",
            str(library_output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        library_result.returncode == 0,
        "DXC failed to compile the DirectX GEMV all-entry library",
    )
    _require(
        library_output_path.is_file() and library_output_path.stat().st_size > 0,
        "DXC did not emit the DirectX GEMV all-entry library",
    )
    library_diagnostics = _dxc_diagnostics(library_result)
    library_diagnostic_counts = Counter(
        (
            diagnostic.get("severity"),
            diagnostic.get("message"),
            diagnostic.get("sourceLine"),
        )
        for diagnostic in library_diagnostics
    )
    expected_library_diagnostic_counts = Counter(
        (
            "warning",
            GEMV_DIRECTX_LIBRARY_NUMTHREADS_WARNING_MESSAGE,
            str(entry["sourceExpression"]),
        )
        for entry in compute_entries
    )
    _require(
        library_diagnostic_counts == expected_library_diagnostic_counts,
        "DXC diagnostics changed for the DirectX GEMV all-entry library",
    )
    library_output_hash = _sha256(library_output_path)
    library_output_size = library_output_path.stat().st_size
    warning_source_counts = Counter(
        str(entry["sourceExpression"]) for entry in compute_entries
    )
    source_expression_by_size = {
        tuple(entry["workgroupSize"]): str(entry["sourceExpression"])
        for entry in compute_entries
    }
    library_allowed_warnings = [
        {
            "classification": "library-profile-numthreads-ignored",
            "severity": "warning",
            "message": GEMV_DIRECTX_LIBRARY_NUMTHREADS_WARNING_MESSAGE,
            "sourceExpression": source_expression_by_size[size],
            "count": warning_source_counts[source_expression_by_size[size]],
        }
        for size in GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]

    return {
        "name": "gemv-directx-compiler-frontier",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_GEMV_SOURCE,
        "sourceHash": MLX_GEMV_SHA256,
        "sourceSizeBytes": MLX_GEMV_SOURCE_SIZE_BYTES,
        "target": "directx",
        "artifactStatus": "translated",
        "artifactPackaging": "single-aggregate-artifact",
        "artifact": _relpath(generated_path, mlx_root),
        "artifactHash": {"algorithm": "sha256", "value": generated_hash},
        "artifactSizeBytes": generated_size,
        "artifactProvenance": {
            "intermediate": "crossgl",
            "pipeline": "single-file-translate",
        },
        "projectDiagnosticCount": 0,
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "unsupportedSpecializationCount": 0,
        "workgroupSizeRule": list(GEMV_WORKGROUP_SIZE_RULE),
        "reportWorkgroupSizeRule": list(GEMV_REPORT_WORKGROUP_SIZE_RULE),
        "reportWorkgroupSizeRuleCount": 1,
        **execution_evidence,
        "materializationResidueCount": 0,
        "bareValueDiscardCount": 0,
        "computeEntryPointCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "compiler": "dxc",
        "entryProfile": GEMV_DIRECTX_ENTRY_PROFILE,
        "entryProfileCompilerEntryPoints": list(GEMV_DIRECTX_COMPILER_ENTRY_POINTS),
        "entryProfileCompiledEntryPointCount": len(entry_profile_runs),
        "entryProfileCompilerRuns": entry_profile_runs,
        "entryProfileDiagnosticCount": 0,
        "entryProfileUnusedValueWarningCount": 0,
        "libraryProfile": GEMV_DIRECTX_LIBRARY_PROFILE,
        "libraryExports": list(GEMV_DIRECTX_EXPECTED_ENTRY_POINTS),
        "libraryExportCount": len(GEMV_DIRECTX_EXPECTED_ENTRY_POINTS),
        "libraryExportSetHash": {
            "algorithm": "sha256",
            "value": hashlib.sha256(library_exports.encode("utf-8")).hexdigest(),
        },
        "libraryCompilerRun": {
            "profile": GEMV_DIRECTX_LIBRARY_PROFILE,
            "status": "compiled",
            "output": _relpath(library_output_path, mlx_root),
            "outputHash": {"algorithm": "sha256", "value": library_output_hash},
            "outputSizeBytes": library_output_size,
            "allowedWarningCounts": {
                "libraryNumthreads": GEMV_DIRECTX_EXPECTED_LIBRARY_WARNING_COUNT,
            },
            "unusedValueWarningCount": 0,
            "stdout": _relpath(library_result.stdout_path, mlx_root),
            "stderr": _relpath(library_result.stderr_path, mlx_root),
        },
        "libraryAllowedWarnings": library_allowed_warnings,
        "libraryUnusedValueWarningCount": 0,
        "compilerCoveredEntryPointCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "uncompiledEntryPointCount": 0,
        "compilerCoverageStatus": "all-exported-entry-points-compiled",
        "libraryCodeGenerationScope": "all-exported-functions",
        "wholeArtifactSemanticValidityClaimed": False,
        "libraryExecutionSemanticsEstablished": False,
        "observedNumthreadsDirectives": [
            source_expression_by_size[size]
            for size in GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
        ],
        "numthreadsDirectiveCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "numthreadsContractEstablished": True,
        "exactWorkgroupSizeEstablished": True,
        "requiredWaveSize": 32,
        "requiredWaveSizeEstablished": False,
        "executionContractBlockedBy": list(GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES),
        "runtimeExecutionAttempted": False,
        "runtimeIntegrationIncluded": False,
        "numericalParityClaimed": False,
        "runtimeParityClaimed": False,
    }


def _check_gemv_opengl_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "gemv-opengl-frontier.toml"
    report_path = report_dir / "gemv-opengl-frontier.json"
    output_dir = work_dir / "out-gemv-opengl-frontier"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    report_path.unlink(missing_ok=True)
    _write_project_config(
        config_path,
        include=MLX_GEMV_SOURCE,
        targets=("opengl",),
        output_dir=_relpath(output_dir, mlx_root),
        workgroup_size_rules={
            MLX_GEMV_SOURCE: GEMV_WORKGROUP_SIZE_RULE,
        },
        metal_source_options={
            "max_template_specializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
    )
    result = _run_command(
        "translate-gemv-opengl-frontier",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--no-format",
        ],
        log_dir=log_dir,
        check=False,
        timeout_seconds=GEMV_OPENGL_TRANSLATION_TIMEOUT_SECONDS,
    )
    _require(
        result.returncode == 1,
        "OpenGL GEMV translation must remain fail-closed at its pinned "
        "workgroup-pointer frontier",
    )
    _require(
        report_path.is_file(),
        "OpenGL GEMV translation did not produce a project report",
    )

    payload = _load_json(report_path)
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        "OpenGL GEMV translation report kind changed",
    )
    _require_gemv_project_workgroup_size_rule(payload, target="OpenGL")
    summary = payload.get("summary", {})
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("skippedCount") == 0
        and summary.get("targetCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 0
        and summary.get("failedCount") == 1,
        "OpenGL GEMV translation report did not retain one failed unit/artifact",
    )
    _require(
        summary.get("diagnosticCounts") == {"error": 1, "note": 0, "warning": 0}
        and summary.get("diagnosticsByCode")
        == {GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE: 1}
        and summary.get("missingCapabilityCounts")
        == {GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY: 1},
        "OpenGL GEMV workgroup-pointer diagnostic summary changed",
    )
    _require(
        summary.get("artifactProvenanceByPipeline") == {"single-file-translate": 1}
        and summary.get("artifactProvenanceByIntermediate") == {"crossgl": 1},
        "OpenGL GEMV report provenance summary changed",
    )

    units = payload.get("units", [])
    _require(
        isinstance(units, list) and len(units) == 1,
        "OpenGL GEMV report must contain one source unit",
    )
    unit = units[0]
    expected_source_hash = {"algorithm": "sha256", "value": MLX_GEMV_SHA256}
    _require(
        isinstance(unit, Mapping)
        and unit.get("id") == MLX_GEMV_SOURCE
        and unit.get("path") == MLX_GEMV_SOURCE
        and unit.get("sourceBackend") == "metal"
        and unit.get("extension") == ".metal"
        and unit.get("sourceHash") == expected_source_hash
        and unit.get("sourceSizeBytes") == MLX_GEMV_SOURCE_SIZE_BYTES,
        "OpenGL GEMV source-unit provenance changed at the pinned MLX commit",
    )

    diagnostics = payload.get("diagnostics", [])
    _require(
        isinstance(diagnostics, list) and len(diagnostics) == 1,
        "OpenGL GEMV report must contain one workgroup-pointer diagnostic",
    )
    diagnostic = diagnostics[0]
    _require(
        isinstance(diagnostic, Mapping)
        and diagnostic.get("severity") == "error"
        and diagnostic.get("code") == GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE
        and diagnostic.get("message") == GEMV_OPENGL_EXPECTED_MESSAGE
        and diagnostic.get("sourceBackend") == "metal"
        and diagnostic.get("target") == "opengl"
        and diagnostic.get("missingCapabilities")
        == [GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY],
        "OpenGL GEMV workgroup-pointer diagnostic contract changed",
    )
    location = diagnostic.get("location", {})
    _require(
        isinstance(location, Mapping) and location.get("file") == MLX_GEMV_SOURCE,
        "OpenGL GEMV diagnostic source changed",
    )
    details = diagnostic.get("details", {})
    _require(isinstance(details, Mapping), "OpenGL GEMV diagnostic details are missing")

    artifacts = payload.get("artifacts", [])
    _require(
        isinstance(artifacts, list) and len(artifacts) == 1,
        "OpenGL GEMV report must contain one failed artifact record",
    )
    artifact = artifacts[0]
    artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
    _require(
        isinstance(artifact, Mapping)
        and artifact.get("source") == MLX_GEMV_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "opengl"
        and artifact.get("status") == "failed"
        and artifact.get("error") == GEMV_OPENGL_EXPECTED_MESSAGE
        and isinstance(artifact_path, str),
        "OpenGL GEMV failed artifact contract changed",
    )
    _require(
        details
        == {
            "sourcePath": MLX_GEMV_SOURCE,
            "targetArtifact": artifact_path,
            "workgroupPointer": GEMV_OPENGL_EXPECTED_POINTER_EVIDENCE,
        },
        "OpenGL GEMV diagnostic lost exact workgroup-pointer provenance",
    )
    _require(
        artifact.get("sourceHash") == expected_source_hash
        and artifact.get("sourceSizeBytes") == MLX_GEMV_SOURCE_SIZE_BYTES
        and artifact.get("provenance")
        == {"intermediate": "crossgl", "pipeline": "single-file-translate"},
        "OpenGL GEMV failed artifact provenance changed",
    )

    materialization = artifact.get("templateMaterialization", {})
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("specializationCount")
        == GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "OpenGL GEMV artifact specialization count changed",
    )
    specializations = materialization.get("specializations", [])
    _require(
        isinstance(specializations, list)
        and len(specializations) == GEMV_EXPECTED_SPECIALIZATION_COUNT
        and materialization.get("unsupported") == [],
        "OpenGL GEMV artifact did not retain complete materialization evidence",
    )

    generated_path = (mlx_root / artifact_path).resolve()
    _require(
        _is_relative_to(generated_path, output_dir.resolve()),
        "OpenGL GEMV artifact path escaped its output directory",
    )
    _require(
        not generated_path.exists(),
        "OpenGL GEMV emitted GLSL despite its fail-closed project diagnostic",
    )
    _require(
        not output_dir.exists()
        or not any(
            path.is_file() or path.is_symlink() for path in output_dir.rglob("*")
        ),
        "OpenGL GEMV emitted target files despite its fail-closed project diagnostic",
    )

    return {
        "name": "gemv-opengl-frontier",
        "status": "blocked-as-expected",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_GEMV_SOURCE,
        "sourceHash": MLX_GEMV_SHA256,
        "target": "opengl",
        "artifactStatus": "failed",
        "artifactEmitted": False,
        "emittedTargetFileCount": 0,
        "nativeValidationAttempted": False,
        "nativeValidationStatus": "not-run-no-artifact",
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "workgroupSizeRule": list(GEMV_WORKGROUP_SIZE_RULE),
        "reportWorkgroupSizeRule": list(GEMV_REPORT_WORKGROUP_SIZE_RULE),
        "reportWorkgroupSizeRuleCount": 1,
        "workgroupSizeRuleConfigured": True,
        "diagnosticCode": GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE,
        "missingCapability": GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY,
        "workgroupPointer": dict(GEMV_OPENGL_EXPECTED_POINTER_EVIDENCE),
        "provenanceStatus": "concrete-backing-preserved",
        "accessRangeStatus": "unprovable",
        "trackedIssues": list(GEMV_OPENGL_FRONTIER_TRACKED_ISSUES),
        "translationBlockedBy": [GEMV_OPENGL_BACKING_RANGE_ISSUE],
        "executionContractBlockedBy": list(GEMV_OPENGL_EXECUTION_TRACKED_ISSUES),
        "maxTemplateSpecializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        "runtimeExecutionAttempted": False,
        "runtimeIntegrationIncluded": False,
        "runnableArtifactClaimed": False,
        "numericalParityClaimed": False,
        "runtimeParityClaimed": False,
    }


def _check_gemv_vulkan_toolchain(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    required_tools = {
        "spirv-as": shutil.which("spirv-as"),
        "spirv-val": shutil.which("spirv-val"),
    }
    missing_tools = sorted(
        name for name, resolved in required_tools.items() if resolved is None
    )
    _require(
        not missing_tools,
        "Vulkan GEMV validation requires: " + ", ".join(missing_tools),
    )

    config_path = config_dir / "gemv-vulkan.toml"
    report_path = report_dir / "gemv-vulkan.json"
    _write_project_config(
        config_path,
        include=MLX_GEMV_SOURCE,
        targets=("vulkan",),
        output_dir=_relpath(work_dir / "out-gemv-vulkan", mlx_root),
        metal_source_options={
            "max_template_specializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
    )
    result = _run_command(
        "translate-gemv-vulkan",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--no-format",
        ],
        log_dir=log_dir,
        check=False,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, Mapping), "Vulkan GEMV summary must be an object")
    if result.returncode != 0:
        diagnostics = [
            item for item in payload.get("diagnostics", []) if isinstance(item, Mapping)
        ]
        messages = [
            str(item.get("message"))
            for item in diagnostics
            if isinstance(item.get("message"), str)
        ]
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"Vulkan GEMV translation failed{detail}")
    _require(
        summary.get("translatedCount") == 1 and summary.get("failedCount") == 0,
        "Vulkan GEMV report did not contain one clean translated artifact",
    )

    artifact = next(
        (
            item
            for item in payload.get("artifacts", [])
            if isinstance(item, Mapping)
            and item.get("source") == MLX_GEMV_SOURCE
            and item.get("target") == "vulkan"
            and item.get("status") == "translated"
        ),
        None,
    )
    _require(isinstance(artifact, Mapping), "Vulkan GEMV artifact is missing")
    artifact_path = artifact.get("path")
    _require(isinstance(artifact_path, str), "Vulkan GEMV artifact path is missing")
    generated_path = mlx_root / artifact_path
    _require(
        generated_path.is_file(),
        f"Vulkan GEMV artifact is missing: {artifact_path}",
    )

    materialization = artifact.get("templateMaterialization", {})
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("specializationCount")
        == GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "Vulkan GEMV artifact did not materialize the complete specialization set",
    )
    generated = generated_path.read_text(encoding="utf-8")
    entry_point_count = len(
        re.findall(r"(?m)^[ \t]*OpEntryPoint[ \t]+GLCompute\b", generated)
    )
    _require(
        entry_point_count == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "Vulkan GEMV artifact did not emit the complete entry-point set",
    )

    warning_lines = [
        line
        for line in generated.splitlines()
        if line.lstrip().startswith("; WARNING:")
    ]
    _require(
        not warning_lines,
        "Vulkan GEMV artifact emitted a semantic warning: "
        + (warning_lines[0] if warning_lines else ""),
    )
    generated_without_warnings = "\n".join(
        line
        for line in generated.splitlines()
        if not line.lstrip().startswith("; WARNING:")
    )
    residue = re.search(
        r"BinaryOpNode|IdentifierNode|LiteralNode|PrimitiveType|\b(?:acc_type|nullptr)\b",
        generated_without_warnings,
    )
    _require(
        residue is None,
        "Vulkan GEMV artifact retained unresolved materialization text outside "
        f"tracked warnings: {residue.group(0) if residue else ''}",
    )

    output_path = work_dir / "validation" / "gemv-vulkan.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assembly_result = _run_command(
        "assemble-gemv-vulkan",
        [
            str(required_tools["spirv-as"]),
            "--target-env",
            "vulkan1.1",
            str(generated_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        assembly_result.returncode == 0,
        "Vulkan GEMV assembly failed; inspect assemble-gemv-vulkan logs",
    )
    _require(
        output_path.is_file(),
        "Vulkan GEMV assembly succeeded without producing SPIR-V",
    )
    validation_result = _run_command(
        "validate-gemv-vulkan-spirv",
        [
            str(required_tools["spirv-val"]),
            "--target-env",
            "vulkan1.1",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        validation_result.returncode == 0,
        "Vulkan GEMV SPIR-V validation failed; inspect "
        "validate-gemv-vulkan-spirv logs",
    )
    tool_warning_lines = [
        line
        for command_result in (assembly_result, validation_result)
        for path in (command_result.stdout_path, command_result.stderr_path)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "warning:" in line.lower()
    ]
    _require(
        not tool_warning_lines,
        "Vulkan GEMV SPIR-V tools emitted a warning: "
        + (tool_warning_lines[0] if tool_warning_lines else ""),
    )

    diagnostic_counts = summary.get("diagnosticCounts", {})
    report_warning_count = (
        diagnostic_counts.get("warning", 0)
        if isinstance(diagnostic_counts, Mapping)
        else 0
    )
    _require(
        report_warning_count in (0, len(warning_lines)),
        "Vulkan GEMV report warning count does not match artifact warnings",
    )
    return {
        "name": "gemv-vulkan-toolchain",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_GEMV_SOURCE,
        "target": "vulkan",
        "specializationCount": materialization.get("specializationCount"),
        "entryPointCount": entry_point_count,
        "structuralValidationStatus": "validated",
        "assembler": "spirv-as",
        "spirvValidator": "spirv-val",
        "semanticReadinessStatus": "no-known-codegen-fallbacks",
        "semanticWarningCount": len(warning_lines),
        "semanticWarningsByIssue": {},
        "semanticBlockers": list(VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES),
        "reportWarningCount": report_warning_count,
        "reportWarningTransportTrackedBy": None,
        "structuralValidationOutput": _relpath(output_path, mlx_root),
        "runtimeIntegrationIncluded": False,
    }


def _runtime_readiness_fixture(
    target: str, variant: str | None = None
) -> dict[str, Any]:
    default_variant = RUNTIME_READINESS_DEFAULT_VARIANTS.get(target)
    _require(
        default_variant is not None,
        f"runtime readiness variant is not configured for target: {target}",
    )
    variant = variant or default_variant
    variant_spec = ARANGE_RUNTIME_VARIANTS.get(variant)
    _require(
        variant_spec is not None,
        f"runtime readiness variant is not configured: {variant}",
    )
    if variant == default_variant:
        entry_point = RUNTIME_READINESS_ENTRY_POINTS.get(target)
    else:
        _require(
            target == "vulkan" and variant in VULKAN_ARANGE_RUNTIME_VARIANTS,
            f"runtime readiness variant {variant} is only configured for Vulkan",
        )
        entry_point = f"arange{variant}"
    _require(
        entry_point is not None,
        f"runtime readiness entry point is not configured for target: {target}",
    )
    fixture_id = f"mlx-arange-{target}-runtime-readiness"
    if variant != default_variant:
        fixture_id = f"mlx-arange-{target}-{variant}-runtime-readiness"
    return {
        "id": fixture_id,
        "selector": {
            "source": MLX_ARANGE_SOURCE,
            "target": target,
        },
        "entryPoint": entry_point,
        "inputs": [
            {
                "name": "start",
                "kind": "scalar",
                "dtype": variant,
                "value": variant_spec["start"],
            },
            {
                "name": "step",
                "kind": "scalar",
                "dtype": variant,
                "value": variant_spec["step"],
            },
        ],
        "expectedOutputs": [
            {
                "name": "out",
                "kind": "buffer",
                "dtype": variant,
                "shape": [4],
                "values": list(variant_spec["expected"]),
            }
        ],
        "runtimeAdapter": {
            "dispatch": {
                "globalSize": [4, 1, 1],
            }
        },
        "metadata": {
            "repository": "mlx",
            "source": MLX_ARANGE_SOURCE,
            "purpose": "runtime-readiness-metadata-probe",
        },
    }


def _runtime_readiness_fixtures(targets: Sequence[str]) -> list[dict[str, Any]]:
    fixtures = []
    for target in targets:
        variants = (
            VULKAN_ARANGE_RUNTIME_VARIANTS
            if target == "vulkan"
            else (RUNTIME_READINESS_DEFAULT_VARIANTS[target],)
        )
        fixtures.extend(
            _runtime_readiness_fixture(target, variant) for variant in variants
        )
    return fixtures


def _runtime_readiness_fixture_metadata(targets: Sequence[str]) -> dict[str, Any]:
    return {
        "kind": "crosstl-project-runtime-fixture-metadata",
        "metadata": {
            "repository": "mlx",
            "fixtureSet": "reduced-arange-runtime-readiness",
            "scope": "artifact-execution-metadata-readiness",
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "trackedIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
        },
        "fixtures": _runtime_readiness_fixtures(targets),
    }


def _runtime_fixture_execution_adapter_id(target: str) -> str:
    return f"mlx-arange-reference-{target}"


def _runtime_fixture_execution_metadata(targets: Sequence[str]) -> dict[str, Any]:
    metadata = _runtime_readiness_fixture_metadata(targets)
    metadata["metadata"] = {
        **metadata["metadata"],
        "scope": "reference-runtime-fixture-execution",
        "runtimeFixtureExecutionIncluded": True,
    }
    metadata["adapters"] = [
        {
            "id": _runtime_fixture_execution_adapter_id(target),
            "executor": _runtime_fixture_execution_adapter_id(target),
            "adapterKind": RUNTIME_FIXTURE_EXECUTION_ADAPTER_KIND,
            "platformRequirements": {"requiredTools": []},
            "metadata": {
                "target": target,
                "scope": "reference-runtime-fixture-execution",
            },
        }
        for target in targets
    ]
    metadata["fixtures"] = [
        {
            **fixture,
            "adapter": _runtime_fixture_execution_adapter_id(
                str(fixture["selector"]["target"])
            ),
        }
        for fixture in metadata["fixtures"]
        if isinstance(fixture.get("selector"), Mapping)
    ]
    return metadata


def _native_runtime_execution_adapter_id(target: str) -> str:
    return f"mlx-arange-native-{target}"


def _native_runtime_execution_metadata(targets: Sequence[str]) -> dict[str, Any]:
    metadata = _runtime_readiness_fixture_metadata(targets)
    metadata["metadata"] = {
        **metadata["metadata"],
        "scope": NATIVE_RUNTIME_EXECUTION_SCOPE,
        "nativeRuntimeExecutionIncluded": True,
    }
    metadata["adapters"] = [
        {
            "id": _native_runtime_execution_adapter_id(target),
            "executor": target,
            "adapterKind": f"{target}-native-runtime",
            "platformRequirements": {"requiredTools": []},
            "metadata": {
                "target": target,
                "scope": NATIVE_RUNTIME_EXECUTION_SCOPE,
            },
        }
        for target in targets
    ]
    metadata["fixtures"] = [
        {
            **fixture,
            "adapter": _native_runtime_execution_adapter_id(
                str(fixture["selector"]["target"])
            ),
        }
        for fixture in metadata["fixtures"]
        if isinstance(fixture.get("selector"), Mapping)
    ]
    return metadata


class MlxArangeReferenceRuntime(RuntimeParityAdapter):
    """Reference executor for MLX reduced arange runtime fixtures."""

    name = RUNTIME_FIXTURE_EXECUTION_ADAPTER_KIND

    def __init__(self, target: str):
        self.target = target

    def prepare_buffers(self, state):
        prepared = dict(state.resource_values)
        for resource in state.plan.resource_bindings:
            value = resource.value
            if value is None or resource.source == "expectedOutput":
                continue
            prepared[value.name] = value.values
        return prepared

    def dispatch(self, state, prepared_buffers):
        start = _runtime_fixture_scalar(prepared_buffers.get("start"), default=0)
        step = _runtime_fixture_scalar(prepared_buffers.get("step"), default=1)
        output = state.request.fixture.expected_outputs[0]
        count = _runtime_fixture_output_count(state, output)
        return {output.name: [start + index * step for index in range(count)]}

    def collect_outputs(self, state, dispatch_result):
        outputs = {}
        for output in state.request.fixture.expected_outputs:
            values = dispatch_result.get(output.name, [])
            outputs[output.name] = {
                "dtype": output.dtype,
                "shape": list(output.shape),
                "values": values,
            }
        return outputs


def _runtime_fixture_scalar(value: Any, *, default: int | float) -> int | float:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return default
        value = value[0]
    if isinstance(value, float):
        return value
    return int(value)


def _runtime_fixture_output_count(state: Any, output: Any) -> int:
    if output.shape:
        return int(output.shape[0])
    if isinstance(output.values, Sequence) and not isinstance(
        output.values, (str, bytes, bytearray)
    ):
        return len(output.values)
    dispatch = state.plan.dispatch
    if dispatch is not None and dispatch.global_size:
        return int(dispatch.global_size[0])
    return 1


def _runtime_fixture_execution_executors(targets: Sequence[str]) -> dict[str, Any]:
    return {
        _runtime_fixture_execution_adapter_id(target): MlxArangeReferenceRuntime(target)
        for target in targets
    }


def _diagnostics_by_code(diagnostics: Sequence[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        code = diagnostic.get("code")
        if isinstance(code, str) and code:
            counts[code] += 1
    return dict(sorted(counts.items()))


def _runtime_plan_diagnostics(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics: list[Mapping[str, Any]] = []
    for test_case in plan.get("testCases", []):
        if not isinstance(test_case, Mapping):
            continue
        for diagnostic in test_case.get("diagnostics", []):
            if isinstance(diagnostic, Mapping):
                diagnostics.append(diagnostic)
    return diagnostics


def _runtime_report_diagnostics(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics: list[Mapping[str, Any]] = []
    for result in report.get("results", []):
        if not isinstance(result, Mapping):
            continue
        for diagnostic in result.get("diagnostics", []):
            if isinstance(diagnostic, Mapping):
                diagnostics.append(diagnostic)
    runtime_report = report.get("runtimeTestReport")
    if isinstance(runtime_report, Mapping):
        diagnostics.extend(_runtime_report_diagnostics(runtime_report))
    return diagnostics


def _runtime_report_results_for_target(
    runtime_report: Mapping[str, Any], target: str
) -> list[Mapping[str, Any]]:
    results = []
    for result in runtime_report.get("results", []):
        if not isinstance(result, Mapping):
            continue
        artifact = result.get("artifact")
        if isinstance(artifact, Mapping) and artifact.get("target") == target:
            results.append(result)
            continue
        fixture = result.get("fixture")
        if isinstance(fixture, Mapping):
            selector = fixture.get("selector")
            if isinstance(selector, Mapping) and selector.get("target") == target:
                results.append(result)
    return results


def _require_native_runtime_results(
    runtime_report: Mapping[str, Any],
    target: str,
) -> None:
    target_results = _runtime_report_results_for_target(runtime_report, target)
    _require(
        bool(target_results)
        and all(result.get("status") == "passed" for result in target_results),
        f"{target} native runtime execution was required for every MLX arange fixture",
    )


def _error_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.get("severity") == "error"
    ]


def _execute_runtime_fixtures_for_report(
    *,
    mlx_root: Path,
    report_dir: Path,
    name: str,
    runtime_artifact_manifest_path: Path,
    targets: Sequence[str],
) -> dict[str, Any]:
    metadata_path = report_dir / f"{name}.runtime-fixture-execution-metadata.json"
    manifest_path = report_dir / f"{name}.runtime-fixture-execution-manifest.json"
    plan_path = report_dir / f"{name}.runtime-fixture-execution-plan.json"
    report_path = report_dir / f"{name}.runtime-fixture-execution-report.json"
    metadata = _runtime_fixture_execution_metadata(targets)
    _write_json(metadata_path, metadata)
    manifest = build_runtime_test_manifest(
        runtime_artifact_manifest_path,
        metadata_path,
        project_root=mlx_root,
    )
    _write_json(manifest_path, manifest)
    plan = build_project_test_runner_plan(
        runtime_artifact_manifest_path,
        manifest,
        selected_targets=targets,
        project_root=mlx_root,
    )
    _write_json(plan_path, plan)
    report = execute_project_test_runner_plan(
        plan,
        project_root=mlx_root,
        runtime_executors=_runtime_fixture_execution_executors(targets),
    )
    _write_json(report_path, report)
    project_runner_summary = report.get("summary", {})
    _require(
        isinstance(project_runner_summary, dict),
        "runtime fixture execution project-runner summary missing",
    )
    runtime_report = report.get("runtimeTestReport", {})
    _require(
        isinstance(runtime_report, Mapping),
        "runtime fixture execution runtime report missing",
    )
    summary = runtime_report.get("summary", {})
    _require(isinstance(summary, dict), "runtime fixture execution summary missing")
    diagnostics = _runtime_report_diagnostics(report)
    diagnostics_by_code = _diagnostics_by_code(diagnostics)
    failed_count = int(summary.get("failedCount", 0))
    skipped_count = int(summary.get("skippedCount", 0))
    status = "passed" if failed_count == 0 and skipped_count == 0 else "failed"
    if status == "failed" and RUNTIME_READINESS_TRACKED_ISSUES:
        status = "blocked-by-tracked-issues"
    return {
        "name": f"{name}-runtime-fixture-execution",
        "status": status,
        "fixtureMetadata": _relpath(metadata_path, mlx_root),
        "runtimeTestManifest": _relpath(manifest_path, mlx_root),
        "projectTestRunnerPlan": _relpath(plan_path, mlx_root),
        "projectTestRunnerReport": _relpath(report_path, mlx_root),
        "targets": list(targets),
        "summary": summary,
        "projectRunnerSummary": project_runner_summary,
        "diagnosticsByCode": diagnostics_by_code,
        "runtimeFixtureExecutionIncluded": True,
        "runtimeIntegrationIncluded": False,
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
    }


def _execute_native_runtime_fixtures_for_report(
    *,
    mlx_root: Path,
    report_dir: Path,
    name: str,
    runtime_artifact_manifest_path: Path,
    targets: Sequence[str],
    required_native_runtime_targets: Sequence[str] = (),
) -> dict[str, Any]:
    metadata_path = report_dir / f"{name}.native-runtime-execution-metadata.json"
    manifest_path = report_dir / f"{name}.native-runtime-execution-manifest.json"
    plan_path = report_dir / f"{name}.native-runtime-execution-plan.json"
    report_path = report_dir / f"{name}.native-runtime-execution-report.json"
    metadata = _native_runtime_execution_metadata(targets)
    _write_json(metadata_path, metadata)
    manifest = build_runtime_test_manifest(
        runtime_artifact_manifest_path,
        metadata_path,
        project_root=mlx_root,
    )
    _write_json(manifest_path, manifest)
    plan = build_project_test_runner_plan(
        runtime_artifact_manifest_path,
        manifest,
        selected_targets=targets,
        project_root=mlx_root,
    )
    _write_json(plan_path, plan)
    report = execute_project_test_runner_plan(
        plan,
        project_root=mlx_root,
        runtime_executors=native_runtime_parity_adapters(
            runtimes={"vulkan": VulkanComputeRuntime()}
        ),
    )
    _write_json(report_path, report)
    project_runner_summary = report.get("summary", {})
    _require(
        isinstance(project_runner_summary, dict),
        "native runtime execution project-runner summary missing",
    )
    runtime_report = report.get("runtimeTestReport", {})
    _require(
        isinstance(runtime_report, Mapping),
        "native runtime execution runtime report missing",
    )
    summary = runtime_report.get("summary", {})
    _require(isinstance(summary, dict), "native runtime execution summary missing")
    diagnostics = _runtime_report_diagnostics(report)
    diagnostics_by_code = _diagnostics_by_code(diagnostics)
    failed_count = int(summary.get("failedCount", 0))
    passed_count = int(summary.get("passedCount", 0))
    unavailable_count = int(summary.get("unavailableCount", 0))
    skipped_count = int(summary.get("skippedCount", 0))
    for target in required_native_runtime_targets:
        _require_native_runtime_results(runtime_report, target)
    status = "passed"
    if failed_count:
        status = "blocked-by-tracked-issues"
    elif unavailable_count or skipped_count:
        status = "blocked-by-runtime-driver"
    return {
        "name": f"{name}-native-runtime-execution",
        "status": status,
        "fixtureMetadata": _relpath(metadata_path, mlx_root),
        "runtimeTestManifest": _relpath(manifest_path, mlx_root),
        "projectTestRunnerPlan": _relpath(plan_path, mlx_root),
        "projectTestRunnerReport": _relpath(report_path, mlx_root),
        "targets": list(targets),
        "summary": summary,
        "passedCount": passed_count,
        "projectRunnerSummary": project_runner_summary,
        "diagnosticsByCode": diagnostics_by_code,
        "nativeRuntimeExecutionIncluded": True,
        "runtimeIntegrationIncluded": False,
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
    }


def _plan_runtime_readiness_for_report(
    *,
    mlx_root: Path,
    report_dir: Path,
    name: str,
    artifact_report: Path,
    targets: Sequence[str],
    required_native_runtime_targets: Sequence[str] = (),
) -> dict[str, Any]:
    _require(
        artifact_report.is_file(),
        f"runtime readiness artifact report is missing: {artifact_report}",
    )
    metadata_path = report_dir / f"{name}.fixture-metadata.json"
    runtime_artifact_manifest_path = (
        report_dir / f"{name}.runtime-artifact-manifest.json"
    )
    manifest_path = report_dir / f"{name}.runtime-test-manifest.json"
    plan_path = report_dir / f"{name}.runtime-test-plan.json"
    metadata = _runtime_readiness_fixture_metadata(targets)
    _write_json(metadata_path, metadata)
    runtime_artifact_manifest = build_runtime_artifact_manifest(artifact_report)
    _write_json(runtime_artifact_manifest_path, runtime_artifact_manifest)
    manifest = build_runtime_test_manifest(
        runtime_artifact_manifest_path,
        metadata_path,
        project_root=mlx_root,
    )
    _write_json(manifest_path, manifest)
    plan = plan_runtime_test_manifest(
        runtime_artifact_manifest_path,
        manifest,
        project_root=mlx_root,
    )
    _write_json(plan_path, plan)
    runtime_fixture_execution = _execute_runtime_fixtures_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name=name,
        runtime_artifact_manifest_path=runtime_artifact_manifest_path,
        targets=targets,
    )
    native_runtime_execution = _execute_native_runtime_fixtures_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name=name,
        runtime_artifact_manifest_path=runtime_artifact_manifest_path,
        targets=targets,
        required_native_runtime_targets=required_native_runtime_targets,
    )

    runtime_artifact_diagnostics_by_code = _diagnostics_by_code(
        runtime_artifact_manifest.get("runtimeDiagnostics", [])
    )
    diagnostic_counts = manifest.get("diagnosticCounts", {})
    _require(
        isinstance(diagnostic_counts, dict),
        "runtime readiness diagnostic counts must be an object",
    )
    _require(
        diagnostic_counts.get("error", 0) == 0,
        "runtime readiness manifest reported fixture or artifact selection errors",
    )
    diagnostics_by_code = _diagnostics_by_code(manifest.get("diagnostics", []))
    metadata_gap_codes = sorted(
        code
        for code in diagnostics_by_code
        if code in RUNTIME_READINESS_DIAGNOSTIC_CODES
    )
    plan_diagnostics = _runtime_plan_diagnostics(plan)
    plan_diagnostics_by_code = _diagnostics_by_code(plan_diagnostics)
    plan_blocker_codes = sorted(
        code
        for code in _diagnostics_by_code(_error_diagnostics(plan_diagnostics))
        if code in RUNTIME_READINESS_PLAN_DIAGNOSTIC_CODES
    )
    if metadata_gap_codes:
        _require(
            RUNTIME_READINESS_TRACKED_ISSUES,
            "runtime readiness manifest reported artifact execution metadata gaps "
            "without tracked issue references",
        )
    if plan_blocker_codes:
        _require(
            RUNTIME_READINESS_TRACKED_ISSUES,
            "runtime readiness plan reported adapter setup blockers without "
            "tracked issue references",
        )
    status = (
        "blocked-by-tracked-issues"
        if metadata_gap_codes or plan_blocker_codes
        else "planned"
    )
    plan_summary = plan.get("summary", {})
    _require(isinstance(plan_summary, dict), "runtime readiness plan summary missing")
    manifest_summary = manifest.get("summary", {})
    _require(
        isinstance(manifest_summary, dict),
        "runtime readiness manifest summary missing",
    )
    return {
        "name": name,
        "status": status,
        "artifactReport": _relpath(artifact_report, mlx_root),
        "fixtureMetadata": _relpath(metadata_path, mlx_root),
        "runtimeArtifactManifest": _relpath(runtime_artifact_manifest_path, mlx_root),
        "runtimeTestManifest": _relpath(manifest_path, mlx_root),
        "runtimeTestPlan": _relpath(plan_path, mlx_root),
        "targets": list(targets),
        "testCount": manifest_summary.get("testCount", 0),
        "runtimeArtifactSummary": runtime_artifact_manifest.get("summary", {}),
        "runtimeArtifactDiagnosticCounts": runtime_artifact_manifest.get(
            "runtimeDiagnosticCounts", {}
        ),
        "runtimeArtifactDiagnosticsByCode": runtime_artifact_diagnostics_by_code,
        "diagnosticCounts": diagnostic_counts,
        "diagnosticsByCode": diagnostics_by_code,
        "runtimePlanDiagnosticsByCode": plan_diagnostics_by_code,
        "runtimePlanSummary": plan_summary,
        "metadataGapCodes": metadata_gap_codes,
        "planBlockerCodes": plan_blocker_codes,
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeFixtureExecutionIncluded": True,
        "runtimeFixtureExecution": runtime_fixture_execution,
        "nativeRuntimeExecutionIncluded": True,
        "nativeRuntimeExecution": native_runtime_execution,
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
    }


def _plan_reduced_runtime_readiness(
    mlx_root: Path,
    report_dir: Path,
    *,
    require_vulkan_native_runtime: bool,
    require_opengl_native_runtime: bool,
) -> dict[str, Any]:
    reports = [
        _plan_runtime_readiness_for_report(
            mlx_root=mlx_root,
            report_dir=report_dir,
            name="directx-runtime-readiness",
            artifact_report=report_dir / "directx-frontier.json",
            targets=("directx",),
            required_native_runtime_targets=(),
        ),
        _plan_runtime_readiness_for_report(
            mlx_root=mlx_root,
            report_dir=report_dir,
            name="vulkan-runtime-readiness",
            artifact_report=report_dir / "vulkan-frontier.json",
            targets=("vulkan",),
            required_native_runtime_targets=(
                ("vulkan",) if require_vulkan_native_runtime else ()
            ),
        ),
        _plan_runtime_readiness_for_report(
            mlx_root=mlx_root,
            report_dir=report_dir,
            name="opengl-runtime-readiness",
            artifact_report=report_dir / "arange-opengl.json",
            targets=("opengl",),
            required_native_runtime_targets=(
                ("opengl",) if require_opengl_native_runtime else ()
            ),
        ),
    ]
    status = (
        "blocked-by-tracked-issues"
        if any(report["status"] == "blocked-by-tracked-issues" for report in reports)
        else "planned"
    )
    diagnostics_by_code: Counter[str] = Counter()
    runtime_artifact_diagnostics_by_code: Counter[str] = Counter()
    runtime_plan_diagnostics_by_code: Counter[str] = Counter()
    runtime_fixture_execution_by_status: Counter[str] = Counter()
    runtime_fixture_execution_summary: Counter[str] = Counter()
    native_runtime_execution_by_status: Counter[str] = Counter()
    native_runtime_execution_summary: Counter[str] = Counter()
    for report in reports:
        diagnostics_by_code.update(report.get("diagnosticsByCode", {}))
        runtime_artifact_diagnostics_by_code.update(
            report.get("runtimeArtifactDiagnosticsByCode", {})
        )
        runtime_plan_diagnostics_by_code.update(
            report.get("runtimePlanDiagnosticsByCode", {})
        )
        runtime_fixture_execution = report.get("runtimeFixtureExecution", {})
        if isinstance(runtime_fixture_execution, Mapping):
            runtime_fixture_execution_by_status.update(
                [str(runtime_fixture_execution.get("status", "unknown"))]
            )
            execution_summary = runtime_fixture_execution.get("summary", {})
            if isinstance(execution_summary, Mapping):
                for key in (
                    "fixtureCount",
                    "resultCount",
                    "passedCount",
                    "skippedCount",
                    "unavailableCount",
                    "translationFailedCount",
                    "runtimeFailedCount",
                    "comparisonFailedCount",
                    "failedCount",
                ):
                    if key in execution_summary:
                        runtime_fixture_execution_summary[key] += int(
                            execution_summary.get(key, 0)
                        )
        native_runtime_execution = report.get("nativeRuntimeExecution", {})
        if isinstance(native_runtime_execution, Mapping):
            native_runtime_execution_by_status.update(
                [str(native_runtime_execution.get("status", "unknown"))]
            )
            execution_summary = native_runtime_execution.get("summary", {})
            if isinstance(execution_summary, Mapping):
                for key in (
                    "fixtureCount",
                    "passedCount",
                    "skippedCount",
                    "unavailableCount",
                    "translationFailedCount",
                    "runtimeFailedCount",
                    "comparisonFailedCount",
                    "failedCount",
                ):
                    if key in execution_summary:
                        native_runtime_execution_summary[key] += int(
                            execution_summary.get(key, 0)
                        )
    return {
        "name": "runtime-readiness",
        "status": status,
        "reports": reports,
        "targets": ["directx", "opengl", "vulkan"],
        "testCount": sum(int(report.get("testCount", 0)) for report in reports),
        "diagnosticsByCode": dict(sorted(diagnostics_by_code.items())),
        "runtimeArtifactDiagnosticsByCode": dict(
            sorted(runtime_artifact_diagnostics_by_code.items())
        ),
        "runtimePlanDiagnosticsByCode": dict(
            sorted(runtime_plan_diagnostics_by_code.items())
        ),
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeFixtureExecutionIncluded": True,
        "runtimeFixtureExecutionByStatus": dict(
            sorted(runtime_fixture_execution_by_status.items())
        ),
        "runtimeFixtureExecutionSummary": dict(
            sorted(runtime_fixture_execution_summary.items())
        ),
        "nativeRuntimeExecutionIncluded": True,
        "nativeRuntimeExecutionByStatus": dict(
            sorted(native_runtime_execution_by_status.items())
        ),
        "nativeRuntimeExecutionSummary": dict(
            sorted(native_runtime_execution_summary.items())
        ),
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
    }


def _translate_full_corpus(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "full-corpus.toml"
    report_path = report_dir / "full-corpus.json"
    output_dir = work_dir / "out-full-corpus"
    _write_project_config(
        config_path,
        include=f"{MLX_METAL_KERNEL_ROOT}/**/*.metal",
        targets=FULL_CORPUS_TARGETS,
        output_dir=_relpath(output_dir, mlx_root),
        metal_source_options={
            "max_template_specializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        },
    )
    result = _run_command(
        "translate-full-corpus",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--validate",
        ],
        log_dir=log_dir,
        check=False,
        timeout_seconds=FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS,
    )
    if not report_path.is_file() and result.returncode:
        _require(
            FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
            "full-corpus translation failed before writing a report without "
            "tracked issue references",
        )
        return {
            "name": "full-corpus",
            "status": "blocked-by-tracked-issues",
            "report": _relpath(report_path, mlx_root),
            "reportProduced": False,
            "unitCount": EXPECTED_METAL_KERNEL_COUNT,
            "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "targets": list(FULL_CORPUS_TARGETS),
            "returncode": result.returncode,
            "timeoutSeconds": FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS,
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
            "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "maxTemplateMaterializationWork": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        }
    _require(
        report_path.is_file(),
        "full-corpus translation did not produce a project report",
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, dict), "full-corpus summary must be an object")
    diagnostic_counts = summary.get("diagnosticCounts", {})
    _require(
        isinstance(diagnostic_counts, dict),
        "full-corpus diagnostic counts must be an object",
    )
    failed_count = summary.get("failedCount")
    _require(
        summary.get("unitCount") == EXPECTED_METAL_KERNEL_COUNT,
        "full-corpus translation must scan {} units; found {}".format(
            EXPECTED_METAL_KERNEL_COUNT,
            summary.get("unitCount"),
        ),
    )
    _require(
        summary.get("artifactCount") == FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "full-corpus translation must emit {} artifacts; found {}".format(
            FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            summary.get("artifactCount"),
        ),
    )
    artifacts_by_target = summary.get("artifactsByTarget", {})
    _require(
        isinstance(artifacts_by_target, dict),
        "full-corpus artifactsByTarget must be an object",
    )
    target_counts: dict[str, dict[str, int]] = {}
    for target in FULL_CORPUS_TARGETS:
        target_summary = artifacts_by_target.get(target, {})
        _require(
            isinstance(target_summary, dict),
            f"full-corpus target summary is missing for {target}",
        )
        target_counts[target] = {
            "translatedCount": target_summary.get("translatedCount", 0),
            "failedCount": target_summary.get("failedCount", 0),
        }
    validation = payload.get("validation", {})
    _require(isinstance(validation, dict), "full-corpus validation must be an object")
    artifact_validation = validation.get("summary", {})
    _require(
        isinstance(artifact_validation, dict),
        "full-corpus validation summary must be an object",
    )
    fence_contracts = _validate_atomic_fence_contract_report(
        mlx_root,
        output_dir,
        payload,
        exact_report=False,
    )
    diagnostics = payload.get("diagnostics", [])
    _require(isinstance(diagnostics, list), "full-corpus diagnostics must be a list")
    error_diagnostics_by_code = Counter(
        diagnostic.get("code")
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping)
        and diagnostic.get("severity") == "error"
        and isinstance(diagnostic.get("code"), str)
    )
    expected_error_diagnostics_by_code = Counter(
        {
            **{
                contract["diagnosticCode"]: 1
                for contract in MLX_FENCE_TARGET_CONTRACTS.values()
            },
            "project.validate.failed-artifact": (
                FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
            ),
        }
    )
    expected_target_counts = {
        target: {
            "translatedCount": EXPECTED_METAL_KERNEL_COUNT - 1,
            "failedCount": 1,
        }
        for target in FULL_CORPUS_TARGETS
    }
    expected_fence_only_result = (
        failed_count == FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
        and summary.get("translatedCount")
        == FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
        and result.returncode == 1
        and target_counts == expected_target_counts
        and artifact_validation.get("failedCount")
        == FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
        and error_diagnostics_by_code == expected_error_diagnostics_by_code
    )
    if not expected_fence_only_result:
        _require(
            FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
            "full-corpus translation reported failures beyond the expected fence "
            "contract without tracked issue references",
        )
        unexpected_error_diagnostics = (
            error_diagnostics_by_code - expected_error_diagnostics_by_code
        )
        return {
            "name": "full-corpus",
            "status": "blocked-by-tracked-issues",
            "report": _relpath(report_path, mlx_root),
            "unitCount": EXPECTED_METAL_KERNEL_COUNT,
            "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "translatedCount": summary.get("translatedCount", 0),
            "failedCount": summary.get("failedCount", 0),
            "diagnosticCounts": diagnostic_counts,
            "diagnosticsByCode": summary.get("diagnosticsByCode", {}),
            "targets": list(FULL_CORPUS_TARGETS),
            "targetCounts": target_counts,
            "validationFailedCount": artifact_validation.get("failedCount", 0),
            "expectedFenceFailureCount": FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT,
            "unexpectedFailedCount": max(
                int(summary.get("failedCount", 0))
                - FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT,
                0,
            ),
            "unexpectedErrorDiagnosticsByCode": dict(
                sorted(unexpected_error_diagnostics.items())
            ),
            "fenceContract": {
                "status": "blocked-as-expected",
                "source": MLX_FENCE_SOURCE,
                "targetContracts": fence_contracts,
                "trackedIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
            },
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
            "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "maxTemplateMaterializationWork": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        }
    return {
        "name": "full-corpus",
        "status": "passed-with-expected-fence-blockers",
        "report": _relpath(report_path, mlx_root),
        "unitCount": EXPECTED_METAL_KERNEL_COUNT,
        "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "translatedCount": FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT,
        "failedCount": FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT,
        "targets": list(FULL_CORPUS_TARGETS),
        "targetCounts": target_counts,
        "validationFailedCount": artifact_validation.get("failedCount", 0),
        "fenceContract": {
            "status": "blocked-as-expected",
            "source": MLX_FENCE_SOURCE,
            "targetContracts": fence_contracts,
            "trackedIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
        },
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
        "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
        "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK,
    }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    mlx_root = Path(args.mlx_root).resolve()
    require_metal_toolchain = bool(getattr(args, "require_metal_toolchain", False))
    require_directx_gemv_compiler_frontier = bool(
        getattr(args, "require_directx_gemv_compiler_frontier", False)
    )
    require_opengl_frontier_toolchain = bool(
        getattr(args, "require_opengl_frontier_toolchain", False)
    )
    require_opengl_gemv_frontier = bool(
        getattr(args, "require_opengl_gemv_frontier", False)
    )
    require_opengl_native_runtime = bool(
        getattr(args, "require_opengl_native_runtime", False)
    )
    require_vulkan_gemv_toolchain = bool(
        getattr(args, "require_vulkan_gemv_toolchain", False)
    )
    _require(
        not require_directx_gemv_compiler_frontier
        or args.mode == REDUCED_FRONTIER_MODE,
        "--require-directx-gemv-compiler-frontier is only valid in "
        "reduced-frontier mode",
    )
    _require(
        not require_opengl_frontier_toolchain or args.mode == REDUCED_FRONTIER_MODE,
        "--require-opengl-frontier-toolchain is only valid in reduced-frontier mode",
    )
    _require(
        not require_opengl_gemv_frontier or args.mode == REDUCED_FRONTIER_MODE,
        "--require-opengl-gemv-frontier is only valid in reduced-frontier mode",
    )
    _require(
        not require_opengl_native_runtime or args.mode == REDUCED_FRONTIER_MODE,
        "--require-opengl-native-runtime is only valid in reduced-frontier mode",
    )
    _require(
        not require_vulkan_gemv_toolchain or args.mode == REDUCED_FRONTIER_MODE,
        "--require-vulkan-gemv-toolchain is only valid in reduced-frontier mode",
    )
    work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    if work_dir.exists() and not args.no_clean:
        shutil.rmtree(work_dir)
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = [
        _verify_mlx_checkout(mlx_root, args.python, log_dir),
        _scan_metal_kernels(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
        ),
    ]
    checks.append(
        _check_metal_roundtrip(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
            require_metal_toolchain=require_metal_toolchain,
        )
    )
    if args.mode == REDUCED_FRONTIER_MODE:
        checks.append(
            _check_atomic_fence_contract(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
        checks.append(
            _check_reference_accessor_lvalue_identity(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_directx_toolchain=args.require_directx_toolchain,
                require_opengl_toolchain=require_opengl_frontier_toolchain,
            )
        )
        checks.append(
            _check_template_member_buffer_pointer(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
        checks.append(
            _translate_directx_frontier(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_directx_toolchain=args.require_directx_toolchain,
            )
        )
        checks.append(
            _translate_vulkan_frontier(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_toolchain=args.require_vulkan_toolchain,
                run_optional_toolchain=(
                    not args.require_directx_toolchain
                    and not args.require_vulkan_toolchain
                ),
            )
        )
        checks.append(
            _check_arange_opengl(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
        if require_directx_gemv_compiler_frontier:
            checks.append(
                _check_gemv_directx_compiler_frontier(
                    mlx_root,
                    work_dir,
                    config_dir,
                    report_dir,
                    log_dir,
                    args.python,
                )
            )
        checks.append(
            _check_opengl_frontier(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_toolchain=require_opengl_frontier_toolchain,
            )
        )
        checks.append(
            _check_fft_opengl_workgroup_pointer_frontier(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
        if require_opengl_gemv_frontier:
            checks.append(
                _check_gemv_opengl_frontier(
                    mlx_root,
                    work_dir,
                    config_dir,
                    report_dir,
                    log_dir,
                    args.python,
                )
            )
        if require_vulkan_gemv_toolchain:
            checks.append(
                _check_gemv_vulkan_toolchain(
                    mlx_root,
                    work_dir,
                    config_dir,
                    report_dir,
                    log_dir,
                    args.python,
                )
            )
        checks.append(
            _plan_reduced_runtime_readiness(
                mlx_root,
                report_dir,
                require_vulkan_native_runtime=args.require_vulkan_native_runtime,
                require_opengl_native_runtime=require_opengl_native_runtime,
            )
        )
    elif args.mode == FULL_CORPUS_MODE:
        checks.append(
            _translate_full_corpus(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
    else:
        raise PortingCheckError(f"unsupported MLX porting mode: {args.mode}")
    reference_accessor_included = args.mode == REDUCED_FRONTIER_MODE
    template_member_pointer_included = args.mode == REDUCED_FRONTIER_MODE
    fft_opengl_pointer_frontier_included = args.mode == REDUCED_FRONTIER_MODE
    return {
        "schema_version": 1,
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "mode": args.mode,
            "sourceRoot": MLX_METAL_KERNEL_ROOT,
            "metalRoundTripSource": MLX_METAL_ROUNDTRIP_SOURCE,
            "metalRoundTripIncluded": True,
            "metalToolchainRequired": require_metal_toolchain,
            "frontierSources": list(MLX_REDUCED_FRONTIER_SOURCES),
            "nonFenceFrontierSources": list(MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES),
            "blockedFrontierSources": list(MLX_BLOCKED_REDUCED_FRONTIER_SOURCES),
            "blockedFrontierIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
            "directxTranslatedFrontierSources": list(
                MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES
            ),
            "directxWorkgroupBlockedFrontierSources": list(
                MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
            ),
            "vulkanTranslatedFrontierSources": list(
                MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
            ),
            "openglTranslatedFrontierSources": list(
                MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
            ),
            "openglWorkgroupBlockedFrontierSources": list(
                MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
            ),
            "dynamicWorkgroupTrackedIssue": MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE,
            "fullCorpusTargets": list(FULL_CORPUS_TARGETS),
            "fullCorpusExpectedUnitCount": EXPECTED_METAL_KERNEL_COUNT,
            "fullCorpusExpectedArtifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "fullCorpusExpectedTranslatedArtifactCount": (
                FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
            ),
            "fullCorpusExpectedFenceFailureCount": (
                FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
            ),
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "runtimeReadinessIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "runtimeFixtureExecutionIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "nativeRuntimeExecutionIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "referenceAccessorProofIncluded": reference_accessor_included,
            "referenceAccessorTargets": (
                list(REFERENCE_ACCESSOR_TARGETS) if reference_accessor_included else []
            ),
            "referenceAccessorDirectxToolchainRequired": bool(
                reference_accessor_included and args.require_directx_toolchain
            ),
            "referenceAccessorOpenglToolchainRequired": bool(
                reference_accessor_included and require_opengl_frontier_toolchain
            ),
            "templateMemberBufferPointerProofIncluded": (
                template_member_pointer_included
            ),
            "templateMemberBufferPointerTargets": (
                list(TEMPLATE_MEMBER_POINTER_TARGETS)
                if template_member_pointer_included
                else []
            ),
            "templateMemberBufferPointerNativeValidationIncluded": False,
            "openglFrontierToolchainRequired": require_opengl_frontier_toolchain,
            "directxGemvCompilerFrontierRequired": (
                require_directx_gemv_compiler_frontier
            ),
            "fftOpenGLWorkgroupPointerFrontierIncluded": (
                fft_opengl_pointer_frontier_included
            ),
            "openglGemvFrontierRequired": require_opengl_gemv_frontier,
            "openglNativeRuntimeRequired": require_opengl_native_runtime,
            "vulkanGemvToolchainRequired": require_vulkan_gemv_toolchain,
            "runtimeParityClaimed": False,
        },
        "trackedIssues": list(FULL_CORPUS_TRACKED_ISSUES),
        "resolvedFrontierIssues": list(RESOLVED_FRONTIER_ISSUES),
        "checks": checks,
        "status": "passed",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pinned MLX project-porting checks through CrossTL."
    )
    parser.add_argument("--mlx-root", required=True, help="Path to the MLX checkout")
    parser.add_argument(
        "--mode",
        choices=(REDUCED_FRONTIER_MODE, FULL_CORPUS_MODE),
        default=REDUCED_FRONTIER_MODE,
        help=(
            "Harness scope to run. The default reduced frontier is the pull "
            "request gate; full-corpus is intended for scheduled and manual "
            "artifact-generation scouts."
        ),
    )
    parser.add_argument(
        "--work-dir",
        help=(
            "Generated config/report/output directory. Defaults to "
            "<mlx-root>/.crosstl-mlx-porting."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke `python -m crosstl`.",
    )
    parser.add_argument(
        "--require-metal-toolchain",
        action="store_true",
        help=(
            "Fail unless the generated Metal round-trip artifact compiles "
            "natively with the macOS Metal compiler."
        ),
    )
    parser.add_argument(
        "--require-directx-toolchain",
        action="store_true",
        help="Fail unless the DirectX HLSL smoke check runs successfully.",
    )
    parser.add_argument(
        "--require-directx-gemv-compiler-frontier",
        action="store_true",
        help=(
            "Translate pinned GEMV to DirectX and compile the three configured "
            "representative entries with DXC."
        ),
    )
    parser.add_argument(
        "--require-vulkan-toolchain",
        action="store_true",
        help="Fail unless the Vulkan SPIR-V smoke check runs successfully.",
    )
    parser.add_argument(
        "--require-vulkan-native-runtime",
        action="store_true",
        help="Fail unless the MLX-generated Vulkan arange fixture executes natively.",
    )
    parser.add_argument(
        "--require-opengl-native-runtime",
        action="store_true",
        help=(
            "Fail unless the selected MLX OpenGL arangeuint32 artifact executes "
            "natively and passes exact output comparison."
        ),
    )
    parser.add_argument(
        "--require-opengl-frontier-toolchain",
        action="store_true",
        help=(
            "Translate the pinned OpenGL frontier and require native GLSL and "
            "SPIR-V 1.3 validation."
        ),
    )
    parser.add_argument(
        "--require-opengl-gemv-frontier",
        action="store_true",
        help=(
            "Require pinned GEMV OpenGL translation to materialize all source "
            "specializations and stop at the exact tracked workgroup-pointer frontier."
        ),
    )
    parser.add_argument(
        "--require-vulkan-gemv-toolchain",
        action="store_true",
        help=(
            "Materialize pinned GEMV for Vulkan, require SPIR-V validation, "
            "and verify the exact tracked semantic blockers."
        ),
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing files in the generated work directory.",
    )
    parser.add_argument(
        "--summary",
        help="Summary JSON path. Defaults to <work-dir>/summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mlx_root = Path(args.mlx_root).resolve()
    work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    summary_path = (
        Path(args.summary).resolve() if args.summary else work_dir / "summary.json"
    )
    try:
        summary = run_checks(args)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except PortingCheckError as exc:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repository": {
                        "name": "ml-explore/mlx",
                        "url": MLX_REPOSITORY,
                        "commit": MLX_COMMIT,
                    },
                    "scope": {
                        "mode": args.mode,
                        "shaderArtifactsOnly": True,
                        "runtimeIntegrationIncluded": False,
                    },
                    "status": "failed",
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"MLX project-porting checks failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    print(f"MLX project-porting checks passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
