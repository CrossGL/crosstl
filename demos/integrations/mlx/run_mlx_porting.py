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
    load_dispatch_contract,
    load_project_translation_checkpoint,
    native_runtime_parity_adapters,
)
from crosstl.project.directx_toolchain import (
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
)
from crosstl.project.runtime_verification import (
    RuntimeParityAdapter,
    build_runtime_test_manifest,
    plan_runtime_test_manifest,
)

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_REFERENCE_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_CORPUS_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
# The reduced frontier remains an exact historical proof until each fixture and
# runtime contract has been remeasured against the current corpus.
MLX_COMMIT = MLX_REFERENCE_COMMIT
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
MLX_LAYER_NORM_SHA256 = (
    "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"
)
MLX_LAYER_NORM_DISPATCH_CONTRACT_SOURCE = (
    Path(__file__).resolve().parent / "contracts" / "layer_norm.dispatch.json"
)
MLX_LAYER_NORM_DISPATCH_NORMALIZED_SHA256 = (
    "17924e1eb885da0b91bed4ac67df39a72ab2f9448a40988efef2efd1f0f1bc93"
)
MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY = (
    "sha256:527140d8720a414ff5eaeb670ee9312571c5ca1a13460353d0c177193f5a8f98"
)
MLX_LAYER_NORM_DISPATCH_VARIANTS = {
    "layer_normfloat32": {
        "artifactId": (
            "sha256:6c2a76fa651fc945ff5f320801d70b925a3b289c2927990ab4161535edc8d6bf"
        ),
        "dispatchVariantId": (
            "sha256:9f67f3aa323b6a35566bcce84c678121ba050a49843e10187bd5abf4ab3dd01c"
        ),
        "workgroupSize": [544, 1, 1],
        "subgroupWidth": 32,
        "specializationConstants": {},
        "specializationCount": 3,
    },
    "vjp_layer_normfloat32": {
        "artifactId": (
            "sha256:7f6bf28c1536b59c8bd51d9f55dc80cb337698f15e1433f378e1692625e45d54"
        ),
        "dispatchVariantId": (
            "sha256:eed664cf712ef3aeb953e3539389a6f3adcd853d8374aa16da343055080edbf1"
        ),
        "workgroupSize": [1024, 1, 1],
        "subgroupWidth": 32,
        "specializationConstants": {"20": True},
        "specializationCount": 4,
    },
}
MLX_LOGSUMEXP_SOURCE = "mlx/backend/metal/kernels/logsumexp.metal"
MLX_LOGSUMEXP_SHA256 = (
    "f9bec5e1e5a23d20bedf9ff8d29a8c03bbb5144bc5d751bbfe906d32ee894817"
)
MLX_LOGSUMEXP_DISPATCH_CONTRACT_SOURCE = (
    Path(__file__).resolve().parent / "contracts" / "logsumexp.dispatch.json"
)
MLX_LOGSUMEXP_DISPATCH_NORMALIZED_SHA256 = (
    "b1153ba5a68cb9fdb1bdcb04f552f258c6adaf127e6bcbedd8ef8c152067b3d5"
)
MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY = (
    "sha256:db762a188e05786e206d9aa5a340b6f9095a8a3e938b85a7c04836f300e97c95"
)
MLX_LOGSUMEXP_NATIVE_LOADER_DISPATCH_CONTRACT_SOURCE = (
    Path(__file__).resolve().parent
    / "contracts"
    / "logsumexp.native-loader.dispatch.json"
)
MLX_LOGSUMEXP_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY = (
    "sha256:3cfc400f25cf49cb16d028fdba59ebe8b56b729ade919f711de4b8b67bfa5ab4"
)
MLX_LOGSUMEXP_SOFTWARE_OPENGL_ARTIFACT = {
    "sha256": "813762d4535fdd693ca0a48c3c3f5dc79f6cc298050faae6e180d3cc9f1d60e5",
    "size_bytes": 4676,
}
MLX_LOGSUMEXP_DISPATCH_VARIANTS = {
    "block-float32-axis-32": {
        "entryPoint": "block_logsumexp_float32",
        "artifactId": (
            "sha256:f51ab67b8a9ad3240e3fbb52f6de00cdf4a8532e58790758e59aa50fb95e2c52"
        ),
        "dispatchVariantId": (
            "sha256:0a9cafa696d2fddd6284c44673b9120120a70630a0a18714e27284f8a6189c59"
        ),
        "inputs": {"axisSize": 32, "dtype": "float32", "nRows": 1},
        "workgroupSize": [32, 1, 1],
        "dispatchWorkgroupCount": [1, 1, 1],
        "specializationConstants": {},
    },
    "block-float32-axis-1025": {
        "entryPoint": "block_logsumexp_float32",
        "artifactId": (
            "sha256:ae512c102a88628c05a49f28a872c44ab582bacf74584e8ca7e6ae765263afe0"
        ),
        "dispatchVariantId": (
            "sha256:35754405ceaa5e50614c3fef95a66390b99580c9bd2394c0ba5279322fde7446"
        ),
        "inputs": {"axisSize": 1025, "dtype": "float32", "nRows": 1},
        "workgroupSize": [288, 1, 1],
        "dispatchWorkgroupCount": [1, 1, 1],
        "specializationConstants": {},
    },
}
MLX_METAL_ROUNDTRIP_SOURCE = MLX_FENCE_SOURCE
MLX_RANDOM_SOURCE = "mlx/backend/metal/kernels/random.metal"
MLX_QUANTIZED_SOURCE = "mlx/backend/metal/kernels/quantized.metal"
MLX_QUANTIZED_SELECTED_ENTRY_POINT = "affine_quantize_float_gs_32_b_2"
MLX_RMS_NORM_SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
MLX_RMS_NORM_SHA256 = "5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee"
MLX_RMS_NORM_DISPATCH_CONTRACT_SOURCE = (
    Path(__file__).resolve().parent / "contracts" / "rms_norm.dispatch.json"
)
MLX_RMS_NORM_DISPATCH_NORMALIZED_SHA256 = (
    "ea91c5b5655e776e6db3537bf6539d1f1983c0a4ed74bf90f96a57ad3611e91f"
)
MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY = (
    "sha256:ea238af83b140c33d43b79b5efd1814c398bbc09ac70cbef21375b1e8ff9a1eb"
)
MLX_RMS_NORM_DISPATCH_VARIANTS = {
    "forward-float32-axis-32": {
        "entryPoint": "rmsfloat32",
        "artifactId": (
            "sha256:00c05fccf276cf11f3fb9b617b8fe0bb3c5f8766c0e4ca1ed990c093e700422e"
        ),
        "dispatchVariantId": (
            "sha256:4306831dce3a9a479ef63093a7f2722358caf58b6fcf1a47ed808a1c28dc9ebb"
        ),
        "inputs": {
            "axisSize": 32,
            "dtype": "float32",
            "hasW": False,
            "isVjp": False,
        },
        "workgroupSize": [32, 1, 1],
        "dispatchWorkgroupCount": [2, 1, 1],
        "specializationConstants": {},
    },
    "forward-float32-axis-256": {
        "entryPoint": "rmsfloat32",
        "artifactId": (
            "sha256:1ef80b00c1a7a2f7967177bc003a961f3e4448358716d2deb79778fc3cbfb68e"
        ),
        "dispatchVariantId": (
            "sha256:f27b102d2ee4f473afef2448e42103223922ff29fd703e6d269d827f280e8bf7"
        ),
        "inputs": {
            "axisSize": 256,
            "dtype": "float32",
            "hasW": False,
            "isVjp": False,
        },
        "workgroupSize": [64, 1, 1],
        "dispatchWorkgroupCount": [2, 1, 1],
        "specializationConstants": {},
    },
    "forward-float32-axis-512": {
        "entryPoint": "rmsfloat32",
        "artifactId": (
            "sha256:a9b6980b1645e867b2502052d6d5f37a03447258bb4f819535901bf45a01d5da"
        ),
        "dispatchVariantId": (
            "sha256:369be00b341be529b913101657bd6a78d8b8c689e9dc7c6d27e694809dd9d098"
        ),
        "inputs": {
            "axisSize": 512,
            "dtype": "float32",
            "hasW": False,
            "isVjp": False,
        },
        "workgroupSize": [128, 1, 1],
        "dispatchWorkgroupCount": [2, 1, 1],
        "specializationConstants": {},
    },
    "forward-float16-axis-32": {
        "entryPoint": "rmsfloat16",
        "artifactId": (
            "sha256:b694e5240f2a87bfae8af862878251a45cfaeaf39fd810bf2df8a5e3724bdad7"
        ),
        "dispatchVariantId": (
            "sha256:51de914a20a4defb1d1b79ed26def94a994894228d6c64cae0e1132553809dbb"
        ),
        "inputs": {
            "axisSize": 32,
            "dtype": "float16",
            "hasW": False,
            "isVjp": False,
        },
        "workgroupSize": [32, 1, 1],
        "dispatchWorkgroupCount": [2, 1, 1],
        "specializationConstants": {},
    },
    "forward-bfloat16-axis-32": {
        "entryPoint": "rmsbfloat16",
        "artifactId": (
            "sha256:13655322998b557a5143ac9b871dc898f0ae43c07cf9659aecd505156ed9318b"
        ),
        "dispatchVariantId": (
            "sha256:635d28f468bf3e15a0ec7285aacc23d0de32f9596d511beb98adb3ecc2c2abf4"
        ),
        "inputs": {
            "axisSize": 32,
            "dtype": "bfloat16",
            "hasW": False,
            "isVjp": False,
        },
        "workgroupSize": [32, 1, 1],
        "dispatchWorkgroupCount": [2, 1, 1],
        "specializationConstants": {},
    },
    "forward-float32-axis-4099": {
        "entryPoint": "rms_loopedfloat32",
        "artifactId": (
            "sha256:b81c2043b10bde966cb6f4dbfa198d2b93a3e456f3026030b69557c4a8983729"
        ),
        "dispatchVariantId": (
            "sha256:392aee49734fcc2fd3ff4fd232a49f1a97a2bb79f3c22cc80eeabeb1f1ca1959"
        ),
        "inputs": {
            "axisSize": 4099,
            "dtype": "float32",
            "hasW": False,
            "isVjp": False,
        },
        "workgroupSize": [1024, 1, 1],
        "dispatchWorkgroupCount": [1, 1, 1],
        "specializationConstants": {},
    },
    "vjp-float32-axis-32-has-w-false": {
        "entryPoint": "vjp_rmsfloat32",
        "artifactId": (
            "sha256:ef832e1ceb8c864a13aee3460d23658f4fffba18db1b800461628ba6ebe38e0a"
        ),
        "dispatchVariantId": (
            "sha256:1a25dca51070c3b6fc96f162e6c152049d388833ba02b1f3d10cc1928c5661c4"
        ),
        "inputs": {
            "axisSize": 32,
            "dtype": "float32",
            "hasW": False,
            "isVjp": True,
        },
        "workgroupSize": [32, 1, 1],
        "dispatchWorkgroupCount": [800, 1, 1],
        "specializationConstants": {"20": False},
    },
    "vjp-float32-axis-32-has-w-true": {
        "entryPoint": "vjp_rmsfloat32",
        "artifactId": (
            "sha256:a9be06b43a6156fb9ee1f9a6955d03d6bda0940c2a8223b58f564c2d12bd0cd0"
        ),
        "dispatchVariantId": (
            "sha256:26177a77e484a56b7b2572516e8e1360714c88b668881237a9cf499129e34f35"
        ),
        "inputs": {
            "axisSize": 32,
            "dtype": "float32",
            "hasW": True,
            "isVjp": True,
        },
        "workgroupSize": [32, 1, 1],
        "dispatchWorkgroupCount": [800, 1, 1],
        "specializationConstants": {"20": True},
    },
    "vjp-float32-axis-256-has-w-false": {
        "entryPoint": "vjp_rmsfloat32",
        "artifactId": (
            "sha256:4455fc9204f826fc5d0d7f016bcaf970b75ea31d5f1598d13ceca6a2baa369e7"
        ),
        "dispatchVariantId": (
            "sha256:f26ffb357ecfabef6216ff45661f298af4f164a0494da838ca579ea0812e73d2"
        ),
        "inputs": {
            "axisSize": 256,
            "dtype": "float32",
            "hasW": False,
            "isVjp": True,
        },
        "workgroupSize": [64, 1, 1],
        "dispatchWorkgroupCount": [800, 1, 1],
        "specializationConstants": {"20": False},
    },
    "vjp-float32-axis-256-has-w-true": {
        "entryPoint": "vjp_rmsfloat32",
        "artifactId": (
            "sha256:0944044e2f050bedde1d05e1ae5648628e7144c04752dd49b2e2bb7bcd807b7b"
        ),
        "dispatchVariantId": (
            "sha256:a010324f59769fde9d71cc8968852ae1e6c8b0ddc213f035523c2c5f2e12d413"
        ),
        "inputs": {
            "axisSize": 256,
            "dtype": "float32",
            "hasW": True,
            "isVjp": True,
        },
        "workgroupSize": [64, 1, 1],
        "dispatchWorkgroupCount": [800, 1, 1],
        "specializationConstants": {"20": True},
    },
    "vjp-float32-axis-8192-has-w-false": {
        "entryPoint": "vjp_rms_loopedfloat32",
        "artifactId": (
            "sha256:3bd55b546fc00ddf8412f092da4793c0272eec0ad7c130065ad7c1677f60cdce"
        ),
        "dispatchVariantId": (
            "sha256:fa4d57c473cd5799c46a982e3ea339debc06de21fb6a255adc11bbb546f9329b"
        ),
        "inputs": {
            "axisSize": 8192,
            "dtype": "float32",
            "hasW": False,
            "isVjp": True,
        },
        "workgroupSize": [1024, 1, 1],
        "dispatchWorkgroupCount": [4, 1, 1],
        "specializationConstants": {"20": False},
    },
    "vjp-float32-axis-8192-has-w-true": {
        "entryPoint": "vjp_rms_loopedfloat32",
        "artifactId": (
            "sha256:345d524ffec14682b6d0325bc97b624b89d83dc257a17ed49bea5e11e24573f3"
        ),
        "dispatchVariantId": (
            "sha256:8fc64f93a1be95f7acac67d6595c9f9c66c01b87cf952f29052835c20d4d765b"
        ),
        "inputs": {
            "axisSize": 8192,
            "dtype": "float32",
            "hasW": True,
            "isVjp": True,
        },
        "workgroupSize": [1024, 1, 1],
        "dispatchWorkgroupCount": [4, 1, 1],
        "specializationConstants": {"20": True},
    },
}
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
MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES = tuple(
    source
    for source in MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    if source not in {MLX_LAYER_NORM_SOURCE, MLX_LOGSUMEXP_SOURCE, MLX_RMS_NORM_SOURCE}
)
MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES = tuple(
    source
    for source in MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    if source not in MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
)
MLX_DIRECTX_NATIVE_BFLOAT16_STORAGE_SOURCES = (
    MLX_ARANGE_SOURCE,
    MLX_BINARY_TWO_SOURCE,
    MLX_RANDOM_SOURCE,
    MLX_ROPE_SOURCE,
    MLX_TERNARY_SOURCE,
)
MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE = {
    source: {
        "bfloat16Lowering": {
            "status": "exact",
            "approximationUsed": False,
            "registerRepresentation": "uint-low-16-bits",
            "storageRepresentation": (
                "native-uint16"
                if source in MLX_DIRECTX_NATIVE_BFLOAT16_STORAGE_SOURCES
                else "not-required"
            ),
            "roundingMode": "round-to-nearest-ties-to-even",
        },
        "requiredCapabilities": (
            ["directx.native-16bit-types"]
            if source in MLX_DIRECTX_NATIVE_BFLOAT16_STORAGE_SOURCES
            else []
        ),
    }
    for source in MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES
}
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
MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM = 0
MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM = 2_147_483_647
MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS = {
    MLX_BINARY_TWO_SOURCE: (
        "offset + i",
        "a_idx",
        "b_idx",
        "out_idx",
        "out_idx++",
        "idx.x",
        "idx.y",
    ),
    MLX_ROPE_SOURCE: (
        "batch_idx * offset_stride",
        "freq_stride * pos.x",
        "in_index_1",
        "in_index_2",
        "out_index_1",
        "out_index_2",
    ),
    MLX_TERNARY_SOURCE: (
        "offset + i",
        "a_idx",
        "b_idx",
        "c_idx",
        "bidx",
        "cidx",
        "out_idx",
        "out_idx++",
        "idx.x",
        "idx.y",
        "idx.z",
    ),
}
MLX_OPENGL_INDEX_RANGE_ASSERTIONS = tuple(
    {
        "source": source,
        "expression": expression,
        "minimum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
        "maximum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
    }
    for source in MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
    for expression in MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS[source]
)
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
# Aggregate artifacts retain every source entry when one workgroup contract applies.
# LayerNorm, LogSumExp, and RMSNorm contribute bounded, entry-scoped artifacts.
MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES = tuple(
    source
    for source in MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    if source in MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES
    or source in {MLX_LAYER_NORM_SOURCE, MLX_LOGSUMEXP_SOURCE, MLX_RMS_NORM_SOURCE}
)
MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS = {
    source: (
        len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
        if source == MLX_LAYER_NORM_SOURCE
        else (
            len(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
            if source == MLX_LOGSUMEXP_SOURCE
            else (
                len(MLX_RMS_NORM_DISPATCH_VARIANTS)
                if source == MLX_RMS_NORM_SOURCE
                else MLX_DIRECTX_FRONTIER_ENTRY_POINT_COUNTS[source]
            )
        )
    )
    for source in MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
}
MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT = sum(
    MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS.values()
)
MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT = (
    len(MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES)
    + len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
    + len(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
    + len(MLX_RMS_NORM_DISPATCH_VARIANTS)
)
MLX_DIRECTX_TOOLCHAIN_WARNING_CONTRACTS: tuple[dict[str, Any], ...] = ()
DIRECTX_TOOLCHAIN_WARNING_TRACKED_ISSUES: tuple[str, ...] = ()
MLX_DIRECTX_TOOLCHAIN_WARNING_EVIDENCE = {
    "status": "warning-clean",
    "validatedRunCount": MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT,
    "warningRunCount": 0,
    "observedWarningCount": 0,
    "uniqueContractCount": 0,
    "contracts": [],
}
MLX_DIRECTX_CONTEXTUAL_NARROWING_EVIDENCE = {
    "status": "resolved",
    "issue": "https://github.com/CrossGL/crosstl/issues/1801",
    "compiler": "dxc",
    "profile": "cs_6_2",
    "compilerArguments": ["-enable-16bit-types"],
    "resolvedWarningContracts": [
        {
            "classification": "native-int16-destination-conversion",
            "source": MLX_ARANGE_SOURCE,
            "validatedEntryPointCount": MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS[
                MLX_ARANGE_SOURCE
            ],
            "generatedSourceLines": [
                {
                    "text": (
                        "arangeint16_out[index] = int16_t((uint(arangeint16_start) + "
                        "(index * uint(arangeint16_step))));"
                    ),
                    "occurrencesPerArtifact": 1,
                }
            ],
            "observedWarningCount": 0,
            "warningsAsErrors": True,
        },
        {
            "classification": "uint64-local-destination-conversion",
            "source": MLX_ROPE_SOURCE,
            "validatedEntryPointCount": MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS[
                MLX_ROPE_SOURCE
            ],
            "generatedSourceLines": [
                {
                    "text": "index_1 = uint(((2 * pos.x) + (pos.y * stride)));",
                    "occurrencesPerArtifact": 3,
                },
                {
                    "text": "index_1 = uint((pos.x + (pos.y * stride)));",
                    "occurrencesPerArtifact": 3,
                },
            ],
            "observedWarningCount": 0,
            "warningsAsErrors": True,
        },
    ],
    "runtimeExecutionAttempted": False,
    "numericalParityClaimed": False,
}
MLX_DIRECTX_NATIVE_16_BIT_ARITHMETIC_EVIDENCE = {
    "status": "resolved",
    "issue": "https://github.com/CrossGL/crosstl/issues/1802",
    "compiler": "dxc",
    "profile": "cs_6_2",
    "compilerArguments": ["-enable-16bit-types"],
    "source": MLX_ARANGE_SOURCE,
    "validatedEntryPointCount": MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS[
        MLX_ARANGE_SOURCE
    ],
    "generatedSourceLine": (
        "arangefloat16_out[index] = (arangefloat16_start + "
        "(float16_t(index) * arangefloat16_step));"
    ),
    "observedWarningCount": 0,
    "warningsAsErrors": True,
    "runtimeExecutionAttempted": False,
    "numericalParityClaimed": False,
}
MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS = {
    source: MLX_DIRECTX_FRONTIER_ENTRY_POINT_COUNTS[source]
    for source in MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
}
MLX_DIRECTX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS = {
    source: MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS[source]
    for source in MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
}
MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE = (
    "project.translate.workgroup-size-entry-ambiguous"
)
MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_MESSAGE = (
    "A multi-entry compute artifact cannot apply one workgroup size when the "
    "emitted entry points do not declare a shared source size. Select one entry "
    "point or emit independently specialized artifacts."
)
MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE = (
    "https://github.com/CrossGL/crosstl/issues/1793"
)
MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE = {
    MLX_ARG_REDUCE_SOURCE: {
        "specializationCount": 39,
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
        "specializationCount": 14,
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
MLX_REFERENCE_TARGETS = tuple(MLX_FENCE_TARGET_CONTRACTS)
EXPECTED_METAL_KERNEL_COUNT = 40
FULL_CORPUS_EXPECTED_UNIT_COUNT = 42
FULL_CORPUS_TARGETS = ("directx", "opengl")
FULL_CORPUS_EXPECTED_ARTIFACT_COUNT = FULL_CORPUS_EXPECTED_UNIT_COUNT * len(
    FULL_CORPUS_TARGETS
)
FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT = len(FULL_CORPUS_TARGETS)
FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT = (
    FULL_CORPUS_EXPECTED_ARTIFACT_COUNT - FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
)
FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS = 4096
FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK = 131072
FULL_CORPUS_JOB_TIMEOUT_SECONDS = 120
FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS = 900
GEMV_MAX_TEMPLATE_SPECIALIZATIONS = 4096
GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK = 2097152
GEMV_EXPECTED_SPECIALIZATION_COUNT = 226
GEMV_EXPECTED_ENTRY_POINT_COUNT = 224
GEMV_DIRECTX_TRANSLATION_TIMEOUT_SECONDS = 900
GEMV_DIRECTX_ENTRY_PROFILE = "cs_6_0"
GEMV_DIRECTX_LIBRARY_PROFILE = "lib_6_6"
GEMV_SUBGROUP_WIDTH = 32
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
GEMV_OPENGL_TRANSLATION_TIMEOUT_SECONDS = 1500
GEMV_OPENGL_INDEX_RANGE_ASSERTIONS = (
    {
        "expression": "batch_offsets.x",
        "minimum": 0,
        "maximum": (1 << 32) - 1,
    },
    {
        "expression": "batch_offsets.y",
        "minimum": 0,
        "maximum": (1 << 32) - 1,
    },
    {
        "expression": "buffer_load(index_batch_strides, 0) * tid.z",
        "minimum": 0,
        "maximum": (1 << 31) - 1,
    },
    {
        "expression": "buffer_load(index_batch_strides, batch_ndim) * tid.z",
        "minimum": 0,
        "maximum": (1 << 31) - 1,
    },
    {
        "expression": "uint64(bm + tm) * marix_ld + out_col + tn",
        "minimum": 0,
        "maximum": (1 << 32) - 1,
    },
)
GEMV_OPENGL_SUBGROUP_WIDTH_FALLBACK_ISSUE = (
    "https://github.com/CrossGL/crosstl/issues/1894"
)
GEMV_OPENGL_PORTABILITY_TRACKED_ISSUES = (GEMV_OPENGL_SUBGROUP_WIDTH_FALLBACK_ISSUE,)
GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1786",
)
GEMV_OPENGL_SUBGROUP_WIDTH_ENFORCEMENT = {
    "mechanism": "glsl-subgroup-size-guard",
    "shaderExtension": "GL_KHR_shader_subgroup_basic",
    "hostExtension": "GL_KHR_shader_subgroup",
    "hostQuery": "GL_SUBGROUP_SIZE_KHR",
    "artifactMarker": "CROSSTL_REQUIRED_SUBGROUP_WIDTH",
    "mismatchBehavior": "reject-before-dispatch",
}
FFT_MAX_TEMPLATE_SPECIALIZATIONS = 4096
FFT_MAX_TEMPLATE_MATERIALIZATION_WORK = 2097152
FFT_OPENGL_TRANSLATION_TIMEOUT_SECONDS = 900
FFT_OPENGL_EXPECTED_SPECIALIZATION_COUNT = 99
FFT_OPENGL_EXPECTED_FUNCTION_CONSTANT_COUNT = 22
FFT_INDEX_RANGE_ASSERTIONS = tuple(
    {
        "source": MLX_FFT_SOURCE,
        "expression": expression,
        "minimum": 0,
        "maximum": (1 << 32) - 1,
    }
    for expression in (
        "batch_idx + index",
        "batch_idx + index + 1",
        "batch_idx + index + next_in",
        "batch_idx + index + next_out",
    )
)
FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS = tuple(
    {
        "source": MLX_FFT_SOURCE,
        "entry_point": f"*mem_{extent}_*",
        "function": "*",
        "parameter": "*",
        "minimum": 0,
        "maximum": extent - 1,
    }
    for extent in (256, 512, 1024, 2048, 4096)
)
FFT_DIRECTX_ENTRY_POINT = "fft_mem_256_float2_float2"
FFT_DIRECTX_WORKGROUP_SIZE = (1, 1, 64)
FFT_DIRECTX_TRANSLATION_TIMEOUT_SECONDS = 300
FFT_DIRECTX_EXPECTED_SPECIALIZATION_COUNT = 24
FFT_DIRECTX_EXPECTED_FUNCTION_CONSTANT_COUNT = 21
FFT_DIRECTX_GENERATED_SHA256 = (
    "459d779fb07557c81931cadc0bf6c4020d8f08adafc62986504b8f1faf872feb"
)
FFT_DIRECTX_GENERATED_SIZE_BYTES = 116260
FFT_DIRECTX_SPECIALIZATION_CONSTANTS = {
    "0": False,
    "1": True,
    "2": 4,
    "3": 256,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    "10": 4,
    "11": 0,
    "12": 0,
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 0,
    "21": 0,
}
FFT_DIRECTX_REACHABLE_SPECIALIZATION_CONSTANTS = {
    int(selector): value
    for selector, value in FFT_DIRECTX_SPECIALIZATION_CONSTANTS.items()
    if selector != "3"
}
FFT_DIRECTX_WORKGROUP_ACCESS_ASSERTIONS = (
    {
        "source": MLX_FFT_SOURCE,
        "entry_point": FFT_DIRECTX_ENTRY_POINT,
        "function": "*",
        "parameter": "*",
        "minimum": 0,
        "maximum": 255,
    },
)
OPENGL_QUANTIZED_INDEX_TYPE_RESOLVED_ISSUE = (
    "https://github.com/CrossGL/crosstl/issues/1515"
)
MLX_OPENGL_QUANTIZED_INDEX_RANGE_ASSERTION_EXPRESSIONS = (
    "in_index + i",
    "gindex",
    "out_index / writes_per_reduce",
)
MLX_OPENGL_QUANTIZED_INDEX_RANGE_ASSERTIONS = tuple(
    {
        "source": MLX_QUANTIZED_SOURCE,
        "expression": expression,
        "minimum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
        "maximum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
    }
    for expression in MLX_OPENGL_QUANTIZED_INDEX_RANGE_ASSERTION_EXPRESSIONS
)
MLX_DIRECTX_QUANTIZED_FRONTIER_EVIDENCE = {
    "status": "translated-dxc-validated",
    "commit": MLX_COMMIT,
    "source": MLX_QUANTIZED_SOURCE,
    "target": "directx",
    "configured_project_target": "directx-12",
    "compiler_target_profiles": ["directx-12"],
    "selected_entry_point": MLX_QUANTIZED_SELECTED_ENTRY_POINT,
    "artifact_status": "translated",
    "artifact_count": 1,
    "translation_diagnostic_count": 0,
    "required_capabilities": [],
    "generated_hlsl": {
        "sha256": "52569209d98f1bf2ae7fa645f2e4858a420f3920368e14aecb98c2ba9939ac8f",
        "size_bytes": 4357,
    },
    "materialization": {
        "reachable_specialization_count": 6,
        "concrete_specialization_count": 3,
        "pruned_candidate_count": 110861,
    },
    "native_16_bit_emission": {
        "status": "not-required-for-selected-entry",
        "issue": "https://github.com/CrossGL/crosstl/issues/1799",
        "support_status": "resolved-and-validated-elsewhere",
        "reason": "unreachable-native-width-materializations-pruned",
    },
    "concrete_static_assertion_evaluation": {
        "status": "resolved-for-selected-entry",
        "issue": "https://github.com/CrossGL/crosstl/issues/1800",
        "remaining_static_assertion_count": 0,
    },
    "compiler_validation": {
        "compiler": "dxc",
        "profile": "cs_6_0",
        "compiler_arguments": [],
        "warnings_as_errors": True,
        "status": "passed",
        "observed_failure_count": 0,
        "contextual_narrowing": {
            "status": "not-required-for-selected-entry",
            "issue": "https://github.com/CrossGL/crosstl/issues/1801",
            "resource": "out_",
            "resource_element_type": "uint",
            "source_specialized_type": "uint32_t",
            "generated_value_type": "uint",
            "conversion": "not-required",
            "generated_store": "out_[uint((out_index / writes_per_reduce))] = output;",
        },
    },
    "runtime_execution_attempted": False,
    "numerical_parity_claimed": False,
}
MLX_DIRECTX_QUANTIZED_PRIVATE_POINTER_BOUNDARY_EVIDENCE = {
    "status": "translated-dxc-validated",
    "commit": MLX_COMMIT,
    "source": MLX_QUANTIZED_SOURCE,
    "target": "directx",
    "selected_entry_point": "affine_gather_qmv_fast_float_gs_32_b_2",
    "project_translation": {
        "unit_count": 1,
        "artifact_record_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "emitted_target_file_count": 1,
        "project_diagnostic_count": 0,
    },
    "materialization": {
        "reachable_specialization_count": 11,
        "concrete_specialization_count": 8,
        "pruned_candidate_count": 110861,
    },
    "source_contract": {
        "helper": "load_vector<T, U, values_per_thread, bits>",
        "caller_array": "thread U x_thread[values_per_thread]",
        "specialized_extent": 16,
        "loop_step": 4,
    },
    "private_array_aliasing": {
        "status": "passed",
        "helper": "load_vector_float_float_16_2",
        "parameter_mode": "inout",
        "base_offset_parameter": "x_thread_base",
        "extent": 16,
        "writes_per_iteration": 4,
    },
    "weight_byte_view": {
        "status": "passed",
        "helper": "qdot_float_16_2",
        "backing_element_type": "uint32_t",
        "view_element_type": "uint8_t",
        "access": "read",
        "lane_read_count": 4,
        "composed_offset_terms": [
            "w_offset * 4",
            "ws_offset",
            "row * in_vec_size_w",
            "wl_offset",
        ],
    },
    "index_helper_materialization": {
        "status": "passed",
        "helper": "elem_to_loc_uint32_t",
        "source_index_type": "uint32_t",
        "generated_index_type": "uint",
        "resource_offsets": ["x_shape_offset", "x_strides_offset"],
    },
    "pointer_reference_offset_writeback": {
        "status": "passed",
        "helper": "adjust_matrix_offsets_float",
        "offsets": [
            "x_offset",
            "w_offset",
            "scales_offset",
            "biases_offset",
            "y_offset",
        ],
        "downstream_helper": "qmv_fast_impl_float_32_2",
    },
    "execution_contract": {
        "status": "source-verified-and-emitted",
        "workgroup_size": [32, 2, 1],
        "subgroup_width": 32,
        "subgroup_width_enforcement": "WaveSize(32)",
        "minimum_shader_model": "6.6",
        "host_dispatch_provenance": {
            "source": "mlx/backend/metal/quantized.cpp",
            "function": "gather_qmv",
            "workgroup_expression": "MTL::Size group_dims(bk, 2, 1)",
            "bk": 32,
        },
    },
    "generated_hlsl": {
        "sha256": "b7d6251d27fcdafc003c85975bf5c5774a1fca0a3d4602b9e9ea5ef62673f76e",
        "size_bytes": 15835,
    },
    "compiler_validation": {
        "compiler": "dxc",
        "profile": "cs_6_6",
        "compiler_arguments": [],
        "warnings_as_errors": True,
        "status": "passed",
        "observed_failure_count": 0,
    },
    "artifact_emitted": True,
    "native_validation_attempted": True,
    "native_validation_status": "passed",
    "tracked_by": [
        "https://github.com/CrossGL/crosstl/issues/1497",
        "https://github.com/CrossGL/crosstl/issues/1518",
        "https://github.com/CrossGL/crosstl/issues/1546",
        "https://github.com/CrossGL/crosstl/issues/1786",
    ],
    "runtime_execution_attempted": False,
    "numerical_parity_claimed": False,
}
MLX_OPENGL_QUANTIZED_FRONTIER_EVIDENCE = {
    "status": "translated-toolchain-validated-native-loader-executed",
    "commit": MLX_CORPUS_COMMIT,
    "source": MLX_QUANTIZED_SOURCE,
    "source_sha256": "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd",
    "target": "opengl",
    "selected_entry_point": MLX_QUANTIZED_SELECTED_ENTRY_POINT,
    "project_translation": {
        "unit_count": 1,
        "artifact_record_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "emitted_target_file_count": 1,
        "project_diagnostic_count": 0,
        "workgroup_size": [32, 1, 1],
        "subgroup_width_rule_configured": False,
        "max_template_specializations": 128,
        "max_template_materialization_work": 4096,
    },
    "materialization": {
        "reachable_specialization_count": 9,
        "concrete_specialization_count": 3,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 104702,
        "selected_parameters": {
            "T": "float",
            "bits": 2,
            "group_size": 32,
            "has_global_scale": False,
        },
    },
    "index_range_assertion_evidence": {
        "assertion_count": len(MLX_OPENGL_QUANTIZED_INDEX_RANGE_ASSERTIONS),
        "assertions": [
            dict(assertion) for assertion in MLX_OPENGL_QUANTIZED_INDEX_RANGE_ASSERTIONS
        ],
        "inclusive_bounds": {
            "minimum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
            "maximum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
        },
        "contract_kind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtime_enforced": False,
    },
    "software_subgroup": {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "activation": "explicit-target-scoped",
        "operations": [
            "WaveActiveMin(float)",
            "WaveActiveMax(float)",
            "WaveShuffleDown(uint,int)",
        ],
        "artifact_marker": "CROSSTL_SOFTWARE_SUBGROUP_WIDTH",
        "control_barrier_instruction_count": 8,
        "group_non_uniform_instruction_count": 0,
        "hardware_subgroup_extensions_emitted": False,
        "hardware_subgroup_marker_emitted": False,
        "hardware_subgroup_execution_metadata_emitted": False,
        "unsupported_contract_behavior": "reject-before-artifact-emission",
    },
    "generated_glsl": {
        "sha256": "e4d8e5931bfc93f81e2c3686c102a1d676c9a3dcdfd6447e90918aa7581beecb",
        "size_bytes": 6642,
    },
    "required_capabilities": [],
    "artifact_emitted": True,
    "native_validation_attempted": True,
    "native_validation_status": "passed",
    "toolchain_validation": {
        "compiler": "glslangValidator",
        "compiler_target": "OpenGL/SPIR-V 1.3",
        "validator": "spirv-val",
        "validator_target": "SPIR-V 1.3",
        "status": "passed",
        "observed_failure_count": 0,
    },
    "runtime_package": {
        "artifact_count": 1,
        "ready_load_unit_count": 1,
        "blocked_load_unit_count": 0,
        "resources": [
            {"name": "wBuffer", "binding": 0, "access": "read"},
            {"name": "out_Buffer", "binding": 1, "access": "read_write"},
            {"name": "scalesBuffer", "binding": 2, "access": "read_write"},
            {"name": "biasesBuffer", "binding": 3, "access": "read_write"},
        ],
    },
    "runtime_execution": {
        "platform": "Linux Mesa software OpenGL",
        "context": "EGL surfaceless",
        "adapter": "OpenGLRuntimeParityAdapter",
        "runtime": "OpenGLComputeRuntime",
        "workgroup_count": [1, 1, 1],
        "global_size": [32, 1, 1],
        "input": {
            "dtype": "float32",
            "shape": [32],
            "values": "[0.0,1.0,2.0,3.0] repeated 8 times",
        },
        "outputs": {
            "out_Buffer": {
                "dtype": "uint32",
                "shape": [8],
                "values": [27, 27, 27, 27, 27, 27, 27, 27],
            },
            "scalesBuffer": {
                "dtype": "float32",
                "shape": [1],
                "values": [-1.0],
            },
            "biasesBuffer": {
                "dtype": "float32",
                "shape": [1],
                "values": [3.0],
            },
        },
        "status": "passed",
    },
    "resolved_by": [OPENGL_QUANTIZED_INDEX_TYPE_RESOLVED_ISSUE],
    "tracked_issues": ["https://github.com/CrossGL/crosstl/issues/1894"],
    "runtime_execution_attempted": True,
    "runtime_integration_included": True,
    "mlx_host_runtime_integration_included": False,
    "numerical_parity_claimed": True,
    "runtime_parity_claimed": True,
    "parity_scope": "one deterministic affine_quantize_float_gs_32_b_2 workload",
}
MLX_OPENGL_LOGSUMEXP_SOFTWARE_RUNTIME_EVIDENCE = {
    "status": "translated-toolchain-validated-native-loader-required",
    "commit": MLX_CORPUS_COMMIT,
    "source": MLX_LOGSUMEXP_SOURCE,
    "source_sha256": MLX_LOGSUMEXP_SHA256,
    "selected_entry_point": "block_logsumexp_float32",
    "selected_workload": "block-float32-axis-32",
    "dispatch_contract": {
        "path": (
            "demos/integrations/mlx/contracts/" "logsumexp.native-loader.dispatch.json"
        ),
        "content_identity": MLX_LOGSUMEXP_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY,
        "artifact_id": MLX_LOGSUMEXP_DISPATCH_VARIANTS["block-float32-axis-32"][
            "artifactId"
        ],
        "workgroup_size": [32, 1, 1],
        "subgroup_width": 32,
        "dispatch_workgroup_count": [1, 1, 1],
    },
    "project_translation": {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "project_diagnostic_count": 0,
        "subgroup_width_rule_configured": False,
    },
    "materialization": {
        "concrete_specialization_count": 1,
        "reachable_specialization_count": 4,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 76,
        "selected_parameters": {"AccT": "float", "N_READS": "4", "T": "float"},
    },
    "software_subgroup": {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "activation": "explicit-target-scoped",
        "operations": ["WaveActiveMax(float)", "WaveActiveSum(float)"],
        "builtin_substitutions": {
            "gl_NumSubgroups": "1u",
            "gl_SubgroupID": "0u",
            "gl_SubgroupSize": "CROSSTL_SOFTWARE_SUBGROUP_WIDTH",
            "gl_SubgroupInvocationID": "gl_LocalInvocationIndex",
        },
        "artifact_marker": "CROSSTL_SOFTWARE_SUBGROUP_WIDTH",
        "control_barrier_instruction_count": 10,
        "group_non_uniform_instruction_count": 0,
        "hardware_subgroup_extensions_emitted": False,
        "hardware_subgroup_marker_emitted": False,
        "hardware_subgroup_execution_metadata_emitted": False,
        "unsupported_contract_behavior": "reject-before-artifact-emission",
    },
    "generated_glsl": dict(MLX_LOGSUMEXP_SOFTWARE_OPENGL_ARTIFACT),
    "toolchain_validation": {
        "compiler": "glslangValidator",
        "compiler_target": "OpenGL/SPIR-V 1.3",
        "validator": "spirv-val",
        "validator_target": "SPIR-V 1.3",
        "status": "passed",
        "observed_failure_count": 0,
    },
    "runtime_package": {
        "artifact_count": 1,
        "ready_load_unit_count": 1,
        "blocked_load_unit_count": 0,
        "resources": [
            {
                "name": "in_Buffer",
                "binding": 0,
                "access": "read",
                "layout": "std430-float32",
            },
            {
                "name": "out_Buffer",
                "binding": 1,
                "access": "read_write",
                "layout": "std430-float32",
            },
            {
                "name": "block_logsumexp_float32_axis_size_Args",
                "binding": 2,
                "access": "read",
                "layout": "std140-int32-16-byte-block",
            },
        ],
    },
    "workload": {
        "axis_size": 32,
        "input_dtype": "float32",
        "input_shape": [32],
        "input_values": "[(index - 16) / 8.0 for index in range(32)]",
        "expected_output": 3.9978051379373145,
        "absolute_tolerance": 0.00005,
        "relative_tolerance": 0.00005,
    },
    "runtime_execution": {
        "platform": "ubuntu-latest",
        "runtime": "Mesa headless EGL software OpenGL",
        "adapter": "OpenGLRuntimeParityAdapter",
        "workgroup_count": [1, 1, 1],
        "global_size": [32, 1, 1],
        "status": "required-on-ci",
        "test": (
            "tests/test_translator/test_mlx_logsumexp_native_loader.py::"
            "test_pinned_mlx_logsumexp_executes_through_opengl_native_loader"
        ),
    },
    "remaining_scope": {
        "workload": "block-float32-axis-1025",
        "axis_size": 1025,
        "workgroup_size": [288, 1, 1],
        "status": "hardware-subgroup-only",
        "reason": "software-mode-requires-exactly-one-32-thread-subgroup",
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1894",
    },
    "runtime_execution_attempted": True,
    "runtime_integration_included": True,
    "mlx_host_runtime_integration_included": False,
    "selected_workload_numerical_parity_verified": True,
    "full_mlx_test_suite_included": False,
    "numerical_parity_claimed": False,
    "runtime_parity_claimed": False,
}
MLX_CURRENT_ARG_REDUCE_SHA256 = (
    "3d413c4d7eb5a6c397a52487721f445e08ba206a997a90fddcb7e51bf126d1f2"
)
MLX_ARG_REDUCE_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY = (
    "sha256:9c9a196f8bf4c12264422136921d2c6cbf02e1f524e1791ef277406144df1c62"
)
MLX_ARG_REDUCE_NATIVE_LOADER_DISPATCH_VARIANTS = {
    "argmin-float32-axis-32-two-rows": {
        "artifact_id": (
            "sha256:3e7aebfb8869a3c9169b5941a5457b582cc72a098ba3bfd1aec8e4f73c4f8e25"
        ),
        "dispatch_variant_id": (
            "sha256:c000ab737f30b46a98bd1354ccffe502ed0b1460a77b8ba7940469d266d71ae3"
        ),
        "entry_point": "argmin_float32",
        "inputs": {
            "argMax": False,
            "axisSize": 32,
            "dtype": "float32",
            "nRows": 2,
        },
        "workgroup_size": [32, 1, 1],
        "dispatch_workgroup_count": [1, 2, 1],
    },
    "argmax-float32-axis-32-two-rows": {
        "artifact_id": (
            "sha256:1281be909c51f5c74d898e4c1e9992de3572d0135abc42e6bb8ad8308332fa82"
        ),
        "dispatch_variant_id": (
            "sha256:5f6dbeea6ab134cc1d25a3914e643600b4c6eb474eae60a1dac869e1c79b1f06"
        ),
        "entry_point": "argmax_float32",
        "inputs": {
            "argMax": True,
            "axisSize": 32,
            "dtype": "float32",
            "nRows": 2,
        },
        "workgroup_size": [32, 1, 1],
        "dispatch_workgroup_count": [1, 2, 1],
    },
}
MLX_ARG_REDUCE_NATIVE_RUNTIME_EVIDENCE = {
    "status": "translated-packaged-and-cross-target-native-runtime-required",
    "commit": MLX_CORPUS_COMMIT,
    "source": MLX_ARG_REDUCE_SOURCE,
    "source_sha256": MLX_CURRENT_ARG_REDUCE_SHA256,
    "selected_entry_points": ["argmin_float32", "argmax_float32"],
    "dispatch_contract": {
        "path": (
            "demos/integrations/mlx/contracts/" "arg_reduce.native-loader.dispatch.json"
        ),
        "content_identity": MLX_ARG_REDUCE_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY,
        "workload_count": len(MLX_ARG_REDUCE_NATIVE_LOADER_DISPATCH_VARIANTS),
        "host_formula": (
            "roundUp(min(ceilDiv(axisSize, 4), " "maxThreadsPerWorkgroup), simdWidth)"
        ),
        "bounded_axis_size": 32,
        "subgroup_width": 32,
        "variants": MLX_ARG_REDUCE_NATIVE_LOADER_DISPATCH_VARIANTS,
    },
    "project_translation": {
        "unit_count": 1,
        "artifact_count_by_target": {"directx": 2, "opengl": 2},
        "translated_count_by_target": {"directx": 2, "opengl": 2},
        "failed_count_by_target": {"directx": 0, "opengl": 0},
        "project_diagnostic_count": 0,
    },
    "materialization": {
        "concrete_specialization_count": 5,
        "reachable_specialization_count": 10,
        "dependency_discovery_work_count": 76,
        "pruned_candidate_count": 274,
        "selected_parameters_by_entry": {
            "argmin_float32": {
                "N_READS": "4",
                "Op": "ArgMin<float>",
                "T": "float",
            },
            "argmax_float32": {
                "N_READS": "4",
                "Op": "ArgMax<float>",
                "T": "float",
            },
        },
        "overloaded_helper_selection": {
            "source_name": "elem_to_loc",
            "materialized_name": "elem_to_loc_int64_t",
            "selected_first_parameter": "int64_t",
            "rejected_first_parameter": "uint3",
            "resolution": "signature-aware-call-site-materialization",
        },
    },
    "artifacts": {
        "directx": {
            "argmin_float32": {
                "sha256": (
                    "b63a160f4ca102cc6407d88b84f5c3e6f849840f73e57d372722f9828569a34d"
                ),
                "size_bytes": 6655,
            },
            "argmax_float32": {
                "sha256": (
                    "99359b364f701a420f1ee918f481eaabac7d5ea9b02c5872c915c7672c60b398"
                ),
                "size_bytes": 6657,
            },
            "subgroup_enforcement": "hlsl-wave-size-attribute",
            "compiler": "dxc",
            "compiler_version": "1.9.2602.24",
            "compiler_profile": "cs_6_6",
            "compiler_arguments": ["-enable-16bit-types", "-WX"],
            "compiler_validation_status": "passed",
        },
        "opengl": {
            "argmin_float32": {
                "sha256": (
                    "b74534a5120665ad07755141af2a73702cb5ea504a0526b92306eabedfed4765"
                ),
                "size_bytes": 7581,
            },
            "argmax_float32": {
                "sha256": (
                    "d90e758132832490b7f356c6750d4deb2bdb3341f053a921228a9f24ce8d27d8"
                ),
                "size_bytes": 7587,
            },
            "subgroup_enforcement": "explicit-32-lane-software-subgroup",
            "compiler": "glslangValidator",
            "compiler_target": "OpenGL/SPIR-V 1.3",
            "validator": "spirv-val",
            "control_barrier_instruction_count": 5,
            "group_non_uniform_instruction_count": 0,
            "compiler_validation_status": "passed",
        },
    },
    "directx_relative_shuffle": {
        "configuration": (
            "project.source_options.metal.target_options.directx."
            "relative_wave_shuffle_out_of_range"
        ),
        "policy": "self",
        "default_policy": "undefined",
        "activation": "explicit-target-scoped",
        "selected_value_types": ["float", "uint"],
        "invalid_source_lane_result": "calling-lane-value",
        "invalid_source_lane_reads_emitted": False,
        "source_lane_bounds": {
            "down_valid_when": "delta < laneCount - lane",
            "read_lane": "valid ? lane + delta : lane",
        },
        "wave_read_control_flow": "single-unconditional-selected-lane-read",
    },
    "software_subgroup": {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "activation": "explicit-target-scoped",
        "selected_kernel_operations": [
            "WaveShuffleDown(float, uint)",
            "WaveShuffleDown(uint, uint)",
        ],
        "uniform_loop_contract": {
            "condition": "offset > 0",
            "updates": ["offset /= constant>=2", "offset >>= constant>=1"],
            "helper_call_form": "direct-top-level-statement",
            "unsafe_loop_behavior": "reject-before-artifact-emission",
        },
        "hardware_subgroup_extensions_emitted": False,
        "group_non_uniform_instruction_count": 0,
    },
    "runtime_package": {
        "artifact_count_per_variant_and_target": 1,
        "ready_load_unit_count_per_variant_and_target": 1,
        "blocked_load_unit_count": 0,
        "resource_count_by_target": {"directx": 9, "opengl": 8},
        "resource_element_types": [
            "float32",
            "uint32",
            "int32",
            "int64",
            "uint64",
        ],
        "directx_generated_dispatch_binding": "CrossGLDispatchInfo",
        "scalar_64_bit_layout_reflected": True,
    },
    "workloads": {
        "input_shape": [2, 32],
        "input_values": (
            "deterministic float32 rows with repeated extrema at distinct indices"
        ),
        "argmin_expected_indices": [5, 7],
        "argmax_expected_indices": [3, 2],
        "tie_behavior": "lowest index",
        "output_dtype": "uint32",
    },
    "native_runtime": {
        "directx": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_arg_reduce_native_loader.py::"
                "test_pinned_mlx_arg_reduce_executes_through_directx_native_loader"
            ),
        },
        "opengl": {
            "platform": "ubuntu-latest",
            "runtime": "mesa-headless-egl-software-opengl",
            "status": "required-on-ci",
            "local_linux_arm64_validation": "passed",
            "test": (
                "tests/test_translator/test_mlx_arg_reduce_native_loader.py::"
                "test_pinned_mlx_arg_reduce_executes_through_opengl_native_loader"
            ),
        },
    },
    "metal_roundtrip_boundary": {
        "status": "entry-workgroup-specialization-target-unsupported",
        "diagnostic": "project.translate.workgroup-size-rule-unsupported-target",
        "missing_capability": "execution.workgroup-size-specialization",
    },
    "remaining_scope": {
        "other_axis_sizes_included": False,
        "other_dtypes_included": False,
        "remaining_host_named_entries_included": False,
        "aggregate_directx_opengl_translation_unblocked": False,
        "mlx_host_runtime_integration_included": False,
        "full_mlx_test_suite_included": False,
    },
    "runtime_execution_attempted": True,
    "runtime_integration_included": True,
    "selected_workloads_numerical_parity_verified": True,
    "complete_runtime_coverage_claimed": False,
    "full_mlx_test_suite_included": False,
    "numerical_parity_claimed": False,
    "runtime_parity_claimed": False,
}
MLX_CURRENT_SCALED_DOT_PRODUCT_ATTENTION_SHA256 = (
    "f6fefad1d91b01f05c12095e69f248e255f880d65b5ae9e9d8bc2714da56fb41"
)
MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY = (
    "sha256:97e5ebb69af8da3a0082776015787456f23b8bfdb0cff757f5364db2cfef8d2c"
)
MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_LOADER_DISPATCH_VARIANTS = {
    "vector-float32-b1-h1-q1-k4-d64-v64-nomask": {
        "artifact_id": (
            "sha256:dd0138695bd82e1f8ea49bd667052b484420ee96cb2849c6eed20ba5eae39a89"
        ),
        "dispatch_variant_id": (
            "sha256:8b2abb9f7179e051530697fb8d1956d0ff03a324e7acaa5fcdf4f4dd9f1befbb"
        ),
        "entry_point": "sdpa_vector_float_64_64",
        "inputs": {
            "batchSize": 1,
            "boolMask": False,
            "doCausal": False,
            "dtype": "float32",
            "floatMask": False,
            "hasMask": False,
            "hasSinks": False,
            "keyLength": 4,
            "kvHeads": 1,
            "queryDimension": 64,
            "queryHeads": 1,
            "queryLength": 1,
            "queryTransposed": False,
            "valueDimension": 64,
        },
        "workgroup_size": [1024, 1, 1],
        "dispatch_workgroup_count": [1, 1, 1],
        "specialization_constants": {
            "20": False,
            "21": False,
            "22": False,
            "23": False,
            "24": False,
            "25": False,
        },
    }
}
MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_RUNTIME_EVIDENCE = {
    "status": "translated-packaged-and-cross-target-native-runtime-required",
    "commit": MLX_CORPUS_COMMIT,
    "source": MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
    "source_sha256": MLX_CURRENT_SCALED_DOT_PRODUCT_ATTENTION_SHA256,
    "selected_entry_point": "sdpa_vector_float_64_64",
    "selected_workload": "vector-float32-b1-h1-q1-k4-d64-v64-nomask",
    "dispatch_contract": {
        "path": (
            "demos/integrations/mlx/contracts/"
            "scaled_dot_product_attention.native-loader.dispatch.json"
        ),
        "content_identity": (
            MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY
        ),
        "workload_count": len(
            MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_LOADER_DISPATCH_VARIANTS
        ),
        "host_selection": "one-pass-vector",
        "host_source": "mlx/backend/metal/scaled_dot_product_attention.cpp",
        "implementation_source": "mlx/backend/metal/kernels/sdpa_vector.h",
        "subgroup_width": 32,
        "variants": MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_LOADER_DISPATCH_VARIANTS,
    },
    "project_translation": {
        "unit_count": 1,
        "artifact_count_by_target": {"directx": 1, "opengl": 1},
        "translated_count_by_target": {"directx": 1, "opengl": 1},
        "failed_count_by_target": {"directx": 0, "opengl": 0},
        "project_diagnostic_count": 0,
    },
    "materialization": {
        "concrete_specialization_count": 1,
        "reachable_specialization_count": 4,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 753,
        "selected_parameters": {"D": "64", "T": "float", "V": "64"},
    },
    "specialization_constants": {
        "20": {"name": "has_mask", "value": False},
        "21": {"name": "query_transposed", "value": False},
        "22": {"name": "do_causal", "value": False},
        "23": {"name": "bool_mask", "value": False},
        "24": {"name": "float_mask", "value": False},
        "25": {"name": "has_sinks", "value": False},
    },
    "artifacts": {
        "directx": {
            "target_entry_point": "CSMain",
            "sha256": (
                "2182a09b1e03815f11e36c3ab1addb2138257e0bcf69284f99a0c33ec344816b"
            ),
            "size_bytes": 8721,
            "workgroup_size": [1024, 1, 1],
            "subgroup_enforcement": "hlsl-wave-size-attribute",
            "subgroup_id_lowering": "workgroup-synchronized-physical-wave-allocation",
            "compiler": "dxc",
            "compiler_version": "1.9.2602.24",
            "compiler_profile": "cs_6_6",
            "compiler_arguments": ["-enable-16bit-types", "-WX"],
            "compiled_dxil_size_bytes": 9000,
            "compiler_validation_status": "passed",
            "specialization_materialization": "concrete",
        },
        "opengl": {
            "target_entry_point": "main",
            "sha256": (
                "9b7cb7dc9a76b9fb93c30fd93d13ad639f5493f60fd97b965514db0fe6b4840b"
            ),
            "size_bytes": 12089,
            "workgroup_size": [1024, 1, 1],
            "subgroup_enforcement": "explicit-32-lane-software-subgroup",
            "compiler": "glslangValidator",
            "compiler_target": "OpenGL/SPIR-V 1.3",
            "validator": "spirv-val",
            "control_barrier_instruction_count": 9,
            "group_non_uniform_instruction_count": 0,
            "specialization_constant_false_count": 6,
            "compiler_validation_status": "passed",
            "specialization_materialization": "deferred",
        },
    },
    "software_subgroup": {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "logical_subgroup_count": 32,
        "activation": "explicit-target-scoped",
        "selected_kernel_operations": [
            "WaveActiveMax(float)",
            "WaveActiveSum(float)",
        ],
        "synchronized_subgroup_strided_loop_contract": {
            "initializer": "subgroup-id-semantic",
            "bound": "workgroup-uniform-runtime-value",
            "stride": 32,
            "collective_count_per_loop": 1,
            "inactive_lane_behavior": "typed-collective-identity",
            "unsafe_loop_behavior": "reject-before-artifact-emission",
        },
        "hardware_subgroup_extensions_emitted": False,
        "group_non_uniform_instruction_count": 0,
    },
    "runtime_package": {
        "artifact_count_per_target": 1,
        "ready_load_unit_count_per_target": 1,
        "blocked_load_unit_count": 0,
        "resource_count_by_target": {"directx": 19, "opengl": 18},
        "specialization_constant_count": 6,
        "resource_element_types": ["float32", "int32", "uint32", "uint64"],
        "stored_bool_physical_type": "uint32",
        "optional_placeholder_resources": ["bmask", "fmask", "sinks"],
        "directx_generated_dispatch_binding": "CrossGLDispatchInfo",
        "opengl_deferred_variant_registry": "ready",
        "opengl_native_registry_header": {
            "available": False,
            "reason": "specialization-requires-deferred-compilation",
        },
    },
    "workload": {
        "batch_size": 1,
        "query_heads": 1,
        "kv_heads": 1,
        "query_length": 1,
        "key_length": 4,
        "query_dimension": 64,
        "value_dimension": 64,
        "scale": 0.125,
        "mask": "none",
        "causal": False,
        "sinks": False,
        "output_element_count": 64,
        "reference": "stable-float32-scaled-dot-product-attention",
        "absolute_tolerance": 0.0002,
        "relative_tolerance": 0.0002,
    },
    "native_runtime": {
        "directx": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/"
                "test_mlx_scaled_dot_product_attention_native_loader.py::"
                "test_pinned_mlx_attention_executes_through_directx_native_loader"
            ),
        },
        "opengl": {
            "platform": "ubuntu-latest",
            "runtime": "mesa-headless-egl-software-opengl",
            "status": "required-on-ci",
            "local_linux_mesa_validation": "passed",
            "local_max_absolute_error": 4.082320426146424e-08,
            "local_max_relative_error": 4.2163276126605175e-06,
            "test": (
                "tests/test_translator/"
                "test_mlx_scaled_dot_product_attention_native_loader.py::"
                "test_pinned_mlx_attention_executes_through_opengl_native_loader"
            ),
        },
    },
    "metal_roundtrip_boundary": {
        "status": "outside-selected-native-loader-proof",
        "aggregate_native_baseline": "separate",
        "selected_entry_compiler_validation_included": False,
    },
    "remaining_scope": {
        "masked_attention_included": False,
        "causal_attention_included": False,
        "sinks_included": False,
        "two_pass_attention_included": False,
        "full_attention_path_included": False,
        "other_dimensions_included": False,
        "other_dtypes_included": False,
        "remaining_host_named_entries_included": False,
        "aggregate_directx_opengl_translation_unblocked": False,
        "mlx_host_runtime_integration_included": False,
        "full_mlx_test_suite_included": False,
    },
    "runtime_execution_attempted": True,
    "runtime_integration_included": True,
    "selected_workload_numerical_parity_verified": True,
    "complete_runtime_coverage_claimed": False,
    "full_mlx_test_suite_included": False,
    "numerical_parity_claimed": False,
    "runtime_parity_claimed": False,
}
MLX_CURRENT_SOFTMAX_SHA256 = (
    "d19231c66973edc3944f12529d1cc393029e7f7262b914907c710ef9dbcb39e2"
)
MLX_SOFTMAX_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY = (
    "sha256:1ef1b663b0ff87dbd193bf73dded4dd1c8008e03dcd9c7e8d9ff8aaad832b006"
)
MLX_SOFTMAX_NATIVE_LOADER_DISPATCH_VARIANTS = {
    "block-float32-axis-32-two-rows": {
        "artifact_id": (
            "sha256:fa3eb57ad31078c7300855a29c8d67d5f6ac9595e33b651085a9e5fcf3883ea2"
        ),
        "dispatch_variant_id": (
            "sha256:59a3029f50afad7ed80bd0533849cf33f1cdd8d2e5586b6b17b82d0e979119e8"
        ),
        "inputs": {"axisSize": 32, "dtype": "float32", "nRows": 2},
        "workgroup_size": [32, 1, 1],
        "dispatch_workgroup_count": [2, 1, 1],
    },
    "block-float32-axis-2049": {
        "artifact_id": (
            "sha256:956659b706f518421d9f99b08d028ec0a9ceeb359a556a7777c565dfb8e5df37"
        ),
        "dispatch_variant_id": (
            "sha256:fa751ebed377b5371cb0a4f91c38eaf8ac95e6430f03ac4a21ef4bddcda91a0d"
        ),
        "inputs": {"axisSize": 2049, "dtype": "float32", "nRows": 1},
        "workgroup_size": [544, 1, 1],
        "dispatch_workgroup_count": [1, 1, 1],
    },
}
MLX_SOFTMAX_NATIVE_RUNTIME_EVIDENCE = {
    "status": "translated-packaged-and-cross-target-native-runtime-required",
    "commit": MLX_CORPUS_COMMIT,
    "source": MLX_SOFTMAX_SOURCE,
    "source_sha256": MLX_CURRENT_SOFTMAX_SHA256,
    "selected_entry_point": "block_softmax_float32",
    "dispatch_contract": {
        "path": (
            "demos/integrations/mlx/contracts/" "softmax.native-loader.dispatch.json"
        ),
        "content_identity": MLX_SOFTMAX_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY,
        "workload_count": len(MLX_SOFTMAX_NATIVE_LOADER_DISPATCH_VARIANTS),
        "host_formula": "32 * ceilDiv(ceilDiv(axisSize, 4), 32)",
        "block_limit": 4096,
        "subgroup_width": 32,
        "variants": MLX_SOFTMAX_NATIVE_LOADER_DISPATCH_VARIANTS,
    },
    "project_translation": {
        "unit_count": 1,
        "guarded_artifact_count_by_target": {"directx": 2, "opengl": 2},
        "software_opengl_artifact_count": 2,
        "translated_count_by_target": {"directx": 2, "opengl": 2},
        "failed_count_by_target": {"directx": 0, "opengl": 0},
        "project_diagnostic_count": 0,
    },
    "materialization": {
        "concrete_specialization_count": 2,
        "reachable_specialization_count": 5,
        "dependency_discovery_work_count": 11,
        "pruned_candidate_count": 131,
        "selected_parameters": {
            "AccT": "float",
            "N_READS": "SOFTMAX_N_READS",
            "T": "float",
        },
    },
    "guarded_artifacts": {
        "directx": {
            "block-float32-axis-32-two-rows": {
                "sha256": (
                    "de5ae241c037cf9a0b37f456d777d51c544c8f725cf2efe8a51c45ae968a0fc2"
                ),
                "size_bytes": 4213,
                "subgroup_id_lowering": "fixed-single-wave-group-index-quotient",
            },
            "block-float32-axis-2049": {
                "sha256": (
                    "3a3f443fdb6df38e37bda4828601b967cc6aa216c0805e6dc060be7e7bddd02c"
                ),
                "size_bytes": 4784,
                "subgroup_id_lowering": (
                    "workgroup-synchronized-physical-wave-allocation"
                ),
            },
            "subgroup_enforcement": "hlsl-wave-size-attribute",
            "compiler": "dxc",
            "compiler_profile": "cs_6_6",
            "compiler_arguments": ["-enable-16bit-types"],
            "warnings_as_errors": True,
            "compiler_validation_status": "passed",
        },
        "opengl": {
            "block-float32-axis-32-two-rows": {
                "sha256": (
                    "5a1da27d4acc858937ddbfc73aed27b2270c9b17e02d6cf9ebb920980dbc5a51"
                ),
                "size_bytes": 4663,
            },
            "block-float32-axis-2049": {
                "sha256": (
                    "9d185604faa31f5dd013b2e009683658c063f0105f516fb02891ea3ab2b5485f"
                ),
                "size_bytes": 4664,
            },
            "subgroup_enforcement": "glsl-subgroup-size-guard",
            "compiler": "glslangValidator",
            "validator": "spirv-val",
        },
    },
    "software_opengl_artifacts": {
        "block-float32-axis-32-two-rows": {
            "sha256": (
                "f69dad597cefc34f7908799aaf0ba2eac47a0dcdd91e5f2bf3d7247172fa84b9"
            ),
            "size_bytes": 5585,
            "workgroup_size": [32, 1, 1],
            "logical_subgroup_count": 1,
            "masked_collective_count": 0,
            "control_barrier_instruction_count": 11,
            "group_non_uniform_instruction_count": 0,
        },
        "block-float32-axis-2049": {
            "sha256": (
                "eb195e15089f4e7bade380af55e8b7e167c4b89f80b2f25675eb71196a5468ce"
            ),
            "size_bytes": 7204,
            "workgroup_size": [544, 1, 1],
            "logical_subgroup_count": 17,
            "masked_collective_count": 2,
            "control_barrier_instruction_count": 11,
            "group_non_uniform_instruction_count": 0,
        },
        "compiler": "glslangValidator",
        "compiler_target": "OpenGL/SPIR-V 1.3",
        "validator": "spirv-val",
        "validator_target": "SPIR-V 1.3",
        "status": "passed",
    },
    "software_subgroup": {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "activation": "explicit-target-scoped",
        "selected_kernel_operations": [
            "WaveActiveMax(float)",
            "WaveActiveSum(float)",
        ],
        "masked_reduction_operations": [
            "WaveActiveSum",
            "WaveActiveMin",
            "WaveActiveMax",
        ],
        "typed_mask_identities": {
            "WaveActiveSum": {"float": "0.0", "int": "0", "uint": "0u"},
            "WaveActiveMin": {
                "float": "+infinity",
                "int": "INT_MAX",
                "uint": "UINT_MAX",
            },
            "WaveActiveMax": {
                "float": "-infinity",
                "int": "INT_MIN",
                "uint": "0u",
            },
        },
        "masked_shuffle_supported": False,
        "artifact_marker": "CROSSTL_SOFTWARE_SUBGROUP_WIDTH",
        "hardware_subgroup_extensions_emitted": False,
        "hardware_subgroup_marker_emitted": False,
        "hardware_subgroup_execution_metadata_emitted": False,
        "unsupported_contract_behavior": "reject-before-artifact-emission",
    },
    "runtime_package": {
        "artifact_count_per_variant_and_target": 1,
        "ready_load_unit_count_per_variant_and_target": 1,
        "blocked_load_unit_count": 0,
        "resource_count": 3,
        "resources": [
            {"binding": 0, "role": "input", "dtype": "float32"},
            {"binding": 1, "role": "output", "dtype": "float32"},
            {"binding": 2, "role": "axis_size", "dtype": "int32"},
        ],
    },
    "workloads": {
        "block-float32-axis-32-two-rows": {
            "input_shape": [2, 32],
            "input_values": "row0=(index-16)/8; row1=(((index*7)%19)-9)/5",
            "reference": "row-wise stable exp(x-max)/sum(exp(x-max))",
            "output_element_count": 64,
        },
        "block-float32-axis-2049": {
            "input_shape": [2049],
            "input_values": "(((index*13)%37)-18)/7",
            "reference": "stable exp(x-max)/sum(exp(x-max))",
            "output_element_count": 2049,
        },
        "absolute_tolerance": 0.00005,
        "relative_tolerance": 0.00005,
    },
    "native_runtime": {
        "directx": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_softmax_native_loader.py::"
                "test_pinned_mlx_softmax_executes_through_directx_native_loader"
            ),
        },
        "opengl": {
            "platform": "ubuntu-latest",
            "runtime": "mesa-headless-egl-software-opengl",
            "status": "required-on-ci",
            "local_linux_arm64_validation": "passed",
            "test": (
                "tests/test_translator/test_mlx_softmax_native_loader.py::"
                "test_pinned_mlx_softmax_executes_through_opengl_native_loader"
            ),
        },
    },
    "metal_roundtrip_boundary": {
        "status": "entry-scoped-target-unsupported",
        "diagnostic": "project.translate.entry-point-target-unsupported",
        "missing_capability": "artifact.entry-point-selection",
    },
    "remaining_scope": {
        "looped_axis_sizes_above_4096_included": False,
        "float16_and_bfloat16_included": False,
        "precise_half_and_bfloat16_entries_included": False,
        "other_axis_sizes_included": False,
        "mlx_host_runtime_integration_included": False,
        "full_mlx_test_suite_included": False,
    },
    "runtime_execution_attempted": True,
    "runtime_integration_included": True,
    "selected_workload_numerical_parity_verified": True,
    "complete_runtime_coverage_claimed": False,
    "full_mlx_test_suite_included": False,
    "numerical_parity_claimed": False,
    "runtime_parity_claimed": False,
}
MLX_CURRENT_RMS_NORM_SHA256 = (
    "b2e04e377fdad1d645581f9beeaf9cbb06d1ad32926161e06cbc15240caf12bf"
)
MLX_RMS_NORM_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY = (
    "sha256:f733849c0ded2be96de6b5b6df6df662751c733fa13c2036baf9a834db3981c5"
)
MLX_RMS_NORM_NATIVE_RUNTIME_EVIDENCE = {
    "status": "translated-packaged-and-native-runtime-required",
    "commit": MLX_CORPUS_COMMIT,
    "source": MLX_RMS_NORM_SOURCE,
    "source_sha256": MLX_CURRENT_RMS_NORM_SHA256,
    "selected_entry_point": "rmsfloat32",
    "selected_workload": "forward-float32-axis-32",
    "dispatch_contract": {
        "path": (
            "demos/integrations/mlx/contracts/" "rms_norm.native-loader.dispatch.json"
        ),
        "content_identity": MLX_RMS_NORM_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY,
        "artifact_id": (
            "sha256:00c05fccf276cf11f3fb9b617b8fe0bb3c5f8766c0e4ca1ed990c093e700422e"
        ),
        "workload_count": 1,
        "workgroup_size": [32, 1, 1],
        "subgroup_width": 32,
        "dispatch_workgroup_count": [2, 1, 1],
        "function_constants": {},
    },
    "entry_scoped_specialization_ownership": {
        "constant_name": "has_w",
        "constant_id": 20,
        "reachable_from_selected_entry": False,
        "artifact_specialization_constant_count": 0,
        "runtime_manifest_specialization_constant_count": 0,
        "reachable_vjp_constants_preserved": True,
        "resolved_issue": "https://github.com/CrossGL/crosstl/issues/1795",
    },
    "project_translation": {
        "unit_count": 1,
        "artifact_count_by_target": {"directx": 1, "opengl": 1},
        "translated_count_by_target": {"directx": 1, "opengl": 1},
        "failed_count_by_target": {"directx": 0, "opengl": 0},
        "project_diagnostic_count": 0,
        "workgroup_size": [32, 1, 1],
        "directx_subgroup_width": 32,
        "opengl_subgroup_width_rule_configured": False,
    },
    "materialization": {
        "concrete_specialization_count": 1,
        "reachable_specialization_count": 4,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 168,
        "selected_parameters": {"N_READS": "RMS_N_READS", "T": "float"},
    },
    "artifacts": {
        "directx": {
            "target_entry_point": "CSMain",
            "sha256": (
                "83f7b6e437122b2afe3dbc5d7649f6bc882d671947bcc72579ae5aa568fb2a5b"
            ),
            "size_bytes": 3486,
            "subgroup_enforcement": "hlsl-wave-size-attribute",
            "compiler": "dxc",
            "compiler_profile": "cs_6_6",
            "compiler_arguments": ["-enable-16bit-types"],
            "native_runtime": {
                "platform": "windows-latest",
                "runtime": "direct3d-12-warp",
                "status": "required-on-ci",
                "test": (
                    "tests/test_translator/test_mlx_rms_norm_native_loader.py::"
                    "test_pinned_mlx_rms_norm_executes_through_directx_native_loader"
                ),
            },
        },
        "opengl": {
            "target_entry_point": "main",
            "sha256": (
                "3180aba83b64add0ae3c2d471b9297eb5bada4c4ff2bd5c91a3db3698cf0df78"
            ),
            "size_bytes": 4393,
            "subgroup_enforcement": "explicit-32-lane-software-subgroup",
            "compiler": "glslangValidator",
            "validator": "spirv-val",
            "control_barrier_instruction_count": 6,
            "group_non_uniform_instruction_count": 0,
            "native_runtime": {
                "platform": "ubuntu-latest",
                "runtime": "mesa-headless-egl-software-opengl",
                "status": "required-on-ci",
                "test": (
                    "tests/test_translator/test_mlx_rms_norm_native_loader.py::"
                    "test_pinned_mlx_rms_norm_executes_through_opengl_native_loader"
                ),
            },
        },
    },
    "software_subgroup": {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "activation": "explicit-target-scoped",
        "operations": ["WaveActiveSum(float)"],
        "artifact_marker": "CROSSTL_SOFTWARE_SUBGROUP_WIDTH",
        "control_barrier_instruction_count": 6,
        "group_non_uniform_instruction_count": 0,
        "hardware_subgroup_extensions_emitted": False,
        "hardware_subgroup_marker_emitted": False,
        "hardware_subgroup_execution_metadata_emitted": False,
        "unsupported_contract_behavior": "reject-before-artifact-emission",
    },
    "runtime_package": {
        "artifact_count_by_target": {"directx": 1, "opengl": 1},
        "ready_load_unit_count_by_target": {"directx": 1, "opengl": 1},
        "blocked_load_unit_count_by_target": {"directx": 0, "opengl": 0},
        "resource_count": 6,
        "resources": [
            {"binding": 0, "role": "input", "dtype": "float32"},
            {"binding": 1, "role": "weight", "dtype": "float32"},
            {"binding": 2, "role": "output", "dtype": "float32"},
            {"binding": 3, "role": "epsilon", "dtype": "float32"},
            {"binding": 4, "role": "axis_size", "dtype": "uint32"},
            {"binding": 5, "role": "weight_stride", "dtype": "uint32"},
        ],
    },
    "workload": {
        "dtype": "float32",
        "shape": [2, 32],
        "axis_size": 32,
        "row_count": 2,
        "weight_shape": [32],
        "epsilon": 0.00001,
        "weight_stride": 1,
        "input_values": "row0=(index-16)/8; row1=((index%9)-4)*0.3125",
        "weight_values": "0.5+(index%5)*0.125",
        "reference": "x*w*rsqrt(mean(x*x)+epsilon)",
        "output_element_count": 64,
        "absolute_tolerance": 0.00003,
        "relative_tolerance": 0.00003,
    },
    "remaining_scope": {
        "forward_entries_other_than_rmsfloat32_included": False,
        "vjp_entries_included": False,
        "float16_and_bfloat16_included": False,
        "looped_entries_included": False,
        "other_axis_sizes_included": False,
        "historical_compiler_dispatch_record_count": 12,
        "mlx_host_runtime_integration_included": False,
    },
    "runtime_execution_attempted": True,
    "runtime_integration_included": True,
    "selected_workload_numerical_parity_verified": True,
    "complete_runtime_coverage_claimed": False,
    "full_mlx_test_suite_included": False,
    "numerical_parity_claimed": False,
    "runtime_parity_claimed": False,
    "resolved_by": ["https://github.com/CrossGL/crosstl/issues/1795"],
}
REDUCED_FRONTIER_MODE = "reduced-frontier"
FULL_CORPUS_MODE = "full-corpus"
FRONTIER_VALIDATION_TRACKED_ISSUES: tuple[str, ...] = ()
FULL_CORPUS_TRANSLATION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1376",
    "https://github.com/CrossGL/crosstl/issues/1676",
    "https://github.com/CrossGL/crosstl/issues/1479",
    "https://github.com/CrossGL/crosstl/issues/1490",
    "https://github.com/CrossGL/crosstl/issues/1497",
    "https://github.com/CrossGL/crosstl/issues/1544",
    "https://github.com/CrossGL/crosstl/issues/1546",
    "https://github.com/CrossGL/crosstl/issues/1554",
    "https://github.com/CrossGL/crosstl/issues/1559",
    "https://github.com/CrossGL/crosstl/issues/1562",
    "https://github.com/CrossGL/crosstl/issues/1669",
    "https://github.com/CrossGL/crosstl/issues/1671",
)
FULL_CORPUS_VALIDATION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1670",
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
    *DIRECTX_TOOLCHAIN_WARNING_TRACKED_ISSUES,
    *FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
    *FULL_CORPUS_VALIDATION_TRACKED_ISSUES,
    *OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES,
    *OPENGL_SCALED_DOT_PRODUCT_ATTENTION_TRACKED_ISSUES,
    *GEMV_OPENGL_PORTABILITY_TRACKED_ISSUES,
    *RUNTIME_READINESS_TRACKED_ISSUES,
    *FENCE_CONTRACT_TRACKED_ISSUES,
    *VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES,
    VULKAN_GEMV_REPORTING_TRACKED_ISSUE,
    *FULL_CORPUS_SEMANTIC_TRACKED_ISSUES,
    *METAL_ROUNDTRIP_SEMANTIC_TRACKED_ISSUES,
)
RESOLVED_FRONTIER_ISSUES = (
    OPENGL_QUANTIZED_INDEX_TYPE_RESOLVED_ISSUE,
    MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
    "https://github.com/CrossGL/crosstl/issues/1576",
    "https://github.com/CrossGL/crosstl/issues/1799",
    "https://github.com/CrossGL/crosstl/issues/1800",
    "https://github.com/CrossGL/crosstl/issues/1801",
    "https://github.com/CrossGL/crosstl/issues/1802",
    "https://github.com/CrossGL/crosstl/issues/1672",
    "https://github.com/CrossGL/crosstl/issues/1659",
    "https://github.com/CrossGL/crosstl/issues/1516",
    "https://github.com/CrossGL/crosstl/issues/1476",
    "https://github.com/CrossGL/crosstl/issues/1472",
    "https://github.com/CrossGL/crosstl/issues/1312",
    "https://github.com/CrossGL/crosstl/issues/1792",
    "https://github.com/CrossGL/crosstl/issues/1790",
    "https://github.com/CrossGL/crosstl/issues/1789",
    "https://github.com/CrossGL/crosstl/issues/1787",
    "https://github.com/CrossGL/crosstl/issues/1784",
    "https://github.com/CrossGL/crosstl/issues/1750",
    "https://github.com/CrossGL/crosstl/issues/1726",
    "https://github.com/CrossGL/crosstl/issues/1474",
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


def _normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    variant_specialization_constants: (
        Mapping[str, Mapping[str, bool | int | float]] | None
    ) = None,
    metal_source_options: Mapping[str, int] | None = None,
    metal_target_options: Mapping[str, Mapping[str, int]] | None = None,
    entry_points: Mapping[str, str] | None = None,
    workgroup_size_rules: Mapping[str, Sequence[str | int]] | None = None,
    entry_workgroup_size_rules: (
        Mapping[str, Mapping[str, Sequence[str | int]]] | None
    ) = None,
    subgroup_width_rules: Mapping[str, str | int] | None = None,
    index_range_assertions: Sequence[Mapping[str, str | int]] | None = None,
    workgroup_access_assertions: Sequence[Mapping[str, str | int]] | None = None,
    dispatch_contracts: Sequence[str] | None = None,
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
    ]
    if dispatch_contracts:
        contract_list = ", ".join(json.dumps(path) for path in dispatch_contracts)
        lines.append(f"dispatch_contracts = [{contract_list}]")
    lines.extend(("", "[project.sources]", '"**/*.metal" = "metal"', ""))
    for assertion in index_range_assertions or ():
        lines.append("[[project.index_range_assertions]]")
        for field in ("source", "function", "expression", "minimum", "maximum"):
            if field in assertion:
                lines.append(f"{field} = {json.dumps(assertion[field])}")
        lines.append("")
    for assertion in workgroup_access_assertions or ():
        lines.append("[[project.workgroup_access_assertions]]")
        for field in (
            "source",
            "entry_point",
            "function",
            "parameter",
            "minimum",
            "maximum",
        ):
            if field in assertion:
                lines.append(f"{field} = {json.dumps(assertion[field])}")
        lines.append("")
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
    if variant_specialization_constants:
        for variant, constants in variant_specialization_constants.items():
            lines.append(
                "[project.variants." f"{json.dumps(variant)}.specialization_constants]"
            )
            for selector, value in constants.items():
                lines.append(f"{json.dumps(str(selector))} = {json.dumps(value)}")
            lines.append("")
    if workgroup_size_rules:
        lines.append("[project.workgroup_size_rules]")
        for source, components in workgroup_size_rules.items():
            rule = ", ".join(json.dumps(component) for component in components)
            lines.append(f"{json.dumps(source)} = [{rule}]")
        lines.append("")
    if entry_workgroup_size_rules:
        for source, entry_rules in entry_workgroup_size_rules.items():
            lines.append(
                "[project.entry_workgroup_size_rules." f"{json.dumps(source)}]"
            )
            for entry_pattern, components in entry_rules.items():
                rule = ", ".join(json.dumps(component) for component in components)
                lines.append(f"{json.dumps(entry_pattern)} = [{rule}]")
            lines.append("")
    if subgroup_width_rules:
        lines.append("[project.subgroup_width_rules]")
        for source, expression in subgroup_width_rules.items():
            lines.append(f"{json.dumps(source)} = {json.dumps(str(expression))}")
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


def _prepare_layer_norm_dispatch_contract(
    mlx_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    source_path = MLX_LAYER_NORM_DISPATCH_CONTRACT_SOURCE
    _require(
        source_path.is_file(), f"LayerNorm dispatch contract is missing: {source_path}"
    )
    _require(
        _normalized_text_sha256(source_path)
        == MLX_LAYER_NORM_DISPATCH_NORMALIZED_SHA256,
        "LayerNorm dispatch contract normalized content changed",
    )
    layer_norm_source = mlx_root / MLX_LAYER_NORM_SOURCE
    _require(
        layer_norm_source.is_file()
        and _sha256(layer_norm_source) == MLX_LAYER_NORM_SHA256,
        "pinned LayerNorm source identity changed",
    )

    manifest = load_dispatch_contract(source_path)
    content_identity = manifest.content_identity.to_json()
    expected_identity = {
        "algorithm": "sha256",
        "value": MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix("sha256:"),
    }
    _require(
        content_identity == expected_identity,
        "LayerNorm dispatch contract identity changed",
    )
    evaluation = manifest.evaluate().to_json()
    variants = evaluation.get("variants")
    variants_by_entry = {
        variant.get("entryPoint"): variant
        for variant in variants or []
        if isinstance(variant, Mapping)
    }
    _require(
        isinstance(variants, list)
        and len(variants) == len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
        and set(variants_by_entry) == set(MLX_LAYER_NORM_DISPATCH_VARIANTS),
        "LayerNorm dispatch contract variant set changed",
    )
    for entry_point, expected in MLX_LAYER_NORM_DISPATCH_VARIANTS.items():
        variant = variants_by_entry[entry_point]
        _require(
            variant.get("artifactId") == expected["artifactId"]
            and variant.get("variantId") == expected["dispatchVariantId"]
            and variant.get("source") == MLX_LAYER_NORM_SOURCE
            and variant.get("workgroupSize") == expected["workgroupSize"]
            and variant.get("subgroupWidth") == expected["subgroupWidth"]
            and variant.get("specializationConstants")
            == expected["specializationConstants"],
            f"LayerNorm dispatch contract changed for {entry_point}",
        )

    destination = work_dir / "contracts" / "layer_norm.dispatch.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return {
        "path": _relpath(destination, mlx_root),
        "contentIdentity": content_identity,
        "variantCount": len(variants),
    }


def _prepare_logsumexp_dispatch_contract(
    mlx_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    source_path = MLX_LOGSUMEXP_DISPATCH_CONTRACT_SOURCE
    _require(
        source_path.is_file(),
        f"LogSumExp dispatch contract is missing: {source_path}",
    )
    _require(
        _normalized_text_sha256(source_path)
        == MLX_LOGSUMEXP_DISPATCH_NORMALIZED_SHA256,
        "LogSumExp dispatch contract normalized content changed",
    )
    logsumexp_source = mlx_root / MLX_LOGSUMEXP_SOURCE
    _require(
        logsumexp_source.is_file()
        and _sha256(logsumexp_source) == MLX_LOGSUMEXP_SHA256,
        "pinned LogSumExp source identity changed",
    )

    manifest = load_dispatch_contract(source_path)
    content_identity = manifest.content_identity.to_json()
    expected_identity = {
        "algorithm": "sha256",
        "value": MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY.removeprefix("sha256:"),
    }
    _require(
        content_identity == expected_identity,
        "LogSumExp dispatch contract identity changed",
    )
    evaluation = manifest.evaluate().to_json()
    variants = evaluation.get("variants")
    variants_by_workload = {
        variant.get("workload", {}).get("id"): variant
        for variant in variants or []
        if isinstance(variant, Mapping) and isinstance(variant.get("workload"), Mapping)
    }
    _require(
        isinstance(variants, list)
        and len(variants) == len(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
        and set(variants_by_workload) == set(MLX_LOGSUMEXP_DISPATCH_VARIANTS),
        "LogSumExp dispatch contract variant set changed",
    )
    for workload_id, expected in MLX_LOGSUMEXP_DISPATCH_VARIANTS.items():
        variant = variants_by_workload[workload_id]
        _require(
            variant.get("artifactId") == expected["artifactId"]
            and variant.get("variantId") == expected["dispatchVariantId"]
            and variant.get("source") == MLX_LOGSUMEXP_SOURCE
            and variant.get("entryPoint") == expected["entryPoint"]
            and variant.get("workload", {}).get("inputs") == expected["inputs"]
            and variant.get("workgroupSize") == expected["workgroupSize"]
            and variant.get("subgroupWidth") == 32
            and variant.get("specializationConstants")
            == expected["specializationConstants"]
            and variant.get("dispatch", {}).get("workgroupCount")
            == expected["dispatchWorkgroupCount"],
            f"LogSumExp dispatch contract changed for {workload_id}",
        )

    destination = work_dir / "contracts" / "logsumexp.dispatch.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return {
        "path": _relpath(destination, mlx_root),
        "contentIdentity": content_identity,
        "variantCount": len(variants),
    }


def _prepare_rms_norm_dispatch_contract(
    mlx_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    source_path = MLX_RMS_NORM_DISPATCH_CONTRACT_SOURCE
    _require(
        source_path.is_file(), f"RMSNorm dispatch contract is missing: {source_path}"
    )
    _require(
        _normalized_text_sha256(source_path) == MLX_RMS_NORM_DISPATCH_NORMALIZED_SHA256,
        "RMSNorm dispatch contract normalized content changed",
    )
    rms_norm_source = mlx_root / MLX_RMS_NORM_SOURCE
    _require(
        rms_norm_source.is_file() and _sha256(rms_norm_source) == MLX_RMS_NORM_SHA256,
        "pinned RMSNorm source identity changed",
    )

    manifest = load_dispatch_contract(source_path)
    content_identity = manifest.content_identity.to_json()
    expected_identity = {
        "algorithm": "sha256",
        "value": MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix("sha256:"),
    }
    _require(
        content_identity == expected_identity,
        "RMSNorm dispatch contract identity changed",
    )
    evaluation = manifest.evaluate().to_json()
    variants = evaluation.get("variants")
    variants_by_workload = {
        variant.get("workload", {}).get("id"): variant
        for variant in variants or []
        if isinstance(variant, Mapping) and isinstance(variant.get("workload"), Mapping)
    }
    _require(
        isinstance(variants, list)
        and len(variants) == len(MLX_RMS_NORM_DISPATCH_VARIANTS)
        and set(variants_by_workload) == set(MLX_RMS_NORM_DISPATCH_VARIANTS),
        "RMSNorm dispatch contract variant set changed",
    )
    for workload_id, expected in MLX_RMS_NORM_DISPATCH_VARIANTS.items():
        variant = variants_by_workload[workload_id]
        _require(
            variant.get("artifactId") == expected["artifactId"]
            and variant.get("variantId") == expected["dispatchVariantId"]
            and variant.get("source") == MLX_RMS_NORM_SOURCE
            and variant.get("entryPoint") == expected["entryPoint"]
            and variant.get("workload", {}).get("inputs") == expected["inputs"]
            and variant.get("workgroupSize") == expected["workgroupSize"]
            and variant.get("subgroupWidth") == 32
            and variant.get("specializationConstants")
            == expected["specializationConstants"]
            and variant.get("dispatch", {}).get("workgroupCount")
            == expected["dispatchWorkgroupCount"],
            f"RMSNorm dispatch contract changed for {workload_id}",
        )

    destination = work_dir / "contracts" / "rms_norm.dispatch.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return {
        "path": _relpath(destination, mlx_root),
        "contentIdentity": content_identity,
        "variantCount": len(variants),
    }


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


def _verify_mlx_checkout(
    mlx_root: Path,
    python: str,
    log_dir: Path,
    *,
    expected_commit: str = MLX_COMMIT,
) -> dict[str, Any]:
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
        revision == expected_commit,
        f"MLX checkout must be pinned to {expected_commit}; found {revision}",
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
    *,
    expected_unit_count: int = EXPECTED_METAL_KERNEL_COUNT,
    targets: Sequence[str] = MLX_REFERENCE_TARGETS,
) -> dict[str, Any]:
    config_path = config_dir / "scan-metal-kernels.toml"
    report_path = report_dir / "scan-metal-kernels.json"
    _write_project_config(
        config_path,
        include=f"{MLX_METAL_KERNEL_ROOT}/**/*.metal",
        targets=targets,
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
        summary.get("unitCount") == expected_unit_count,
        "expected {} MLX Metal kernels, found {}".format(
            expected_unit_count,
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
    targets: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected_targets = tuple(targets or MLX_FENCE_TARGET_CONTRACTS)
    _require(
        all(target in MLX_FENCE_TARGET_CONTRACTS for target in selected_targets),
        "fence contract contains an unsupported target",
    )
    summary = payload.get("summary", {})
    _require(isinstance(summary, Mapping), "fence contract summary must be an object")
    expected_diagnostic_codes = {
        contract["diagnosticCode"]: 1
        for target, contract in MLX_FENCE_TARGET_CONTRACTS.items()
        if target in selected_targets
    }
    expected_missing_capabilities = {
        contract["missingCapability"]: 1
        for target, contract in MLX_FENCE_TARGET_CONTRACTS.items()
        if target in selected_targets
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
            and summary.get("artifactCount") == len(selected_targets)
            and summary.get("translatedCount") == 0
            and summary.get("failedCount") == len(selected_targets),
            "fence contract translation must report one failed artifact per target",
        )
        _require(
            summary.get("diagnosticCounts")
            == {"error": len(selected_targets), "note": 0, "warning": 0},
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
            len(diagnostics) == len(selected_targets)
            and len(artifacts) == len(selected_targets),
            "fence contract report must contain one diagnostic and artifact per target",
        )

    target_results: dict[str, dict[str, Any]] = {}
    output_root = output_dir.resolve()
    for target in selected_targets:
        contract = MLX_FENCE_TARGET_CONTRACTS[target]
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
    generated = artifact_path.read_text(encoding="utf-8")
    profile = dxc_profile_for_source("cs_6_0", generated)
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    output_path = work_dir / "validation" / "reference-accessor.dxil"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    result = _run_command(
        "validate-reference-accessor-directx",
        [
            dxc,
            "-T",
            profile,
            *compiler_arguments,
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
        "profile": profile,
        **(
            {
                "compilerArguments": list(compiler_arguments),
                "minimumShaderModel": "6.2",
            }
            if compiler_arguments
            else {}
        ),
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


_DXC_REPORT_WARNING_PATTERN = re.compile(
    r"^.+:\d+:\d+:\s+warning:\s+(?P<message>.+)$", re.IGNORECASE
)


def _dxc_report_warnings(run: Mapping[str, Any]) -> list[tuple[str, str]]:
    stderr = run.get("stderr", "")
    _require(
        isinstance(stderr, str),
        "DirectX toolchain validation recorded non-text stderr",
    )
    warnings: list[tuple[str, str]] = []
    lines = stderr.splitlines()
    for index, line in enumerate(lines):
        if "warning:" not in line.lower():
            continue
        match = _DXC_REPORT_WARNING_PATTERN.match(line)
        _require(match is not None, f"DXC emitted an unrecognized warning: {line}")
        _require(
            index + 1 < len(lines) and lines[index + 1].strip(),
            f"DXC warning omitted its generated source line: {line}",
        )
        warnings.append((match.group("message"), lines[index + 1].strip()))
    return warnings


def _directx_toolchain_warning_evidence(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contracts_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for contract in MLX_DIRECTX_TOOLCHAIN_WARNING_CONTRACTS:
        contracts_by_source.setdefault(contract["source"], []).append(contract)

    observed_counts: Counter[str] = Counter()
    warning_run_count = 0
    for run in runs:
        source = run.get("source")
        _require(
            isinstance(source, str),
            "DirectX toolchain validation omitted a source path",
        )
        warnings = Counter(_dxc_report_warnings(run))
        expected_contracts = contracts_by_source.get(source, [])
        expected_warnings: Counter[tuple[str, str]] = Counter()
        for contract in expected_contracts:
            for source_line in contract["sourceLines"]:
                expected_warnings[
                    (contract["message"], source_line["text"])
                ] += source_line["occurrencesPerRun"]
        _require(
            warnings == expected_warnings,
            f"DirectX toolchain warnings changed for {source}: "
            f"expected {dict(expected_warnings)}, observed {dict(warnings)}",
        )
        if warnings:
            warning_run_count += 1
        for contract in expected_contracts:
            observed_counts[contract["classification"]] += sum(
                warnings[(contract["message"], source_line["text"])]
                for source_line in contract["sourceLines"]
            )

    contracts = [
        {
            **contract,
            "observedCount": observed_counts[contract["classification"]],
        }
        for contract in MLX_DIRECTX_TOOLCHAIN_WARNING_CONTRACTS
    ]
    return {
        "status": "warning-clean" if runs else "not-run",
        "validatedRunCount": len(runs),
        "warningRunCount": warning_run_count,
        "observedWarningCount": sum(observed_counts.values()),
        "uniqueContractCount": len(contracts),
        "contracts": contracts,
    }


def _require_frontier_project_join(
    payload: Mapping[str, Any],
    *,
    target: str,
    sources: Sequence[str],
    index_range_assertions: Sequence[Mapping[str, str | int]] | None = None,
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
    if index_range_assertions is not None:
        expected_assertions = [dict(assertion) for assertion in index_range_assertions]
        _require(
            project.get("indexRangeAssertions") == expected_assertions
            and type(project.get("indexRangeAssertionCount")) is int
            and project.get("indexRangeAssertionCount") == len(expected_assertions),
            f"{target.title()} frontier index-range assertion contract changed",
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
    index_range_assertions: Sequence[Mapping[str, str | int]] | None = None,
) -> dict[str, Mapping[str, Any]]:
    units_by_source = _require_frontier_project_join(
        payload,
        target=target,
        sources=sources,
        index_range_assertions=index_range_assertions,
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


def _require_layer_norm_dispatch_frontier_report(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    validated: bool,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], list[Mapping[str, Any]]]:
    project = payload.get("project")
    contract_records = (
        project.get("dispatchContracts") if isinstance(project, Mapping) else None
    )
    _require(
        isinstance(project, Mapping)
        and project.get("includePatterns") == [MLX_LAYER_NORM_SOURCE]
        and project.get("targets") == ["directx"]
        and project.get("dispatchContractFiles") == [contract["path"]]
        and project.get("dispatchContractCount") == 1
        and project.get("dispatchVariantCount") == len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
        and isinstance(contract_records, list)
        and len(contract_records) == 1,
        "LayerNorm dispatch frontier project metadata changed",
    )
    contract_record = contract_records[0]
    expected_manifest_path = str((mlx_root / str(contract["path"])).resolve())
    _require(
        isinstance(contract_record, Mapping)
        and contract_record.get("path") == contract["path"]
        and contract_record.get("schemaVersion") == 1
        and contract_record.get("contentIdentity") == contract["contentIdentity"]
        and contract_record.get("manifest", {}).get("provenance", {}).get("commit")
        == MLX_COMMIT
        and contract_record.get("evaluation", {}).get("manifestSource")
        == expected_manifest_path
        and contract_record.get("evaluation", {}).get("variantCount")
        == len(MLX_LAYER_NORM_DISPATCH_VARIANTS),
        "LayerNorm dispatch frontier did not retain replayable manifest provenance",
    )

    summary = payload.get("summary")
    artifact_count = len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("artifactCount") == artifact_count
        and summary.get("translatedCount") == artifact_count
        and summary.get("failedCount") == 0
        and summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0}
        and summary.get("artifactsByTarget", {}).get("directx")
        == {
            "artifactCount": artifact_count,
            "translatedCount": artifact_count,
            "failedCount": 0,
        }
        and payload.get("diagnostics") == [],
        "LayerNorm dispatch frontier accounting changed",
    )
    units = payload.get("units")
    _require(
        isinstance(units, list)
        and len(units) == 1
        and units[0].get("path") == MLX_LAYER_NORM_SOURCE
        and units[0].get("sourceBackend") == "metal"
        and units[0].get("sourceHash")
        == {"algorithm": "sha256", "value": MLX_LAYER_NORM_SHA256},
        "LayerNorm dispatch frontier source identity changed",
    )

    dispatch_plan = project.get("dispatchArtifactPlan")
    planned_artifacts = (
        dispatch_plan.get("artifacts") if isinstance(dispatch_plan, Mapping) else None
    )
    _require(
        isinstance(dispatch_plan, Mapping)
        and dispatch_plan.get("kind") == "crosstl-dispatch-artifact-plan"
        and dispatch_plan.get("schemaVersion") == 1
        and dispatch_plan.get("sourceUnitCount") == 1
        and dispatch_plan.get("artifactCount") == artifact_count
        and dispatch_plan.get("dispatchVariantCount") == artifact_count
        and isinstance(planned_artifacts, list)
        and len(planned_artifacts) == artifact_count,
        "LayerNorm dispatch artifact plan changed",
    )
    plan_by_entry = {
        record.get("entryPoint"): record
        for record in planned_artifacts
        if isinstance(record, Mapping)
    }
    _require(
        set(plan_by_entry) == set(MLX_LAYER_NORM_DISPATCH_VARIANTS),
        "LayerNorm dispatch artifact plan entry set changed",
    )

    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == artifact_count,
        "LayerNorm dispatch artifact records are incomplete",
    )
    output_root = output_dir.resolve()
    artifacts_by_entry: dict[str, Mapping[str, Any]] = {}
    generated_evidence: dict[str, Any] = {}
    for artifact in artifacts:
        entry = artifact.get("entryPoint") if isinstance(artifact, Mapping) else None
        entry_point = entry.get("source") if isinstance(entry, Mapping) else None
        _require(
            isinstance(entry_point, str)
            and entry_point in MLX_LAYER_NORM_DISPATCH_VARIANTS
            and entry_point not in artifacts_by_entry,
            "LayerNorm dispatch artifact entry identity changed",
        )
        expected = MLX_LAYER_NORM_DISPATCH_VARIANTS[entry_point]
        plan = plan_by_entry[entry_point]
        variant_name = "dispatch-" + expected["artifactId"].removeprefix("sha256:")
        _require(
            artifact.get("source") == MLX_LAYER_NORM_SOURCE
            and artifact.get("sourceBackend") == "metal"
            and artifact.get("sourceHash")
            == {"algorithm": "sha256", "value": MLX_LAYER_NORM_SHA256}
            and artifact.get("target") == "directx"
            and artifact.get("status") == "translated"
            and artifact.get("variant") == variant_name
            and entry.get("target") == "CSMain"
            and entry.get("stage") == "compute"
            and artifact.get("dispatchArtifact") == plan
            and plan.get("artifactId") == expected["artifactId"]
            and plan.get("dispatchVariantIds") == [expected["dispatchVariantId"]]
            and plan.get("manifestContentIdentities")
            == [MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY]
            and plan.get("source") == MLX_LAYER_NORM_SOURCE
            and plan.get("workgroupSize") == expected["workgroupSize"]
            and plan.get("subgroupWidth") == expected["subgroupWidth"]
            and plan.get("specializationConstants")
            == expected["specializationConstants"],
            f"LayerNorm dispatch artifact contract changed for {entry_point}",
        )

        execution = artifact.get("execution")
        execution_entries = (
            execution.get("entryPoints") if isinstance(execution, Mapping) else None
        )
        _require(
            isinstance(execution, Mapping)
            and execution.get("sourceEntryPoints") == [entry_point]
            and execution.get("provenance", {}).get("kind") == "host-dispatch-contract"
            and execution.get("provenance", {}).get("artifactId")
            == expected["artifactId"]
            and execution.get("subgroupWidthProvenance", {}).get("kind")
            == "host-dispatch-contract"
            and execution.get("subgroupWidthEnforcement")
            == {
                "mechanism": "hlsl-wave-size-attribute",
                "minimumShaderModel": "6.6",
                "entryProfiles": [{"entryPoint": "CSMain", "profile": "cs_6_6"}],
            }
            and isinstance(execution_entries, list)
            and len(execution_entries) == 1
            and execution_entries[0].get("sourceEntryPoint") == entry_point
            and execution_entries[0].get("materializedEntryPoint") == entry_point
            and execution_entries[0].get("targetEntryPoint") == "CSMain"
            and execution_entries[0].get("workgroupSize") == expected["workgroupSize"]
            and execution_entries[0].get("subgroupWidth") == expected["subgroupWidth"],
            f"LayerNorm execution metadata changed for {entry_point}",
        )

        constants = artifact.get("specializationConstants") or []
        constants_by_id = {
            str(record.get("id")): record
            for record in constants
            if isinstance(record, Mapping)
        }
        _require(
            set(constants_by_id) == set(expected["specializationConstants"]),
            f"LayerNorm specialization inputs changed for {entry_point}",
        )
        for constant_id, value in expected["specializationConstants"].items():
            record = constants_by_id[constant_id]
            _require(
                record.get("concreteValue") is value
                and record.get("deferred") is False
                and record.get("valueProvenance", {}).get("kind")
                == "host-dispatch-contract"
                and record.get("valueProvenance", {}).get("artifactId")
                == expected["artifactId"],
                f"LayerNorm specialization provenance changed for {entry_point}",
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
        _require(
            isinstance(materialization, Mapping)
            and materialization.get("status") == "materialized"
            and materialization.get("specializationCount")
            == expected["specializationCount"]
            and isinstance(specializations, list)
            and len(specializations) == expected["specializationCount"]
            and materialization.get("unsupported") == []
            and host_names == [entry_point],
            f"LayerNorm materialization changed for {entry_point}",
        )

        artifact_path_value = artifact.get("path")
        _require(
            isinstance(artifact_path_value, str) and artifact_path_value,
            f"LayerNorm artifact path is missing for {entry_point}",
        )
        artifact_path = (mlx_root / artifact_path_value).resolve()
        _require(
            _is_relative_to(artifact_path, output_root) and artifact_path.is_file(),
            f"LayerNorm artifact is missing or escaped output for {entry_point}",
        )
        generated = artifact_path.read_text(encoding="utf-8")
        generated_hash = _sha256(artifact_path)
        _require(
            artifact.get("generatedHash")
            == {"algorithm": "sha256", "value": generated_hash}
            and artifact.get("generatedSizeBytes") == artifact_path.stat().st_size
            and len(re.findall(r"\bvoid\s+CSMain\s*\(", generated)) == 1
            and re.search(r"\[\s*WaveSize\s*\(\s*32\s*\)\s*\]", generated) is not None
            and re.search(
                r"\[\s*numthreads\s*\(\s*{}\s*,\s*1\s*,\s*1\s*\)\s*\]".format(
                    expected["workgroupSize"][0]
                ),
                generated,
            )
            is not None,
            f"LayerNorm generated HLSL contract changed for {entry_point}",
        )
        if expected["specializationConstants"]:
            _require(
                re.search(r"\bstatic\s+const\s+bool\s+has_w\s*=\s*true\s*;", generated)
                is not None,
                "LayerNorm VJP artifact did not materialize has_w=true",
            )
        else:
            _require(
                re.search(r"\bhas_w\b", generated) is None,
                "LayerNorm forward artifact retained an unreachable function constant",
            )
        generated_evidence[entry_point] = {
            "artifactId": expected["artifactId"],
            "dispatchVariantId": expected["dispatchVariantId"],
            "workgroupSize": list(expected["workgroupSize"]),
            "subgroupWidth": expected["subgroupWidth"],
            "specializationConstants": dict(expected["specializationConstants"]),
            "generatedHlsl": {
                "sha256": generated_hash,
                "sizeBytes": artifact_path.stat().st_size,
            },
        }
        artifacts_by_entry[entry_point] = artifact

    toolchain_runs: list[Mapping[str, Any]] = []
    if validated:
        validation = payload.get("validation")
        runs = (
            validation.get("toolchainRuns") if isinstance(validation, Mapping) else None
        )
        _require(
            isinstance(validation, Mapping)
            and isinstance(validation.get("summary"), Mapping)
            and validation["summary"].get("failedCount") == 0
            and isinstance(runs, list),
            "LayerNorm dispatch DXC validation changed",
        )
        toolchain_runs = [
            run
            for run in runs
            if isinstance(run, Mapping) and run.get("target") == "directx"
        ]
        _require(
            len(toolchain_runs) == artifact_count
            and all(run.get("status") == "ok" for run in toolchain_runs),
            "LayerNorm dispatch DXC did not validate both artifacts",
        )

    evidence = {
        "status": "translated-dxc-validated" if validated else "translated",
        "source": MLX_LAYER_NORM_SOURCE,
        "sourceSha256": MLX_LAYER_NORM_SHA256,
        "target": "directx",
        "dispatchContract": {
            "path": contract["path"],
            "contentIdentity": MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY,
            "variantCount": artifact_count,
            "resolvedIssue": MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
        },
        "artifactCount": artifact_count,
        "variants": generated_evidence,
        "dxcValidatedArtifactCount": len(toolchain_runs),
        "runtimeExecutionAttempted": False,
        "numericalParityClaimed": False,
    }
    return artifacts_by_entry, evidence, toolchain_runs


def _require_logsumexp_dispatch_frontier_report(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    validated: bool,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], list[Mapping[str, Any]]]:
    project = payload.get("project")
    contract_records = (
        project.get("dispatchContracts") if isinstance(project, Mapping) else None
    )
    artifact_count = len(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
    _require(
        isinstance(project, Mapping)
        and project.get("includePatterns") == [MLX_LOGSUMEXP_SOURCE]
        and project.get("targets") == ["directx"]
        and project.get("dispatchContractFiles") == [contract["path"]]
        and project.get("dispatchContractCount") == 1
        and project.get("dispatchVariantCount") == artifact_count
        and isinstance(contract_records, list)
        and len(contract_records) == 1,
        "LogSumExp dispatch frontier project metadata changed",
    )
    contract_record = contract_records[0]
    expected_manifest_path = str((mlx_root / str(contract["path"])).resolve())
    evaluation = (
        contract_record.get("evaluation")
        if isinstance(contract_record, Mapping)
        else None
    )
    evaluated_variants = (
        evaluation.get("variants") if isinstance(evaluation, Mapping) else None
    )
    _require(
        isinstance(contract_record, Mapping)
        and contract_record.get("path") == contract["path"]
        and contract_record.get("schemaVersion") == 1
        and contract_record.get("contentIdentity") == contract["contentIdentity"]
        and contract_record.get("manifest", {}).get("provenance", {}).get("commit")
        == MLX_COMMIT
        and isinstance(evaluation, Mapping)
        and evaluation.get("manifestSource") == expected_manifest_path
        and evaluation.get("variantCount") == artifact_count
        and isinstance(evaluated_variants, list)
        and len(evaluated_variants) == artifact_count,
        "LogSumExp dispatch frontier did not retain replayable manifest provenance",
    )
    evaluation_by_workload = {
        variant.get("workload", {}).get("id"): variant
        for variant in evaluated_variants
        if isinstance(variant, Mapping) and isinstance(variant.get("workload"), Mapping)
    }
    _require(
        set(evaluation_by_workload) == set(MLX_LOGSUMEXP_DISPATCH_VARIANTS),
        "LogSumExp dispatch evaluation workload set changed",
    )

    summary = payload.get("summary")
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("artifactCount") == artifact_count
        and summary.get("translatedCount") == artifact_count
        and summary.get("failedCount") == 0
        and summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0}
        and summary.get("artifactsByTarget", {}).get("directx")
        == {
            "artifactCount": artifact_count,
            "translatedCount": artifact_count,
            "failedCount": 0,
        }
        and payload.get("diagnostics") == [],
        "LogSumExp dispatch frontier accounting changed",
    )
    units = payload.get("units")
    _require(
        isinstance(units, list)
        and len(units) == 1
        and units[0].get("path") == MLX_LOGSUMEXP_SOURCE
        and units[0].get("sourceBackend") == "metal"
        and units[0].get("sourceHash")
        == {"algorithm": "sha256", "value": MLX_LOGSUMEXP_SHA256},
        "LogSumExp dispatch frontier source identity changed",
    )

    dispatch_plan = project.get("dispatchArtifactPlan")
    planned_artifacts = (
        dispatch_plan.get("artifacts") if isinstance(dispatch_plan, Mapping) else None
    )
    expected_artifact_ids = {
        expected["artifactId"] for expected in MLX_LOGSUMEXP_DISPATCH_VARIANTS.values()
    }
    _require(
        isinstance(dispatch_plan, Mapping)
        and dispatch_plan.get("kind") == "crosstl-dispatch-artifact-plan"
        and dispatch_plan.get("schemaVersion") == 1
        and dispatch_plan.get("sourceUnitCount") == 1
        and dispatch_plan.get("artifactCount") == artifact_count
        and dispatch_plan.get("dispatchVariantCount") == artifact_count
        and isinstance(planned_artifacts, list)
        and len(planned_artifacts) == artifact_count,
        "LogSumExp dispatch artifact plan changed",
    )
    plan_by_artifact_id = {
        record.get("artifactId"): record
        for record in planned_artifacts
        if isinstance(record, Mapping)
    }
    _require(
        set(plan_by_artifact_id) == expected_artifact_ids,
        "LogSumExp dispatch artifact plan identity set changed",
    )

    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == artifact_count,
        "LogSumExp dispatch artifact records are incomplete",
    )
    artifacts_by_id = {
        artifact.get("dispatchArtifact", {}).get("artifactId"): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and isinstance(artifact.get("dispatchArtifact"), Mapping)
    }
    _require(
        set(artifacts_by_id) == expected_artifact_ids,
        "LogSumExp dispatch artifact identity set changed",
    )

    output_root = output_dir.resolve()
    artifacts_by_workload: dict[str, Mapping[str, Any]] = {}
    generated_evidence: dict[str, Any] = {}
    expected_template_parameters = {"AccT": "float", "N_READS": "4", "T": "float"}
    for workload_id, expected in MLX_LOGSUMEXP_DISPATCH_VARIANTS.items():
        artifact_id = expected["artifactId"]
        artifact = artifacts_by_id[artifact_id]
        plan = plan_by_artifact_id[artifact_id]
        evaluated = evaluation_by_workload[workload_id]
        entry = artifact.get("entryPoint")
        entry_point = entry.get("source") if isinstance(entry, Mapping) else None
        _require(
            evaluated.get("artifactId") == artifact_id
            and evaluated.get("variantId") == expected["dispatchVariantId"]
            and evaluated.get("entryPoint") == expected["entryPoint"]
            and evaluated.get("workload", {}).get("inputs") == expected["inputs"]
            and evaluated.get("workgroupSize") == expected["workgroupSize"]
            and evaluated.get("subgroupWidth") == 32
            and evaluated.get("specializationConstants") == {}
            and evaluated.get("dispatch", {}).get("workgroupCount")
            == expected["dispatchWorkgroupCount"],
            f"LogSumExp evaluated dispatch changed for {workload_id}",
        )
        variant_name = "dispatch-" + artifact_id.removeprefix("sha256:")
        _require(
            artifact.get("source") == MLX_LOGSUMEXP_SOURCE
            and artifact.get("sourceBackend") == "metal"
            and artifact.get("sourceHash")
            == {"algorithm": "sha256", "value": MLX_LOGSUMEXP_SHA256}
            and artifact.get("target") == "directx"
            and artifact.get("status") == "translated"
            and artifact.get("variant") == variant_name
            and entry_point == expected["entryPoint"]
            and entry.get("target") == "CSMain"
            and entry.get("stage") == "compute"
            and artifact.get("dispatchArtifact") == plan
            and plan.get("artifactId") == artifact_id
            and plan.get("dispatchVariantIds") == [expected["dispatchVariantId"]]
            and plan.get("manifestContentIdentities")
            == [MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY]
            and plan.get("source") == MLX_LOGSUMEXP_SOURCE
            and plan.get("entryPoint") == expected["entryPoint"]
            and plan.get("workgroupSize") == expected["workgroupSize"]
            and plan.get("subgroupWidth") == 32
            and plan.get("specializationConstants") == {},
            f"LogSumExp dispatch artifact contract changed for {workload_id}",
        )

        execution = artifact.get("execution")
        execution_entries = (
            execution.get("entryPoints") if isinstance(execution, Mapping) else None
        )
        _require(
            isinstance(execution, Mapping)
            and execution.get("sourceEntryPoints") == [expected["entryPoint"]]
            and execution.get("provenance", {}).get("kind") == "host-dispatch-contract"
            and execution.get("provenance", {}).get("artifactId") == artifact_id
            and execution.get("subgroupWidthProvenance", {}).get("kind")
            == "host-dispatch-contract"
            and execution.get("subgroupWidthEnforcement")
            == {
                "mechanism": "hlsl-wave-size-attribute",
                "minimumShaderModel": "6.6",
                "entryProfiles": [{"entryPoint": "CSMain", "profile": "cs_6_6"}],
            }
            and isinstance(execution_entries, list)
            and len(execution_entries) == 1
            and execution_entries[0].get("sourceEntryPoint") == expected["entryPoint"]
            and execution_entries[0].get("materializedEntryPoint")
            == expected["entryPoint"]
            and execution_entries[0].get("targetEntryPoint") == "CSMain"
            and execution_entries[0].get("workgroupSize") == expected["workgroupSize"]
            and execution_entries[0].get("subgroupWidth") == 32,
            f"LogSumExp execution metadata changed for {workload_id}",
        )
        _require(
            not artifact.get("specializationConstants"),
            f"LogSumExp artifact gained specialization constants for {workload_id}",
        )

        materialization = artifact.get("templateMaterialization")
        specializations = (
            materialization.get("specializations")
            if isinstance(materialization, Mapping)
            else None
        )
        _require(
            isinstance(materialization, Mapping)
            and materialization.get("status") == "materialized"
            and materialization.get("specializationCount") == 1
            and isinstance(specializations, list)
            and len(specializations) == 1
            and specializations[0].get("hostName") == expected["entryPoint"]
            and specializations[0].get("parameters") == expected_template_parameters
            and materialization.get("unsupported") == [],
            f"LogSumExp materialization changed for {workload_id}",
        )

        artifact_path_value = artifact.get("path")
        _require(
            isinstance(artifact_path_value, str) and artifact_path_value,
            f"LogSumExp artifact path is missing for {workload_id}",
        )
        artifact_path = (mlx_root / artifact_path_value).resolve()
        _require(
            _is_relative_to(artifact_path, output_root) and artifact_path.is_file(),
            f"LogSumExp artifact is missing or escaped output for {workload_id}",
        )
        generated = artifact_path.read_text(encoding="utf-8")
        generated_hash = _sha256(artifact_path)
        normalized_hash = _normalized_text_sha256(artifact_path)
        _require(
            artifact.get("generatedHash")
            == {"algorithm": "sha256", "value": generated_hash}
            and artifact.get("generatedSizeBytes") == artifact_path.stat().st_size
            and len(re.findall(r"\bvoid\s+CSMain\s*\(", generated)) == 1
            and re.search(r"\[\s*WaveSize\s*\(\s*32\s*\)\s*\]", generated) is not None
            and re.search(
                r"\[\s*numthreads\s*\(\s*{}\s*,\s*1\s*,\s*1\s*\)\s*\]".format(
                    expected["workgroupSize"][0]
                ),
                generated,
            )
            is not None
            and all(
                fragment in generated
                for fragment in (
                    "WaveActiveMax",
                    "WaveActiveSum",
                    "GroupMemoryBarrierWithGroupSync",
                    "exp(",
                    "log(",
                    "out_[gid]",
                )
            ),
            f"LogSumExp generated HLSL contract changed for {workload_id}",
        )
        generated_evidence[workload_id] = {
            "entryPoint": expected["entryPoint"],
            "artifactId": artifact_id,
            "dispatchVariantId": expected["dispatchVariantId"],
            "inputs": dict(expected["inputs"]),
            "workgroupSize": list(expected["workgroupSize"]),
            "dispatchWorkgroupCount": list(expected["dispatchWorkgroupCount"]),
            "subgroupWidth": 32,
            "generatedHlsl": {
                "normalizedSha256": normalized_hash,
                "contentSha256": generated_hash,
                "sizeBytes": artifact_path.stat().st_size,
            },
        }
        artifacts_by_workload[workload_id] = artifact

    toolchain_runs: list[Mapping[str, Any]] = []
    if validated:
        validation = payload.get("validation")
        runs = (
            validation.get("toolchainRuns") if isinstance(validation, Mapping) else None
        )
        _require(
            isinstance(validation, Mapping)
            and isinstance(validation.get("summary"), Mapping)
            and validation["summary"].get("failedCount") == 0
            and isinstance(runs, list),
            "LogSumExp dispatch DXC validation changed",
        )
        toolchain_runs = [
            run
            for run in runs
            if isinstance(run, Mapping) and run.get("target") == "directx"
        ]
        artifact_paths = {
            artifact["path"] for artifact in artifacts_by_workload.values()
        }
        _require(
            len(toolchain_runs) == artifact_count
            and all(run.get("status") == "ok" for run in toolchain_runs)
            and {run.get("path") for run in toolchain_runs} == artifact_paths,
            "LogSumExp dispatch DXC did not validate every artifact",
        )

    evidence = {
        "status": "translated-dxc-validated" if validated else "translated",
        "source": MLX_LOGSUMEXP_SOURCE,
        "sourceSha256": MLX_LOGSUMEXP_SHA256,
        "target": "directx",
        "testSources": [
            "python/tests/test_ops.py::test_logsumexp",
            "python/tests/test_autograd.py::test_logsumexp_grad",
        ],
        "dispatchContract": {
            "path": contract["path"],
            "contentIdentity": MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY,
            "variantCount": artifact_count,
            "resolvedIssue": MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
        },
        "artifactCount": artifact_count,
        "variants": generated_evidence,
        "dxcValidatedArtifactCount": len(toolchain_runs),
        "runtimeExecutionAttempted": False,
        "numericalParityClaimed": False,
    }
    return artifacts_by_workload, evidence, toolchain_runs


def _require_rms_norm_dispatch_frontier_report(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    validated: bool,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], list[Mapping[str, Any]]]:
    project = payload.get("project")
    contract_records = (
        project.get("dispatchContracts") if isinstance(project, Mapping) else None
    )
    artifact_count = len(MLX_RMS_NORM_DISPATCH_VARIANTS)
    _require(
        isinstance(project, Mapping)
        and project.get("includePatterns") == [MLX_RMS_NORM_SOURCE]
        and project.get("targets") == ["directx"]
        and project.get("dispatchContractFiles") == [contract["path"]]
        and project.get("dispatchContractCount") == 1
        and project.get("dispatchVariantCount") == artifact_count
        and isinstance(contract_records, list)
        and len(contract_records) == 1,
        "RMSNorm dispatch frontier project metadata changed",
    )
    contract_record = contract_records[0]
    expected_manifest_path = str((mlx_root / str(contract["path"])).resolve())
    evaluation = (
        contract_record.get("evaluation")
        if isinstance(contract_record, Mapping)
        else None
    )
    evaluated_variants = (
        evaluation.get("variants") if isinstance(evaluation, Mapping) else None
    )
    _require(
        isinstance(contract_record, Mapping)
        and contract_record.get("path") == contract["path"]
        and contract_record.get("schemaVersion") == 1
        and contract_record.get("contentIdentity") == contract["contentIdentity"]
        and contract_record.get("manifest", {}).get("provenance", {}).get("commit")
        == MLX_COMMIT
        and isinstance(evaluation, Mapping)
        and evaluation.get("manifestSource") == expected_manifest_path
        and evaluation.get("variantCount") == artifact_count
        and isinstance(evaluated_variants, list)
        and len(evaluated_variants) == artifact_count,
        "RMSNorm dispatch frontier did not retain replayable manifest provenance",
    )
    evaluation_by_workload = {
        variant.get("workload", {}).get("id"): variant
        for variant in evaluated_variants
        if isinstance(variant, Mapping) and isinstance(variant.get("workload"), Mapping)
    }
    _require(
        set(evaluation_by_workload) == set(MLX_RMS_NORM_DISPATCH_VARIANTS),
        "RMSNorm dispatch evaluation workload set changed",
    )

    summary = payload.get("summary")
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("artifactCount") == artifact_count
        and summary.get("translatedCount") == artifact_count
        and summary.get("failedCount") == 0
        and summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0}
        and summary.get("artifactsByTarget", {}).get("directx")
        == {
            "artifactCount": artifact_count,
            "translatedCount": artifact_count,
            "failedCount": 0,
        }
        and payload.get("diagnostics") == [],
        "RMSNorm dispatch frontier accounting changed",
    )
    units = payload.get("units")
    _require(
        isinstance(units, list)
        and len(units) == 1
        and units[0].get("path") == MLX_RMS_NORM_SOURCE
        and units[0].get("sourceBackend") == "metal"
        and units[0].get("sourceHash")
        == {"algorithm": "sha256", "value": MLX_RMS_NORM_SHA256},
        "RMSNorm dispatch frontier source identity changed",
    )

    dispatch_plan = project.get("dispatchArtifactPlan")
    planned_artifacts = (
        dispatch_plan.get("artifacts") if isinstance(dispatch_plan, Mapping) else None
    )
    _require(
        isinstance(dispatch_plan, Mapping)
        and dispatch_plan.get("kind") == "crosstl-dispatch-artifact-plan"
        and dispatch_plan.get("schemaVersion") == 1
        and dispatch_plan.get("sourceUnitCount") == 1
        and dispatch_plan.get("artifactCount") == artifact_count
        and dispatch_plan.get("dispatchVariantCount") == artifact_count
        and isinstance(planned_artifacts, list)
        and len(planned_artifacts) == artifact_count,
        "RMSNorm dispatch artifact plan changed",
    )
    plan_by_artifact_id = {
        record.get("artifactId"): record
        for record in planned_artifacts
        if isinstance(record, Mapping)
    }
    expected_artifact_ids = {
        expected["artifactId"] for expected in MLX_RMS_NORM_DISPATCH_VARIANTS.values()
    }
    _require(
        set(plan_by_artifact_id) == expected_artifact_ids,
        "RMSNorm dispatch artifact plan identity set changed",
    )

    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == artifact_count,
        "RMSNorm dispatch artifact records are incomplete",
    )
    artifacts_by_id = {
        artifact.get("dispatchArtifact", {}).get("artifactId"): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and isinstance(artifact.get("dispatchArtifact"), Mapping)
    }
    _require(
        set(artifacts_by_id) == expected_artifact_ids,
        "RMSNorm dispatch artifact identity set changed",
    )

    output_root = output_dir.resolve()
    artifacts_by_workload: dict[str, Mapping[str, Any]] = {}
    generated_evidence: dict[str, Any] = {}
    for workload_id, expected in MLX_RMS_NORM_DISPATCH_VARIANTS.items():
        artifact_id = expected["artifactId"]
        artifact = artifacts_by_id[artifact_id]
        plan = plan_by_artifact_id[artifact_id]
        evaluated = evaluation_by_workload[workload_id]
        entry = artifact.get("entryPoint")
        entry_point = entry.get("source") if isinstance(entry, Mapping) else None
        _require(
            evaluated.get("artifactId") == artifact_id
            and evaluated.get("variantId") == expected["dispatchVariantId"]
            and evaluated.get("entryPoint") == expected["entryPoint"]
            and evaluated.get("workload", {}).get("inputs") == expected["inputs"]
            and evaluated.get("workgroupSize") == expected["workgroupSize"]
            and evaluated.get("subgroupWidth") == 32
            and evaluated.get("specializationConstants")
            == expected["specializationConstants"]
            and evaluated.get("dispatch", {}).get("workgroupCount")
            == expected["dispatchWorkgroupCount"],
            f"RMSNorm evaluated dispatch changed for {workload_id}",
        )
        variant_name = "dispatch-" + artifact_id.removeprefix("sha256:")
        _require(
            artifact.get("source") == MLX_RMS_NORM_SOURCE
            and artifact.get("sourceBackend") == "metal"
            and artifact.get("sourceHash")
            == {"algorithm": "sha256", "value": MLX_RMS_NORM_SHA256}
            and artifact.get("target") == "directx"
            and artifact.get("status") == "translated"
            and artifact.get("variant") == variant_name
            and entry_point == expected["entryPoint"]
            and entry.get("target") == "CSMain"
            and entry.get("stage") == "compute"
            and artifact.get("dispatchArtifact") == plan
            and plan.get("artifactId") == artifact_id
            and plan.get("dispatchVariantIds") == [expected["dispatchVariantId"]]
            and plan.get("manifestContentIdentities")
            == [MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY]
            and plan.get("source") == MLX_RMS_NORM_SOURCE
            and plan.get("entryPoint") == expected["entryPoint"]
            and plan.get("workgroupSize") == expected["workgroupSize"]
            and plan.get("subgroupWidth") == 32
            and plan.get("specializationConstants")
            == expected["specializationConstants"],
            f"RMSNorm dispatch artifact contract changed for {workload_id}",
        )

        execution = artifact.get("execution")
        execution_entries = (
            execution.get("entryPoints") if isinstance(execution, Mapping) else None
        )
        _require(
            isinstance(execution, Mapping)
            and execution.get("sourceEntryPoints") == [expected["entryPoint"]]
            and execution.get("provenance", {}).get("kind") == "host-dispatch-contract"
            and execution.get("provenance", {}).get("artifactId") == artifact_id
            and execution.get("subgroupWidthProvenance", {}).get("kind")
            == "host-dispatch-contract"
            and execution.get("subgroupWidthEnforcement")
            == {
                "mechanism": "hlsl-wave-size-attribute",
                "minimumShaderModel": "6.6",
                "entryProfiles": [{"entryPoint": "CSMain", "profile": "cs_6_6"}],
            }
            and isinstance(execution_entries, list)
            and len(execution_entries) == 1
            and execution_entries[0].get("sourceEntryPoint") == expected["entryPoint"]
            and execution_entries[0].get("materializedEntryPoint")
            == expected["entryPoint"]
            and execution_entries[0].get("targetEntryPoint") == "CSMain"
            and execution_entries[0].get("workgroupSize") == expected["workgroupSize"]
            and execution_entries[0].get("subgroupWidth") == 32,
            f"RMSNorm execution metadata changed for {workload_id}",
        )

        constants = artifact.get("specializationConstants") or []
        constants_by_id = {
            str(record.get("id")): record
            for record in constants
            if isinstance(record, Mapping)
        }
        _require(
            set(constants_by_id) == set(expected["specializationConstants"]),
            f"RMSNorm specialization inputs changed for {workload_id}",
        )
        for constant_id, value in expected["specializationConstants"].items():
            record = constants_by_id[constant_id]
            _require(
                record.get("concreteValue") is value
                and record.get("deferred") is False
                and record.get("valueProvenance", {}).get("kind")
                == "host-dispatch-contract"
                and record.get("valueProvenance", {}).get("artifactId") == artifact_id,
                f"RMSNorm specialization provenance changed for {workload_id}",
            )

        materialization = artifact.get("templateMaterialization")
        specializations = (
            materialization.get("specializations")
            if isinstance(materialization, Mapping)
            else None
        )
        _require(
            isinstance(materialization, Mapping)
            and materialization.get("status") == "materialized"
            and materialization.get("specializationCount") == 1
            and isinstance(specializations, list)
            and len(specializations) == 1
            and specializations[0].get("hostName") == expected["entryPoint"]
            and materialization.get("unsupported") == [],
            f"RMSNorm materialization changed for {workload_id}",
        )

        artifact_path_value = artifact.get("path")
        _require(
            isinstance(artifact_path_value, str) and artifact_path_value,
            f"RMSNorm artifact path is missing for {workload_id}",
        )
        artifact_path = (mlx_root / artifact_path_value).resolve()
        _require(
            _is_relative_to(artifact_path, output_root) and artifact_path.is_file(),
            f"RMSNorm artifact is missing or escaped output for {workload_id}",
        )
        generated = artifact_path.read_text(encoding="utf-8")
        generated_hash = _sha256(artifact_path)
        normalized_hash = _normalized_text_sha256(artifact_path)
        _require(
            artifact.get("generatedHash")
            == {"algorithm": "sha256", "value": generated_hash}
            and artifact.get("generatedSizeBytes") == artifact_path.stat().st_size
            and len(re.findall(r"\bvoid\s+CSMain\s*\(", generated)) == 1
            and re.search(r"\[\s*WaveSize\s*\(\s*32\s*\)\s*\]", generated) is not None
            and re.search(
                r"\[\s*numthreads\s*\(\s*{}\s*,\s*1\s*,\s*1\s*\)\s*\]".format(
                    expected["workgroupSize"][0]
                ),
                generated,
            )
            is not None,
            f"RMSNorm generated HLSL contract changed for {workload_id}",
        )
        if expected["specializationConstants"]:
            has_w = str(expected["specializationConstants"]["20"]).lower()
            _require(
                re.search(
                    rf"\bstatic\s+const\s+bool\s+has_w\s*=\s*{has_w}\s*;",
                    generated,
                )
                is not None,
                f"RMSNorm VJP artifact did not materialize has_w={has_w}",
            )
        else:
            _require(
                re.search(r"\bhas_w\b", generated) is None,
                "RMSNorm forward artifact retained an unreachable function constant",
            )
        generated_evidence[workload_id] = {
            "entryPoint": expected["entryPoint"],
            "artifactId": artifact_id,
            "dispatchVariantId": expected["dispatchVariantId"],
            "inputs": dict(expected["inputs"]),
            "workgroupSize": list(expected["workgroupSize"]),
            "dispatchWorkgroupCount": list(expected["dispatchWorkgroupCount"]),
            "subgroupWidth": 32,
            "specializationConstants": dict(expected["specializationConstants"]),
            "generatedHlsl": {
                "normalizedSha256": normalized_hash,
                "contentSha256": generated_hash,
                "sizeBytes": artifact_path.stat().st_size,
            },
        }
        artifacts_by_workload[workload_id] = artifact

    toolchain_runs: list[Mapping[str, Any]] = []
    if validated:
        validation = payload.get("validation")
        runs = (
            validation.get("toolchainRuns") if isinstance(validation, Mapping) else None
        )
        _require(
            isinstance(validation, Mapping)
            and isinstance(validation.get("summary"), Mapping)
            and validation["summary"].get("failedCount") == 0
            and isinstance(runs, list),
            "RMSNorm dispatch DXC validation changed",
        )
        toolchain_runs = [
            run
            for run in runs
            if isinstance(run, Mapping) and run.get("target") == "directx"
        ]
        artifact_paths = {artifact["path"] for artifact in artifacts_by_id.values()}
        _require(
            len(toolchain_runs) == artifact_count
            and all(run.get("status") == "ok" for run in toolchain_runs)
            and {run.get("path") for run in toolchain_runs} == artifact_paths,
            "RMSNorm dispatch DXC did not validate every artifact",
        )

    evidence = {
        "status": "translated-dxc-validated" if validated else "translated",
        "source": MLX_RMS_NORM_SOURCE,
        "sourceSha256": MLX_RMS_NORM_SHA256,
        "target": "directx",
        "testSources": [
            "python/tests/test_fast.py::test_rms_norm",
            "python/tests/test_fast.py::test_rms_norm_grad",
        ],
        "dispatchContract": {
            "path": contract["path"],
            "contentIdentity": MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY,
            "variantCount": artifact_count,
            "resolvedIssue": MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
        },
        "artifactCount": artifact_count,
        "variants": generated_evidence,
        "dxcValidatedArtifactCount": len(toolchain_runs),
        "runtimeExecutionAttempted": False,
        "numericalParityClaimed": False,
    }
    return artifacts_by_workload, evidence, toolchain_runs


def _require_directx_bfloat16_lowering_evidence(
    artifacts_by_source: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    _require(
        set(artifacts_by_source) == set(MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE),
        "DirectX bfloat16 lowering evidence source set changed",
    )
    evidence: dict[str, dict[str, Any]] = {}
    for source, expected in MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE.items():
        artifact = artifacts_by_source[source]
        lowering = artifact.get("bfloat16Lowering")
        required_capabilities = artifact.get("requiredCapabilities")
        _require(
            isinstance(lowering, Mapping)
            and dict(lowering) == expected["bfloat16Lowering"]
            and required_capabilities == expected["requiredCapabilities"],
            f"DirectX bfloat16 lowering contract changed for {source}",
        )
        evidence[source] = {
            "bfloat16Lowering": dict(lowering),
            "requiredCapabilities": list(required_capabilities),
        }
    return evidence


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
    expected_error_diagnostics = {
        MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE: len(sources),
    }
    if validated:
        expected_error_diagnostics["project.validate.failed-artifact"] = len(sources)
    expected_warning_diagnostics: dict[str, int] = {}
    if validated:
        validation = payload.get("validation")
        toolchains = (
            validation.get("toolchains") if isinstance(validation, Mapping) else None
        )
        target_toolchains = [
            record
            for record in toolchains or []
            if isinstance(record, Mapping) and record.get("target") == target
        ]
        _require(
            isinstance(toolchains, list) and len(target_toolchains) == 1,
            f"{target.title()} dynamic-workgroup toolchain status changed",
        )
        toolchain = target_toolchains[0]
        tools = toolchain.get("tools")
        status = toolchain.get("status")
        _require(
            status in {"available", "unavailable"}
            and isinstance(tools, list)
            and bool(tools)
            and all(isinstance(tool, Mapping) for tool in tools)
            and (
                (
                    status == "available"
                    and any(tool.get("available") is True for tool in tools)
                )
                or (
                    status == "unavailable"
                    and all(tool.get("available") is False for tool in tools)
                )
            ),
            f"{target.title()} dynamic-workgroup toolchain evidence changed",
        )
        if status == "unavailable":
            expected_warning_diagnostics["project.validate.toolchain-unavailable"] = 1
    expected_diagnostics = {
        **expected_error_diagnostics,
        **expected_warning_diagnostics,
    }
    expected_error_count = sum(expected_error_diagnostics.values())
    expected_warning_count = sum(expected_warning_diagnostics.values())
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == len(sources)
        and summary.get("artifactCount") == len(sources)
        and summary.get("translatedCount") == 0
        and summary.get("failedCount") == len(sources)
        and summary.get("diagnosticCounts")
        == {
            "error": expected_error_count,
            "note": 0,
            "warning": expected_warning_count,
        }
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
        == Counter(expected_error_diagnostics),
        f"{target.title()} dynamic-workgroup diagnostics changed",
    )
    warning_diagnostics = [
        diagnostic
        for diagnostic in payload.get("diagnostics", [])
        if isinstance(diagnostic, Mapping) and diagnostic.get("severity") == "warning"
    ]
    _require(
        len(warning_diagnostics) == expected_warning_count
        and Counter(str(diagnostic.get("code")) for diagnostic in warning_diagnostics)
        == Counter(expected_warning_diagnostics)
        and all(
            diagnostic.get("target") == target
            and diagnostic.get("missingCapabilities") == ["toolchain.validation"]
            for diagnostic in warning_diagnostics
        ),
        f"{target.title()} dynamic-workgroup warning diagnostics changed",
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
    index_range_assertions: Sequence[Mapping[str, str | int]] | None = None,
    dispatch_contracts: Sequence[str] | None = None,
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
        index_range_assertions=index_range_assertions,
        dispatch_contracts=dispatch_contracts,
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
    clean_artifacts = _require_clean_frontier_report(
        mlx_root,
        clean_output_dir,
        clean_payload,
        target="directx",
        sources=MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES,
    )
    bfloat16_lowering_evidence = _require_directx_bfloat16_lowering_evidence(
        clean_artifacts
    )

    layer_norm_contract = _prepare_layer_norm_dispatch_contract(mlx_root, work_dir)
    layer_norm_output_dir = work_dir / "out-directx-layer-norm-dispatch-frontier"
    (
        _layer_norm_result,
        layer_norm_payload,
        _layer_norm_config,
        layer_norm_report_path,
    ) = _run_frontier_project(
        mlx_root=mlx_root,
        config_dir=config_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        python=python,
        command_name="directx-layer-norm-dispatch-frontier",
        target="directx",
        sources=(MLX_LAYER_NORM_SOURCE,),
        output_dir=layer_norm_output_dir,
        dispatch_contracts=(str(layer_norm_contract["path"]),),
    )
    (
        layer_norm_artifacts,
        layer_norm_evidence,
        _layer_norm_runs,
    ) = _require_layer_norm_dispatch_frontier_report(
        mlx_root,
        layer_norm_output_dir,
        layer_norm_payload,
        contract=layer_norm_contract,
        validated=False,
    )

    logsumexp_contract = _prepare_logsumexp_dispatch_contract(mlx_root, work_dir)
    logsumexp_output_dir = work_dir / "out-directx-logsumexp-dispatch-frontier"
    (
        _logsumexp_result,
        logsumexp_payload,
        _logsumexp_config,
        logsumexp_report_path,
    ) = _run_frontier_project(
        mlx_root=mlx_root,
        config_dir=config_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        python=python,
        command_name="directx-logsumexp-dispatch-frontier",
        target="directx",
        sources=(MLX_LOGSUMEXP_SOURCE,),
        output_dir=logsumexp_output_dir,
        dispatch_contracts=(str(logsumexp_contract["path"]),),
    )
    (
        logsumexp_artifacts,
        logsumexp_evidence,
        _logsumexp_runs,
    ) = _require_logsumexp_dispatch_frontier_report(
        mlx_root,
        logsumexp_output_dir,
        logsumexp_payload,
        contract=logsumexp_contract,
        validated=False,
    )

    rms_norm_contract = _prepare_rms_norm_dispatch_contract(mlx_root, work_dir)
    rms_norm_output_dir = work_dir / "out-directx-rms-norm-dispatch-frontier"
    (
        _rms_norm_result,
        rms_norm_payload,
        _rms_norm_config,
        rms_norm_report_path,
    ) = _run_frontier_project(
        mlx_root=mlx_root,
        config_dir=config_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        python=python,
        command_name="directx-rms-norm-dispatch-frontier",
        target="directx",
        sources=(MLX_RMS_NORM_SOURCE,),
        output_dir=rms_norm_output_dir,
        dispatch_contracts=(str(rms_norm_contract["path"]),),
    )
    (
        rms_norm_artifacts,
        rms_norm_evidence,
        _rms_norm_runs,
    ) = _require_rms_norm_dispatch_frontier_report(
        mlx_root,
        rms_norm_output_dir,
        rms_norm_payload,
        contract=rms_norm_contract,
        validated=False,
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
            sources=MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
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
        sources=MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
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
                sources=MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES,
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
            sources=MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES,
        )
        _require_directx_bfloat16_lowering_evidence(artifacts)
        validation = toolchain_payload.get("validation")
        runs = (
            validation.get("toolchainRuns") if isinstance(validation, Mapping) else None
        )
        _require(
            isinstance(runs, list), "DirectX frontier toolchainRuns must be a list"
        )
        aggregate_runs = [
            run
            for run in runs
            if isinstance(run, Mapping) and run.get("target") == "directx"
        ]
        layer_norm_toolchain_output = (
            work_dir / "out-directx-layer-norm-dispatch-toolchain"
        )
        (
            _layer_norm_toolchain_result,
            layer_norm_toolchain_payload,
            _layer_norm_toolchain_config,
            _layer_norm_toolchain_report,
        ) = _run_frontier_project(
            mlx_root=mlx_root,
            config_dir=config_dir,
            report_dir=report_dir,
            log_dir=log_dir,
            python=python,
            command_name="validate-directx-layer-norm-dispatch-toolchain",
            target="directx",
            sources=(MLX_LAYER_NORM_SOURCE,),
            output_dir=layer_norm_toolchain_output,
            run_toolchains=True,
            dispatch_contracts=(str(layer_norm_contract["path"]),),
        )
        (
            layer_norm_artifacts,
            layer_norm_evidence,
            layer_norm_runs,
        ) = _require_layer_norm_dispatch_frontier_report(
            mlx_root,
            layer_norm_toolchain_output,
            layer_norm_toolchain_payload,
            contract=layer_norm_contract,
            validated=True,
        )
        logsumexp_toolchain_output = (
            work_dir / "out-directx-logsumexp-dispatch-toolchain"
        )
        (
            _logsumexp_toolchain_result,
            logsumexp_toolchain_payload,
            _logsumexp_toolchain_config,
            _logsumexp_toolchain_report,
        ) = _run_frontier_project(
            mlx_root=mlx_root,
            config_dir=config_dir,
            report_dir=report_dir,
            log_dir=log_dir,
            python=python,
            command_name="validate-directx-logsumexp-dispatch-toolchain",
            target="directx",
            sources=(MLX_LOGSUMEXP_SOURCE,),
            output_dir=logsumexp_toolchain_output,
            run_toolchains=True,
            dispatch_contracts=(str(logsumexp_contract["path"]),),
        )
        (
            logsumexp_artifacts,
            logsumexp_evidence,
            logsumexp_runs,
        ) = _require_logsumexp_dispatch_frontier_report(
            mlx_root,
            logsumexp_toolchain_output,
            logsumexp_toolchain_payload,
            contract=logsumexp_contract,
            validated=True,
        )
        rms_norm_toolchain_output = work_dir / "out-directx-rms-norm-dispatch-toolchain"
        (
            _rms_norm_toolchain_result,
            rms_norm_toolchain_payload,
            _rms_norm_toolchain_config,
            _rms_norm_toolchain_report,
        ) = _run_frontier_project(
            mlx_root=mlx_root,
            config_dir=config_dir,
            report_dir=report_dir,
            log_dir=log_dir,
            python=python,
            command_name="validate-directx-rms-norm-dispatch-toolchain",
            target="directx",
            sources=(MLX_RMS_NORM_SOURCE,),
            output_dir=rms_norm_toolchain_output,
            run_toolchains=True,
            dispatch_contracts=(str(rms_norm_contract["path"]),),
        )
        (
            rms_norm_artifacts,
            rms_norm_evidence,
            rms_norm_runs,
        ) = _require_rms_norm_dispatch_frontier_report(
            mlx_root,
            rms_norm_toolchain_output,
            rms_norm_toolchain_payload,
            contract=rms_norm_contract,
            validated=True,
        )
        directx_runs = [
            *aggregate_runs,
            *layer_norm_runs,
            *logsumexp_runs,
            *rms_norm_runs,
        ]
        artifact_paths = {
            *(artifact["path"] for artifact in artifacts.values()),
            *(artifact["path"] for artifact in layer_norm_artifacts.values()),
            *(artifact["path"] for artifact in logsumexp_artifacts.values()),
            *(artifact["path"] for artifact in rms_norm_artifacts.values()),
        }
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
    layer_norm_entry_by_path = {
        artifact["path"]: entry_point
        for entry_point, artifact in layer_norm_artifacts.items()
    }
    logsumexp_workload_by_path = {
        artifact["path"]: workload_id
        for workload_id, artifact in logsumexp_artifacts.items()
    }
    rms_norm_workload_by_path = {
        artifact["path"]: workload_id
        for workload_id, artifact in rms_norm_artifacts.items()
    }
    for run in directx_runs:
        if run.get("status") != "ok":
            continue
        source = run.get("source")
        _require(
            source in directx_entry_points_by_source,
            f"DirectX toolchain validation reported an unexpected source: {source}",
        )
        if source == MLX_LAYER_NORM_SOURCE:
            entry_point = layer_norm_entry_by_path.get(run.get("path"))
        elif source == MLX_LOGSUMEXP_SOURCE:
            entry_point = logsumexp_workload_by_path.get(run.get("path"))
        elif source == MLX_RMS_NORM_SOURCE:
            entry_point = rms_norm_workload_by_path.get(run.get("path"))
        else:
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
    warning_evidence = _directx_toolchain_warning_evidence(directx_runs)
    return {
        "name": "directx-frontier",
        "status": "passed-with-bounded-dispatch-and-pending-contracts",
        "scope": "target-split-frontier",
        "report": _relpath(report_path, mlx_root),
        "layerNormDispatchReport": _relpath(layer_norm_report_path, mlx_root),
        "logsumexpDispatchReport": _relpath(logsumexp_report_path, mlx_root),
        "rmsNormDispatchReport": _relpath(rms_norm_report_path, mlx_root),
        "workgroupBlockedReport": _relpath(blocked_report_path, mlx_root),
        "sources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "unitCount": len(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "artifactCount": (
            len(MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES)
            + len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
            + len(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
            + len(MLX_RMS_NORM_DISPATCH_VARIANTS)
            + len(MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES)
        ),
        "translatedSources": list(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
        "translatedArtifactCount": MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT,
        "workgroupBlockedSources": list(MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES),
        "workgroupBlockedArtifactCount": len(
            MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
        ),
        "target": "directx",
        "toolchainRuns": len(directx_runs),
        "directxToolchainRequired": require_directx_toolchain,
        "directxToolchainSources": list(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
        "directxToolchainArtifactCount": MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT,
        "directxToolchainExpectedEntryPointCounts": dict(
            MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
        ),
        "directxToolchainExpectedEntryPointCount": (
            MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT
        ),
        "directxToolchainValidatedSources": directx_validated_sources,
        "directxToolchainValidatedArtifactCount": (
            MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT if directx_runs else 0
        ),
        "directxToolchainValidatedEntryPointCounts": (
            directx_validated_entry_point_counts
        ),
        "directxToolchainValidatedEntryPointCount": sum(
            directx_validated_entry_point_counts.values()
        ),
        "directxValidationStatus": (
            "validated" if run_toolchains and directx_runs else "not-required"
        ),
        "directxToolchainWarningEvidence": warning_evidence,
        "contextualNarrowingEvidence": MLX_DIRECTX_CONTEXTUAL_NARROWING_EVIDENCE,
        "native16BitArithmeticEvidence": MLX_DIRECTX_NATIVE_16_BIT_ARITHMETIC_EVIDENCE,
        "bfloat16LoweringEvidence": bfloat16_lowering_evidence,
        "layerNormDispatchEvidence": layer_norm_evidence,
        "logsumexpDispatchEvidence": logsumexp_evidence,
        "rmsNormDispatchEvidence": rms_norm_evidence,
        "dynamicWorkgroupDispatchEvidence": dispatch_evidence,
        "semanticReadinessStatus": "not-established",
        "trackedIssues": [
            *FRONTIER_VALIDATION_TRACKED_ISSUES,
            *DIRECTX_TOOLCHAIN_WARNING_TRACKED_ISSUES,
        ],
        "resolvedIssues": [MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE],
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
        index_range_assertions=MLX_OPENGL_INDEX_RANGE_ASSERTIONS,
    )
    artifacts_by_source = _require_clean_frontier_report(
        mlx_root,
        clean_output_dir,
        payload,
        target="opengl",
        sources=MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES,
        validated=False,
        index_range_assertions=MLX_OPENGL_INDEX_RANGE_ASSERTIONS,
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
        "indexRangeAssertionEvidence": {
            "assertionCount": len(MLX_OPENGL_INDEX_RANGE_ASSERTIONS),
            "inclusiveBounds": {
                "minimum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
                "maximum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
            },
            "expressionsBySource": {
                source: list(expressions)
                for source, expressions in (
                    MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS.items()
                )
            },
            "contractKind": "explicit-host-runtime-portability-preconditions",
            "inferred": False,
            "runtimeEnforced": False,
        },
        "dynamicWorkgroupDispatchEvidence": dispatch_evidence,
        "trackedIssues": [],
        "resolvedIssues": [MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE],
        "runtimeIntegrationIncluded": False,
    }


def _check_fft_directx_toolchain(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_toolchain: bool,
) -> dict[str, Any]:
    config_path = config_dir / "fft-directx-toolchain.toml"
    report_path = report_dir / "fft-directx-toolchain.json"
    output_dir = work_dir / "out-fft-directx"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_project_config(
        config_path,
        include=MLX_FFT_SOURCE,
        targets=("directx",),
        output_dir=_relpath(output_dir, mlx_root),
        specialization_constants=FFT_DIRECTX_SPECIALIZATION_CONSTANTS,
        metal_source_options={
            "max_template_specializations": FFT_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": FFT_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
        entry_points={MLX_FFT_SOURCE: FFT_DIRECTX_ENTRY_POINT},
        entry_workgroup_size_rules={
            MLX_FFT_SOURCE: {
                FFT_DIRECTX_ENTRY_POINT: FFT_DIRECTX_WORKGROUP_SIZE,
            }
        },
        index_range_assertions=FFT_INDEX_RANGE_ASSERTIONS,
        workgroup_access_assertions=FFT_DIRECTX_WORKGROUP_ACCESS_ASSERTIONS,
    )
    result = _run_command(
        "translate-fft-directx",
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
        timeout_seconds=FFT_DIRECTX_TRANSLATION_TIMEOUT_SECONDS,
    )
    _require(result.returncode == 0, "DirectX FFT project translation failed")
    _require(
        report_path.is_file(),
        "DirectX FFT translation did not produce a project report",
    )

    payload = _load_json(report_path)
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        "DirectX FFT translation report kind changed",
    )
    summary = payload.get("summary", {})
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0,
        "DirectX FFT translation report did not retain one translated artifact",
    )
    _require(
        summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0}
        and summary.get("diagnosticsByCode") == {}
        and summary.get("missingCapabilityCounts") == {}
        and payload.get("diagnostics") == [],
        "DirectX FFT project translation must have zero diagnostics",
    )

    expected_workgroup_assertions = [
        {
            "source": assertion["source"],
            "entryPoint": assertion["entry_point"],
            "function": assertion["function"],
            "parameter": assertion["parameter"],
            "minimum": assertion["minimum"],
            "maximum": assertion["maximum"],
        }
        for assertion in FFT_DIRECTX_WORKGROUP_ACCESS_ASSERTIONS
    ]
    project = payload.get("project", {})
    _require(
        isinstance(project, Mapping)
        and project.get("indexRangeAssertionCount") == len(FFT_INDEX_RANGE_ASSERTIONS)
        and project.get("indexRangeAssertions") == list(FFT_INDEX_RANGE_ASSERTIONS)
        and project.get("workgroupAccessAssertionCount")
        == len(FFT_DIRECTX_WORKGROUP_ACCESS_ASSERTIONS)
        and project.get("workgroupAccessAssertions") == expected_workgroup_assertions,
        "DirectX FFT portability preconditions changed",
    )

    units = payload.get("units", [])
    _require(
        isinstance(units, list)
        and len(units) == 1
        and isinstance(units[0], Mapping)
        and units[0].get("id") == MLX_FFT_SOURCE
        and units[0].get("path") == MLX_FFT_SOURCE
        and units[0].get("sourceBackend") == "metal"
        and units[0].get("sourceHash")
        == {"algorithm": "sha256", "value": MLX_FFT_SHA256}
        and units[0].get("sourceSizeBytes") == MLX_FFT_SOURCE_SIZE_BYTES,
        "DirectX FFT source-unit identity changed at the pinned MLX commit",
    )

    artifacts = payload.get("artifacts", [])
    _require(
        isinstance(artifacts, list) and len(artifacts) == 1,
        "DirectX FFT report must contain one translated artifact record",
    )
    artifact = artifacts[0]
    artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
    entry = artifact.get("entryPoint") if isinstance(artifact, Mapping) else None
    execution = artifact.get("execution") if isinstance(artifact, Mapping) else None
    execution_entries = (
        execution.get("entryPoints") if isinstance(execution, Mapping) else None
    )
    expected_rule_path = (
        f'project.entry_workgroup_size_rules["{MLX_FFT_SOURCE}"].'
        f"{FFT_DIRECTX_ENTRY_POINT}"
    )
    _require(
        isinstance(artifact, Mapping)
        and artifact.get("source") == MLX_FFT_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "directx"
        and artifact.get("status") == "translated"
        and artifact.get("provenance")
        == {"intermediate": "crossgl", "pipeline": "entry-scoped-translate"}
        and artifact.get("sourceHash")
        == {"algorithm": "sha256", "value": MLX_FFT_SHA256}
        and artifact.get("sourceSizeBytes") == MLX_FFT_SOURCE_SIZE_BYTES
        and isinstance(artifact_path, str)
        and entry
        == {
            "source": FFT_DIRECTX_ENTRY_POINT,
            "stage": "compute",
            "target": "CSMain",
        },
        "DirectX FFT translated artifact contract changed",
    )
    _require(
        isinstance(execution, Mapping)
        and execution.get("sourceEntryPoints") == [FFT_DIRECTX_ENTRY_POINT]
        and execution.get("provenance")
        == {
            "kind": "materialized-template-entry-rules",
            "path": f'project.entry_workgroup_size_rules["{MLX_FFT_SOURCE}"]',
        }
        and isinstance(execution_entries, list)
        and len(execution_entries) == 1
        and execution_entries[0].get("sourceEntryPoint") == FFT_DIRECTX_ENTRY_POINT
        and execution_entries[0].get("materializedEntryPoint")
        == FFT_DIRECTX_ENTRY_POINT
        and execution_entries[0].get("targetEntryPoint") == "CSMain"
        and execution_entries[0].get("workgroupSize")
        == list(FFT_DIRECTX_WORKGROUP_SIZE)
        and execution_entries[0].get("rule")
        == {
            "components": [str(value) for value in FFT_DIRECTX_WORKGROUP_SIZE],
            "entryPattern": FFT_DIRECTX_ENTRY_POINT,
            "path": expected_rule_path,
            "sourcePattern": MLX_FFT_SOURCE,
        },
        "DirectX FFT execution contract changed",
    )

    specialization_constants = artifact.get("specializationConstants", [])
    constants_by_id = {
        record.get("id"): record
        for record in specialization_constants
        if isinstance(record, Mapping)
    }
    _require(
        len(specialization_constants) == FFT_DIRECTX_EXPECTED_FUNCTION_CONSTANT_COUNT
        and set(constants_by_id) == set(FFT_DIRECTX_REACHABLE_SPECIALIZATION_CONSTANTS)
        and all(
            constants_by_id[constant_id].get("concreteValue") == expected_value
            and constants_by_id[constant_id].get("deferred") is False
            for constant_id, expected_value in (
                FFT_DIRECTX_REACHABLE_SPECIALIZATION_CONSTANTS.items()
            )
        ),
        "DirectX FFT function-constant materialization changed",
    )
    materialization = artifact.get("templateMaterialization", {})
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and len(materialization.get("specializations", []))
        == FFT_DIRECTX_EXPECTED_SPECIALIZATION_COUNT
        and materialization.get("unsupported") == [],
        "DirectX FFT template materialization evidence changed",
    )

    generated_path = (mlx_root / artifact_path).resolve()
    _require(
        _is_relative_to(generated_path, output_dir.resolve()),
        "DirectX FFT artifact path escaped its output directory",
    )
    _require(
        generated_path.is_file() and generated_path.stat().st_size > 0,
        f"DirectX FFT artifact is missing: {artifact_path}",
    )
    generated_hash = _sha256(generated_path)
    generated_size = generated_path.stat().st_size
    _require(
        artifact.get("generatedHash")
        == {"algorithm": "sha256", "value": generated_hash}
        and artifact.get("generatedSizeBytes") == generated_size
        and generated_hash == FFT_DIRECTX_GENERATED_SHA256
        and generated_size == FFT_DIRECTX_GENERATED_SIZE_BYTES,
        "DirectX FFT generated artifact identity changed",
    )
    generated = generated_path.read_text(encoding="utf-8")
    _require(
        len(re.findall(r"\bvoid\s+CSMain\s*\(", generated)) == 1
        and "groupshared float2 fft_mem_256_float2_float2_shared_in[256];" in generated
        and "[numthreads(1, 1, 64)]" in generated
        and "crossglNumWorkGroups * uint3(1, 1, 64)" in generated
        and "static const bool inv_ = false;" in generated
        and "static const bool is_power_of_2_ = true;" in generated
        and "static const int elems_per_thread_ = 4;" in generated
        and "static const int radix_4_steps_ = 4;" in generated
        and "rader_m_" not in generated,
        "DirectX FFT generated execution or specialization contract changed",
    )
    _require(
        re.search(r"\bgroupshared\s+[^;]*\*", generated) is None
        and re.search(r"\bcrosstl_ptr_buf\s*[,)]", generated) is None
        and "nullptr" not in generated,
        "DirectX FFT artifact retained a first-class workgroup pointer",
    )

    dxc = shutil.which("dxc")
    _require(
        not require_toolchain or dxc is not None,
        "DirectX FFT validation requires dxc",
    )
    native_validation_attempted = dxc is not None
    native_validation_status = "not-run-tool-unavailable"
    native_validation_output = None
    profile = dxc_profile_for_source("cs_6_0", generated)
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    _require(
        profile == "cs_6_2" and compiler_arguments == ("-enable-16bit-types",),
        "DirectX FFT compiler requirements changed",
    )
    if native_validation_attempted:
        output_path = work_dir / "validation" / "fft-directx-256.dxil"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        compile_result = _run_command(
            "validate-fft-directx",
            [
                str(dxc),
                "-WX",
                "-T",
                profile,
                *compiler_arguments,
                "-E",
                "CSMain",
                str(generated_path),
                "-Fo",
                str(output_path),
            ],
            log_dir=log_dir,
            check=False,
        )
        _require(
            compile_result.returncode == 0,
            "DirectX FFT native compilation failed; inspect "
            "validate-fft-directx logs",
        )
        _require(
            output_path.is_file() and output_path.stat().st_size > 0,
            "DirectX FFT compilation succeeded without producing DXIL",
        )
        native_validation_status = "validated"
        native_validation_output = _relpath(output_path, mlx_root)

    return {
        "name": "fft-directx-toolchain",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_FFT_SOURCE,
        "sourceHash": MLX_FFT_SHA256,
        "target": "directx",
        "selectedEntryPoint": FFT_DIRECTX_ENTRY_POINT,
        "targetEntryPoint": "CSMain",
        "artifactStatus": "translated",
        "artifactEmitted": True,
        "generatedHash": generated_hash,
        "generatedSizeBytes": generated_size,
        "nativeValidationAttempted": native_validation_attempted,
        "nativeValidationStatus": native_validation_status,
        "nativeValidationOutput": native_validation_output,
        "nativeCompiler": "dxc",
        "entryProfile": profile,
        "compilerArguments": list(compiler_arguments),
        "warningsAsErrors": True,
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": FFT_DIRECTX_EXPECTED_SPECIALIZATION_COUNT,
        "configuredFunctionConstantCount": len(FFT_DIRECTX_SPECIALIZATION_CONSTANTS),
        "reachableFunctionConstantCount": FFT_DIRECTX_EXPECTED_FUNCTION_CONSTANT_COUNT,
        "specializationConstants": dict(FFT_DIRECTX_SPECIALIZATION_CONSTANTS),
        "workgroupSize": list(FFT_DIRECTX_WORKGROUP_SIZE),
        "indexRangeAssertions": list(FFT_INDEX_RANGE_ASSERTIONS),
        "workgroupAccessAssertions": expected_workgroup_assertions,
        "toolchainRequired": require_toolchain,
        "trackedIssues": [],
        "maxTemplateSpecializations": FFT_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": FFT_MAX_TEMPLATE_MATERIALIZATION_WORK,
        "runtimeIntegrationIncluded": False,
        "numericalParityClaimed": False,
        "runtimeParityClaimed": False,
    }


def _check_fft_opengl_toolchain(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_toolchain: bool,
) -> dict[str, Any]:
    config_path = config_dir / "fft-opengl-toolchain.toml"
    report_path = report_dir / "fft-opengl-toolchain.json"
    output_dir = work_dir / "out-fft-opengl"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_project_config(
        config_path,
        include=MLX_FFT_SOURCE,
        targets=("opengl",),
        output_dir=_relpath(output_dir, mlx_root),
        metal_source_options={
            "max_template_specializations": FFT_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": FFT_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
        index_range_assertions=FFT_INDEX_RANGE_ASSERTIONS,
        workgroup_access_assertions=FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS,
    )
    result = _run_command(
        "translate-fft-opengl",
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
        result.returncode == 0,
        "OpenGL FFT project translation failed",
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
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0,
        "OpenGL FFT translation report did not retain one translated artifact",
    )
    _require(
        summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0}
        and summary.get("diagnosticsByCode") == {}
        and summary.get("missingCapabilityCounts") == {}
        and payload.get("diagnostics") == [],
        "OpenGL FFT project translation must have zero diagnostics",
    )

    project = payload.get("project", {})
    expected_workgroup_assertions = [
        {
            "source": assertion["source"],
            "entryPoint": assertion["entry_point"],
            "function": assertion["function"],
            "parameter": assertion["parameter"],
            "minimum": assertion["minimum"],
            "maximum": assertion["maximum"],
        }
        for assertion in FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS
    ]
    _require(
        isinstance(project, Mapping)
        and project.get("indexRangeAssertionCount") == len(FFT_INDEX_RANGE_ASSERTIONS)
        and project.get("indexRangeAssertions") == list(FFT_INDEX_RANGE_ASSERTIONS)
        and project.get("workgroupAccessAssertionCount")
        == len(FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS)
        and project.get("workgroupAccessAssertions") == expected_workgroup_assertions,
        "OpenGL FFT portability preconditions changed",
    )

    units = payload.get("units", [])
    _require(
        isinstance(units, list)
        and len(units) == 1
        and isinstance(units[0], Mapping)
        and units[0].get("id") == MLX_FFT_SOURCE
        and units[0].get("path") == MLX_FFT_SOURCE
        and units[0].get("sourceBackend") == "metal"
        and units[0].get("sourceHash")
        == {"algorithm": "sha256", "value": MLX_FFT_SHA256}
        and units[0].get("sourceSizeBytes") == MLX_FFT_SOURCE_SIZE_BYTES,
        "OpenGL FFT source-unit identity changed at the pinned MLX commit",
    )

    artifacts = payload.get("artifacts", [])
    _require(
        isinstance(artifacts, list) and len(artifacts) == 1,
        "OpenGL FFT report must contain one translated artifact record",
    )
    artifact = artifacts[0]
    artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
    _require(
        isinstance(artifact, Mapping)
        and artifact.get("source") == MLX_FFT_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "opengl"
        and artifact.get("status") == "translated"
        and artifact.get("provenance")
        == {"intermediate": "crossgl", "pipeline": "single-file-translate"}
        and isinstance(artifact_path, str),
        "OpenGL FFT translated artifact contract changed",
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
        generated_path.is_file() and generated_path.stat().st_size > 0,
        f"OpenGL FFT artifact is missing: {artifact_path}",
    )
    generated_hash = _sha256(generated_path)
    generated_size = generated_path.stat().st_size
    _require(
        artifact.get("generatedHash")
        == {"algorithm": "sha256", "value": generated_hash}
        and artifact.get("generatedSizeBytes") == generated_size,
        "OpenGL FFT artifact hash or size does not match the emitted file",
    )
    generated = generated_path.read_text(encoding="utf-8")
    _require(
        generated.count("#version 450 core") == 1
        and generated.count("void main()") == 1,
        "OpenGL FFT artifact is not one standalone compute shader",
    )
    _require(
        re.search(r"(?<![A-Za-z0-9_])post_in\s*\(", generated) is None,
        "OpenGL FFT artifact retained an unresolved member-helper call",
    )
    radix_bodies = re.findall(
        r"(?m)^void radix_butterfly_2_radix2\([^;{]*\)\s*\{",
        generated,
    )
    _require(
        len(radix_bodies) == 1,
        "OpenGL FFT artifact emitted duplicate radix helper bodies",
    )

    required_tools = {
        "glslangValidator": shutil.which("glslangValidator"),
        "spirv-val": shutil.which("spirv-val"),
    }
    missing_tools = sorted(
        name for name, resolved in required_tools.items() if resolved is None
    )
    _require(
        not require_toolchain or not missing_tools,
        "OpenGL FFT validation requires: " + ", ".join(missing_tools),
    )
    native_validation_attempted = not missing_tools
    native_validation_status = "not-run-tool-unavailable"
    native_validation_output = None
    if native_validation_attempted:
        output_path = work_dir / "validation" / "fft-opengl.spv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        compile_result = _run_command(
            "validate-fft-opengl",
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
            "OpenGL FFT native compilation failed; inspect validate-fft-opengl logs",
        )
        _require(
            output_path.is_file(),
            "OpenGL FFT compilation succeeded without producing SPIR-V",
        )
        validation_result = _run_command(
            "validate-fft-opengl-spirv",
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
            "OpenGL FFT SPIR-V validation failed; inspect validation logs",
        )
        native_validation_status = "validated"
        native_validation_output = _relpath(output_path, mlx_root)

    return {
        "name": "fft-opengl-toolchain",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_FFT_SOURCE,
        "sourceHash": MLX_FFT_SHA256,
        "target": "opengl",
        "artifactStatus": "translated",
        "artifactEmitted": True,
        "generatedHash": generated_hash,
        "generatedSizeBytes": generated_size,
        "nativeValidationAttempted": native_validation_attempted,
        "nativeValidationStatus": native_validation_status,
        "nativeValidationOutput": native_validation_output,
        "nativeCompiler": "glslangValidator",
        "spirvValidator": "spirv-val",
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": FFT_OPENGL_EXPECTED_SPECIALIZATION_COUNT,
        "functionConstantCount": FFT_OPENGL_EXPECTED_FUNCTION_CONSTANT_COUNT,
        "indexRangeAssertions": list(FFT_INDEX_RANGE_ASSERTIONS),
        "workgroupAccessAssertions": expected_workgroup_assertions,
        "toolchainRequired": require_toolchain,
        "trackedIssues": [],
        "maxTemplateSpecializations": FFT_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": FFT_MAX_TEMPLATE_MATERIALIZATION_WORK,
        "runtimeIntegrationIncluded": False,
        "numericalParityClaimed": False,
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


def _require_gemv_project_opengl_contracts(payload: Mapping[str, Any]) -> None:
    project = payload.get("project")
    expected_subgroup_rules = {MLX_GEMV_SOURCE: str(GEMV_SUBGROUP_WIDTH)}
    expected_index_assertions = [
        {"source": MLX_GEMV_SOURCE, **assertion}
        for assertion in GEMV_OPENGL_INDEX_RANGE_ASSERTIONS
    ]
    _require(
        isinstance(project, Mapping)
        and project.get("subgroupWidthRules") == expected_subgroup_rules
        and project.get("subgroupWidthRuleCount") == 1,
        "OpenGL GEMV report did not retain the exact subgroup-width rule",
    )
    _require(
        project.get("indexRangeAssertions") == expected_index_assertions
        and project.get("indexRangeAssertionCount") == len(expected_index_assertions),
        "OpenGL GEMV report did not retain the exact index-range assertions",
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
    entry_profile = dxc_profile_for_source(GEMV_DIRECTX_ENTRY_PROFILE, generated)
    library_profile = dxc_profile_for_source(
        GEMV_DIRECTX_LIBRARY_PROFILE,
        generated,
    )
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
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
                entry_profile,
                *compiler_arguments,
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
                "profile": entry_profile,
                **(
                    {
                        "compilerArguments": list(compiler_arguments),
                        "minimumShaderModel": "6.2",
                    }
                    if compiler_arguments
                    else {}
                ),
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
            library_profile,
            *compiler_arguments,
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
        **(
            {
                "compilerArguments": list(compiler_arguments),
                "minimumShaderModel": "6.2",
            }
            if compiler_arguments
            else {}
        ),
        "entryProfile": entry_profile,
        "entryProfileCompilerEntryPoints": list(GEMV_DIRECTX_COMPILER_ENTRY_POINTS),
        "entryProfileCompiledEntryPointCount": len(entry_profile_runs),
        "entryProfileCompilerRuns": entry_profile_runs,
        "entryProfileDiagnosticCount": 0,
        "entryProfileUnusedValueWarningCount": 0,
        "libraryProfile": library_profile,
        "libraryExports": list(GEMV_DIRECTX_EXPECTED_ENTRY_POINTS),
        "libraryExportCount": len(GEMV_DIRECTX_EXPECTED_ENTRY_POINTS),
        "libraryExportSetHash": {
            "algorithm": "sha256",
            "value": hashlib.sha256(library_exports.encode("utf-8")).hexdigest(),
        },
        "libraryCompilerRun": {
            "profile": library_profile,
            **(
                {
                    "compilerArguments": list(compiler_arguments),
                    "minimumShaderModel": "6.2",
                }
                if compiler_arguments
                else {}
            ),
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


def _check_gemv_opengl_toolchain(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    required_tools = {
        "glslangValidator": shutil.which("glslangValidator"),
        "spirv-val": shutil.which("spirv-val"),
    }
    missing_tools = sorted(
        name for name, resolved in required_tools.items() if resolved is None
    )
    _require(
        not missing_tools,
        "OpenGL GEMV validation requires: " + ", ".join(missing_tools),
    )

    config_path = config_dir / "gemv-opengl.toml"
    report_path = report_dir / "gemv-opengl.json"
    output_dir = work_dir / "out-gemv-opengl"
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
        subgroup_width_rules={MLX_GEMV_SOURCE: GEMV_SUBGROUP_WIDTH},
        index_range_assertions=tuple(
            {"source": MLX_GEMV_SOURCE, **assertion}
            for assertion in GEMV_OPENGL_INDEX_RANGE_ASSERTIONS
        ),
        metal_source_options={
            "max_template_specializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
    )
    result = _run_command(
        "translate-gemv-opengl",
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
        result.returncode == 0,
        "OpenGL GEMV project translation failed",
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
    _require_gemv_project_opengl_contracts(payload)
    summary = payload.get("summary", {})
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("skippedCount") == 0
        and summary.get("targetCount") == 1
        and summary.get("artifactCount") == GEMV_EXPECTED_ENTRY_POINT_COUNT
        and summary.get("translatedCount") == GEMV_EXPECTED_ENTRY_POINT_COUNT
        and summary.get("failedCount") == 0,
        "OpenGL GEMV translation did not emit every entry-scoped artifact",
    )
    _require(
        summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0}
        and summary.get("diagnosticsByCode") == {}
        and summary.get("missingCapabilityCounts") == {},
        "OpenGL GEMV translation reported diagnostics",
    )
    _require(
        summary.get("artifactProvenanceByPipeline")
        == {"entry-scoped-translate": GEMV_EXPECTED_ENTRY_POINT_COUNT}
        and summary.get("artifactProvenanceByIntermediate")
        == {"crossgl": GEMV_EXPECTED_ENTRY_POINT_COUNT},
        "OpenGL GEMV report provenance summary changed",
    )
    _require(
        summary.get("sourceMapCount") == GEMV_EXPECTED_ENTRY_POINT_COUNT
        and summary.get("sourceRemapCount") == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "OpenGL GEMV report did not retain source provenance for every artifact",
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
        diagnostics == [],
        "OpenGL GEMV report retained project diagnostics",
    )

    artifacts = payload.get("artifacts", [])
    _require(
        isinstance(artifacts, list)
        and len(artifacts) == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "OpenGL GEMV report must contain one artifact for every source entry",
    )

    workgroup_rule_path = f"project.workgroup_size_rules[{json.dumps(MLX_GEMV_SOURCE)}]"
    subgroup_rule_path = f"project.subgroup_width_rules[{json.dumps(MLX_GEMV_SOURCE)}]"
    expected_workgroup_rule = {
        "components": list(GEMV_REPORT_WORKGROUP_SIZE_RULE),
        "path": workgroup_rule_path,
        "sourcePattern": MLX_GEMV_SOURCE,
    }
    expected_subgroup_rule = {
        "expression": str(GEMV_SUBGROUP_WIDTH),
        "path": subgroup_rule_path,
        "sourcePattern": MLX_GEMV_SOURCE,
    }
    source_entry_points = set()
    resolved_workgroup_sizes = set()
    generated_artifacts = []
    for artifact in artifacts:
        artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
        _require(
            isinstance(artifact, Mapping)
            and artifact.get("source") == MLX_GEMV_SOURCE
            and artifact.get("sourceBackend") == "metal"
            and artifact.get("target") == "opengl"
            and artifact.get("status") == "translated"
            and isinstance(artifact_path, str),
            "OpenGL GEMV translated artifact contract changed",
        )
        _require(
            artifact.get("sourceHash") == expected_source_hash
            and artifact.get("sourceSizeBytes") == MLX_GEMV_SOURCE_SIZE_BYTES
            and artifact.get("provenance")
            == {"intermediate": "crossgl", "pipeline": "entry-scoped-translate"},
            "OpenGL GEMV artifact provenance changed",
        )

        generated_path = (mlx_root / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_dir.resolve()),
            "OpenGL GEMV artifact path escaped its output directory",
        )
        _require(
            generated_path.is_file() and generated_path.stat().st_size > 0,
            f"OpenGL GEMV artifact is missing: {artifact_path}",
        )
        _require(
            artifact.get("generatedHash")
            == {"algorithm": "sha256", "value": _sha256(generated_path)}
            and artifact.get("generatedSizeBytes") == generated_path.stat().st_size,
            "OpenGL GEMV artifact hash or size does not match the emitted file",
        )

        entry_point = artifact.get("entryPoint")
        source_entry = (
            entry_point.get("source") if isinstance(entry_point, Mapping) else None
        )
        _require(
            isinstance(source_entry, str)
            and bool(source_entry)
            and entry_point.get("stage") == "compute"
            and entry_point.get("target") == "main"
            and source_entry not in source_entry_points,
            "OpenGL GEMV artifact entry identity changed",
        )
        source_entry_points.add(source_entry)

        execution = artifact.get("execution")
        execution_entries = (
            execution.get("entryPoints") if isinstance(execution, Mapping) else None
        )
        _require(
            isinstance(execution_entries, list)
            and len(execution_entries) == 1
            and execution.get("sourceEntryPoints") == [source_entry]
            and execution.get("provenance")
            == {"kind": "materialized-template-rule", "path": workgroup_rule_path}
            and execution.get("subgroupWidthProvenance")
            == {"kind": "materialized-template-rule", "path": subgroup_rule_path}
            and execution.get("subgroupWidthEnforcement")
            == GEMV_OPENGL_SUBGROUP_WIDTH_ENFORCEMENT
            and _is_sha256_contract_identity(execution.get("identity")),
            "OpenGL GEMV artifact execution provenance changed",
        )
        execution_entry = execution_entries[0]
        workgroup_size = execution_entry.get("workgroupSize")
        _require(
            isinstance(execution_entry, Mapping)
            and execution_entry.get("sourceEntryPoint") == source_entry
            and execution_entry.get("materializedEntryPoint") == source_entry
            and execution_entry.get("targetEntryPoint") == "main"
            and execution_entry.get("rule") == expected_workgroup_rule
            and execution_entry.get("subgroupWidth") == GEMV_SUBGROUP_WIDTH
            and execution_entry.get("subgroupWidthRule") == expected_subgroup_rule
            and _is_sha256_contract_identity(execution_entry.get("identity"))
            and isinstance(workgroup_size, list)
            and len(workgroup_size) == 3,
            "OpenGL GEMV artifact execution contract changed",
        )
        resolved_workgroup_sizes.add(tuple(workgroup_size))

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
            "OpenGL GEMV artifact did not retain complete materialization evidence",
        )

        generated = generated_path.read_text(encoding="utf-8")
        marker = f"#define CROSSTL_REQUIRED_SUBGROUP_WIDTH {GEMV_SUBGROUP_WIDTH}u"
        guard = "if (gl_SubgroupSize != CROSSTL_REQUIRED_SUBGROUP_WIDTH)"
        local_size = tuple(workgroup_size)
        local_size_pattern = re.compile(
            r"layout\s*\(\s*local_size_x\s*=\s*"
            + str(local_size[0])
            + r"\s*,\s*local_size_y\s*=\s*"
            + str(local_size[1])
            + r"\s*,\s*local_size_z\s*=\s*"
            + str(local_size[2])
            + r"\s*\)\s*in\s*;"
        )
        _require(
            generated.count(marker) == 1
            and generated.count(guard) == 1
            and "#extension GL_KHR_shader_subgroup_basic : require" in generated
            and local_size_pattern.search(generated) is not None,
            "OpenGL GEMV generated execution declarations changed",
        )
        generated_artifacts.append((source_entry, artifact_path, generated_path))

    _require(
        len(source_entry_points) == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "OpenGL GEMV source-entry coverage changed",
    )
    _require(
        resolved_workgroup_sizes == set(GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES),
        "OpenGL GEMV resolved workgroup-size coverage changed",
    )

    validation_dir = work_dir / "validation" / "gemv-opengl"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_runs = []
    for index, (source_entry, artifact_path, generated_path) in enumerate(
        generated_artifacts,
        start=1,
    ):
        output_path = validation_dir / f"{source_entry}.spv"
        output_path.unlink(missing_ok=True)
        command_name = f"gemv-opengl-{index:03d}"
        compile_result = _run_command(
            f"compile-{command_name}",
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
            f"glslangValidator failed to compile OpenGL GEMV entry {source_entry}",
        )
        _require(
            output_path.is_file() and output_path.stat().st_size > 0,
            f"glslangValidator emitted no SPIR-V for OpenGL GEMV entry {source_entry}",
        )
        validation_result = _run_command(
            f"validate-{command_name}",
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
            f"spirv-val rejected OpenGL GEMV entry {source_entry}",
        )
        validation_runs.append(
            {
                "sourceEntryPoint": source_entry,
                "artifact": artifact_path,
                "compiledArtifact": _relpath(output_path, mlx_root),
                "compiledHash": {
                    "algorithm": "sha256",
                    "value": _sha256(output_path),
                },
                "compiledSizeBytes": output_path.stat().st_size,
            }
        )

    return {
        "name": "gemv-opengl-toolchain",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_GEMV_SOURCE,
        "sourceHash": MLX_GEMV_SHA256,
        "sourceSizeBytes": MLX_GEMV_SOURCE_SIZE_BYTES,
        "target": "opengl",
        "artifactStatus": "translated",
        "artifactPackaging": "entry-scoped-artifacts",
        "artifactCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "emittedTargetFileCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "reportExecutionEntryCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "sourceMapCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "sourceRemapCount": GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "nativeValidationAttempted": True,
        "nativeValidationStatus": "validated",
        "nativeCompiler": "glslangValidator",
        "spirvValidator": "spirv-val",
        "targetEnvironments": ["opengl", "spirv1.3", "spv1.3"],
        "toolchainValidatedArtifactCount": len(validation_runs),
        "toolchainValidationRuns": validation_runs,
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "unsupportedSpecializationCount": 0,
        "workgroupSizeRule": list(GEMV_WORKGROUP_SIZE_RULE),
        "reportWorkgroupSizeRule": list(GEMV_REPORT_WORKGROUP_SIZE_RULE),
        "reportWorkgroupSizeRuleCount": 1,
        "workgroupSizeRuleConfigured": True,
        "resolvedWorkgroupSizes": [
            list(size) for size in GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
        ],
        "subgroupWidth": GEMV_SUBGROUP_WIDTH,
        "subgroupWidthRuleConfigured": True,
        "subgroupWidthEnforcement": dict(GEMV_OPENGL_SUBGROUP_WIDTH_ENFORCEMENT),
        "subgroupWidthFallbackTrackedBy": GEMV_OPENGL_SUBGROUP_WIDTH_FALLBACK_ISSUE,
        "indexRangeAssertionEvidence": {
            "assertionCount": len(GEMV_OPENGL_INDEX_RANGE_ASSERTIONS),
            "assertions": [
                dict(assertion) for assertion in GEMV_OPENGL_INDEX_RANGE_ASSERTIONS
            ],
            "contractKind": "explicit-host-runtime-portability-preconditions",
            "inferred": False,
            "runtimeEnforced": False,
        },
        "provenanceStatus": "concrete-backing-preserved",
        "accessRangeStatus": "statically-proven-under-project-contracts",
        "trackedIssues": list(GEMV_OPENGL_PORTABILITY_TRACKED_ISSUES),
        "translationBlockedBy": [],
        "executionContractBlockedBy": [],
        "maxTemplateSpecializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        "runtimeExecutionAttempted": False,
        "runtimeIntegrationIncluded": False,
        "runnableArtifactClaimed": False,
        "compilerValidatedArtifactClaimed": True,
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


def _load_full_corpus_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return load_project_translation_checkpoint(checkpoint_path)
    except ValueError as exc:
        raise PortingCheckError(
            f"full-corpus checkpoint is invalid: {checkpoint_path}"
        ) from exc


def _full_corpus_checkpoint_summary(
    checkpoint_path: Path,
    *,
    mlx_root: Path,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        return {
            "path": _relpath(checkpoint_path, mlx_root),
            "produced": False,
        }

    checkpoint = _load_full_corpus_checkpoint(checkpoint_path)
    _require(
        checkpoint.get("schemaVersion") == 1,
        "full-corpus checkpoint schema version changed",
    )
    _require(
        checkpoint.get("kind") == "crosstl-project-translation-checkpoint",
        "full-corpus checkpoint kind changed",
    )
    state = checkpoint.get("state")
    _require(
        state in {"running", "interrupted", "complete"},
        "full-corpus checkpoint state is invalid",
    )
    plan = checkpoint.get("plan")
    _require(isinstance(plan, Mapping), "full-corpus checkpoint plan must be an object")
    completed = plan.get("completed")
    _require(
        isinstance(completed, list),
        "full-corpus checkpoint completed jobs must be a list",
    )
    active = plan.get("active")
    _require(
        active is None or isinstance(active, Mapping),
        "full-corpus checkpoint active coordinate must be an object or null",
    )
    artifact_matrix = checkpoint.get("artifactMatrix")
    _require(
        isinstance(artifact_matrix, Mapping),
        "full-corpus checkpoint artifact matrix must be an object",
    )
    diagnostics = checkpoint.get("diagnostics")
    _require(
        isinstance(diagnostics, list),
        "full-corpus checkpoint diagnostics must be a list",
    )
    diagnostic_counts = Counter(
        diagnostic.get("severity")
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping)
        and isinstance(diagnostic.get("severity"), str)
    )
    diagnostics_by_code = Counter(
        diagnostic.get("code")
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping) and isinstance(diagnostic.get("code"), str)
    )
    last_completed = None
    if completed:
        last_record = completed[-1]
        if isinstance(last_record, Mapping) and isinstance(
            last_record.get("coordinate"), Mapping
        ):
            last_completed = dict(last_record["coordinate"])
    return {
        "path": _relpath(checkpoint_path, mlx_root),
        "produced": True,
        "state": state,
        "resumable": state in {"running", "interrupted"},
        "jobCount": plan.get("jobCount"),
        "completedCount": plan.get("completedCount"),
        "pendingCount": plan.get("pendingCount"),
        "activeCoordinate": dict(active) if active is not None else None,
        "lastCompletedCoordinate": last_completed,
        "artifactMatrix": dict(artifact_matrix),
        "diagnosticCounts": dict(sorted(diagnostic_counts.items())),
        "diagnosticsByCode": dict(sorted(diagnostics_by_code.items())),
        "checkpointHash": checkpoint.get("checkpointHash"),
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
    checkpoint_path = report_dir / "full-corpus.checkpoint.json"
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
    resume_checkpoint = False
    if checkpoint_path.is_file():
        checkpoint = _load_full_corpus_checkpoint(checkpoint_path)
        resume_checkpoint = checkpoint.get("state") in {"running", "interrupted"}
    report_path.unlink(missing_ok=True)
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
        "--checkpoint",
        str(checkpoint_path),
        "--job-timeout-seconds",
        str(FULL_CORPUS_JOB_TIMEOUT_SECONDS),
        "--validate",
    ]
    if resume_checkpoint:
        command.append("--resume")
    result = _run_command(
        "translate-full-corpus",
        command,
        log_dir=log_dir,
        check=False,
        timeout_seconds=FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS,
    )
    checkpoint_summary = _full_corpus_checkpoint_summary(
        checkpoint_path,
        mlx_root=mlx_root,
    )
    if not report_path.is_file() and result.returncode:
        if result.returncode == 124:
            _require(
                checkpoint_summary["produced"],
                "full-corpus translation timed out without a progress checkpoint",
            )
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
            "unitCount": FULL_CORPUS_EXPECTED_UNIT_COUNT,
            "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "targets": list(FULL_CORPUS_TARGETS),
            "returncode": result.returncode,
            "jobTimeoutSeconds": FULL_CORPUS_JOB_TIMEOUT_SECONDS,
            "timeoutSeconds": FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS,
            "checkpoint": checkpoint_summary,
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
        summary.get("unitCount") == FULL_CORPUS_EXPECTED_UNIT_COUNT,
        "full-corpus translation must scan {} units; found {}".format(
            FULL_CORPUS_EXPECTED_UNIT_COUNT,
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
        targets=FULL_CORPUS_TARGETS,
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
                for target, contract in MLX_FENCE_TARGET_CONTRACTS.items()
                if target in FULL_CORPUS_TARGETS
            },
            "project.validate.failed-artifact": (
                FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
            ),
        }
    )
    expected_target_counts = {
        target: {
            "translatedCount": FULL_CORPUS_EXPECTED_UNIT_COUNT - 1,
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
            "unitCount": FULL_CORPUS_EXPECTED_UNIT_COUNT,
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
            "checkpoint": checkpoint_summary,
            "jobTimeoutSeconds": FULL_CORPUS_JOB_TIMEOUT_SECONDS,
            "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "maxTemplateMaterializationWork": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        }
    return {
        "name": "full-corpus",
        "status": "passed-with-expected-fence-blockers",
        "report": _relpath(report_path, mlx_root),
        "unitCount": FULL_CORPUS_EXPECTED_UNIT_COUNT,
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
        "checkpoint": checkpoint_summary,
        "jobTimeoutSeconds": FULL_CORPUS_JOB_TIMEOUT_SECONDS,
        "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK,
    }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    mlx_root = Path(args.mlx_root).resolve()
    full_corpus = args.mode == FULL_CORPUS_MODE
    expected_commit = MLX_CORPUS_COMMIT if full_corpus else MLX_REFERENCE_COMMIT
    expected_unit_count = (
        FULL_CORPUS_EXPECTED_UNIT_COUNT if full_corpus else EXPECTED_METAL_KERNEL_COUNT
    )
    scan_targets = FULL_CORPUS_TARGETS if full_corpus else MLX_REFERENCE_TARGETS
    require_metal_toolchain = bool(getattr(args, "require_metal_toolchain", False))
    require_directx_gemv_compiler_frontier = bool(
        getattr(args, "require_directx_gemv_compiler_frontier", False)
    )
    require_opengl_frontier_toolchain = bool(
        getattr(args, "require_opengl_frontier_toolchain", False)
    )
    require_opengl_gemv_toolchain = bool(
        getattr(args, "require_opengl_gemv_toolchain", False)
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
        not require_opengl_gemv_toolchain or args.mode == REDUCED_FRONTIER_MODE,
        "--require-opengl-gemv-toolchain is only valid in reduced-frontier mode",
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
        _verify_mlx_checkout(
            mlx_root,
            args.python,
            log_dir,
            expected_commit=expected_commit,
        ),
        _scan_metal_kernels(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
            expected_unit_count=expected_unit_count,
            targets=scan_targets,
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
        if args.require_directx_toolchain:
            checks.append(
                _check_fft_directx_toolchain(
                    mlx_root,
                    work_dir,
                    config_dir,
                    report_dir,
                    log_dir,
                    args.python,
                    require_toolchain=True,
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
            _check_fft_opengl_toolchain(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_toolchain=require_opengl_frontier_toolchain,
            )
        )
        if require_opengl_gemv_toolchain:
            checks.append(
                _check_gemv_opengl_toolchain(
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
    fft_opengl_toolchain_included = args.mode == REDUCED_FRONTIER_MODE
    fft_directx_toolchain_included = bool(
        args.mode == REDUCED_FRONTIER_MODE and args.require_directx_toolchain
    )
    return {
        "schema_version": 1,
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": expected_commit,
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
                MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
            ),
            "directxTranslatedFrontierArtifactCount": (
                MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT
            ),
            "directxWorkgroupBlockedFrontierSources": list(
                MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
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
            "hostDispatchImportResolvedIssue": MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
            "fullCorpusTargets": list(FULL_CORPUS_TARGETS),
            "fullCorpusExpectedUnitCount": FULL_CORPUS_EXPECTED_UNIT_COUNT,
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
            "fftDirectXToolchainIncluded": fft_directx_toolchain_included,
            "fftDirectXToolchainRequired": fft_directx_toolchain_included,
            "fftOpenGLToolchainIncluded": fft_opengl_toolchain_included,
            "fftOpenGLToolchainRequired": bool(
                fft_opengl_toolchain_included and require_opengl_frontier_toolchain
            ),
            "openglGemvToolchainRequired": require_opengl_gemv_toolchain,
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
        "--require-opengl-gemv-toolchain",
        action="store_true",
        help=(
            "Translate every pinned GEMV OpenGL entry and require native GLSL and "
            "SPIR-V 1.3 validation."
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
    expected_commit = (
        MLX_CORPUS_COMMIT if args.mode == FULL_CORPUS_MODE else MLX_REFERENCE_COMMIT
    )
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
                        "commit": expected_commit,
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
