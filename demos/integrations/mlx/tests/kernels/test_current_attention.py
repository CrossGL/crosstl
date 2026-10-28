"""Translate the pinned MLX steel attention entries and compare native results.

The entries (``steel_attention_{T}_bq32_bk{32,16}_bd{D}_bv{V}_wm4_wn{1,2}_mask{M}``)
compute ``softmax(scale * Q K^T + mask) V`` with threadgroup Q, K and V tiles,
simdgroup matrix fragments and an online base-2 softmax. The gate covers the
head dimensions 64, 72, 80, 96, 128 and 256 and the asymmetric query and value
head dimensions (96, 64), with the key block ``scaled_dot_product_attention.cpp``
selects for each. Every element type and both mask types meet several of them.
Head dimension 192 is not covered: its 16-bit tiles fill 21.5 KiB of threadgroup
memory, which the OpenGL and bfloat16 DirectX outputs widen to 32-bit words,
past the 32 KiB limit. Head dimension 256 uses the ``wn2`` entries, whose two
simdgroups per row each multiply half of the head dimension and add their
partial scores through the K/V tile viewed as float32; the gate covers their
float32 entries. Thirteen entries freeze the aligned, masked
constants that host code selects for sequence lengths that are multiples of the
blocks. Four more freeze the other sets the host selects: unaligned sequences
with attention sinks under float and boolean masks, and causal masks, unaligned
with the logsumexp and aligned with sinks and the logsumexp. The gate fills
``AttnParams`` and ``AttnMaskParams`` as the host does, and dispatches the
upstream ``(NQ, H, B)`` grid of ``(32, 4, WN)`` threadgroups.

With ``WN == 1`` the kernel widens its tiles to float32, and the ``wn2``
entries are float32 already. Both accumulate in float32, so every case must
match a float64 evaluation of the definition on the decoded inputs to within
``1e-5 * (1 + |reference|)``, the tolerance of the other attention gates.
float16 and bfloat16 outputs must also be one of the values that their final
store can round that interval to. A saved logsumexp is
float32 and must meet the same tolerance. Every case must leave the output
guards, and the ``lse`` buffer when it is unused, unchanged. On Metal the original
MLX source, compiled and dispatched with the same constants and buffers, must
meet the same reference.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    pack_storage_records,
    translate_project,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeConstantBinding,
    RuntimeExecutionState,
    RuntimeSpecializationConstant,
)
from demos.integrations.mlx.tests.kernels.test_current_cross_entropy import (
    _decode,
    _element_layout,
    _runtime_value,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_native_loader_dispatch_integration import _executor

np = pytest.importorskip("numpy")

MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
SOURCE = "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal"
SOURCE_SHA256 = "ade46f59394199b17ae4a04323d2840838d594d22173490df732f1c3b0b8f605"
HEADER_HASHES = {
    "kernels/steel_attention.h": (
        "0519f2ec5f6b2917a7685df85ecc5555ca8acd7e070a14c9945ad28adadec36c"
    ),
    "attn.h": "67722214ff487ec95df675e2f135ccd7fbe639b6eb6ab98b14bcefa58d66ba95",
    "loader.h": "e435f413d71d44ca4ad72a59d192046be77388f63936179c58882286f4cc5dab",
    "mma.h": "6f22866c76a1f15cf27681d3a8700f3419412a954d00e0810c87f7dac82ddeab",
    "params.h": "f03c6ed73d7298fa69b75a49e7eec54954b6fafd59530a1f175934215412ed80",
    "transforms.h": "6479114663c5923a7ee2f9fc1cf9d5e99e680d47d793251749989056a8a63141",
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ATTENTION"
CONSTANT_IDS = {
    "align_Q": 200,
    "align_K": 201,
    "has_mask": 300,
    "do_causal": 301,
    "has_sinks": 302,
    "save_lse": 303,
}
SPECIALIZATIONS = {
    "align_Q": True,
    "align_K": True,
    "has_mask": True,
    "do_causal": False,
    "has_sinks": False,
    "save_lse": False,
}
SIGNED_MAX = 2**31 - 1
PLATFORM_TARGETS = {"darwin": "metal", "win32": "directx", "linux": "opengl"}
GATE_TOOLS = {
    "metal": ("xcrun",),
    "directx": ("dxc",),
    "opengl": ("glslangValidator",),
}
# -98765 is not a binary16 or bfloat16 value; 3 * 2^13 is a value of both.
GUARD = {"float32": -98765.0, "float16": -24576.0, "bfloat16": -24576.0}
GUARDS = 4
TOLERANCE = 1e-5
# Accessor element names in the resource helpers the OpenGL ranges name.
HELPER_ELEMENTS = {
    "float32": "float",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "bool": "bool",
}
SIGNIFICAND_BITS = {"float16": 11, "bfloat16": 8}
# frexp exponent of the smallest normal value of each type.
MIN_NORMAL_EXPONENT = {"float16": -13, "bfloat16": -125}
ORIGINAL_FLAGS = (
    "-std=metal3.2",
    "-fno-fast-math",
    "-Wall",
    "-Wextra",
    "-Wno-c++17-extensions",
    "-Wno-c++20-extensions",
)
BINDINGS = ("Q", "K", "V", "O", "params", "mask_params", "mask", "sinks", "lse")


@dataclass(frozen=True)
class Variant:
    """One instantiated entry and the function constants it is frozen with.

    ``aligned`` sets ``align_Q`` and ``align_K``, which the host derives from
    the sequence lengths, so it also selects the cases. A causal mask is the
    host's ``mask="causal"``, which is not an array mask, and the host returns
    the logsumexp only without an array mask.
    """

    dtype: str
    bq: int
    bk: int
    head_dim: int
    boolean_mask: bool = False
    aligned: bool = True
    has_mask: bool = True
    causal: bool = False
    sinks: bool = False
    lse: bool = False
    wn: int = 1
    # The value head dimension when it differs from the query's.
    bv: int = 0

    def __post_init__(self):
        assert self.has_mask or not self.boolean_mask
        assert not (self.causal and self.has_mask)
        assert not (self.lse and self.has_mask)
        # The host splits only head dimension 256 across two simdgroups.
        assert self.wn == (2 if self.head_dim == 256 else 1)
        # The host's block shapes: (96, 64) is its only asymmetric pair, and
        # head dimensions from 128 halve the key block.
        assert self.bv in {0, 64} and (not self.bv or self.head_dim == 96)
        blocks = {16} if self.head_dim >= 128 else {32}
        if self.head_dim == 256 and self.dtype != "float32":
            blocks.add(32)
        assert self.bk in blocks

    @property
    def value_dim(self):
        return self.bv or self.head_dim

    @property
    def workgroup(self):
        return (32, 4, self.wn)

    @property
    def mask_dtype(self):
        return "bool" if self.boolean_mask else self.dtype

    @property
    def specializations(self):
        return {
            "align_Q": self.aligned,
            "align_K": self.aligned,
            "has_mask": self.has_mask,
            "do_causal": self.causal,
            "has_sinks": self.sinks,
            "save_lse": self.lse,
        }

    @property
    def cases(self):
        return CASES if self.aligned else UNALIGNED_CASES

    @property
    def name(self):
        flags = (
            ("unaligned", not self.aligned),
            ("unmasked", not self.has_mask),
            ("causal", self.causal),
            ("sinks", self.sinks),
            ("lse", self.lse),
        )
        suffix = "".join(f"-{flag}" for flag, chosen in flags if chosen)
        dims = f"bd{self.head_dim}" + (f"-bv{self.bv}" if self.bv else "")
        return f"{self.dtype}-{dims}-mask{self.mask_dtype}{suffix}"

    @property
    def entry(self):
        mask = "bool_" if self.boolean_mask else self.dtype
        return (
            f"steel_attention_{self.dtype}_bq{self.bq}_bk{self.bk}"
            f"_bd{self.head_dim}_bv{self.value_dim}"
            f"_wm4_wn{self.wn}_mask{mask}"
        )


# Instantiations in steel_attention.metal, all with (32, 4, 1) threadgroups.
# Head dimension 128 halves the key block to bound the tiles. The float32
# float-mask entries cover every head dimension; the narrow element types and
# the boolean masks each cover more than one head dimension.
VARIANTS = (
    Variant("float32", 32, 32, 64),
    Variant("float32", 32, 32, 80),
    Variant("float32", 32, 16, 128),
    Variant("float16", 32, 32, 64),
    Variant("bfloat16", 32, 32, 64),
    Variant("float32", 32, 32, 64, boolean_mask=True),
    Variant("float16", 32, 16, 128, boolean_mask=True),
    Variant("bfloat16", 32, 32, 80, boolean_mask=True),
    # Other function-constant sets the host selects: unaligned sequences with
    # attention sinks, and causal masks with and without the logsumexp.
    Variant("float32", 32, 32, 64, aligned=False, sinks=True),
    Variant("float16", 32, 16, 128, boolean_mask=True, aligned=False, sinks=True),
    Variant(
        "float32", 32, 32, 80, aligned=False, has_mask=False, causal=True, lse=True
    ),
    Variant("bfloat16", 32, 32, 64, has_mask=False, causal=True, sinks=True, lse=True),
    # Head dimension 256 splits each row across two simdgroups that exchange
    # their partial scores through the K/V tile reinterpreted as float32.
    Variant("float32", 32, 16, 256, wn=2),
    Variant("float32", 32, 16, 256, boolean_mask=True, wn=2),
    # Head dimensions 72 and 96, and the asymmetric (96, 64) entry.
    Variant("float32", 32, 32, 72),
    Variant("float16", 32, 32, 96, boolean_mask=True),
    Variant("float32", 32, 32, 96, bv=64),
)


@dataclass(frozen=True)
class Case:
    """One dispatch: shapes, the Q/O layout and how the mask broadcasts."""

    batches: int
    heads: int
    kv_heads: int
    query_length: int
    key_length: int
    sequence_major: bool
    mask_per_head: bool
    seed: int

    @property
    def name(self):
        layout = "blhd" if self.sequence_major else "bhld"
        mask = "head-mask" if self.mask_per_head else "shared-mask"
        return (
            f"b{self.batches}-h{self.heads}-kv{self.kv_heads}-"
            f"q{self.query_length}-k{self.key_length}-{layout}-{mask}"
        )


CASES = (
    Case(1, 2, 2, 64, 64, False, True, 7),
    # Grouped query heads, a longer key sequence (qL_off = 64), the
    # sequence-major Q/O layout MLX passes through strides and a mask
    # broadcast over heads with a zero head stride.
    Case(2, 4, 2, 32, 96, True, False, 11),
)
# Partial last query and key blocks for both key block sizes. The second case
# has a single, partial query block, so its first block is also its last.
UNALIGNED_CASES = (
    Case(1, 2, 2, 50, 77, False, True, 13),
    Case(2, 4, 2, 21, 70, True, False, 17),
)


def _opengl_index_assertions(variant, source=SOURCE):
    """GLSL indexes with 32-bit ints, so 64-bit buffer offsets need ranges.

    The kernel offsets Q, K, V, O and the mask by ``int64_t`` strides from
    ``AttnParams``; MLX buffers stay below 2^31 elements. The threadgroup tiles
    need no assertion: their view offsets are proven from the loaders and the
    fragment coordinates. A boolean mask is read through its own accessor, and
    a narrow entry stores its float32 logsumexp through the float accessor.
    """
    element = HELPER_ELEMENTS[variant.dtype]
    accesses = [f"load_{element}", f"store_{element}"]
    if variant.boolean_mask:
        accesses.append("load_bool")
    if variant.lse and element != "float":
        accesses.append("store_float")
    return tuple(
        {
            "source": source,
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}",
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for access in accesses
    )


def _config(root, output, target="metal", variant=VARIANTS[0]):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (variant.entry,)},
        specialization_constants=variant.specializations,
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: variant.workgroup},
        index_range_assertions=(
            _opengl_index_assertions(variant) if target == "opengl" else ()
        ),
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                "target_options": {
                    "directx": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": 32,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": 32,
                    },
                },
            }
        },
    )


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda variant: variant.name)
@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_attention_target_contract_is_explicit(tmp_path, target, variant):
    config = _config(tmp_path, "out", target, variant)
    assert list(config.targets) == [target]
    assert config.entry_points == {SOURCE: (variant.entry,)}
    assert config.workgroup_size_rules == {
        SOURCE: tuple(str(size) for size in variant.workgroup)
    }
    assert config.freeze_specialization_constants is True
    assert config.specialization_constants == variant.specializations
    assert not config.workgroup_access_assertions
    ranges = [item.to_json() for item in config.index_range_assertions]
    if target != "opengl":
        assert not ranges
        return
    assert ranges == list(_opengl_index_assertions(variant))
    element = HELPER_ELEMENTS[variant.dtype]
    assert {item["function"] for item in ranges} == {
        f"crosstl_resource_load_{element}",
        f"crosstl_resource_store_{element}",
        *(("crosstl_resource_load_bool",) if variant.boolean_mask else ()),
        *(
            ("crosstl_resource_store_float",)
            if variant.lse and variant.dtype != "float32"
            else ()
        ),
    }
    assert all(
        item["minimum"] == 0 and item["maximum"] == SIGNED_MAX for item in ranges
    )


def test_attention_variants_span_types_shapes_and_masks():
    """Each element type and each mask type meets several block shapes.

    Translating an entry proves that it is instantiated upstream. Every other
    function constant is set by at least one entry and cleared by another.
    """
    shapes = {(32, 32, 64), (32, 32, 80), (32, 16, 128), (32, 16, 256)}
    assert len({variant.name for variant in VARIANTS}) == len(VARIANTS)
    assert Variant("float32", 32, 32, 64).specializations == SPECIALIZATIONS
    default = [v for v in VARIANTS if v.specializations == SPECIALIZATIONS]
    assert len({variant.entry for variant in default}) == len(default) == 13
    assert {variant.dtype for variant in VARIANTS} == set(SIGNIFICAND_BITS) | {
        "float32"
    }
    for boolean in (False, True):
        chosen = [v for v in default if v.boolean_mask is boolean]
        assert {v.dtype for v in chosen} == {"float32", "float16", "bfloat16"}
    float32 = {(v.bq, v.bk, v.head_dim) for v in default if v.dtype == "float32"}
    assert float32 == shapes | {(32, 32, 72), (32, 32, 96)}
    boolean = {(v.bq, v.bk, v.head_dim) for v in default if v.boolean_mask}
    assert boolean == shapes | {(32, 32, 96)}
    # Every head dimension the host dispatches to these entries except 192,
    # and its asymmetric (query, value) pair.
    assert {(v.head_dim, v.value_dim) for v in VARIANTS} == {
        *((d, d) for d in (64, 72, 80, 96, 128, 256)),
        (96, 64),
    }
    assert {v.workgroup for v in VARIANTS} == {(32, 4, 1), (32, 4, 2)}
    for name in CONSTANT_IDS:
        values = {variant.specializations[name] for variant in VARIANTS}
        assert values == {False, True}, name
    # Each other set covers a narrow element type, and the causal sets both
    # alignments.
    others = [v for v in VARIANTS if v not in default]
    assert {v.dtype for v in others} == {"float32", "float16", "bfloat16"}
    assert {v.aligned for v in others if v.causal} == {False, True}
    assert any(v.boolean_mask and v.sinks for v in others)
    # Every entry runs two cases of its alignment.
    assert len(CASES) == len(UNALIGNED_CASES) == 2


def _strides(shape):
    strides, size = [], 1
    for extent in reversed(shape):
        strides.insert(0, size)
        size *= extent
    return strides


def _quantized(dtype, values):
    """Round float32 ``values`` to ``dtype`` with ties to even, as float32."""
    values = np.asarray(values, dtype=np.float32)
    if dtype == "float32":
        return values
    if dtype == "float16":
        return values.astype(np.float16).astype(np.float32)
    assert np.all(np.isfinite(values))
    bits = values.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32)


def _round(dtype, value):
    """Round a float64 value to ``dtype`` with ties to even, in one step."""
    if value == 0 or not math.isfinite(value):
        return value
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    quantum = 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])
    return round(value / quantum) * quantum


def _case_data(case, variant):
    """Inputs, AttnParams as the host fills them, and the float64 reference.

    Inputs are rounded to the entry's element type first, so the reference
    evaluates the definition on the values the kernel decodes. A boolean mask
    keeps three quarters of the keys; every third query row also drops its
    whole first key block, which the online softmax must then discard, and
    every row keeps its last key so that the softmax is defined. A causal row
    ``i`` keeps the keys ``j <= i + qL_off``. A sink adds one logit per query
    head to every row's softmax without a value. The logsumexp is the natural
    logarithm of each row's softmax denominator, the sink included.
    """
    bq, bk = variant.bq, variant.bk
    assert (case.query_length % bq == 0) is variant.aligned
    assert (case.key_length % bk == 0) is variant.aligned
    assert case.key_length >= 2 * bk
    assert case.key_length >= case.query_length
    assert case.heads % case.kv_heads == 0
    rng = np.random.default_rng(case.seed)
    b, h, kv, ql, kl, d, dv = (
        case.batches,
        case.heads,
        case.kv_heads,
        case.query_length,
        case.key_length,
        variant.head_dim,
        variant.value_dim,
    )
    dtype = variant.dtype
    q = _quantized(dtype, rng.uniform(-1, 1, (b, h, ql, d)))
    k = _quantized(dtype, rng.uniform(-1, 1, (b, kv, kl, d)))
    v = _quantized(dtype, rng.uniform(-1, 1, (b, kv, kl, dv)))
    mask_heads = h if case.mask_per_head else 1
    if not variant.has_mask:
        mask = np.zeros(1, dtype=np.float32)
        additive = np.zeros((1, 1, ql, kl))
    elif variant.boolean_mask:
        mask = rng.random((b, mask_heads, ql, kl)) < 0.75
        mask[:, :, ::3, :bk] = False
        mask[..., -1] = True
        additive = np.where(mask, 0.0, -np.inf)
    else:
        mask = _quantized(dtype, rng.uniform(-2, 0, (b, mask_heads, ql, kl)))
        additive = mask.astype(np.float64)
    if variant.causal:
        rows = np.arange(ql)[:, None] + (kl - ql)
        additive = np.where(rows >= np.arange(kl)[None, :], additive, -np.inf)
    sinks = _quantized(dtype, rng.uniform(-1, 2, h)) if variant.sinks else None
    scale = np.float32(1.0 / math.sqrt(d))
    group = h // kv
    keys = np.repeat(k.astype(np.float64), group, axis=1)
    values = np.repeat(v.astype(np.float64), group, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), keys)
    scores = scores * np.float64(scale) + additive
    peak = scores.max(axis=-1)
    if sinks is not None:
        sink = np.broadcast_to(sinks.astype(np.float64)[None, :, None], peak.shape)
        peak = np.maximum(peak, sink)
    weights = np.exp(scores - peak[..., None])
    denominator = weights.sum(axis=-1)
    if sinks is not None:
        denominator = denominator + np.exp(sink - peak)
    reference = np.einsum("bhqk,bhkd->bhqd", weights, values)
    reference /= denominator[..., None]
    lse = peak + np.log(denominator)
    if case.sequence_major:
        q_storage = np.ascontiguousarray(q.transpose(0, 2, 1, 3))
        reference = reference.transpose(0, 2, 1, 3)
        q_strides = [ql * h * d, d, h * d]
        o_strides = [ql * h * dv, dv, h * dv]
    else:
        q_storage = q
        q_strides = _strides([b, h, ql, d])[:3]
        o_strides = _strides([b, h, ql, dv])[:3]
    k_strides = _strides([b, kv, kl, d])[:3]
    if variant.has_mask:
        mask_strides = _strides([b, mask_heads, ql, kl])[:3]
        if not case.mask_per_head:
            mask_strides[1] = 0
    else:
        mask_strides = [0, 0, 0]
    params = dict(
        B=b,
        H=h,
        D=d,
        qL=ql,
        kL=kl,
        gqa_factor=group,
        scale=float(scale),
        NQ=-(-ql // bq),
        NK=-(-kl // bk),
        NQ_aligned=ql // bq,
        NK_aligned=kl // bk,
        qL_rem=ql % bq,
        kL_rem=kl % bk,
        qL_off=kl - ql,
        Q_strides=q_strides,
        K_strides=k_strides,
        V_strides=_strides([b, kv, kl, dv])[:3],
        O_strides=o_strides,
    )
    resources = {
        "Q": q_storage.ravel(),
        "K": k.ravel(),
        "V": v.ravel(),
        "mask": mask.ravel(),
        "sinks": sinks,
    }
    geometry = {
        "workgroupCount": [params["NQ"], h, b],
        "workgroupSize": list(variant.workgroup),
    }
    return (
        resources,
        params,
        {"M_strides": mask_strides},
        reference.ravel(),
        lse.ravel() if variant.lse else None,
        geometry,
    )


def _pairs():
    return [
        pytest.param(case, variant, id=f"{case.name}-{variant.name}")
        for variant in VARIANTS
        for case in variant.cases
    ]


@pytest.mark.parametrize(("case", "variant"), _pairs())
def test_attention_reference_follows_the_definition(case, variant):
    resources, params, mask_params, reference, lse, geometry = _case_data(case, variant)
    head_dim = variant.head_dim
    rows = case.batches * case.heads * case.query_length
    assert reference.shape == (rows * variant.value_dim,)
    assert resources["V"].shape == (
        resources["K"].size // head_dim * variant.value_dim,
    )
    # Weighted averages of values in [-1, 1] stay in [-1, 1]; a sink only
    # moves weight to a zero value.
    assert np.all(np.abs(reference) <= 1.0)
    assert params["qL_off"] == case.key_length - case.query_length
    assert params["gqa_factor"] * case.kv_heads == case.heads
    for length, block, count, aligned, remainder in (
        ("qL", variant.bq, "NQ", "NQ_aligned", "qL_rem"),
        ("kL", variant.bk, "NK", "NK_aligned", "kL_rem"),
    ):
        assert params[aligned] * block + params[remainder] == params[length]
        assert params[count] == params[aligned] + (params[remainder] > 0)
        assert (params[remainder] == 0) is variant.aligned
    assert geometry["workgroupCount"] == [params["NQ"], case.heads, case.batches]
    assert params["D"] == head_dim
    if variant.has_mask:
        assert (mask_params["M_strides"][1] == 0) is (not case.mask_per_head)
    else:
        assert mask_params["M_strides"] == [0, 0, 0]
    if case.sequence_major:
        assert params["Q_strides"][1] == head_dim
        assert params["Q_strides"][2] == case.heads * head_dim
        assert params["O_strides"][2] == case.heads * variant.value_dim
    narrow = ["Q", "K", "V"]
    narrow += ["mask"] if variant.has_mask and not variant.boolean_mask else []
    narrow += ["sinks"] if variant.sinks else []
    for name in narrow:
        values = resources[name]
        assert values.dtype == np.float32
        assert np.array_equal(_quantized(variant.dtype, values), values), name
    assert (resources["sinks"] is None) is (not variant.sinks)
    if variant.boolean_mask:
        mask = resources["mask"].reshape(
            case.batches, -1, case.query_length, case.key_length
        )
        assert mask.dtype == np.bool_
        assert mask.any(axis=-1).all()
        assert not mask[:, :, ::3, : variant.bk].any()
        assert mask[:, :, 1::3, : variant.bk].any()
        assert 0.6 < mask.mean() < 0.8
    assert (lse is None) is (not variant.lse)
    if variant.lse:
        assert lse.shape == (rows,) and np.all(np.isfinite(lse))


def test_attention_reference_masks_causal_rows_and_counts_sinks():
    """The causal reference ignores later keys, and a sink lowers every weight."""
    variant = next(v for v in VARIANTS if v.causal and v.sinks)
    case = variant.cases[0]
    _, _, _, reference, lse, _ = _case_data(case, variant)
    plain = replace(variant, causal=False, sinks=False, lse=False, has_mask=False)
    unmasked = _case_data(case, plain)[3]
    assert not np.allclose(reference, unmasked)
    # The last query row sees every key, so only the sink changes it.
    d, ql = variant.value_dim, case.query_length
    shape = (case.batches, case.heads, ql, d)
    last = reference.reshape(shape)[:, :, -1]
    plain_last = unmasked.reshape(shape)[:, :, -1]
    assert np.all(np.abs(last) <= np.abs(plain_last) + 1e-12)
    assert lse is not None and lse.shape == (case.batches * case.heads * ql,)


@pytest.mark.parametrize("dtype", sorted(SIGNIFICAND_BITS))
def test_attention_rounding_matches_numpy(dtype):
    rng = np.random.default_rng(3)
    samples = np.concatenate(
        [rng.uniform(-1, 1, 4096), [1.0, -1.0, 2.0**-14, 2.0**-20, 1 + 2.0**-8]]
    )
    rounded = [_round(dtype, float(value)) for value in samples]
    if dtype == "float16":
        assert rounded == samples.astype(np.float16).astype(np.float64).tolist()
    # Ties go to even, and rounding to the element type is idempotent.
    quantum = 2.0 ** (1 - SIGNIFICAND_BITS[dtype])
    assert _round(dtype, 1 + quantum / 2) == 1.0
    assert _round(dtype, 1 + 3 * quantum / 2) == 1 + 2 * quantum
    narrow = _quantized(dtype, samples).astype(np.float64)
    assert [_round(dtype, float(value)) for value in narrow] == narrow.tolist()
    # The vectorized input rounding agrees with the scalar bound rounding.
    single = samples.astype(np.float32)
    assert _quantized(dtype, single).astype(np.float64).tolist() == [
        _round(dtype, float(value)) for value in single
    ]


def _record(binding, values):
    words = pack_storage_records(binding["scalarLayout"], [values])
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


def _stored(binding, dtype, values):
    """Logical values in the storage the descriptor reflects for ``binding``.

    Metal stores the element type itself, DirectX stores 16-bit types as
    ``uint16`` words and OpenGL widens them to float32, which then holds the
    narrow values exactly.
    """
    values = [float(value) for value in values]
    element_type, encoding = _element_layout(binding)
    assert element_type in {dtype, "float32", "uint16"}, (binding["name"], element_type)
    if encoding is not None:
        assert encoding["logicalElementType"] == dtype, (binding["name"], encoding)
    return _runtime_value(dtype, element_type, values, encoding)


def _stored_mask(binding, target, values):
    """Booleans as Metal ``bool`` or as the 32-bit words DirectX and OpenGL use."""
    element_type, encoding = _element_layout(binding)
    assert encoding is None, binding
    if target == "metal":
        assert element_type == "bool", binding
        items = [bool(value) for value in values]
    else:
        assert element_type == "uint32", binding
        items = [int(bool(value)) for value in values]
    return {"dtype": element_type, "shape": [len(items)], "values": items}


def _dispatch_values(bindings, case, variant, target):
    resources, params, mask_params, reference, lse, geometry = _case_data(case, variant)
    dtype, guard = variant.dtype, GUARD[variant.dtype]
    wide = GUARD["float32"]
    values = {
        name: _stored(bindings[name], dtype, resources[name])
        for name in ("Q", "K", "V")
    }
    # Buffers a cleared constant leaves unused hold guards.
    sinks = resources["sinks"] if variant.sinks else [guard]
    lse_initial = [wide] * (len(lse) + GUARDS) if variant.lse else [wide]
    values.update(
        O=_stored(bindings["O"], dtype, [guard] * (len(reference) + GUARDS)),
        mask=(
            _stored_mask(bindings["mask"], target, resources["mask"])
            if variant.boolean_mask
            else _stored(
                bindings["mask"],
                dtype,
                resources["mask"] if variant.has_mask else [guard],
            )
        ),
        sinks=_stored(bindings["sinks"], dtype, sinks),
        lse=_stored(bindings["lse"], "float32", lse_initial),
        params=_record(bindings["params"], params),
        mask_params=_record(bindings["mask_params"], mask_params),
    )
    inputs = {bindings[name]["name"]: value for name, value in values.items()}
    outputs = {
        bindings["O"]["name"]: _stored(
            bindings["O"],
            dtype,
            [
                _round(dtype, value) if dtype != "float32" else value
                for value in reference
            ]
            + [guard] * GUARDS,
        ),
        bindings["lse"]["name"]: (
            _stored(bindings["lse"], "float32", list(lse) + [wide] * GUARDS)
            if variant.lse
            else values["lse"]
        ),
    }
    return inputs, outputs, geometry, reference, lse


def _check(label, actual, outputs, reference, lse, bindings, dtype):
    """Compare the decoded outputs with the float64 references.

    float32 results must lie within ``TOLERANCE * (1 + |reference|)``. A narrow
    result must be a value that the store can round that interval to, the
    interval's rounded end points included. The logsumexp is float32 for every
    element type.
    """
    name, lse_name = bindings["O"]["name"], bindings["lse"]["name"]
    # DirectX may report 16-bit floating storage by its uint16 word type.
    physical = bindings["O"]["scalarLayout"]["elementType"]
    assert actual[name]["dtype"] in {outputs[name]["dtype"], physical}, label
    output = np.asarray(_decode(dtype, actual[name]), dtype=np.float64)
    assert output.shape == (len(reference) + GUARDS,), label
    assert output[-GUARDS:].tolist() == [GUARD[dtype]] * GUARDS, label
    saved = np.asarray(_decode("float32", actual[lse_name]), dtype=np.float64)
    result = {}
    if lse is None:
        assert saved.tolist() == [GUARD["float32"]], label
    else:
        assert saved.shape == (len(lse) + GUARDS,), label
        assert saved[-GUARDS:].tolist() == [GUARD["float32"]] * GUARDS, label
        np.testing.assert_allclose(
            saved[:-GUARDS], lse, rtol=TOLERANCE, atol=TOLERANCE, err_msg=label
        )
        result["lseMaxAbsError"] = float(np.max(np.abs(saved[:-GUARDS] - lse)))
    output = output[:-GUARDS]
    result["maxAbsError"] = float(np.max(np.abs(output - reference)))
    if dtype == "float32":
        np.testing.assert_allclose(
            output, reference, rtol=TOLERANCE, atol=TOLERANCE, err_msg=label
        )
        return result
    bad = []
    for index, (got, want) in enumerate(zip(output.tolist(), reference.tolist())):
        bound = TOLERANCE * (1 + abs(want))
        low, high = _round(dtype, want - bound), _round(dtype, want + bound)
        if not low <= got <= high:
            bad.append((index, got, want, low, high))
    assert not bad, (label, len(bad), bad[:8])
    exact = [_round(dtype, want) for want in reference.tolist()]
    result["correctlyRoundedFraction"] = float(np.mean(output == np.asarray(exact)))
    return result


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


@pytest.fixture(scope="module")
def attention_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned attention native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    work = tmp_path_factory.mktemp("attention-original")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    assert not subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "mlx/backend/metal/kernels",
        ],
        text=True,
        timeout=30,
    ).strip()
    source = root / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    for name, digest in HEADER_HASHES.items():
        header = root / "mlx/backend/metal/kernels/steel/attn" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest
    original = None
    if target == "metal":
        run(["xcrun", "--sdk", "macosx", "metal", "--version"], work, "metal-version")
        original = compile_metal(source, root, work / "original", flags=ORIGINAL_FLAGS)
    else:
        tool = GATE_TOOLS[target][0]
        run([tool, "--version"], work, f"{tool}-version")
    return root, original, target


@pytest.fixture(
    scope="module",
    # Each worker translates the entries in its group once.
    params=[
        pytest.param(
            variant, id=variant.name, marks=pytest.mark.xdist_group(variant.name)
        )
        for variant in VARIANTS
    ],
)
def translated_attention(attention_source, tmp_path_factory, request):
    """One translation, package and binding table shared by an entry's cases."""
    root, _original, target = attention_source
    variant = request.param
    work = tmp_path_factory.mktemp(f"attention-translation-{variant.name}")
    with tempfile.TemporaryDirectory(prefix=".current-attention-", dir=root) as output:
        try:
            report = translate_project(
                _config(root, Path(output).name, target, variant),
                format_output=False,
            )
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert len(payload["artifacts"]) == 1
            assert payload["project"]["indexRangeAssertions"] == (
                list(_opengl_index_assertions(variant)) if target == "opengl" else []
            )
            descriptor, package = _prepare_native_package(report, work)
        finally:
            shutil.copytree(output, work / "translation", dirs_exist_ok=True)
    constants = descriptor["specializationConstants"]
    assert {item["name"]: item["id"] for item in constants} == CONSTANT_IDS
    assert all(
        item["frozen"] is True
        and item["value"] is variant.specializations[item["name"]]
        for item in constants
    )
    bindings = {
        item["provenance"]["sourceResource"]["parameter"]: item
        for item in descriptor["bindings"]
    }
    assert len(descriptor["bindings"]) == len(bindings) == len(BINDINGS)
    assert set(bindings) == set(BINDINGS)
    params = bindings["params"]["scalarLayout"]
    assert params["elementStrideBytes"] == 152
    assert len(params["structMembers"]) == 18
    assert bindings["mask_params"]["scalarLayout"]["elementStrideBytes"] == 24
    # Metal stores the element type, DirectX 16-bit words, OpenGL float32.
    physical = variant.dtype
    if target == "opengl" or (target == "directx" and physical != "float32"):
        physical = "float32" if target == "opengl" else "uint16"
    for name in ("Q", "K", "V", "O", "sinks"):
        assert bindings[name]["scalarLayout"]["elementType"] == physical, name
    assert bindings["lse"]["scalarLayout"]["elementType"] == "float32"
    return variant, descriptor, package, bindings


def _dispatch_original(executor, state, native, source, module, variant):
    specializations = variant.specializations
    constants = {
        name: NativeRuntimeConstantBinding(
            name=name,
            constant=RuntimeSpecializationConstant(
                name=name,
                constant_id=identifier,
                dtype="bool",
                value=specializations[name],
                kind="function-constant",
            ),
            value=specializations[name],
            source="original-source-specialization",
        )
        for name, identifier in CONSTANT_IDS.items()
    }
    control = replace(
        native,
        artifact_path=source,
        module_path=module,
        entry_point=variant.entry,
        loaded_artifact=None,
        constants=constants,
    )
    actual = executor.runtime_adapter.runtime.dispatch(None, state, control)
    assert state.details["metalRuntime"]["librarySHA256"] == (
        hashlib.sha256(module.read_bytes()).hexdigest()
    )
    return actual


@pytest.mark.parametrize("case_index", (0, 1), ids=("case0", "case1"))
def test_current_attention_executes(
    tmp_path, attention_source, translated_attention, case_index
):
    root, original, target = attention_source
    variant, descriptor, package, bindings = translated_attention
    case = variant.cases[case_index]
    inputs, outputs, geometry, reference, lse = _dispatch_values(
        bindings, case, variant, target
    )
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, geometry, expected_target=target
    )
    assert not request.execution_plan.diagnostics
    assert not request.adapter_contract.specialization_constants
    executor = _executor(target)
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        actual = executor.runtime_adapter.dispatch(state, native)
        result = {
            "case": case.name,
            "target": target,
            "checkedValues": len(reference),
            "generated": _check(
                "generated", actual, outputs, reference, lse, bindings, variant.dtype
            ),
            "moduleSha256": hashlib.sha256(native.module_path.read_bytes()).hexdigest(),
        }
        if target == "metal":
            original_outputs = _dispatch_original(
                executor, state, native, root / SOURCE, original, variant
            )
            result["originalMetal"] = _check(
                "original Metal",
                original_outputs,
                outputs,
                reference,
                lse,
                bindings,
                variant.dtype,
            )
        _write_json(
            tmp_path / "parity.json",
            {
                "sourceCommit": MLX_COMMIT,
                "entryPoint": variant.entry,
                "specializations": variant.specializations,
                "sourceSha256": SOURCE_SHA256,
                "headerSha256": HEADER_HASHES,
                "tolerance": {
                    "rtol": TOLERANCE,
                    "atol": TOLERANCE,
                    "narrowOutputs": "rounded interval end points",
                },
                "result": result,
                "scope": (
                    "Seventeen entry configurations: head dimensions 64 to 256 "
                    "except 192 and (96, 64); float32, float16 and bfloat16; "
                    "float, boolean and causal masks; aligned and unaligned "
                    "sequences; sinks and the logsumexp; the float32 wm4_wn2 "
                    "entries, with frozen constants. Not head dimension 192, "
                    "the float16 and bfloat16 wn2 entries or the upstream suite."
                ),
            },
        )
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
