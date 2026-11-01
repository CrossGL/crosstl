"""Name, check and bound MLX's steel full-attention launches.

``mlx/backend/metal/scaled_dot_product_attention.cpp`` runs queries longer
than eight positions through ``sdpa_full_self_attention_metal``. The portable
host reports no Apple GPU architecture and no NAX unit, so that function never
takes the NAX kernels or pads head dims 72 and 80, and dispatches the pinned
``steel_attention.metal`` ``attention`` kernel with a 32-row query block,
32-row key blocks below head dim 128 and 16-row key blocks from it, and four
simdgroups along the queries, split in two along the head dim at head dim
256. The host names each pipeline by that function's hash name, which fixes
the alignment, mask, causal, sink and logsumexp function constants; the
translated entry has them frozen. ``plan_launch`` checks each launch's
``AttnParams`` and grid against the kernel's addressing and returns the exact
element span of each bound buffer, every one below a signed 32-bit offset.
"""

import math
import re
import struct

SOURCE = "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal"
ENTRY_PREFIXES = ("steel_attention_",)
TYPES = ("float32", "float16", "bfloat16")
ITEM_SIZES = {"float32": 4, "float16": 2, "bfloat16": 2}
SIMD_SIZE = 32
BQ = 32
WM = 4
FRAGMENT = 8
# instantiate_attn_shapes_helper's (bk, bd, bv) with WM = 4 and WN = 1 for
# every type, and the head dim 256 instantiations with WN = 2.
SHAPES = frozenset(
    (dtype, bk, bd, bv, 1)
    for dtype in TYPES
    for bk, bd, bv in (
        (16, 192, 192),
        (16, 128, 128),
        (32, 96, 96),
        (32, 96, 64),
        (32, 80, 80),
        (32, 72, 72),
        (32, 64, 64),
    )
) | frozenset(
    (dtype, bk, 256, 256, 2)
    for dtype, bk in (
        ("float16", 16),
        ("float16", 32),
        ("bfloat16", 16),
        ("bfloat16", 32),
        ("float32", 16),
    )
)
FLAGS = ("align_Q", "align_K", "has_mask", "do_causal", "has_sinks", "save_lse")
MAX_WORKGROUPS = 65535
SIGNED_MAX = (1 << 31) - 1
# AttnParams in params.h order; the host passes each field as an int64 and the
# scale as its binary32 bits.
PARAMS_FIELDS = (
    "B",
    "H",
    "D",
    "qL",
    "kL",
    "gqa_factor",
    "scale",
    "NQ",
    "NK",
    "NQ_aligned",
    "NK_aligned",
    "qL_rem",
    "kL_rem",
    "qL_off",
)
STRIDE_FIELDS = ("Q_strides", "K_strides", "V_strides", "O_strides")
PARAMS_COUNT = len(PARAMS_FIELDS) + 3 * len(STRIDE_FIELDS)
MASK_PARAMS_COUNT = 3
RECORDS = ("params", "mask_params")

_TYPE = "|".join(TYPES)
NAME = re.compile(
    rf"steel_attention_({_TYPE})_bq32_bk(\d+)_bd(\d+)_bv(\d+)_wm4_wn(\d+)"
    rf"_mask({_TYPE}|bool_)" + "".join(rf"_{flag}_([tn])" for flag in FLAGS)
)


def _canonical(number):
    if number != str(int(number)):
        raise ValueError("Steel attention entry numbers must be canonical")
    return int(number)


def signature(entry):
    """The kernel, dtype, tile, mask type, constants and threadgroup of ``entry``."""
    if not isinstance(entry, str):
        raise ValueError("Steel attention entries are names")
    match = NAME.fullmatch(entry)
    if not match:
        raise ValueError(f"Not a steel attention entry: {entry!r}")
    dtype = match.group(1)
    bk, bd, bv, wn = map(_canonical, match.group(2, 3, 4, 5))
    mask_type = match.group(6)
    constants = {
        flag: value == "t" for flag, value in zip(FLAGS, match.group(*range(7, 13)))
    }
    if (dtype, bk, bd, bv, wn) not in SHAPES:
        raise ValueError("No steel attention instantiation for this tile")
    if mask_type not in (dtype, "bool_") or (
        not constants["has_mask"] and mask_type != dtype
    ):
        raise ValueError("Steel attention mask type differs from its hash name")
    source = entry[: match.end(6)]
    return {
        "source": source,
        "dtype": dtype,
        "BK": bk,
        "BD": bd,
        "BV": bv,
        "WN": wn,
        "maskType": mask_type,
        "constants": constants,
        "workgroupSize": (SIMD_SIZE, WM, wn),
    }


def tile(bd):
    """sdpa_full_self_attention_metal's key block and WN on the host device."""
    return (32 if bd < 128 else 16), (2 if bd == 256 else 1)


def entry_name(dtype, bd, bv, mask_type, constants):
    """sdpa_full_self_attention_metal's hash name on the host device."""
    bk, wn = tile(bd)
    name = (
        f"steel_attention_{dtype}_bq{BQ}_bk{bk}_bd{bd}_bv{bv}_wm{WM}_wn{wn}"
        f"_mask{mask_type if constants['has_mask'] else dtype}"
    )
    return name + "".join(
        f"_{flag}_{'t' if constants[flag] else 'n'}" for flag in FLAGS
    )


def source_entry(entry):
    return signature(entry)["source"]


def workgroup_size(entry):
    return tuple(signature(entry)["workgroupSize"])


def constants(entry):
    return dict(signature(entry)["constants"])


def parameters(entry):
    """The buffers the host binds, in the kernel's buffer order."""
    flags = signature(entry)["constants"]
    names = ["Q", "K", "V", "O", "params"]
    if flags["has_mask"]:
        names += ["mask_params", "mask"]
    if flags["has_sinks"]:
        names.append("sinks")
    if flags["save_lse"]:
        names.append("lse")
    return tuple(names)


def outputs(entry):
    return ("O", "lse") if signature(entry)["constants"]["save_lse"] else ("O",)


def records(entry):
    return tuple(name for name in parameters(entry) if name in RECORDS)


def parameter_type(entry, name):
    """The MLX dtype the host binds ``name`` as."""
    info = signature(entry)
    if name not in parameters(entry):
        raise ValueError(f"Steel attention entry binds no {name!r}")
    if name in RECORDS:
        return "int64"
    if name == "lse":
        return "float32"
    if name == "mask":
        return info["maskType"]
    return info["dtype"]


def threadgroup_extents(entry):
    """The elements of the kernel's ``Q_smem`` and ``KV_smem`` arrays.

    steel_attention.h pads each threadgroup row by 16 bytes and keeps the
    queries in registers when WN is 2 and the key block is 32 rows or the
    type is four bytes wide.
    """
    info = signature(entry)
    bk, bd, bv, wn = info["BK"], info["BD"], info["BV"], info["WN"]
    pad = 16 // ITEM_SIZES[info["dtype"]]
    kv = max((bk + pad) * bd, bk * (bv + pad))
    q_in_registers = wn == 2 and (bk == 32 or ITEM_SIZES[info["dtype"]] == 4)
    return {"Q_smem": 1 if q_in_registers else BQ * (bd + pad), "KV_smem": kv}


def opengl_index_assertions(entry):
    """The offsets OpenGL needs proven.

    Every device access stays inside ``plan_launch``'s spans, below 2^31 - 1,
    and every threadgroup access inside its array.
    """
    workgroup = [
        {
            "source": SOURCE,
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{name}",
            "minimum": 0,
            "maximum": extent - 1,
        }
        for name, extent in sorted(threadgroup_extents(entry).items())
    ]
    device = {
        "source": SOURCE,
        "expression": "reference.offset + index",
        "minimum": 0,
        "maximum": SIGNED_MAX,
    }
    return [*workgroup, device]


def _require(condition, message="Steel attention launch differs from its kernel"):
    if not condition:
        raise ValueError(message)


def decode_params(values):
    """``AttnParams``' fields from the host's int64 words."""
    _require(
        len(values) == PARAMS_COUNT and all(type(value) is int for value in values)
    )
    fields = dict(zip(PARAMS_FIELDS, values))
    bits = fields["scale"]
    _require(0 <= bits < 1 << 32, "Steel attention scale is not binary32")
    fields["scale"] = struct.unpack("<f", struct.pack("<I", bits))[0]
    offset = len(PARAMS_FIELDS)
    for index, name in enumerate(STRIDE_FIELDS):
        fields[name] = list(values[offset + 3 * index : offset + 3 * index + 3])
    return fields


def _span(strides, extents, row):
    return sum((extent - 1) * stride for stride, extent in zip(strides, extents)) + row


def _check_params(info, params):
    bq, bk = BQ, info["BK"]
    flags = info["constants"]
    B, H, D, qL, kL, gqa = (
        params[name] for name in ("B", "H", "D", "qL", "kL", "gqa_factor")
    )
    _require(
        all(1 <= value <= SIGNED_MAX for value in (B, H, qL, kL, gqa))
        and D == info["BD"]
        and H % gqa == 0
        and math.isfinite(params["scale"])
    )
    _require(
        params["NQ"] == -(-qL // bq)
        and params["NK"] == -(-kL // bk)
        and params["NQ_aligned"] == qL // bq
        and params["NK_aligned"] == kL // bk
        and params["qL_rem"] == qL - (qL // bq) * bq
        and params["kL_rem"] == kL - (kL // bk) * bk
        and params["qL_off"] == kL - qL
        and flags["align_Q"] == (qL % bq == 0)
        and flags["align_K"] == (kL % bk == 0)
        and (not flags["do_causal"] or qL <= kL)
    )
    for name in STRIDE_FIELDS:
        _require(all(0 <= stride <= SIGNED_MAX for stride in params[name]))
    # sdpa_full_self_attention_metal's output: (B, L, H, V) storage viewed as
    # (B, H, L, V).
    bv = info["BV"]
    _require(params["O_strides"] == [qL * H * bv, bv, H * bv])


def plan_launch(entry, counts, words, threads, execution):
    """Check one launch and return its metadata and exact buffer spans.

    ``counts`` holds each array parameter's bound element count and ``words``
    the int64 words of ``params`` and, with a mask, of ``mask_params``.
    """
    info = signature(entry)
    flags = info["constants"]
    _require(set(words) == set(records(entry)))
    params = decode_params(words["params"])
    _check_params(info, params)
    B, H, qL, kL = params["B"], params["H"], params["qL"], params["kL"]
    kv_heads = H // params["gqa_factor"]
    bd, bv, bk = info["BD"], info["BV"], info["BK"]
    _require(
        execution.get("workgroupSize") == list(info["workgroupSize"])
        and execution.get("workgroupCount") == [params["NQ"], H, B]
        and max(params["NQ"], H, B) <= MAX_WORKGROUPS
        and not any(execution.get("threadGridSize", ()))
    )
    spans = {
        "Q": _span(params["Q_strides"], (B, H, qL), bd),
        "K": _span(params["K_strides"], (B, kv_heads, kL), bd),
        "V": _span(params["V_strides"], (B, kv_heads, kL), bv),
        "O": B * H * qL * bv,
    }
    # The loaders leave the key and value pointers one block past their last
    # read; those offsets stay representable as well.
    _require(
        spans["K"] + bk * params["K_strides"][2] <= SIGNED_MAX
        and spans["V"] + bk * params["V_strides"][2] <= SIGNED_MAX
    )
    if flags["has_mask"]:
        mask_strides = words["mask_params"]
        _require(
            len(mask_strides) == MASK_PARAMS_COUNT
            and all(
                type(value) is int and 0 <= value <= SIGNED_MAX
                for value in mask_strides
            )
        )
        spans["mask"] = _span(mask_strides, (B, H, qL), kL)
    if flags["has_sinks"]:
        spans["sinks"] = H
    if flags["save_lse"]:
        spans["lse"] = B * H * qL
    _require(
        counts == spans and max(spans.values()) <= SIGNED_MAX and threads == spans["O"],
        "Steel attention buffers differ from the kernel's addressing",
    )
    metadata = {
        key: params[key] for key in ("B", "H", "qL", "kL", "gqa_factor", "NQ", "NK")
    }
    return {**metadata, "elements": spans}
