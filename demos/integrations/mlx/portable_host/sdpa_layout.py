"""Name, check and bound MLX's vector scaled-dot-product-attention launches.

``mlx/backend/metal/scaled_dot_product_attention.cpp`` runs queries of at
most eight positions through the pinned ``scaled_dot_product_attention.metal``
vector kernels: ``sdpa_vector`` in one pass, or ``sdpa_vector_2pass_1``
(``sdpa_vector_2pass_1_gqa_<G>`` for long grouped single-token queries)
writing per-block partials that ``sdpa_vector_2pass_2`` combines. The host
names each pipeline by sdpa_vector's and sdpa_vector_2pass's hash names,
which fix the mask, transposed-query, causal and sink function constants
(and, for the first pass, ``blocks``). The first pass's threadgroups are
``(32, gqa_factor, query_length)``, so its host entry appends them as
``@32x<gqa>x<length>``. ``plan_launch`` checks each launch against the
kernels' addressing and returns the exact element span each bound buffer
may be read or written at, every one inside a signed 32-bit offset.
"""

import re

SOURCE = "mlx/backend/metal/kernels/scaled_dot_product_attention.metal"
ENTRY_PREFIXES = ("sdpa_vector_",)
# get_type_string names and the MLX dtypes they carry.
TYPES = {"float": "float32", "float16_t": "float16", "bfloat16_t": "bfloat16"}
TYPE_NAMES = {dtype: name for name, dtype in TYPES.items()}
# instantiate_sdpa_vector_heads' (query, value) head dimensions.
VECTOR_DIMS = frozenset(
    {(64, 64), (96, 64), (96, 96), (128, 128), (192, 128), (192, 192), (256, 256)}
    | {(512, 512)}
)
# instantiate_sdpa_vector_gqa: (gqa factor, head dim) -> heads per thread.
GQA_HEADS = {(8, 64): 8, (8, 128): 4, (12, 128): 4, (16, 128): 2}
# instantiate_sdpa_vector_aggregation's value head dimensions.
AGGREGATION_DIMS = frozenset({64, 96, 128, 192, 256, 512})
MASKS = {
    "nomask": {"has_mask": False, "bool_mask": False, "float_mask": False},
    "boolmask": {"has_mask": True, "bool_mask": True, "float_mask": False},
    "floatmask": {"has_mask": True, "bool_mask": False, "float_mask": True},
}
SIMD_SIZE = 32
# sdpa_vector and sdpa_vector_2pass_2: 32 simdgroups of 32 threads.
WIDE_WORKGROUP = (1024, 1, 1)
# The first pass's reduction reads the partials in simd-width chunks.
BLOCK_ALIGNMENT = 32
MAX_QUERY_LENGTH = 8
# The key count the first pass's translation may assume (parameter_ranges).
MAX_KEYS = 1 << 30
MAX_WORKGROUPS = 65535
SIGNED_MAX = (1 << 31) - 1
SEPARATOR = "@"

_TYPE = "|".join(TYPES)
_FLAGS = r"_(nomask|boolmask|floatmask)_(qt|qnt)_(c|nc)_(sinks|nosinks)"
VECTOR_NAME = re.compile(rf"sdpa_vector_({_TYPE})_(\d+)_(\d+){_FLAGS}")
PASS1_NAME = re.compile(
    rf"sdpa_vector_2pass_1(?:_gqa_(\d+))?_({_TYPE})_(\d+)_(\d+){_FLAGS}_(\d+)"
    rf"{SEPARATOR}32x(\d+)x(\d+)"
)
PASS2_NAME = re.compile(rf"sdpa_vector_2pass_2_({_TYPE})_(\d+)")

MASK_STRIDES = ("mask_kv_seq_stride", "mask_q_seq_stride", "mask_head_stride")
KV_STRIDES = ("k_head_stride", "k_seq_stride", "v_head_stride", "v_seq_stride")


def _canonical(number):
    if number != str(int(number)):
        raise ValueError("SDPA entry numbers must be canonical")
    return int(number)


def _flags(mask, transposed, causal, sinks):
    return {
        **MASKS[mask],
        "query_transposed": transposed == "qt",
        "do_causal": causal == "c",
        "has_sinks": sinks == "sinks",
    }


def _suffix(constants):
    mask = next(
        name
        for name, flags in MASKS.items()
        if all(constants[key] == value for key, value in flags.items())
    )
    return (
        f"_{mask}_{'qt' if constants['query_transposed'] else 'qnt'}"
        f"_{'c' if constants['do_causal'] else 'nc'}"
        f"_{'sinks' if constants['has_sinks'] else 'nosinks'}"
    )


def _vector_signature(match):
    type_name, d, v = match.group(1), *map(_canonical, match.group(2, 3))
    if (d, v) not in VECTOR_DIMS:
        raise ValueError("No sdpa_vector instantiation for these head dimensions")
    constants = _flags(*match.group(4, 5, 6, 7))
    return {
        "kind": "vector",
        "source": f"sdpa_vector_{type_name}_{d}_{v}",
        "dtype": TYPES[type_name],
        "D": d,
        "V": v,
        "constants": constants,
        "workgroupSize": WIDE_WORKGROUP,
    }


def _pass1_signature(match):
    gqa_kernel = match.group(1)
    type_name = match.group(2)
    d, v = map(_canonical, match.group(3, 4))
    constants = _flags(*match.group(5, 6, 7, 8))
    blocks, gqa, length = map(_canonical, match.group(9, 10, 11))
    if (d, v) not in VECTOR_DIMS:
        raise ValueError(
            "No sdpa_vector_2pass_1 instantiation for these head dimensions"
        )
    if (
        blocks < BLOCK_ALIGNMENT
        or blocks % BLOCK_ALIGNMENT
        or blocks > MAX_WORKGROUPS
        or not 1 <= length <= MAX_QUERY_LENGTH
        or gqa < 1
        or gqa * length > SIMD_SIZE
        or (constants["do_causal"] and length < 2)
    ):
        raise ValueError("SDPA first pass blocks or threadgroup differ from its kernel")
    info = {
        "kind": "pass1",
        "source": f"sdpa_vector_2pass_1_{type_name}_{d}_{v}",
        "dtype": TYPES[type_name],
        "D": d,
        "V": v,
        "constants": {**constants, "blocks": blocks},
        "blocks": blocks,
        "gqa": gqa,
        "length": length,
        "workgroupSize": (SIMD_SIZE, gqa, length),
    }
    if gqa_kernel is not None:
        factor = _canonical(gqa_kernel)
        if (
            (factor, d) not in GQA_HEADS
            or d != v
            or factor != gqa
            or length != 1
            or constants["has_mask"]
            or constants["has_sinks"]
            or constants["do_causal"]
        ):
            raise ValueError("No sdpa_vector_2pass_1_gqa kernel for this launch")
        info.update(
            kind="pass1_gqa",
            source=f"sdpa_vector_2pass_1_gqa_{factor}_{type_name}_{d}_{v}",
            heads=GQA_HEADS[(factor, d)],
        )
    return info


def signature(entry):
    """The kernel, dtype, head dimensions, constants and threadgroup of ``entry``."""
    if not isinstance(entry, str):
        raise ValueError("SDPA entries are names")
    match = PASS1_NAME.fullmatch(entry)
    if match:
        return _pass1_signature(match)
    match = PASS2_NAME.fullmatch(entry)
    if match:
        v = _canonical(match.group(2))
        if v not in AGGREGATION_DIMS:
            raise ValueError("No sdpa_vector_2pass_2 instantiation for this head dim")
        return {
            "kind": "pass2",
            "source": entry,
            "dtype": TYPES[match.group(1)],
            "D": v,
            "V": v,
            "constants": {},
            "workgroupSize": WIDE_WORKGROUP,
        }
    match = VECTOR_NAME.fullmatch(entry)
    if match:
        return _vector_signature(match)
    raise ValueError(f"Not an SDPA vector entry: {entry!r}")


def vector_entry(dtype, d, v, constants):
    """sdpa_vector's hash name."""
    return f"sdpa_vector_{TYPE_NAMES[dtype]}_{d}_{v}{_suffix(constants)}"


def pass1_entry(dtype, d, v, constants, blocks, gqa, length, gqa_kernel=False):
    """sdpa_vector_2pass's first hash name and the host's threadgroup suffix."""
    kernel = f"_gqa_{gqa}" if gqa_kernel else ""
    return (
        f"sdpa_vector_2pass_1{kernel}_{TYPE_NAMES[dtype]}_{d}_{v}"
        f"{_suffix(constants)}_{blocks}{SEPARATOR}32x{gqa}x{length}"
    )


def pass2_entry(dtype, v):
    """sdpa_vector_2pass's second kernel name."""
    return f"sdpa_vector_2pass_2_{TYPE_NAMES[dtype]}_{v}"


def source_entry(entry):
    return signature(entry)["source"]


def workgroup_size(entry):
    return tuple(signature(entry)["workgroupSize"])


def constants(entry):
    return dict(signature(entry)["constants"])


def parameters(entry):
    """The buffers the host binds, in the kernel's buffer order."""
    info = signature(entry)
    if info["kind"] == "pass2":
        return ("partials", "sums", "maxs", "out", "blocks")
    flags = info["constants"]
    head = ("queries", "keys", "values", "out")
    if info["kind"] == "vector":
        names = [*head, "gqa_factor", "N", *KV_STRIDES, "scale"]
    else:
        names = [*head, "sums", "maxs", "N", *KV_STRIDES, "scale"]
    if flags["bool_mask"]:
        names.append("bmask")
    if flags["float_mask"]:
        names.append("fmask")
    if flags["has_mask"]:
        names.extend(MASK_STRIDES)
    if flags["has_sinks"]:
        names.append("sinks")
        if info["kind"] == "vector":
            names.append("num_q_heads")
    return tuple(names)


def outputs(entry):
    if signature(entry)["kind"] == "pass2":
        return ("out",)
    if signature(entry)["kind"] == "vector":
        return ("out",)
    return ("out", "sums", "maxs")


def scalars(entry):
    return tuple(
        name
        for name in parameters(entry)
        if name in {"gqa_factor", "N", "scale", "num_q_heads", "blocks"}
        or name in KV_STRIDES
        or name in MASK_STRIDES
    )


def parameter_type(entry, name):
    """The MLX dtype the host binds ``name`` as."""
    info = signature(entry)
    if name not in parameters(entry):
        raise ValueError(f"SDPA entry binds no {name!r}")
    if name in KV_STRIDES:
        return "uint64"
    if name == "scale":
        return "float32"
    if name in {"sums", "maxs"}:
        return "float32"
    if name == "bmask":
        return "bool_"
    if name in scalars(entry):
        return "int32"
    return info["dtype"]


def _require(condition, message="SDPA launch differs from its kernel's dispatch"):
    if not condition:
        raise ValueError(message)


def _int(values, name, minimum=0):
    value = values.get(name)
    _require(type(value) is int and minimum <= value <= SIGNED_MAX)
    return value


def _mask_span(values, heads, keys, length):
    strides = [_int(values, name) for name in MASK_STRIDES]
    kv, q, head = strides
    return (heads - 1) * head + (keys - 1) * kv + (length - 1) * q + 1


def _kv_spans(info, values, kv_heads, keys, step):
    """The key and value spans; each pointer also stays representable one
    ``step`` of keys past its last read, where the kernels leave it."""
    k_head, k_seq, v_head, v_seq = (_int(values, name) for name in KV_STRIDES)
    spans = {
        "keys": (kv_heads - 1) * k_head + (keys - 1) * k_seq + info["D"],
        "values": (kv_heads - 1) * v_head + (keys - 1) * v_seq + info["V"],
    }
    _require(
        keys <= MAX_KEYS
        and spans["keys"] + step * k_seq <= SIGNED_MAX
        and spans["values"] + step * v_seq <= SIGNED_MAX
    )
    return spans


def _plan_vector(info, values, execution):
    groups = execution["workgroupCount"]
    heads, length = groups[0], groups[1]
    gqa = _int(values, "gqa_factor", 1)
    keys = _int(values, "N", 1)
    _require(
        groups[2] == 1
        and 1 <= length <= MAX_QUERY_LENGTH
        and heads % gqa == 0
        and keys >= length
        and gqa * length <= SIMD_SIZE
        and (length > 1 or not info["constants"]["do_causal"])
    )
    spans = {
        "queries": heads * length * info["D"],
        "out": heads * length * info["V"],
        **_kv_spans(info, values, heads // gqa, keys, SIMD_SIZE),
    }
    flags = info["constants"]
    if flags["has_mask"]:
        spans["bmask" if flags["bool_mask"] else "fmask"] = _mask_span(
            values, heads, keys, length
        )
    if flags["has_sinks"]:
        query_heads = _int(values, "num_q_heads", 1)
        _require(heads % query_heads == 0 and query_heads % gqa == 0)
        spans["sinks"] = query_heads
    return spans, {"heads": heads, "length": length, "keys": keys, "gqa": gqa}


def _plan_pass1(info, values, execution):
    kv_heads, batch, blocks = execution["workgroupCount"]
    gqa, length = info["gqa"], info["length"]
    keys = _int(values, "N", 1)
    _require(blocks == info["blocks"] and keys >= length)
    heads = batch * kv_heads * gqa
    partial = heads * length * blocks
    spans = {
        "queries": heads * length * info["D"],
        "out": partial * info["V"],
        "sums": partial,
        "maxs": partial,
        **_kv_spans(
            info,
            values,
            batch * kv_heads,
            keys,
            1 if info["kind"] == "pass1_gqa" else blocks,
        ),
    }
    flags = info["constants"]
    if flags["has_mask"]:
        spans["bmask" if flags["bool_mask"] else "fmask"] = _mask_span(
            values, heads, keys, length
        )
    if flags["has_sinks"]:
        spans["sinks"] = kv_heads * gqa
    return spans, {
        "heads": heads,
        "length": length,
        "keys": keys,
        "gqa": gqa,
        "blocks": blocks,
        "batch": batch,
    }


def _plan_pass2(info, values, execution):
    heads, length, depth = execution["workgroupCount"]
    blocks = _int(values, "blocks", BLOCK_ALIGNMENT)
    _require(
        depth == 1
        and 1 <= length <= MAX_QUERY_LENGTH
        and blocks % BLOCK_ALIGNMENT == 0
        and blocks <= MAX_WORKGROUPS
    )
    partial = heads * length * blocks
    spans = {
        "partials": partial * info["V"],
        "sums": partial,
        "maxs": partial,
        "out": heads * length * info["V"],
    }
    return spans, {"heads": heads, "length": length, "blocks": blocks}


def plan_launch(entry, counts, values, threads, execution):
    """Check one launch and return its metadata and exact buffer spans.

    ``counts`` holds each array parameter's bound element count and
    ``values`` each scalar parameter's value.
    """
    info = signature(entry)
    _require(set(values) == set(scalars(entry)))
    _require(
        execution.get("workgroupSize") == list(info["workgroupSize"])
        and len(execution.get("workgroupCount", ())) == 3
        and all(
            type(value) is int and 1 <= value <= MAX_WORKGROUPS
            for value in execution["workgroupCount"]
        )
        and not any(execution.get("threadGridSize", ()))
    )
    _require(
        info["kind"] == "pass2" or isinstance(values.get("scale"), float),
        "SDPA scale must be a float",
    )
    planner = {
        "vector": _plan_vector,
        "pass1": _plan_pass1,
        "pass1_gqa": _plan_pass1,
        "pass2": _plan_pass2,
    }[info["kind"]]
    spans, metadata = planner(info, values, execution)
    _require(
        counts == spans
        and max(spans.values()) <= SIGNED_MAX
        and threads == spans["out"],
        "SDPA buffers differ from the kernel's addressing",
    )
    return {"kind": info["kind"], **metadata, "elements": spans}


def parameter_ranges(entry):
    """Scalar ranges OpenGL's translation may assume and ``plan_launch`` enforces.

    The first pass steps its keys by ``blocks``; bounding ``N`` proves that
    loop terminates alike in every invocation of a software subgroup.
    """
    if signature(entry)["kind"] == "pass1":
        return {"N": (0, MAX_KEYS)}
    return {}


def opengl_index_assertions(entry):
    """The device offsets OpenGL needs proven, from ``plan_launch``'s bounds.

    Every key and value offset the kernels form, including the pointer after
    its last step, stays inside ``plan_launch``'s span plus one step, below
    2^31 - 1.
    """
    info = signature(entry)
    if info["kind"] == "pass2":
        return []
    if info["kind"] == "vector":
        head, position, pointers = "kv_head_idx", "simd_gid", ("keys + j", "values + j")
    elif info["kind"] == "pass1":
        head, position, pointers = (
            "kv_batch_head_idx",
            "block_idx",
            ("keys + i", "values + i"),
        )
    else:
        head, position = "kv_batch_head_idx", "s0"
        pointers = (
            "kp + i",
            "vp + i",
            "k_seq_stride",
            "v_seq_stride",
            "kp + k_seq_stride",
            "vp + v_seq_stride",
        )
    expressions = [
        f"{head} * k_head_stride + {position} * k_seq_stride + simd_lid * qk_per_thread",
        f"{head} * v_head_stride + {position} * v_seq_stride + simd_lid * v_per_thread",
        *pointers,
    ]
    return [
        {
            "source": SOURCE,
            "expression": expression,
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for expression in expressions
    ]
