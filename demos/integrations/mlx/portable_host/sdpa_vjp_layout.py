"""Name, check and bound MLX's fused attention VJP launches.

``mlx/backend/metal/scaled_dot_product_attention.cpp``'s
``ScaledDotProductAttentionVJP::eval_gpu`` runs the pinned
``scaled_dot_product_attention_vjp.metal`` kernels around steel matrix
products: ``sdpa_vjp_odo`` takes each query row's dot product of the forward
output and its cotangent, and ``sdpa_vjp_blocked`` runs, for each key and
query block, ``sdpa_vjp_ds``, which turns the block's scores and output
gradients into its probabilities and score gradients, and
``sdpa_vjp_reduce_add`` or ``sdpa_vjp_reduce_set``, which sum a block's
grouped-head gradients into float32 accumulators and the gradient outputs.
That file launches each with ``dispatch_threads``; every kernel returns for
the threads outside its grid, so the host launches the covering threadgroups
instead. ``plan_launch`` checks each launch against the kernels' addressing
and returns each bound buffer's exact element span, every one below a signed
32-bit offset.
"""

import math
import re
import struct

SOURCE = "mlx/backend/metal/kernels/scaled_dot_product_attention_vjp.metal"
ENTRY_PREFIXES = ("sdpa_vjp_",)
# get_type_string names and the MLX dtypes they carry.
TYPES = {"float": "float32", "float16_t": "float16", "bfloat16_t": "bfloat16"}
TYPE_NAMES = {dtype: name for name, dtype in TYPES.items()}
# instantiate_odo_shapes' value head dimensions.
ODO_DIMS = frozenset({64, 72, 80, 96, 128, 192, 256, 512})
SIMD_SIZE = 32
# eval_gpu's and sdpa_vjp_blocked's threadgroups. The reduction's is
# (min(dim, 32), 8, 1); every fused head dimension is at least 64.
WORKGROUPS = {"odo": (SIMD_SIZE, 1, 1), "ds": (32, 8, 1), "reduce": (32, 8, 1)}
MAX_WORKGROUPS = 65535
SIGNED_MAX = (1 << 31) - 1
# SDPAVJPTileParams in field order; the host passes each as an int64 and the
# scale as its binary32 bits.
TILE_FIELDS = ("bq", "bk", "qL", "i0", "j0", "scale", "diag_off", "causal")
REDUCE_SCALARS = ("rows", "dim", "group", "acc_rows", "row_off")
INOUT = 2

_TYPE = "|".join(TYPES)
ODO_NAME = re.compile(rf"sdpa_vjp_odo_({_TYPE})_(\d+)")
DS_NAME = re.compile(rf"sdpa_vjp_ds_({_TYPE})")
REDUCE_NAME = re.compile(rf"sdpa_vjp_reduce_(add|set)_({_TYPE})")


def _canonical(number):
    if number != str(int(number)):
        raise ValueError("SDPA VJP entry numbers must be canonical")
    return int(number)


def signature(entry):
    """The kernel kind, dtype and threadgroup of ``entry``."""
    if not isinstance(entry, str):
        raise ValueError("SDPA VJP entries are names")
    if match := ODO_NAME.fullmatch(entry):
        dv = _canonical(match.group(2))
        if dv not in ODO_DIMS:
            raise ValueError("No sdpa_vjp_odo instantiation for this head dim")
        info = {"kind": "odo", "dtype": TYPES[match.group(1)], "Dv": dv}
    elif match := DS_NAME.fullmatch(entry):
        info = {"kind": "ds", "dtype": TYPES[match.group(1)]}
    elif match := REDUCE_NAME.fullmatch(entry):
        info = {
            "kind": "reduce",
            "dtype": TYPES[match.group(2)],
            "accumulate": match.group(1) == "add",
        }
    else:
        raise ValueError(f"Not an SDPA VJP entry: {entry!r}")
    return {**info, "source": entry, "workgroupSize": WORKGROUPS[info["kind"]]}


def odo_entry(dtype, dv):
    return f"sdpa_vjp_odo_{TYPE_NAMES[dtype]}_{dv}"


def ds_entry(dtype):
    return f"sdpa_vjp_ds_{TYPE_NAMES[dtype]}"


def reduce_entry(dtype, accumulate):
    return f"sdpa_vjp_reduce_{'add' if accumulate else 'set'}_{TYPE_NAMES[dtype]}"


def source_entry(entry):
    return signature(entry)["source"]


def workgroup_size(entry):
    return tuple(signature(entry)["workgroupSize"])


def parameters(entry):
    """The buffers the host binds, in the kernel's buffer order."""
    kind = signature(entry)["kind"]
    if kind == "odo":
        return ("o", "cot_o", "odo", "qL")
    if kind == "ds":
        return ("S", "dP", "lse", "odo", "dS", "P", "p")
    return ("src", "acc", "out", *REDUCE_SCALARS)


def outputs(entry):
    kind = signature(entry)["kind"]
    return {"odo": ("odo",), "ds": ("dS", "P"), "reduce": ("acc", "out")}[kind]


def directions(entry):
    """Each parameter's buffer direction: the reduction's accumulator and
    gradient output keep the rows outside its window, so both start as the
    host's storage."""
    kind = signature(entry)["kind"]
    written = {"acc": INOUT, "out": INOUT} if kind == "reduce" else {}
    return {
        name: written.get(name, int(name in outputs(entry)))
        for name in parameters(entry)
    }


def scalars(entry):
    kind = signature(entry)["kind"]
    return {"odo": ("qL",), "ds": (), "reduce": REDUCE_SCALARS}[kind]


def records(entry):
    return ("p",) if signature(entry)["kind"] == "ds" else ()


def parameter_type(entry, name):
    """The MLX dtype the host binds ``name`` as."""
    info = signature(entry)
    if name not in parameters(entry):
        raise ValueError(f"SDPA VJP entry binds no {name!r}")
    if name in records(entry):
        return "int64"
    if name in scalars(entry):
        return "int32"
    if name in {"odo", "lse", "acc"}:
        return "float32"
    return info["dtype"]


def _require(condition, message="SDPA VJP launch differs from its kernel"):
    if not condition:
        raise ValueError(message)


def _int(values, name, minimum=0):
    value = values.get(name)
    _require(type(value) is int and minimum <= value <= SIGNED_MAX)
    return value


def decode_tile(words):
    """``SDPAVJPTileParams``' fields from the host's int64 words."""
    _require(
        len(words) == len(TILE_FIELDS) and all(type(word) is int for word in words)
    )
    fields = dict(zip(TILE_FIELDS, words))
    bits = fields["scale"]
    _require(0 <= bits < 1 << 32, "SDPA VJP scale is not binary32")
    fields["scale"] = struct.unpack("<f", struct.pack("<I", bits))[0]
    return fields


def _groups(grid, workgroup):
    return [-(-extent // width) for extent, width in zip(grid, workgroup)]


def _plan_odo(info, counts, values, words, execution):
    groups = execution["workgroupCount"]
    qL = _int(values, "qL", 1)
    heads = groups[2]
    _require(groups[:2] == [1, qL])
    rows = heads * qL
    spans = {"o": rows * info["Dv"], "cot_o": rows * info["Dv"], "odo": rows}
    return spans, {"heads": heads, "qL": qL}, [SIMD_SIZE, qL, heads]


def _plan_ds(info, counts, values, words, execution):
    tile = decode_tile(words["p"])
    bq, bk, qL, i0, j0 = (tile[name] for name in ("bq", "bk", "qL", "i0", "j0"))
    heads = execution["workgroupCount"][2]
    _require(
        all(1 <= value <= SIGNED_MAX for value in (bq, bk, qL))
        and 0 <= i0 <= qL - bq
        and 0 <= j0 <= SIGNED_MAX
        and -SIGNED_MAX <= tile["diag_off"] <= SIGNED_MAX
        and tile["causal"] in (0, 1)
        and math.isfinite(tile["scale"])
    )
    block = heads * bq * bk
    spans = {
        "S": block,
        "dP": block,
        "lse": heads * qL,
        "odo": heads * qL,
        "dS": block,
        "P": block,
    }
    metadata = {key: tile[key] for key in TILE_FIELDS}
    return spans, {"heads": heads, **metadata}, [bk, bq, heads]


def _plan_reduce(info, counts, values, words, execution):
    rows, dim, group, acc_rows, row_off = (
        _int(values, name, 1 if name != "row_off" else 0) for name in REDUCE_SCALARS
    )
    heads = execution["workgroupCount"][2]
    _require(dim >= SIMD_SIZE and row_off + rows <= acc_rows)
    total = heads * acc_rows * dim
    spans = {"src": heads * group * rows * dim, "acc": total, "out": total}
    metadata = {
        "heads": heads,
        "rows": rows,
        "dim": dim,
        "group": group,
        "accRows": acc_rows,
        "rowOffset": row_off,
        "accumulate": info["accumulate"],
    }
    return spans, metadata, [dim, rows, heads]


def plan_launch(entry, counts, values, words, threads, execution):
    """Check one launch and return its metadata and exact buffer spans.

    ``counts`` holds each array parameter's bound element count, ``values``
    each scalar parameter's value and ``words`` the int64 words of each
    record.
    """
    info = signature(entry)
    _require(set(values) == set(scalars(entry)) and set(words) == set(records(entry)))
    workgroup = list(info["workgroupSize"])
    _require(
        execution.get("workgroupSize") == workgroup
        and len(execution.get("workgroupCount", ())) == 3
        and all(
            type(value) is int and 1 <= value <= MAX_WORKGROUPS
            for value in execution["workgroupCount"]
        )
        and not any(execution.get("threadGridSize", ()))
    )
    planner = {"odo": _plan_odo, "ds": _plan_ds, "reduce": _plan_reduce}[info["kind"]]
    spans, metadata, grid = planner(info, counts, values, words, execution)
    _require(
        execution["workgroupCount"] == _groups(grid, workgroup),
        "SDPA VJP threadgroups do not cover its grid exactly",
    )
    _require(
        counts == spans
        and max(spans.values()) <= SIGNED_MAX
        and threads == spans[outputs(entry)[-1]],
        "SDPA VJP buffers differ from the kernel's addressing",
    )
    return {"kind": info["kind"], **metadata, "grid": grid, "elements": spans}


def opengl_index_assertions(entry):
    """The device offsets OpenGL needs proven, from ``plan_launch``'s spans.

    The row dot product reads row ``bh * qL + i`` of ``heads * qL`` rows of
    ``Dv`` elements; the block kernel indexes rows ``qi`` below
    ``heads * qL`` and elements ``sbase + col`` below ``heads * bq * bk``;
    and the reduction's ``si`` and ``ai`` stay below ``heads * group * rows *
    dim`` and ``heads * acc_rows * dim``. Every span is below 2^31 - 1.
    """
    info = signature(entry)
    expressions = {
        "odo": (f"row * {info.get('Dv')}", "o_i + d", "co_i + d", "row"),
        "ds": ("qi", "sbase + col"),
        "reduce": ("si", "ai"),
    }[info["kind"]]
    return [
        {
            "source": SOURCE,
            "expression": expression,
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for expression in expressions
    ]
