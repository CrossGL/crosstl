"""Name, check and bound MLX's gated-delta recurrence launches.

``gated_delta.inc`` ports ``mlx/backend/metal/gated_delta_update.cpp``
without NAX, so it runs three pinned kernels of
``gated_delta_update.metal`` and ``gated_delta_update_vjp.metal``:

- ``seq_gated_delta_<type>_128_128_<Hk>_<Hv>_<Ckpt>``: the sequential
  recurrence, with ``save_state`` (function constant 200) off for inference.
  The backward's save pass sets it, and its pipeline's hash name appends
  ``_save``; it then also stores the state entering every ``Ckpt``-th step.
- ``gated_delta_fused_chunk_<type>_128_128_<Hk>_<Hv>_8``: the eight-step
  chunked inference recurrence;
- ``seq_gated_delta_vjp_<type>_128_128_<Hk>_<Hv>_<Ckpt>``: the sequential
  backward, which adds its query, key, decay and beta gradients atomically.

Every launch is ``(32, 4, 1)`` threadgroups over a ``(32, Dv, B * Hv)`` grid,
or ``(32, Dv / 8, B * Hv)`` for the fused chunk, so the batch is the grid's
depth over ``Hv``. ``plan_launch`` derives each buffer's exact element span
from the batch and ``T`` and requires every one inside a signed 32-bit
offset.
"""

import re

SOURCES = {
    "forward": "mlx/backend/metal/kernels/gated_delta_update.metal",
    "vjp": "mlx/backend/metal/kernels/gated_delta_update_vjp.metal",
}
ENTRY_PREFIXES = ("seq_gated_delta_", "gated_delta_fused_chunk_")
# get_type_string names and the MLX dtypes they carry.
TYPES = {"float": "float32", "bfloat16_t": "bfloat16", "float16_t": "float16"}
TYPE_NAMES = {dtype: name for name, dtype in TYPES.items()}
# supported_gated_delta_shape's (Hk, Hv) pairs, all with Dk = Dv = 128.
HEADS = frozenset({(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)})
DK = DV = 128
# Every checkpoint interval gated_delta_ckpt() can name, and the fused chunk.
CHECKPOINTS = frozenset({1, 4, 8, 16})
CHUNK = 8
SAVE_STATE_ID = 200
SAVE_SUFFIX = "_save"
SIMD_SIZE = 32
WORKGROUP_SIZE = (32, 4, 1)
MAX_WORKGROUPS = 65535
SIGNED_MAX = (1 << 31) - 1
# ``for (; t + C <= T; t += C)`` must not overflow ``t + C``, so the software
# subgroup translations of the fused chunk assume this bound on ``T``.
T_MAXIMUM = SIGNED_MAX - CHUNK
# The fused chunk kernel holds its tiles in simdgroup_float8x8 values whose
# fragments MLX addresses through its own (fm, fn) lane coordinates.
FRAGMENT_MAPPING = "tile_4x4_row_pair"
FRAGMENT_MAPPING_PROVENANCE = "mlx_gated_delta_lane_coordinates"

_TYPE = "|".join(TYPES)
SEQUENTIAL = re.compile(
    rf"seq_gated_delta_(vjp_)?({_TYPE})_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)({SAVE_SUFFIX})?"
)
FUSED = re.compile(rf"gated_delta_fused_chunk_({_TYPE})_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)")

FORWARD = ("q", "k", "v", "state_in", "g", "beta", "y", "state_out", "T")
BACKWARD = (
    *("q", "k", "v", "g", "b", "cot_o", "cot_h", "state_cache", "T"),
    *("dq", "dk", "dv", "dg", "db", "dh"),
)
# Parameters bound as the entry's storage type; every other one is float32,
# and T is int32.
TYPED = frozenset({"q", "k", "v", "g", "beta", "b", "y", "cot_o"})
# The backward's gradients gated_delta_update_vjp.h declares
# ``device mlx_atomic<float>*``: one-member structs over float32 words.
ATOMIC = ("dq", "dk", "dg", "db")


def _canonical(text):
    if text != str(int(text)):
        raise ValueError("Gated-delta entries spell their numbers canonically")
    return int(text)


def _require_shape(dk, dv, hk, hv):
    if (dk, dv) != (DK, DV) or (hk, hv) not in HEADS:
        raise ValueError("No gated-delta instantiation for these heads")


def signature(entry):
    """The kernel, dtype, heads, interval, constants and threadgroup of ``entry``."""
    if not isinstance(entry, str):
        raise ValueError("Gated-delta entries are names")
    match = SEQUENTIAL.fullmatch(entry)
    if match:
        vjp, type_name, *numbers, save = match.groups()
        dk, dv, hk, hv, checkpoint = (_canonical(number) for number in numbers)
        _require_shape(dk, dv, hk, hv)
        if checkpoint not in CHECKPOINTS or (vjp and save):
            raise ValueError("No gated-delta sequential instantiation for this entry")
        kind = "vjp" if vjp else "sequential"
        source_entry = entry[: -len(SAVE_SUFFIX)] if save else entry
        info = {"kind": kind, "checkpoint": checkpoint, "chunk": None}
        constants = {} if vjp else {"save_state": bool(save)}
    else:
        match = FUSED.fullmatch(entry)
        if not match:
            raise ValueError(f"Not a gated-delta entry: {entry!r}")
        type_name, *numbers = match.groups()
        dk, dv, hk, hv, chunk = (_canonical(number) for number in numbers)
        _require_shape(dk, dv, hk, hv)
        if chunk != CHUNK:
            raise ValueError("No gated-delta fused chunk instantiation for this size")
        source_entry = entry
        info = {"kind": "fused", "checkpoint": None, "chunk": chunk}
        constants = {"save_state": False}
    return {
        **info,
        "source": SOURCES["vjp" if info["kind"] == "vjp" else "forward"],
        "sourceEntry": source_entry,
        "dtype": TYPES[type_name],
        "Dk": dk,
        "Dv": dv,
        "Hk": hk,
        "Hv": hv,
        "constants": constants,
        "workgroupSize": WORKGROUP_SIZE,
    }


def entry_name(kind, dtype, hk, hv, step=1, save=False):
    """The hash name gated_delta_update.cpp gives a pipeline."""
    suffix = f"{TYPE_NAMES[dtype]}_{DK}_{DV}_{hk}_{hv}_{step}"
    if kind == "fused":
        return f"gated_delta_fused_chunk_{suffix}"
    prefix = "seq_gated_delta_vjp_" if kind == "vjp" else "seq_gated_delta_"
    return prefix + suffix + (SAVE_SUFFIX if save else "")


def source_entry(entry):
    return signature(entry)["sourceEntry"]


def workgroup_size(entry):
    return tuple(signature(entry)["workgroupSize"])


def specialization_constants(entry):
    """The function constants an entry's name implies, by Metal index."""
    constants = signature(entry)["constants"]
    return {str(SAVE_STATE_ID): constants["save_state"]} if constants else {}


def parameters(entry):
    """The buffers the host binds, in the kernel's buffer order."""
    info = signature(entry)
    if info["kind"] == "vjp":
        return BACKWARD
    if info["constants"]["save_state"] and info["kind"] == "sequential":
        return (*FORWARD, "state_cache")
    return FORWARD


def outputs(entry):
    info = signature(entry)
    if info["kind"] == "vjp":
        return ("dq", "dk", "dv", "dg", "db", "dh")
    if info["constants"]["save_state"]:
        return ("y", "state_out", "state_cache")
    return ("y", "state_out")


def atomics(entry):
    """Outputs the kernel adds to through ``mlx_atomic<float>`` pointers."""
    return ATOMIC if signature(entry)["kind"] == "vjp" else ()


def initialized(entry):
    """Outputs whose zero fill the kernel keeps or accumulates into.

    gated_delta_update.cpp fills the inference output and every backward
    gradient with zeros before its launch; the backward adds four of them
    atomically.
    """
    info = signature(entry)
    if info["kind"] == "vjp":
        return outputs(entry)
    return () if info["constants"]["save_state"] else ("y",)


def scalars(entry):
    return ("T",)


def parameter_type(entry, name):
    """The MLX dtype the host binds ``name`` as."""
    if name not in parameters(entry):
        raise ValueError(f"Gated-delta entry has no parameter {name!r}")
    if name == "T":
        return "int32"
    return signature(entry)["dtype"] if name in TYPED else "float32"


def _require(
    condition, message="Gated-delta launch differs from its kernel's dispatch"
):
    if not condition:
        raise ValueError(message)


def spans(entry, batch, steps):
    """Each bound array's exact element count for ``batch`` and ``steps``."""
    info = signature(entry)
    hk, hv, dk, dv = info["Hk"], info["Hv"], info["Dk"], info["Dv"]
    keys, values = batch * steps * hk * dk, batch * steps * hv * dv
    gates, state = batch * steps * hv, batch * hv * dv * dk
    if info["kind"] == "vjp":
        checkpoints = -(-steps // info["checkpoint"])
        return {
            "q": keys,
            "k": keys,
            "v": values,
            "g": gates,
            "b": gates,
            "cot_o": values,
            "cot_h": state,
            "state_cache": state * checkpoints,
            "dq": keys,
            "dk": keys,
            "dv": values,
            "dg": gates,
            "db": gates,
            "dh": state,
        }
    counts = {
        "q": keys,
        "k": keys,
        "v": values,
        "state_in": state,
        "g": gates,
        "beta": gates,
        "y": values,
        "state_out": state,
    }
    if "state_cache" in parameters(entry):
        counts["state_cache"] = state * -(-steps // info["checkpoint"])
    return counts


def execution(entry, batch):
    """The workgroup counts and size of ``batch`` sequences' launch."""
    info = signature(entry)
    rows = info["Dv"] // CHUNK if info["kind"] == "fused" else info["Dv"]
    return {
        "workgroupCount": [1, rows // WORKGROUP_SIZE[1], batch * info["Hv"]],
        "workgroupSize": list(WORKGROUP_SIZE),
    }


def plan_launch(entry, counts, values, threads, launch):
    """Check one launch and return its metadata and exact buffer spans.

    ``counts`` holds each array parameter's bound element count and
    ``values`` the scalar ``T``.
    """
    info = signature(entry)
    _require(set(values) == {"T"})
    steps = values["T"]
    _require(
        type(steps) is int and 1 <= steps <= T_MAXIMUM,
        "Gated-delta sequences need between one and 2^31 - 9 steps",
    )
    count = launch.get("workgroupCount", ())
    _require(
        launch.get("workgroupSize") == list(WORKGROUP_SIZE)
        and len(count) == 3
        and all(type(value) is int for value in count)
        and not any(launch.get("threadGridSize", ()))
    )
    depth = count[2]
    _require(
        1 <= depth <= MAX_WORKGROUPS and depth % info["Hv"] == 0,
        "Gated-delta grid depth is not whole batches of value heads",
    )
    batch = depth // info["Hv"]
    _require(
        {key: launch[key] for key in ("workgroupCount", "workgroupSize")}
        == execution(entry, batch)
    )
    expected = spans(entry, batch, steps)
    primary = "dv" if info["kind"] == "vjp" else "y"
    _require(
        counts == expected
        and max(expected.values()) <= SIGNED_MAX
        and threads == expected[primary],
        "Gated-delta buffers differ from the kernel's addressing",
    )
    return {
        "kind": info["kind"],
        "dtype": info["dtype"],
        "batch": batch,
        "steps": steps,
        "keyHeads": info["Hk"],
        "valueHeads": info["Hv"],
        "checkpoint": info["checkpoint"],
        "saveState": info["constants"].get("save_state", False),
        "elements": expected,
    }


def metadata_preconditions(entry, target):
    """The ``T`` bound the software-subgroup fused chunk translations assume."""
    info = signature(entry)
    if info["kind"] != "fused" or target not in {"opengl", "directx"}:
        return []
    return [
        {
            "source": info["source"],
            "entryPoint": info["sourceEntry"],
            "parameter": "T",
            "minimum": 0,
            "maximum": T_MAXIMUM,
        }
    ]


def opengl_index_assertions(entry):
    """The backward's atomic element offsets, inside ``plan_launch``'s spans.

    ``mlx_atomic_fetch_add_explicit`` takes a ``size_t`` offset into a rebased
    atomic pointer; every gradient span is at most 2^31 - 1 elements.
    """
    info = signature(entry)
    if info["kind"] != "vjp":
        return []
    return [
        {
            "source": info["source"],
            "expression": "object + offset",
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
    ]
