"""Name, check and bound MLX's gated-delta launches.

``mlx/backend/metal/gated_delta_update.cpp`` runs the pinned recurrent
gated-delta kernels for the head shapes ``supported_gated_delta_shape``
accepts:

- ``seq_gated_delta_<type>_<Dk>_<Dv>_<Hk>_<Hv>_<Ckpt>`` (``gated_delta_seq``
  of gated_delta_update.metal): one 32-lane SIMD group per state row, each
  lane holding ``Dk / 32`` state entries. ``GatedDeltaUpdate::eval_gpu``
  binds it with function constant 200 (``save_state``) cleared; the
  backward's save pass sets it, and MLX names that pipeline
  ``<name>_save``. The save pass stores the state entering every
  ``Ckpt``-th step into ``state_cache``.
- ``gated_delta_fused_chunk_<type>_<Dk>_<Dv>_<Hk>_<Hv>_8``: the eight-step
  chunk kernel, one SIMD group per eight state columns of one head, which
  MLX picks for ``T > 8`` on a device without NAX.
- ``seq_gated_delta_vjp_<type>_<Dk>_<Dv>_<Hk>_<Hv>_<Ckpt>``
  (gated_delta_update_vjp.metal): the sequential backward, which replays
  each checkpoint segment and adds the query, key, decay and beta gradients
  into zero-filled float32 accumulators through ``mlx_atomic<float>``.

Host entries are MLX's pipeline names. Every kernel runs in ``(32, 4, 1)``
threadgroups over MLX's ``dispatch_threads`` grid, ``(32, Dv, B * Hv)`` for
the sequential kernels and ``(32, Dv / 8, B * Hv)`` for the chunk kernel,
which these threadgroups tile exactly. ``plan`` checks a launch's buffers,
its step count and its geometry against that plan before any package is
translated; every bound buffer stays below ``MAX_ELEMENTS`` elements, so
each offset the kernels form fits a signed 32-bit index.
"""

from types import MappingProxyType

SOURCES = MappingProxyType(
    {
        "forward": "mlx/backend/metal/kernels/gated_delta_update.metal",
        "vjp": "mlx/backend/metal/kernels/gated_delta_update_vjp.metal",
    }
)
# get_type_string names and the MLX dtypes they carry.
TYPES = MappingProxyType(
    {"float": "float32", "bfloat16_t": "bfloat16", "float16_t": "float16"}
)
TYPE_NAMES = MappingProxyType({dtype: name for name, dtype in TYPES.items()})
# supported_gated_delta_shape(): (Dk, Dv, Hk, Hv).
SHAPES = (
    (128, 128, 24, 24),
    (128, 128, 32, 32),
    (128, 128, 16, 32),
    (128, 128, 16, 48),
    (128, 128, 16, 16),
    (128, 128, 16, 64),
)
# Every interval gated_delta_ckpt() can return, and the chunk kernel's steps.
CHECKPOINTS = (1, 4, 8, 16)
CHUNK = 8
SIMD_SIZE = 32
WORKGROUP_SIZE = (32, 4, 1)
# Each chunk kernel SIMD group holds eight state columns.
CHUNK_COLUMNS = 8
# Function constant 200.
SAVE_STATE = "save_state"
SAVE_SUFFIX = "_save"
# The bounds the host's other families use: every bound buffer stays below
# 2**28 elements, far inside OpenGL's signed 32-bit offsets, and one launch
# axis carries at most 65535 workgroups.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
SIGNED_MAX = 2**31 - 1
# for (; t + C <= T; t += C) must not overflow t + C, so OpenGL and DirectX
# check this precondition on the chunk kernel's T.
T_MAXIMUM = SIGNED_MAX - CHUNK
INOUT = 2

FORWARD = ("q", "k", "v", "state_in", "g", "beta", "y", "state_out", "T")
BACKWARD = (
    "q",
    "k",
    "v",
    "g",
    "b",
    "cot_o",
    "cot_h",
    "state_cache",
    "T",
    "dq",
    "dk",
    "dv",
    "dg",
    "db",
    "dh",
)
# Parameters in the gradient's storage type; the others are float32 except T.
_TYPED = frozenset(("q", "k", "v", "g", "beta", "b", "y", "cot_o"))
GRADIENTS = ("dq", "dk", "dv", "dg", "db", "dh")
# The gradients the backward adds atomically, in any order.
ATOMIC_GRADIENTS = frozenset(("dq", "dk", "dg", "db"))
SCALARS = ("T",)


def _entries():
    entries = {}
    for type_name, dtype in TYPES.items():
        for dk, dv, hk, hv in SHAPES:
            suffix = f"{type_name}_{dk}_{dv}_{hk}_{hv}"
            shape = {"dtype": dtype, "dk": dk, "dv": dv, "hk": hk, "hv": hv}
            for ckpt in CHECKPOINTS:
                kernel = f"seq_gated_delta_{suffix}_{ckpt}"
                for save in (False, True):
                    entries[kernel + SAVE_SUFFIX * save] = MappingProxyType(
                        {
                            "kind": "seq",
                            "kernel": kernel,
                            "source": SOURCES["forward"],
                            "save": save,
                            "steps": ckpt,
                            **shape,
                        }
                    )
            kernel = f"gated_delta_fused_chunk_{suffix}_{CHUNK}"
            entries[kernel] = MappingProxyType(
                {
                    "kind": "fused",
                    "kernel": kernel,
                    "source": SOURCES["forward"],
                    "save": False,
                    "steps": CHUNK,
                    **shape,
                }
            )
            for ckpt in CHECKPOINTS:
                kernel = f"seq_gated_delta_vjp_{suffix}_{ckpt}"
                entries[kernel] = MappingProxyType(
                    {
                        "kind": "vjp",
                        "kernel": kernel,
                        "source": SOURCES["vjp"],
                        "save": False,
                        "steps": ckpt,
                        **shape,
                    }
                )
    return MappingProxyType(entries)


# Host entry -> its kernel, source, kind, save flag, step interval and shape.
ENTRIES = _entries()
ENTRY_PREFIXES = ("seq_gated_delta_", "gated_delta_fused_chunk_")


def signature(entry):
    """The kernel, source, kind, constants and head shape of ``entry``."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported gated-delta entry") from error


def is_entry(entry):
    return isinstance(entry, str) and entry in ENTRIES


def entry_name(kind, dtype, dk, dv, hk, hv, steps, save=False):
    """The pipeline name eval_gpu gives one launch."""
    prefix = {
        "seq": "seq_gated_delta",
        "fused": "gated_delta_fused_chunk",
        "vjp": "seq_gated_delta_vjp",
    }[kind]
    if dtype not in TYPE_NAMES:
        raise ValueError("Unsupported gated-delta storage type")
    name = f"{prefix}_{TYPE_NAMES[dtype]}_{dk}_{dv}_{hk}_{hv}_{steps}"
    name += SAVE_SUFFIX if save else ""
    signature(name)
    return name


def source_entry(entry):
    return signature(entry)["kernel"]


def source(entry):
    return signature(entry)["source"]


def workgroup_size(entry):
    signature(entry)
    return WORKGROUP_SIZE


def constants(entry):
    """The function constants the pipeline is built with, frozen per entry."""
    info = signature(entry)
    if info["kind"] == "seq":
        return {SAVE_STATE: info["save"]}
    return {}


def parameters(entry):
    """Every bound source parameter, in source order."""
    info = signature(entry)
    if info["kind"] == "vjp":
        return BACKWARD
    return FORWARD + (("state_cache",) if info["save"] else ())


def inactive(entry):
    """Declared parameters whose function constant is off.

    The inference pipeline clears ``save_state``, so it never touches
    ``state_cache``; MLX leaves that slot unbound.
    """
    info = signature(entry)
    return ("state_cache",) if info["kind"] == "seq" and not info["save"] else ()


def parameter_type(entry, name):
    if name not in parameters(entry) + inactive(entry):
        raise ValueError("Unknown gated-delta parameter")
    if name == "T":
        return "int32"
    return signature(entry)["dtype"] if name in _TYPED else "float32"


def outputs(entry):
    if signature(entry)["kind"] == "vjp":
        return GRADIENTS
    return ("y", "state_out") + (("state_cache",) if signature(entry)["save"] else ())


def directions(entry):
    """0 for inputs, 1 for fresh outputs, ``INOUT`` for initialized outputs.

    eval_gpu fills the forward output with zeros before the inference launch,
    while the backward's save pass writes fresh scratch; the backward fills
    every gradient and accumulator with zeros first.
    """
    info = signature(entry)
    result = {name: 0 for name in parameters(entry)}
    if info["kind"] == "vjp":
        result.update({name: INOUT for name in GRADIENTS})
    else:
        result.update(
            {"y": 1 if info["save"] else INOUT, "state_out": 1, "state_cache": 1}
        )
    return {name: result[name] for name in parameters(entry)}


def scalars(entry):
    signature(entry)
    return SCALARS


def checkpoints(entry, steps):
    """The state slots the save pass stores or the backward replays."""
    return -(-steps // signature(entry)["steps"])


def extents(entry, batch, steps):
    """Every bound buffer's element count for ``batch`` sequences of ``steps``."""
    info = signature(entry)
    dk, dv, hk, hv = info["dk"], info["dv"], info["hk"], info["hv"]
    keys = batch * steps * hk * dk
    values = batch * steps * hv * dv
    gates = batch * steps * hv
    state = batch * hv * dv * dk
    if info["kind"] == "vjp":
        cache = state * checkpoints(entry, steps)
        return {
            "q": keys,
            "k": keys,
            "v": values,
            "g": gates,
            "b": gates,
            "cot_o": values,
            "cot_h": state,
            "state_cache": cache,
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
    if info["save"]:
        counts["state_cache"] = state * checkpoints(entry, steps)
    return counts


def launch(entry, batch):
    """``dispatch_threads``' grid of ``batch`` sequences in whole threadgroups."""
    info = signature(entry)
    columns = info["dv"] // (CHUNK_COLUMNS if info["kind"] == "fused" else 1)
    rows = batch * info["hv"]
    return {
        "workgroupCount": [1, columns // WORKGROUP_SIZE[1], rows],
        "workgroupSize": list(WORKGROUP_SIZE),
    }


def threads(execution):
    count, size = execution["workgroupCount"], execution["workgroupSize"]
    return count[0] * count[1] * count[2] * size[0] * size[1] * size[2]


def plan(entry, counts, steps, launched_threads, execution):
    """Check one launch against eval_gpu; returns its batch and step count.

    ``counts`` holds every bound array's element count and ``steps`` the
    ``T`` the launch binds.
    """
    info = signature(entry)
    names = set(parameters(entry)) - set(SCALARS)
    if (
        not isinstance(counts, dict)
        or set(counts) != names
        or any(type(value) is not int for value in counts.values())
        or type(steps) is not int
        or not 1 <= steps <= T_MAXIMUM
    ):
        raise ValueError("Gated-delta buffers do not match their kernel")
    state = counts["cot_h" if info["kind"] == "vjp" else "state_in"]
    per_batch = info["hv"] * info["dv"] * info["dk"]
    batch = state // per_batch
    if batch < 1 or batch * per_batch != state:
        raise ValueError("Gated-delta state does not hold whole heads")
    if batch * info["hv"] > MAX_GROUPS:
        raise ValueError("Gated-delta launch exceeds its group limit")
    expected = extents(entry, batch, steps)
    if counts != expected or any(
        not 1 <= value <= MAX_ELEMENTS for value in counts.values()
    ):
        raise ValueError("Gated-delta buffers do not match their sequences")
    if execution != launch(entry, batch) or launched_threads != threads(execution):
        raise ValueError("Gated-delta launch does not match the upstream plan")
    return {"batch": batch, "steps": steps}


def metadata_preconditions(entry, target):
    """The chunk kernel's ``T`` bound on the software-subgroup targets."""
    info = signature(entry)
    if info["kind"] != "fused" or target not in {"opengl", "directx"}:
        return ()
    return (
        {
            "source": info["source"],
            "entryPoint": info["kernel"],
            "parameter": "T",
            "minimum": 0,
            "maximum": T_MAXIMUM,
        },
    )


def opengl_index_assertions(entry):
    """OpenGL offsets ``plan`` keeps inside ``[0, INT32_MAX]``.

    ``mlx_atomic_fetch_add_explicit`` takes a ``size_t`` offset into a
    rebased atomic pointer; every atomic update stays inside a gradient
    buffer of at most ``MAX_ELEMENTS`` elements.
    """
    if signature(entry)["kind"] != "vjp":
        return ()
    return ("object + offset",)
