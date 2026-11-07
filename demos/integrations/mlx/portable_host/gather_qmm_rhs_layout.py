"""Storage and launch contracts for MLX's sorted gathered quantized kernels.

``GatherQMM::eval_gpu`` in ``gather_qmm.inc`` sends single-row products whose
right-hand indices are sorted, with at least 16 batches and at least four per
weight matrix, to quantized.cpp's ``gather_qmm_rhs``; ``GatherQQMM::eval_gpu``
sends its globally scaled bfloat16 nvfp4 products of that shape there too. The portable host has no
NAX unit, so ``gather_qmm_rhs`` never takes ``gather_qmm_rhs_nax``: it
broadcasts the input with the indices, runs ``gather_mm_offsets`` to find each
weight matrix's first row, then runs quantized.metal's
``affine_gather_qmm_rhs`` (which binds biases) or fp_quantized.metal's
``fp_gather_qmm_rhs`` (uint8 scales, and the ``_hgs`` instantiations that read
each matrix's nvfp4 global scale). Every launch runs ``(32, 2, 1)``
threadgroups over ``ceil(N / 32)`` column tiles and
``min(M, ceil(M / 16) + E - 1)`` row tiles of 16 rows, for ``E`` weight
matrices. The entry names carry the ``align_N`` and ``align_K`` function
constants as quantized.cpp's hash names do.

``plan`` reads the group offsets the kernels schedule their row tiles from,
requires them to split the rows in order, and bounds the weights, scales,
biases and global scale of every matrix that owns rows.

The kernels find their row tile with subgroup scans of the offsets and return
early on their reduction, which DirectX and OpenGL software subgroups lower
only once reductions of lane-symmetric values count as workgroup-uniform
(CrossGL/crosstl#2546); until then only Metal translates these entries.
"""

import re

from demos.integrations.mlx.portable_host import gather_qmm_layout
from demos.integrations.mlx.portable_host.gather_qmv_layout import (
    AFFINE_SOURCE,
    INACTIVE_GLOBAL_SCALE,
    MAX_ELEMENTS,
    MAX_GROUPS,
    MODES,
    SIMD_SIZE,
    SOURCE,
    check_mode,
)
from demos.integrations.mlx.portable_host.matmul_layout import read_buffers
from demos.integrations.mlx.portable_host.qmm_layout import K_LIMIT
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES
from demos.integrations.mlx.tests.kernels.gemm_fused_model import Loader

LABEL = "Sorted gathered quantized matrix-matrix"
# quantized.cpp gather_qmm_rhs's tile: bm, bn, bk, wm and wn.
BM, BN, BK, WM, WN = 16, 32, 32, 1, 2
THREADS = WM * WN * SIMD_SIZE
TILE = f"_bm_{BM}_bn_{BN}_bk_{BK}_wm_{WM}_wn_{WN}"
PATTERN = re.compile(
    r"((affine|mxfp4|mxfp8|nvfp4)_gather_qmm_rhs_(nt|nn)_(float|float16_t|bfloat16_t)"
    r"_gs_(16|32|64|128)_b_(2|3|4|5|6|8)" + TILE + r"(_hgs)?)"
    r"_align_N_([tn])_align_K_([tn])"
)
ENTRY_PREFIXES = tuple(f"{mode}_gather_qmm_rhs_" for mode in ("affine", *MODES))
# The primitives whose routing a launch follows, as for gather_qmm.
ROUTES = gather_qmm_layout.ROUTES
MATRIX_PARAMETERS = ("w", "scales", "x", "y")
AFFINE_MATRIX_PARAMETERS = ("w", "scales", "biases", "x", "y")
SCALAR_PARAMETERS = ("M", "N", "K", "num_groups")
CONSTANTS = ("align_N", "align_K")


def signature(entry):
    """The mode, transposition, element type, group size, bits and constants."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    base, mode, kernel, source, group, bits, scaled, align_n, align_k = match.groups()
    group, bits = int(group), int(bits)
    check_mode(mode, group, bits, scaled is not None, LABEL)
    return {
        "source": base,
        "mode": mode,
        "transpose": kernel == "nt",
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": group,
        "bits": bits,
        "globalScale": scaled is not None,
        "constants": {"align_N": align_n == "t", "align_K": align_k == "t"},
    }


def source_entry(entry):
    """The instantiated kernel ``entry`` specializes with its constants."""
    return signature(entry)["source"]


def source(entry):
    """The pinned source that instantiates ``entry``."""
    return AFFINE_SOURCE if signature(entry)["mode"] == "affine" else SOURCE


def workgroup_size(entry):
    signature(entry)
    return (SIMD_SIZE, WN, WM)


def routed(m, batches, groups, right_sorted):
    """quantized.cpp GatherQMM's test for gather_qmm_rhs: ``M`` is one row."""
    return m == 1 and batches >= 16 and right_sorted and batches // groups >= 4


def select(
    mode, dtype, n, k, global_scale=False, transpose=True, group=None, bits=None
):
    """quantized.cpp gather_qmm_rhs's kernel and hash name without NAX.

    The block-scaled modes fix their group size and bits; affine products
    name theirs.
    """
    if mode != "affine" and group is None:
        group, bits = MODES[mode]
    if group is None or bits is None:
        raise ValueError(f"{LABEL} affine products name their group and bits")
    check_mode(mode, group, bits, global_scale, LABEL)
    kernel = "nt" if transpose else "nn"
    scaled = "_hgs" if global_scale else ""

    def flag(value):
        return "t" if value else "n"

    return (
        f"{mode}_gather_qmm_rhs_{kernel}_{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
        f"{TILE}{scaled}_align_N_{flag(n % BN == 0)}_align_K_{flag(k % BK == 0)}"
    )


def row_tiles(m, groups):
    """quantized.cpp gather_qmm_rhs's grid height."""
    return min(m, -(-m // BM) + groups - 1)


def matrix_parameters(entry):
    """The buffers the host binds by address."""
    info = signature(entry)
    if info["mode"] == "affine":
        return AFFINE_MATRIX_PARAMETERS
    if info["globalScale"]:
        return (*MATRIX_PARAMETERS, "global_scale")
    return MATRIX_PARAMETERS


def parameter_names(entry):
    return (*matrix_parameters(entry), *SCALAR_PARAMETERS, "offsets")


def parameter_type(entry, name):
    info = signature(entry)
    if name == "w":
        return "uint32"
    if name == "scales":
        return info["dtype"] if info["mode"] == "affine" else "uint8"
    if name == "global_scale":
        return "float32"
    if name in ("x", "y", "biases"):
        return info["dtype"]
    if name in (*SCALAR_PARAMETERS, "offsets"):
        return "int32"
    raise ValueError(f"{LABEL} parameter is unknown")


def read_storage(entry, buffers):
    """Check native buffer identity and read the scalars and group offsets."""
    return read_buffers(
        buffers,
        {name: parameter_type(entry, name) for name in parameter_names(entry)},
        SCALAR_PARAMETERS,
        LABEL,
        matrices=matrix_parameters(entry),
    )


def _check_launch(entry, execution, m, n, groups):
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError(f"{LABEL} launch geometry is missing")
    expected = [-(-n // BN), row_tiles(m, groups), 1]
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or list(execution["workgroupCount"]) != expected
        or max(expected) > MAX_GROUPS
    ):
        raise ValueError(f"{LABEL} launch does not match MLX's grid")


def _check_offsets(offsets, m, groups):
    """gather_mm_offsets' result: each matrix's first row, in order."""
    if len(offsets) != groups or any(type(start) is not int for start in offsets):
        raise ValueError(f"{LABEL} group offsets do not match the groups")
    ends = [*offsets[1:], m]
    if any(not 0 <= start <= end <= m for start, end in zip(offsets, ends)):
        raise ValueError(f"{LABEL} group offsets are not sorted rows")
    return [
        group for group, (start, end) in enumerate(zip(offsets, ends)) if start < end
    ]


def plan(entry, counts, parameters, threads, execution, route=None):
    """Validate one launch against MLX's routing and return its metadata.

    The launch must be a product quantized.cpp sends to gather_qmm_rhs, its
    offsets must split the rows in order, and the input, the output and every
    matrix that owns rows must lie inside their buffers. The returned
    ``parameters`` add the inert global scale of the block-scaled kernels
    without _hgs.
    """
    info = signature(entry)
    if route not in ROUTES:
        raise ValueError(f"{LABEL} route is unknown")
    if set(parameters) != {*SCALAR_PARAMETERS, "offsets"}:
        raise ValueError(f"{LABEL} parameters are missing")
    m, n, k, groups = (parameters[name] for name in SCALAR_PARAMETERS)
    group, bits, transpose = info["groupSize"], info["bits"], info["transpose"]
    # Groups run along the weights' rows: K when transposed, N otherwise.
    grouped = k if transpose else n
    if (
        any(type(value) is not int for value in (m, n, k, groups))
        or not 1 <= min(m, n, k) <= max(m, n, k) <= MAX_ELEMENTS
        or k > K_LIMIT
        or not 1 <= groups <= MAX_GROUPS
        or grouped % group
        or (n * k * bits) % 32
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    gather_qmm_layout.check_route(info, route, transpose, k, LABEL)
    if not routed(1, m, groups, True):
        raise ValueError(f"{LABEL} products take gather_qmm and gather_qmv")
    expected = select(
        info["mode"], info["dtype"], n, k, info["globalScale"], transpose, group, bits
    )
    if expected != entry:
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    _check_launch(entry, execution, m, n, groups)
    if threads != m * n or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    if counts["x"] < m * k:
        raise ValueError(f"{LABEL} input exceeds its buffer")
    owners = _check_offsets(list(parameters["offsets"]), m, groups)
    # Each matrix holds n * k quantized values: words of packed bits, and
    # one scale (and bias) per group.
    extents = {"w": n * k * bits // 32, "scales": n * k // group}
    if info["mode"] == "affine":
        extents["biases"] = extents["scales"]
    if info["globalScale"]:
        extents["global_scale"] = 1
    last = max(owners)
    for name, extent in extents.items():
        if (last + 1) * extent > counts[name]:
            raise ValueError(f"{LABEL} {name} of a gathered matrix exceeds its buffer")
    values = dict(parameters)
    if info["mode"] != "affine" and not info["globalScale"]:
        values["global_scale"] = [INACTIVE_GLOBAL_SCALE]
    return {
        "mode": info["mode"],
        "kernel": "rhs_nt" if transpose else "rhs_nn",
        "route": route,
        "M": m,
        "N": n,
        "K": k,
        "groups": groups,
        "owners": owners,
        "constants": info["constants"],
        "overreadPadding": {},
        "parameters": values,
    }


def x_loader(entry):
    """The activation ``BlockLoader`` of BM rows the translation materializes."""
    info = signature(entry)
    width = BK + 16 // (4 if info["dtype"] == "float32" else 2)
    itype = SOURCE_TYPES[info["dtype"]]
    return Loader(
        f"BlockLoader_{itype}_{BM}_{BK}_{width}_1_{THREADS}__load_unsafe",
        BM,
        BK,
        width,
        THREADS,
    )


def access_assertions(entry):
    """The x loader's ``dst`` stays inside the slots ``BlockLoader`` writes."""
    loader = x_loader(entry)
    slots = [slot for lane in loader.slots() for slot in lane]
    return (
        {
            "source": source(entry),
            "entryPoint": source_entry(entry),
            "function": loader.function,
            "parameter": "self.dst",
            "minimum": min(slots),
            "maximum": max(slots),
        },
    )
