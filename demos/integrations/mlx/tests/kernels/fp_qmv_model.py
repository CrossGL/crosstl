"""The fp_quantized.metal matrix-vector model the fp_qmv gates share.

Inputs, layouts and expectations for ``fp_qmv``, ``fp_qmv_fast`` and their
gather twins: quarters in ``[-1, 1]``, E2M1 and E4M3 codes, E4M3 and E8M0
scales and per-expert global scales; direct dequantized dot products with
their float32 exactness budget; ``qdot`` and ``adjust_matrix_offsets``
expression for expression; and the translation each entry runs.
"""

import math
import struct
from fractions import Fraction
from pathlib import Path

from crosstl.project import (
    ProjectConfig,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    F4E2M1_MAX,
    F8E4M3_MAX,
    SOURCE,
    SOURCE_SHA256,
    _div,
    _e2m1_decode,
    _e4m3_decode,
    _e8m0_decode,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _pack_bitstream,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    ARRAY_TYPES,
    BROADCAST_RHS,
    EXPERT_RHS,
    SCALAR_TYPES,
    SIMD_SIZE,
    WORKGROUP,
    _elem_to_loc,
    _f32,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _below,
    _draws,
    _signed,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact

# (mode, group size, bits) in instantiate_quantized_types' order.
MODES = (("nvfp4", 16, 4), ("mxfp8", 32, 8), ("mxfp4", 32, 4))

# fp_qmv and fp_gather_qmv take their buffers in declaration order.
_BATCH_BUFFERS = ("x_batch_ndims", "x_shape", "x_strides", "w_batch_ndims")
_BATCH_BUFFERS += ("w_shape", "w_strides", "s_strides")

QMV_PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        ("w", "scales", "global_scale", "x", "y", "in_vec_size", "out_vec_size")
        + _BATCH_BUFFERS
    )
)
GATHER_PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        ("w", "scales", "global_scale", "x", "lhs_indices", "rhs_indices", "y")
        + ("in_vec_size", "out_vec_size")
        + _BATCH_BUFFERS
        + ("batch_ndims", "batch_shape", "lhs_strides", "rhs_strides")
    )
)

# Global scales: inv_scale_enc = g / (F8E4M3_MAX * F4E2M1_MAX) is 3000 / 2688
# (inexact, so the product rounds), 4, 2 or 1 / 2. fp_qmv reads one for
# every batch; fp_gather_qmv one per expert.
GLOBAL_SCALES = (3000.0, 10752.0, 5376.0, 1344.0)

# E4M3 exponent fields of the mxfp8 codes: subnormals through 2^2 * 1.875.
E4M3_CODE_FIELDS = tuple(range(10))

# Scale bytes: E4M3 2^-2 through 3.75 for nvfp4, E8M0 2^-3 through 2 for mx.
NV_SCALE_FIELDS = (5, 6, 7, 8)
MX_SCALE_BYTES = (124, 125, 126, 127, 128)

GUARD_COUNT = 8
SENTINEL = -3.25

# Exactly representable poison for every element no case may read.
POISON = 1.5
WORD_POISON = 0xA5A5A5A5
BYTE_POISON = 0xA5
INDEX_POISON = 0x5A5A5A5A

# Expressions OpenGL may evaluate in 32 bits: the batch offsets of
# adjust_matrix_offsets' one-dimensional branches, the elem_to_loc_broadcast
# offsets of its general branches, and gather's index-buffer subscripts. The
# model shows each names an element inside its buffer, and every case's
# buffers hold far fewer than INDEX_BOUND elements.
INDEX_BOUND = 2**20 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "batch_0": (),
    "batch_1": (
        "x_idx * x_strides[0]",
        "w_idx * w_strides[0]",
        "w_idx * s_strides[0]",
        "idx.x",
        "idx.y",
    ),
    "gather": (
        "tid.z * lhs_strides[0]",
        "tid.z * rhs_strides[0]",
        "idx.x",
        "idx.y",
        "x_idx * x_strides[0]",
        "w_idx * w_strides[0]",
        "w_idx * s_strides[0]",
    ),
}


def target_options(info):
    """Per-target options: software subgroups for simd_sum, and correctly
    rounded division for the global scale's ``g / 2688``."""
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    if info["globalScale"]:
        for target in options.values():
            target["binary32_division_profile"] = "rne-flush"
    return options


def config_for(directory, target, entry, info, workgroup=WORKGROUP):
    """The translation of one entry, as the gates and contracts run it."""
    variant = info["variant"]
    assertions = OPENGL_INDEX_RANGE_ASSERTIONS[variant] if target == "opengl" else ()
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup}},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=INDEX_BOUND)
            for expression in assertions
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "target_options": target_options(info),
            }
        },
    )


def _parameters(variant):
    return GATHER_PARAMETERS if variant == "gather" else QMV_PARAMETERS


def _weight_code(draws, bits):
    if bits == 4:
        return _below(draws, 16)
    field = E4M3_CODE_FIELDS[_below(draws, len(E4M3_CODE_FIELDS))]
    return _below(draws, 2) << 7 | field << 3 | _below(draws, 8)


def _scale_byte(draws, mode):
    if mode == "nvfp4":
        field = NV_SCALE_FIELDS[_below(draws, len(NV_SCALE_FIELDS))]
        return field << 3 | _below(draws, 8)
    return MX_SCALE_BYTES[_below(draws, len(MX_SCALE_BYTES))]


def _scale_value(mode, byte):
    """dequantize_scale<float, group_size>: E4M3 for groups of 16, else E8M0."""
    return _e4m3_decode(byte) if mode == "nvfp4" else _e8m0_decode(byte)


def _code_value(bits, byte):
    """Dequantize<bits>: E4M3 of the byte, or E2M1 of its low nibble."""
    return _e4m3_decode(byte) if bits == 8 else _e2m1_decode(byte)


def _experts(layout):
    """The number of global scales fp_gather_qmv may read: one per expert."""
    params = layout["params"]
    return math.prod(params["w_shape"][: params["w_batch_ndims"]])


def _batch_expert(case, z):
    """The rhs index, so the expert and its global scale, of batch ``z``."""
    return EXPERT_RHS[z] if case == "experts" else BROADCAST_RHS[z // 2]


def _inverse_scale(global_scale):
    return _div(global_scale, F8E4M3_MAX * F4E2M1_MAX)


def case_data_for(info, entry, case, layout, label, case_global_scale):
    """Logical storage of every parameter, the grid and the expected outputs.

    ``case_global_scale`` is the one global scale a non-gather entry reads;
    gather reads one per expert.
    """
    dtype, bits, mode = info["dtype"], info["bits"], info["mode"]
    m, n, k = layout["M"], layout["N"], layout["K"]
    draws = _draws(label, entry, case)
    used_rows = {
        batch["x"] + row * k for batch in layout["batches"] for row in range(m)
    }
    x = [POISON] * (layout["xRows"] * k)
    for start in sorted(used_rows):
        x[start : start + k] = [_signed(draws, 4) / 4 for _ in range(k)]
    codes, scales = [], []
    for _ in range(layout["matrices"] * layout["rows"]):
        codes.append([_weight_code(draws, bits) for _ in range(k)])
        scales += [_scale_byte(draws, mode) for _ in range(layout["groups"])]
    w = []
    for row in codes:
        payload = _pack_bitstream(row, bits)
        w += list(struct.unpack(f"<{len(payload) // 4}I", bytes(payload)))
    assert len(w) == layout["matrices"] * layout["rows"] * layout["words"]
    for value in x:
        assert _to_type(dtype, value) == value, (entry, value)
    if not info["globalScale"]:
        global_scale = [POISON]
    elif info["variant"] == "gather":
        global_scale = list(GLOBAL_SCALES[: _experts(layout)])
    else:
        global_scale = [case_global_scale]
    total = len(layout["batches"]) * m * n
    initial = [SENTINEL] * (total + GUARD_COUNT)
    params = layout["params"]
    values = {
        "w": ("uint32", w + [WORD_POISON] * GUARD_COUNT),
        "scales": ("uint8", scales + [BYTE_POISON] * GUARD_COUNT),
        "global_scale": ("float32", global_scale + [POISON] * GUARD_COUNT),
        "x": (dtype, x + [POISON] * GUARD_COUNT),
        "y": (dtype, initial),
        "in_vec_size": ("int32", [k]),
        "out_vec_size": ("int32", [n]),
        **{
            name: (SCALAR_TYPES[name], [value])
            for name, value in params.items()
            if name in SCALAR_TYPES
        },
        **{
            name: (ARRAY_TYPES[name], list(value))
            for name, value in params.items()
            if name in ARRAY_TYPES
        },
    }
    if info["variant"] == "gather":
        values["lhs_indices"] = ("uint32", list(layout["lhs"]))
        values["rhs_indices"] = ("uint32", list(layout["rhs"]))
    assert set(values) == {name for name, _ in _parameters(info["variant"])}, entry
    data = {
        "case": case,
        "layout": layout,
        "codes": codes,
        "values": values,
        "initial": {"y": initial},
        "kinds": {"y": dtype},
        "grid": layout["grid"],
    }
    data["expected"], data["totals"], data["budget"] = _direct(info, data)
    return data


def _fraction_bits(value):
    """The fewest binary fraction digits that hold ``value`` exactly."""
    denominator = Fraction(value).denominator
    assert denominator & (denominator - 1) == 0, value
    return denominator.bit_length() - 1


def _direct(info, data):
    """Each output as a dequantized dot product, its float32 total and budget.

    The budget bounds the kernel's float32 arithmetic whatever its order:
    every product is a multiple of ``2^-f`` and the products' magnitudes sum
    below ``2^(24 - f)``, so every partial sum is exact. It is the largest
    such sum, in units of ``2^(24 - f)``.
    """
    dtype, mode, bits = info["dtype"], info["mode"], info["bits"]
    group_size = info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    x = data["values"]["x"][1]
    scales = data["values"]["scales"][1]
    global_scale = data["values"]["global_scale"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget = 0.0
    for z, batch in enumerate(layout["batches"]):
        factor = None
        if info["globalScale"]:
            expert = (
                _batch_expert(data["case"], z) if info["variant"] == "gather" else 0
            )
            factor = _inverse_scale(global_scale[expert])
        for row in range(m):
            vector = x[batch["x"] + row * k : batch["x"] + (row + 1) * k]
            for col in range(n):
                line = batch["w"] * n + col
                total, bound, digits = 0.0, 0.0, 0
                for index, value in enumerate(vector):
                    scale = _scale_value(
                        mode, scales[line * groups + index // group_size]
                    )
                    term = value * _code_value(bits, data["codes"][line][index]) * scale
                    total += term
                    bound += abs(term)
                    digits = max(digits, _fraction_bits(term))
                share = bound / 2.0 ** (24 - digits)
                assert share < 1 and _f32(total) == total, (info, total, bound)
                budget = max(budget, share)
                index = batch["y"] + row * n + col
                totals[index] = total
                if factor is not None:
                    total = _f32(total * factor)
                expected[index] = _to_type(dtype, total)
    return {"y": expected}, totals, budget


def _checked(value):
    """A float32 intermediate the exactness argument covers."""
    assert _f32(value) == value, value
    return value


def _qdot(bits, byte, base, x_thread, scale, mutation):
    """qdot over one lane's pack, expression for expression."""
    # The sign bit of the E4M3 byte or of the E2M1 low nibble.
    sign = 0xFF if mutation != "sign" else 0x7F if bits == 8 else 0xF7
    accum = 0.0
    if bits == 4:
        # qdot reads the pack as little-endian uint16_t words and hands
        # Dequantize<4> the low byte of each shift; fp4_e2m1 keeps its low
        # nibble.
        for i in range(len(x_thread) // 4):
            ws = byte(base + 2 * i) | byte(base + 2 * i + 1) << 8
            terms = [
                x_thread[4 * i + j] * _code_value(4, (ws >> (4 * j)) & 0xFF & sign)
                for j in range(4)
            ]
            partial = _checked(_checked(terms[0] + terms[1]) + terms[2])
            accum = _checked(accum + _checked(partial + terms[3]))
    else:
        for i, value in enumerate(x_thread):
            accum = _checked(accum + value * _code_value(8, byte(base + i) & sign))
    return _checked(scale * accum)


def _adjust(info, storage, z, mutation, records):
    """adjust_matrix_offsets for ``tid.z``: x, w, scales, global scale, y."""

    def scalar(name):
        return storage[name][0]

    if info["variant"] == "gather":
        if scalar("batch_ndims") == 1:
            lhs = z * storage["lhs_strides"][0]
            rhs = z * storage["rhs_strides"][0]
            records += [
                ("tid.z * lhs_strides[0]", "lhs_indices", lhs),
                ("tid.z * rhs_strides[0]", "rhs_indices", rhs),
            ]
        else:
            lhs, rhs = _elem_to_loc(
                z,
                storage["batch_shape"],
                (storage["lhs_strides"], storage["rhs_strides"]),
                scalar("batch_ndims"),
            )
            records += [("idx.x", "lhs_indices", lhs), ("idx.y", "rhs_indices", rhs)]
        x_idx, w_idx = storage["lhs_indices"][lhs], storage["rhs_indices"][rhs]
    else:
        x_idx = w_idx = z
    if scalar("x_batch_ndims") == 1:
        x_base = x_idx * storage["x_strides"][0]
        records.append(("x_idx * x_strides[0]", "x", x_base))
    else:
        (x_base,) = _elem_to_loc(
            x_idx, storage["x_shape"], (storage["x_strides"],), scalar("x_batch_ndims")
        )
    strides = (storage["w_strides"], storage["s_strides"])
    if scalar("w_batch_ndims") == 1:
        bases = [w_idx * values[0] for values in strides]
        records += [
            ("w_idx * w_strides[0]", "w", bases[0]),
            ("w_idx * s_strides[0]", "scales", bases[1]),
        ]
    else:
        bases = _elem_to_loc(
            w_idx, storage["w_shape"], strides, scalar("w_batch_ndims")
        )
        records += [("idx.x", "w", bases[0]), ("idx.y", "scales", bases[1])]
    if mutation == "batch":
        bases = [0, 0]
    # The gather overload moves global_scale to the expert's; fp_qmv's
    # batched overload leaves it.
    expert = 0
    if info["variant"] == "gather" and info["globalScale"]:
        expert = 0 if mutation == "expert-scale" else w_idx
    rows = storage["x_shape"][scalar("x_batch_ndims")]
    return (x_base, *bases, expert, z * scalar("out_vec_size") * rows)


def translate_entry(root, work, target, config, record):
    """Translate one entry and check it against its pinned artifact."""
    translation = translate_project(
        config,
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    validation = validate_project_report(work / "report.json")
    assert validation["success"] is True, validation
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    assert report["summary"]["failedCount"] == 0
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == list(WORKGROUP)
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    return _prepare_native_package(translation, work)


def bound_parameters(descriptor, info):
    """The kernel parameters the translation reflects.

    An entry without a global scale never reads ``global_scale``, so its
    translation may drop the binding; every other parameter stays.
    """
    parameters = _parameters(info["variant"])
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
    }
    if info["globalScale"] or "global_scale" in bound:
        return parameters
    return tuple(item for item in parameters if item[0] != "global_scale")
