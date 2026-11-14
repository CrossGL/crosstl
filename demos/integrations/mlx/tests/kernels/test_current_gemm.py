"""Translate a complete pinned MLX GEMM entry and compare native results."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from fractions import Fraction
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
from demos.integrations.mlx.tests.kernels import gemm_fused_model as model
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal"
SOURCE_SHA256 = "6c6e67f5c7fc6b378ca2c849c36a79d696c93e65b3a60d3f5b1f13928c79029e"
ENTRY = "steel_gemm_fused_nn_float32_float32_bm32_bn32_bk16_wm2_wn2"
REPRESENTATIVE = model.BY_NAME[ENTRY]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMM"
CONSTANT_IDS = {
    "has_batch": 10,
    "use_out_source": 100,
    "do_axpby": 110,
    "align_M": 200,
    "align_N": 201,
    "align_K": 202,
}
HEADER_HASHES = {
    "loader.h": "c1b82153670b1e18371a6e88fdbcf440af692957775c2f837700315ab8a5112e",
    "gemm.h": "c84af31e2c57154f2a8a24fa7f9fe2449765cce9a66f3f035846ed3c03b6a8b0",
    "mma.h": "9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7",
}
WORKGROUP_LOADERS = {
    "BlockLoader_float_false_16_32_false_32_16_false_32_4_16_4_false_128__load_unsafe": (
        32,
        16,
    ),
    "BlockLoader_float_false_32_16_false_16_32_false_16_4_32_4_false_128__load_unsafe": (
        16,
        32,
    ),
}
SIGNED_MAX = 2**31 - 1
PLATFORM_TARGETS = {"darwin": "metal", "win32": "directx", "linux": "opengl"}
GATE_TOOLS = {
    "metal": ("xcrun",),
    "directx": ("dxc",),
    "opengl": ("glslangValidator", "spirv-val"),
}
GUARD = -98765.0
# -98765 is not a binary16 or bfloat16 value; 3 * 2^13 is a value of both.
GUARDS = {"float32": GUARD, "float16": -24576.0, "bfloat16": -24576.0}
PRECISION = {"float32": 24, "float16": 11, "bfloat16": 8}
# The scalar spelling resource-aggregate helpers carry for each element type.
HELPER_ELEMENTS = {"float32": "float", "float16": "float16", "bfloat16": "bfloat16"}
CASES = (
    (1, 1, 1, 1, 0, 0),
    (4, 5, 7, 1, 0, 0),
    (32, 32, 16, 1, 0, 0),
    (33, 35, 17, 2, 0, 0),
    (33, 35, 17, 2, 3, 1),
)
CASES_BY_VARIANT = {
    "plain": CASES,
    "add": ((4, 5, 7, 1, 0, 0), (33, 35, 17, 2, 3, 1)),
    "axpby": ((4, 5, 7, 1, 0, 0), (33, 35, 17, 2, 3, 1)),
    "broadcast": ((4, 5, 7, 4, 0, 0), (33, 35, 17, 4, 3, 1)),
    "broadcast-axpby": ((4, 5, 7, 4, 0, 0), (33, 35, 17, 4, 3, 1)),
    "aligned": ((32, 32, 16, 1, 0, 0), (64, 64, 32, 2, 3, 1)),
}
ORIGINAL_FLAGS = (
    "-std=metal3.2",
    "-fno-fast-math",
    "-Wall",
    "-Wextra",
    "-Wno-c++17-extensions",
    "-Wno-c++20-extensions",
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _loader_access_assertions(entry=REPRESENTATIVE, source=SOURCE):
    """The threadgroup slots each ``BlockLoader::load_unsafe`` writes.

    ``loader.h`` distributes ``reads`` adjacent elements of a tile row to each
    thread, and ``gemm.h`` pads every threadgroup row by 16 bytes.
    """
    assertions = []
    for loader in entry.loaders:
        slots = [slot for lane in loader.slots() for slot in lane]
        assert len(slots) == len(set(slots)) == loader.rows * loader.columns
        assert set(slots) == {
            row * loader.leading + column
            for row in range(loader.rows)
            for column in range(loader.columns)
        }
        assert max(slots) < loader.rows * loader.leading
        assertions.append(
            {
                "source": source,
                "entryPoint": entry.name,
                "function": loader.function,
                "parameter": "self.dst",
                "minimum": min(slots),
                "maximum": max(slots),
            }
        )
    return tuple(assertions)


def _opengl_index_assertions(entry=REPRESENTATIVE, source=SOURCE):
    """GLSL indexes with 32-bit ints, so 64-bit pointer offsets need ranges.

    Device views index MLX buffers, which stay below 2^31 elements; a loader
    reads ``reads`` adjacent elements past its source offset. Each threadgroup
    view reaches only the slots its loader writes, the ranges the DirectX gate
    asserts for the loaders' ``self.dst``.
    """
    element = HELPER_ELEMENTS[entry.dtype]
    reads = max(loader.reads for loader in entry.loaders)
    device = [
        {"expression": f"source.offset + {offset}", "maximum": SIGNED_MAX}
        for offset in range(reads)
    ] + [
        {
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}",
            "maximum": SIGNED_MAX,
        }
        # Batched variants also read batch_strides, an int64_t buffer.
        for access in (f"load_{element}", f"store_{element}", "load_int64_t")
    ]
    # Each tile's ranges name its threadgroup array, because the helpers that
    # read both tiles are shared when the A and B tiles have the same stride.
    workgroup = [
        {
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{tile}",
            "maximum": loader["maximum"],
        }
        for tile, loader in zip(("As", "Bs"), _loader_access_assertions(entry, source))
    ]
    return tuple(
        {"source": source, **item, "minimum": 0} for item in device + workgroup
    )


def _config(root, output, target="metal", entry=REPRESENTATIVE):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (entry.name,)},
        specialization_constants={name: False for name in CONSTANT_IDS},
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: entry.workgroup_size},
        workgroup_access_assertions=(
            _loader_access_assertions(entry) if target != "metal" else ()
        ),
        index_range_assertions=(
            _opengl_index_assertions(entry) if target == "opengl" else ()
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


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_gemm_target_contract_is_explicit(tmp_path, target):
    config = _config(tmp_path, "out", target)
    assert list(config.targets) == [target]
    assert config.workgroup_size_rules == {SOURCE: ("32", "2", "2")}
    if target == "metal":
        assert not config.workgroup_access_assertions
        assert not config.index_range_assertions
        return
    assertions = config.workgroup_access_assertions
    assert [(item.minimum, item.maximum) for item in assertions] == [
        (0, 635),
        (0, 571),
    ]
    assert {item.function for item in assertions} == set(WORKGROUP_LOADERS)
    assert all(
        item.source == SOURCE
        and item.entry_point == ENTRY
        and item.parameter == "self.dst"
        for item in assertions
    )
    ranges = [item.to_json() for item in config.index_range_assertions]
    if target == "directx":
        assert not ranges
        return
    assert ranges == list(_opengl_index_assertions())
    device = [item for item in ranges if item["maximum"] == SIGNED_MAX]
    assert len(device) == 7
    assert all(item["source"] == SOURCE and item["minimum"] == 0 for item in ranges)
    workgroup = {
        item["indexed"]: item["maximum"]
        for item in ranges
        if item["maximum"] != SIGNED_MAX
    }
    assert workgroup == {"crosstl_workgroup_As": 635, "crosstl_workgroup_Bs": 571}
    assert all("function" not in item for item in ranges if "indexed" in item)


def _frozen_values(variant):
    assert variant in CASES_BY_VARIANT
    return {
        "has_batch": variant in {"broadcast", "broadcast-axpby"},
        "use_out_source": variant in {"add", "axpby", "broadcast-axpby"},
        "do_axpby": variant in {"axpby", "broadcast-axpby"},
        "align_M": variant == "aligned",
        "align_N": variant == "aligned",
        "align_K": variant == "aligned",
    }


def _exact(dtype, value):
    """Whether ``value`` is a finite value of ``dtype`` with no rounding."""
    number = Fraction(value)
    if number == 0:
        return True
    if dtype == "float16" and abs(number) > 65504:
        return False
    if number.denominator & (number.denominator - 1):
        return False
    significand = abs(number.numerator)
    while significand % 2 == 0:
        significand //= 2
    return significand < 2 ** PRECISION[dtype]


def _operand(dtype, name, index):
    """Deterministic operands whose products, sums and epilogues stay exact.

    Quarter integers in [-1.5, 1.5] keep every float32 and binary16 result
    exact. bfloat16 keeps 8 significant bits, so its operands are halves in
    [-1, 1] and its addends halves in [-1.5, 1.5].
    """
    numerator = {
        "A": (index * 3 + 1) % 11 - 5,
        "B": (index * 7 + 2) % 13 - 6,
        "C": (index * 5) % 7 - 3,
    }[name]
    if dtype != "bfloat16":
        return numerator / 4
    if name == "C":
        return numerator / 2
    return ((index * 3 + 1) % 5 - 2 if name == "A" else (index * 7 + 2) % 5 - 2) / 2


def _aligned_cases(entry):
    # align_M, align_N and align_K promise whole tiles: one tile, then 64x64x32,
    # which every instantiated tile divides.
    return ((entry.bm, entry.bn, entry.bk, 1, 0, 0), (64, 64, 32, 2, 3, 1))


def _params(case, entry=REPRESENTATIVE):
    """GEMMParams as ``matmul.cpp`` fills them, with A and B in the entry's layout."""
    m, n, k, batches, padding, extra_groups = case
    a_rows, a_columns = (k, m) if entry.transpose_a else (m, k)
    b_rows, b_columns = (n, k) if entry.transpose_b else (k, n)
    lda, ldb, ldd = a_columns + padding, b_columns + padding, n + padding
    params = dict(
        M=m,
        N=n,
        K=k,
        lda=lda,
        ldb=ldb,
        ldd=ldd,
        tiles_n=-(-n // entry.bn),
        tiles_m=-(-m // entry.bm),
        batch_stride_a=a_rows * lda + padding,
        batch_stride_b=b_rows * ldb + padding,
        batch_stride_d=m * ldd + padding,
        swizzle_log=0,
        gemm_k_iterations_aligned=k // entry.bk,
        batch_ndim=0,
    )
    geometry = {
        "workgroupCount": [
            params["tiles_n"] + extra_groups,
            params["tiles_m"] + extra_groups,
            batches,
        ],
        "workgroupSize": list(entry.workgroup_size),
    }
    return params, geometry


def _a_index(entry, params, batch, row, inner):
    offset = batch * params["batch_stride_a"]
    if entry.transpose_a:
        return offset + inner * params["lda"] + row
    return offset + row * params["lda"] + inner


def _b_index(entry, params, batch, inner, column):
    offset = batch * params["batch_stride_b"]
    if entry.transpose_b:
        return offset + column * params["ldb"] + inner
    return offset + inner * params["ldb"] + column


def _products(entry, params, a, b, a_batch, b_batch, row, column):
    total = sum(
        a[_a_index(entry, params, a_batch, row, inner)]
        * b[_b_index(entry, params, b_batch, inner, column)]
        for inner in range(params["K"])
    )
    # The float accumulator then the cast to the output type round nothing.
    assert _exact(entry.dtype, total), (entry.name, total)
    return total


def _case_data(case, entry=REPRESENTATIVE):
    m, n, k, batches, padding, extra_groups = case
    dtype, guard = entry.dtype, GUARDS[entry.dtype]
    params, geometry = _params(case, entry)
    a = [guard] * (batches * params["batch_stride_a"] + 8)
    b = [guard] * (batches * params["batch_stride_b"] + 8)
    expected = [guard] * (batches * params["batch_stride_d"] + 8)
    for batch in range(batches):
        for row in range(m):
            for inner in range(k):
                index = (batch * m + row) * k + inner
                a[_a_index(entry, params, batch, row, inner)] = _operand(
                    dtype, "A", index
                )
        for inner in range(k):
            for col in range(n):
                index = (batch * k + inner) * n + col
                b[_b_index(entry, params, batch, inner, col)] = _operand(
                    dtype, "B", index
                )
        for row in range(m):
            for col in range(n):
                expected[
                    batch * params["batch_stride_d"] + row * params["ldd"] + col
                ] = _products(entry, params, a, b, batch, batch, row, col)
    return a, b, expected, params, geometry


def _variant_data(case, variant, entry=REPRESENTATIVE):
    a, b, expected, params, geometry = _case_data(case, entry)
    m, n, k, batches, padding, _ = case
    dtype, guard = entry.dtype, GUARDS[entry.dtype]
    constants = _frozen_values(variant)
    if constants["align_M"]:
        assert m % entry.bm == 0 and n % entry.bn == 0 and k % entry.bk == 0
    broadcast = constants["has_batch"]
    batch_shape, batch_strides = [1], [0]
    if broadcast:
        assert batches == 4
        # A broadcasts along the first batch axis, B along the second.
        a = a[: 2 * params["batch_stride_a"]] + [guard] * 8
        b = b[: 2 * params["batch_stride_b"]] + [guard] * 8
        params["batch_ndim"] = 2
        batch_shape = [2, 2]
        batch_strides = [0, params["batch_stride_a"], params["batch_stride_b"], 0]
        for batch in range(batches):
            for row in range(m):
                for col in range(n):
                    expected[
                        batch * params["batch_stride_d"] + row * params["ldd"] + col
                    ] = _products(entry, params, a, b, batch % 2, batch // 2, row, col)
    ldc = 2 * n + padding
    stride_c = m * ldc + padding
    addmm = dict(
        ldc=ldc,
        fdc=2,
        batch_stride_c=stride_c,
        alpha=0.5,
        beta=-0.25,
    )
    c = [0.0]
    if constants["use_out_source"]:
        c_batches = 1 if broadcast else batches
        c = [guard] * (c_batches * stride_c + 8)
        for batch in range(c_batches):
            for row in range(m):
                for col in range(n):
                    logical = (batch * m + row) * n + col
                    c[batch * stride_c + row * ldc + col * 2] = _operand(
                        dtype, "C", logical
                    )
        if broadcast:
            batch_strides.extend([0, 0])
        for batch in range(batches):
            for row in range(m):
                for col in range(n):
                    out = batch * params["batch_stride_d"] + row * params["ldd"] + col
                    value = c[
                        (0 if broadcast else batch) * stride_c + row * ldc + col * 2
                    ]
                    expected[out] = (
                        addmm["alpha"] * expected[out] + addmm["beta"] * value
                        if constants["do_axpby"]
                        else expected[out] + value
                    )
                    assert _exact(dtype, addmm["beta"] * value)
                    assert _exact(dtype, expected[out]), (entry.name, expected[out])
    resources = dict(
        A=a,
        B=b,
        C=c,
        D=[guard] * len(expected),
        params=params,
        addmm_params=addmm,
        batch_shape=batch_shape,
        batch_strides=batch_strides,
    )
    return resources, expected, geometry


def _storage(target, binding, dtype, values):
    """Logical values in the storage the descriptor reflects for ``binding``."""
    layout = binding["scalarLayout"]
    if dtype == "float32":
        assert layout["elementType"] == "float32" and not layout.get("storageEncoding")
        return {"dtype": "float32", "shape": [len(values)], "values": list(values)}
    value = _runtime_value(target, dtype, values)
    # DirectX stores 16-bit floating buffers as uint16 words.
    physical = "uint16" if target == "directx" else value["dtype"]
    assert layout["elementType"] == physical, (binding["name"], layout)
    storage = layout.get("storageEncoding")
    if storage is not None:
        assert storage == {
            "encoding": value["encoding"],
            "logicalElementType": value["dtype"],
        }
    return value


def _dispatch_values(bindings, case, variant, entry=REPRESENTATIVE, target="metal"):
    resources, expected, geometry = _variant_data(case, variant, entry)
    values = {
        name: _storage(target, bindings[name], entry.dtype, resources[name])
        for name in ("A", "B", "C", "D")
    }
    values.update(
        {
            name: {"dtype": dtype, "shape": [len(items)], "values": items}
            for name, (dtype, items) in {
                "params": (
                    "uint32",
                    pack_storage_records(
                        bindings["params"]["scalarLayout"], [resources["params"]]
                    ),
                ),
                "addmm_params": (
                    "uint32",
                    pack_storage_records(
                        bindings["addmm_params"]["scalarLayout"],
                        [resources["addmm_params"]],
                    ),
                ),
                "batch_shape": ("int32", resources["batch_shape"]),
                "batch_strides": ("int64", resources["batch_strides"]),
            }.items()
        }
    )
    inputs = {bindings[name]["name"]: value for name, value in values.items()}
    outputs = {
        bindings["D"]["name"]: _storage(target, bindings["D"], entry.dtype, expected)
    }
    return inputs, outputs, geometry


def _original_outputs(request, source, module, work, specializations, entry):
    executor = _executor("metal")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
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
            entry_point=entry,
            loaded_artifact=None,
            constants=constants,
        )
        actual = executor.runtime_adapter.runtime.dispatch(None, state, control)
        module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
        assert state.details["metalRuntime"]["librarySHA256"] == module_hash
        _write_json(
            work / "original-evidence.json",
            {
                "outputs": actual,
                "request": control.to_json(),
                "sourceCommit": MLX_COMMIT,
                "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "moduleSha256": module_hash,
                "runtime": state.details["metalRuntime"],
                "compileFlags": ORIGINAL_FLAGS,
            },
        )
        return actual
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


@pytest.mark.parametrize("case", CASES)
def test_gemm_case_shapes_and_guards(case):
    m, n, k, batches, padding, extra_groups = case
    a, b, expected, params, geometry = _case_data(case)
    assert len(a) == batches * (m * (k + padding) + padding) + 8
    assert len(b) == batches * (k * (n + padding) + padding) + 8
    assert len(expected) == batches * (m * (n + padding) + padding) + 8
    live = {
        batch * params["batch_stride_d"] + row * params["ldd"] + col
        for batch in range(batches)
        for row in range(m)
        for col in range(n)
    }
    assert sum(value != GUARD for value in expected) == m * n * batches
    assert all(
        expected[index] == GUARD for index in range(len(expected)) if index not in live
    )
    assert geometry["workgroupCount"] == [
        (n + 31) // 32 + extra_groups,
        (m + 31) // 32 + extra_groups,
        batches,
    ]
    assert geometry["workgroupSize"] == [32, 2, 2]
    assert params["gemm_k_iterations_aligned"] == k // 16


@pytest.mark.parametrize(
    "variant,case",
    [(variant, case) for variant, cases in CASES_BY_VARIANT.items() for case in cases],
)
def test_gemm_variant_metadata_and_guards(variant, case):
    resources, expected, geometry = _variant_data(case, variant)
    m, n, k, batches, padding, _ = case
    constants = _frozen_values(variant)
    params = resources["params"]
    assert set(constants) == set(CONSTANT_IDS)
    assert all(type(value) is bool for value in constants.values())
    assert sum(value != GUARD for value in expected) == m * n * batches
    assert resources["D"] == [GUARD] * len(expected)
    for batch in range(batches):
        start = batch * params["batch_stride_d"]
        for row in range(m):
            gap = start + row * params["ldd"] + n
            assert expected[gap : gap + padding] == [GUARD] * padding
        gap = start + m * params["ldd"]
        assert expected[gap : gap + padding] == [GUARD] * padding
    assert expected[-8:] == [GUARD] * 8
    assert geometry["workgroupCount"][2] == batches
    if constants["has_batch"]:
        assert resources["batch_shape"] == [2, 2]
        assert resources["batch_strides"][:4] == [
            0,
            params["batch_stride_a"],
            params["batch_stride_b"],
            0,
        ]
        assert params["batch_ndim"] == 2
        assert len(resources["A"]) == 2 * params["batch_stride_a"] + 8
        assert len(resources["B"]) == 2 * params["batch_stride_b"] + 8
        assert resources["batch_strides"][4:] == (
            [0, 0] if constants["use_out_source"] else []
        )
    if constants["use_out_source"]:
        addmm = resources["addmm_params"]
        assert addmm["fdc"] == 2 and addmm["ldc"] == 2 * n + padding
        assert addmm["alpha"] == 0.5 and addmm["beta"] == -0.25
        assert resources["C"][-8:] == [GUARD] * 8
    if variant == "aligned":
        assert constants["align_M"] and constants["align_N"] and constants["align_K"]
        assert m % 32 == n % 32 == k % 16 == 0


@pytest.mark.parametrize(
    "variant,reference", [("plain", 1.0), ("add", 0.25), ("axpby", 0.6875)]
)
def test_gemm_epilogue_scalar_reference(variant, reference):
    resources, expected, _ = _variant_data((1, 1, 1, 1, 0, 0), variant)
    assert resources["A"][0] == resources["B"][0] == -1.0
    assert expected == [reference] + [GUARD] * 8


@pytest.mark.parametrize(
    "variant,reference",
    [
        ("broadcast", [1.0, 0.25, -0.75, -0.1875]),
        ("broadcast-axpby", [0.6875, 0.3125, -0.1875, 0.09375]),
    ],
)
def test_gemm_broadcast_scalar_reference(variant, reference):
    resources, expected, _ = _variant_data((1, 1, 1, 4, 0, 0), variant)
    assert resources["batch_shape"] == [2, 2]
    assert expected == reference + [GUARD] * 8


def _tile_read_slots():
    """Shared-array slots ``BlockMMA::mma`` reads, following ``steel/gemm/mma.h``.

    ``BaseMMAFrag::get_coord`` gives each lane an 8-by-8 fragment coordinate
    and two adjacent elements; simdgroups split the 32-by-32 output two by two
    and step K by fragments of 8.
    """
    a_slots, b_slots = set(), set()
    lda, ldb = 16 + 4, 32 + 4
    for group in range(4):
        tm, tn = 8 * (group // 2), 8 * (group % 2)
        for lane in range(32):
            qid = lane // 4
            sm = (qid & 4) + (lane // 2) % 4
            sn = (qid & 2) * 2 + (lane % 2) * 2
            for kk in range(0, 16, 8):
                for tile in range(2):
                    for element in range(2):
                        a_slots.add((tm + sm + 16 * tile) * lda + sn + kk + element)
                        b_slots.add((sm + kk) * ldb + tn + sn + 16 * tile + element)
    return a_slots, b_slots


def test_gemm_tile_reads_stay_in_the_loader_slots():
    slots = []
    for rows, columns in WORKGROUP_LOADERS.values():
        slots.append(
            {row * (columns + 4) + col for row in range(rows) for col in range(columns)}
        )
    reads = _tile_read_slots()
    assert [len(item) for item in reads] == [32 * 16, 16 * 32]
    assert reads[0] == slots[0] and reads[1] == slots[1]
    workgroup = [
        item for item in _opengl_index_assertions() if item["maximum"] != SIGNED_MAX
    ]
    assert [(item["minimum"], item["maximum"]) for item in workgroup] == [
        (min(reads[0]), max(reads[0])),
        (min(reads[1]), max(reads[1])),
    ]


def test_gemm_model_derives_the_representative_loaders():
    loaders = REPRESENTATIVE.loaders
    assert [loader.function for loader in loaders] == list(WORKGROUP_LOADERS)
    assert [(loader.rows, loader.columns) for loader in loaders] == list(
        WORKGROUP_LOADERS.values()
    )
    assert [loader.leading for loader in loaders] == [16 + 4, 32 + 4]
    assert _aligned_cases(REPRESENTATIVE) == CASES_BY_VARIANT["aligned"]
    assert model.BY_NAME[ENTRY] is REPRESENTATIVE
    assert len(model.BY_NAME) == len(model.entries()) == 4 * 6 * 4


@pytest.mark.parametrize("entry", model.entries(), ids=lambda entry: entry.name)
def test_gemm_tile_reads_cover_exactly_the_loader_slots(entry):
    """``BlockMMA::mma`` reads exactly the slots the loaders write, per entry."""
    written = [
        {slot for lane in loader.slots() for slot in lane} for loader in entry.loaders
    ]
    assert [len(item) for item in written] == [entry.bm * entry.bk, entry.bk * entry.bn]
    assert list(entry.tile_reads()) == written
    assertions = _loader_access_assertions(entry)
    assert [(item["minimum"], item["maximum"]) for item in assertions] == [
        (min(item), max(item)) for item in written
    ]


def test_gemm_scalar_reference():
    a, b, expected, _, _ = _case_data(CASES[0])
    assert a[0] == -1.0 and b[0] == -1.0
    assert expected == [1.0] + [GUARD] * 8


def prepare_gemm_source(tmp_path_factory, require_env):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(require_env) != "1":
        pytest.skip(f"set {require_env}=1 for the pinned GEMM native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    tmp_path = tmp_path_factory.mktemp("gemm-original")
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
        header = root / "mlx/backend/metal/kernels/steel/gemm" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest
    original = None
    if target == "metal":
        run(
            ["xcrun", "--sdk", "macosx", "metal", "--version"],
            tmp_path,
            "metal-version",
        )
        original = compile_metal(
            source, root, tmp_path / "original", flags=ORIGINAL_FLAGS
        )
    else:
        tool = GATE_TOOLS[target][0]
        run([tool, "--version"], tmp_path, f"{tool}-version")
    return root, original, target


@pytest.fixture(scope="module")
def gemm_source(tmp_path_factory):
    return prepare_gemm_source(tmp_path_factory, REQUIRE_ENV)


def execute_entry(tmp_path, gemm_source, entry, variant, cases, scope):
    """Translate ``entry`` with ``variant``'s frozen constants and run ``cases``.

    Every case must equal the exact reference on the target; on Metal it must
    also equal the original MLX source compiled and dispatched with the same
    specializations and buffers.
    """
    root, original, target = gemm_source
    source = root / SOURCE
    specializations = _frozen_values(variant)
    with tempfile.TemporaryDirectory(prefix=".current-gemm-", dir=root) as directory:
        work = Path(directory)
        try:
            config = replace(
                _config(root, work.name, target, entry),
                specialization_constants=specializations,
            )
            report = translate_project(config, format_output=False)
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert len(payload["artifacts"]) == 1
            assert payload["project"]["indexRangeAssertions"] == (
                list(_opengl_index_assertions(entry)) if target == "opengl" else []
            )
            assert payload["project"]["workgroupAccessAssertions"] == (
                list(_loader_access_assertions(entry)) if target != "metal" else []
            )
            descriptor, package = _prepare_native_package(report, tmp_path)
            constants = descriptor["specializationConstants"]
            assert len(constants) == len(CONSTANT_IDS)
            assert {item["name"]: item["id"] for item in constants} == CONSTANT_IDS
            assert all(
                item["frozen"] is True
                and item["value"] is specializations[item["name"]]
                for item in constants
            )
            bindings = {
                item["provenance"]["sourceResource"]["parameter"]: item
                for item in descriptor["bindings"]
            }
            assert len(descriptor["bindings"]) == len(bindings) == 8
            assert set(bindings) == {
                "A",
                "B",
                "C",
                "D",
                "params",
                "addmm_params",
                "batch_shape",
                "batch_strides",
            }
            assert bindings["params"]["scalarLayout"]["elementStrideBytes"] == 72
            assert len(bindings["params"]["scalarLayout"]["structMembers"]) == 14
            assert bindings["addmm_params"]["scalarLayout"]["elementStrideBytes"] == 24
            results = []
            for case in cases:
                case_work = tmp_path / ("case-" + "-".join(map(str, case)))
                case_work.mkdir()
                inputs, outputs, geometry = _dispatch_values(
                    bindings, case, variant, entry, target
                )
                _write_json(
                    case_work / "inputs.json",
                    {"inputs": inputs, "outputs": outputs, "geometry": geometry},
                )
                request = build_native_loader_dispatch_request(
                    descriptor,
                    package,
                    inputs,
                    outputs,
                    geometry,
                    expected_target=target,
                )
                assert not request.execution_plan.diagnostics
                assert not request.adapter_contract.specialization_constants
                _execute(
                    request,
                    outputs,
                    case_work,
                    validate=(
                        (
                            lambda path, destination, target: compile_metal(
                                path,
                                root,
                                destination,
                                flags=("-std=metal3.1", "-fno-fast-math"),
                            )
                        )
                        if target == "metal"
                        else None
                    ),
                )
                generated = json.loads((case_work / "evidence.json").read_text())
                record = generated["records"]["generated"]
                if target == "metal":
                    original_outputs = _original_outputs(
                        request,
                        source,
                        original,
                        case_work,
                        specializations,
                        entry.name,
                    )
                    assert original_outputs == record["outputs"]
                    assert (
                        record["details"]["metalRuntime"]["librarySHA256"]
                        == record["moduleSha256"]
                    )
                    assert (
                        record["details"]["metalRuntime"]["threadExecutionWidth"] == 32
                    )
                assert not record["request"]["constants"]
                assert record["outputs"] == outputs
                results.append(
                    {
                        "shape": list(case[:4]),
                        "padding": case[4],
                        "extraWorkgroups": case[5],
                        "checkedValues": len(next(iter(outputs.values()))["values"]),
                        "equalOriginalMetal": True if target == "metal" else None,
                        "equalExactReference": True,
                    }
                )
                _write_json(
                    tmp_path / "parity.json",
                    {
                        "sourceCommit": MLX_COMMIT,
                        "target": target,
                        "entryPoint": entry.name,
                        "variant": variant,
                        "specializations": specializations,
                        "sourceSha256": SOURCE_SHA256,
                        "headerSha256": HEADER_HASHES,
                        "cases": results,
                        "complete": len(results) == len(cases),
                        "scope": scope,
                    },
                )
        finally:
            shutil.copytree(work, tmp_path / "translation", dirs_exist_ok=True)


@pytest.mark.parametrize("variant", CASES_BY_VARIANT)
def test_current_gemm_executes(tmp_path, gemm_source, variant):
    execute_entry(
        tmp_path,
        gemm_source,
        REPRESENTATIVE,
        variant,
        CASES_BY_VARIANT[variant],
        "One float32 NN entry with explicit frozen constants; not the upstream suite or full GEMM family.",
    )
