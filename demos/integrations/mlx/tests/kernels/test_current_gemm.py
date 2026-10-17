"""Translate a complete pinned MLX GEMM entry and compare native results."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
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
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal"
SOURCE_SHA256 = "6c6e67f5c7fc6b378ca2c849c36a79d696c93e65b3a60d3f5b1f13928c79029e"
ENTRY = "steel_gemm_fused_nn_float32_float32_bm32_bn32_bk16_wm2_wn2"
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
GUARD = -98765.0
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


def _loader_access_assertions():
    assertions = []
    for function, (rows, columns) in WORKGROUP_LOADERS.items():
        # loader.h distributes four adjacent floats to each of 128 threads.
        # gemm.h pads each row by 16 bytes for this non-transposed float entry.
        leading = columns + 4
        reads = rows * columns // 128
        thread_columns = columns // reads
        thread_rows = 128 // thread_columns
        slots = [
            (lane // thread_columns + row) * leading
            + reads * (lane % thread_columns)
            + element
            for lane in range(128)
            for row in range(0, rows, thread_rows)
            for element in range(reads)
        ]
        assert len(slots) == len(set(slots)) == rows * columns
        assert set(slots) == {
            row * leading + column for row in range(rows) for column in range(columns)
        }
        assert max(slots) < rows * leading
        assertions.append(
            {
                "source": SOURCE,
                "entryPoint": ENTRY,
                "function": function,
                "parameter": "self.dst",
                "minimum": min(slots),
                "maximum": max(slots),
            }
        )
    return tuple(assertions)


def _config(root, output, target="metal"):
    assert target in {"metal", "directx"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (ENTRY,)},
        specialization_constants={name: False for name in CONSTANT_IDS},
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: (32, 2, 2)},
        workgroup_access_assertions=(
            _loader_access_assertions() if target == "directx" else ()
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
                },
            }
        },
    )


@pytest.mark.parametrize("target", ["metal", "directx"])
def test_gemm_target_contract_is_explicit(tmp_path, target):
    config = _config(tmp_path, "out", target)
    assert list(config.targets) == [target]
    assert config.workgroup_size_rules == {SOURCE: ("32", "2", "2")}
    assert not config.index_range_assertions
    if target == "directx":
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
    else:
        assert not config.workgroup_access_assertions


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


def _case_data(case):
    m, n, k, batches, padding, extra_groups = case
    lda, ldb, ldd = k + padding, n + padding, n + padding
    stride_a, stride_b, stride_d = (
        m * lda + padding,
        k * ldb + padding,
        m * ldd + padding,
    )
    a = [GUARD] * (batches * stride_a + 8)
    b = [GUARD] * (batches * stride_b + 8)
    expected = [GUARD] * (batches * stride_d + 8)
    for batch in range(batches):
        for row in range(m):
            for inner in range(k):
                index = (batch * m + row) * k + inner
                a[batch * stride_a + row * lda + inner] = ((index * 3 + 1) % 11 - 5) / 4
        for inner in range(k):
            for col in range(n):
                index = (batch * k + inner) * n + col
                b[batch * stride_b + inner * ldb + col] = ((index * 7 + 2) % 13 - 6) / 4
        for row in range(m):
            for col in range(n):
                # Quarter-integer operands keep every product and sum exact in float32.
                expected[batch * stride_d + row * ldd + col] = sum(
                    a[batch * stride_a + row * lda + inner]
                    * b[batch * stride_b + inner * ldb + col]
                    for inner in range(k)
                )
    params = dict(
        M=m,
        N=n,
        K=k,
        lda=lda,
        ldb=ldb,
        ldd=ldd,
        tiles_n=(n + 31) // 32,
        tiles_m=(m + 31) // 32,
        batch_stride_a=stride_a,
        batch_stride_b=stride_b,
        batch_stride_d=stride_d,
        swizzle_log=0,
        gemm_k_iterations_aligned=k // 16,
        batch_ndim=0,
    )
    geometry = {
        "workgroupCount": [
            params["tiles_n"] + extra_groups,
            params["tiles_m"] + extra_groups,
            batches,
        ],
        "workgroupSize": [32, 2, 2],
    }
    return a, b, expected, params, geometry


def _variant_data(case, variant):
    a, b, expected, params, geometry = _case_data(case)
    m, n, k, batches, padding, _ = case
    constants = _frozen_values(variant)
    if constants["align_M"]:
        assert m % 32 == n % 32 == k % 16 == 0
    broadcast = constants["has_batch"]
    batch_shape, batch_strides = [1], [0]
    if broadcast:
        assert batches == 4
        # A broadcasts along the first batch axis, B along the second.
        a = a[: 2 * params["batch_stride_a"]] + [GUARD] * 8
        b = b[: 2 * params["batch_stride_b"]] + [GUARD] * 8
        params["batch_ndim"] = 2
        batch_shape = [2, 2]
        batch_strides = [0, params["batch_stride_a"], params["batch_stride_b"], 0]
        for batch in range(batches):
            for row in range(m):
                for col in range(n):
                    expected[
                        batch * params["batch_stride_d"] + row * params["ldd"] + col
                    ] = sum(
                        a[
                            (batch % 2) * params["batch_stride_a"]
                            + row * params["lda"]
                            + inner
                        ]
                        * b[
                            (batch // 2) * params["batch_stride_b"]
                            + inner * params["ldb"]
                            + col
                        ]
                        for inner in range(k)
                    )
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
        c = [GUARD] * (c_batches * stride_c + 8)
        for batch in range(c_batches):
            for row in range(m):
                for col in range(n):
                    logical = (batch * m + row) * n + col
                    c[batch * stride_c + row * ldc + col * 2] = (
                        (logical * 5) % 7 - 3
                    ) / 4
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
    resources = dict(
        A=a,
        B=b,
        C=c,
        D=[GUARD] * len(expected),
        params=params,
        addmm_params=addmm,
        batch_shape=batch_shape,
        batch_strides=batch_strides,
    )
    return resources, expected, geometry


def _dispatch_values(bindings, case, variant):
    resources, expected, geometry = _variant_data(case, variant)
    values = {name: ("float32", resources[name]) for name in ("A", "B", "C", "D")}
    values.update(
        {
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
        }
    )
    inputs = {
        bindings[name]["name"]: {"dtype": dtype, "shape": [len(items)], "values": items}
        for name, (dtype, items) in values.items()
    }
    outputs = {
        bindings["D"]["name"]: {
            "dtype": "float32",
            "shape": [len(expected)],
            "values": expected,
        }
    }
    return inputs, outputs, geometry


def _original_outputs(request, source, module, work, specializations):
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
            entry_point=ENTRY,
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


def test_gemm_scalar_reference():
    a, b, expected, _, _ = _case_data(CASES[0])
    assert a[0] == -1.0 and b[0] == -1.0
    assert expected == [1.0] + [GUARD] * 8


@pytest.fixture(scope="module")
def gemm_source(tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned GEMM native gate")
    target = os.environ.get(
        "CROSTL_MLX_CURRENT_TARGET", "directx" if sys.platform == "win32" else "metal"
    )
    assert target in {"metal", "directx"}
    assert sys.platform == {"metal": "darwin", "directx": "win32"}[target]
    assert shutil.which("xcrun" if target == "metal" else "dxc")
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
        run(["dxc", "--version"], tmp_path, "dxc-version")
    return root, original, target


@pytest.mark.parametrize("variant", CASES_BY_VARIANT)
def test_current_gemm_executes(tmp_path, gemm_source, variant):
    root, original, target = gemm_source
    source = root / SOURCE
    specializations = _frozen_values(variant)
    with tempfile.TemporaryDirectory(prefix=".current-gemm-", dir=root) as directory:
        work = Path(directory)
        try:
            config = replace(
                _config(root, work.name, target),
                specialization_constants=specializations,
            )
            report = translate_project(config, format_output=False)
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert len(payload["artifacts"]) == 1
            assert payload["project"]["indexRangeAssertions"] == []
            assert payload["project"]["workgroupAccessAssertions"] == (
                list(_loader_access_assertions()) if target == "directx" else []
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
            for case in CASES_BY_VARIANT[variant]:
                case_work = tmp_path / ("case-" + "-".join(map(str, case)))
                case_work.mkdir()
                inputs, outputs, geometry = _dispatch_values(bindings, case, variant)
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
                        request, source, original, case_work, specializations
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
                        "entryPoint": ENTRY,
                        "variant": variant,
                        "specializations": specializations,
                        "sourceSha256": SOURCE_SHA256,
                        "headerSha256": HEADER_HASHES,
                        "cases": results,
                        "complete": len(results) == len(CASES_BY_VARIANT[variant]),
                        "scope": (
                            "One float32 NN entry with explicit frozen constants; not the upstream suite or full GEMM family."
                        ),
                    },
                )
        finally:
            shutil.copytree(work, tmp_path / "translation", dirs_exist_ok=True)
