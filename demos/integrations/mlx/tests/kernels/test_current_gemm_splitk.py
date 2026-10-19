"""Execute the pinned MLX split-K GEMM entries natively.

``steel_gemm_splitk.metal`` instantiates ``gemm_splitk`` for four element
types, four tile shapes, four transpositions and four alignment pairs, plus
the ``gemm_splitk_accum`` kernels that sum the partitions. This gate
translates every float32, float16 and bfloat16 entry and runs it with the
partition plan ``matmul.cpp`` dispatches, comparing every partition word,
guards included, with an exact reference; on Metal it also compares with the
original source. complex64 entries are not covered: their ``BlockMMA``
partial specialization is tracked by #2261.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    pack_storage_records,
    translate_project,
)
from crosstl.project.runtime_verification import RuntimeExecutionState
from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused
from demos.integrations.mlx.tests.kernels import gemm_splitk_model as model
from demos.integrations.mlx.tests.kernels.test_current_gemm import (
    GATE_TOOLS,
    GUARDS,
    HEADER_HASHES,
    HELPER_ELEMENTS,
    MLX_COMMIT,
    ORIGINAL_FLAGS,
    PLATFORM_TARGETS,
    SIGNED_MAX,
    _exact,
    _loader_access_assertions,
    _operand,
    _storage,
    _write_json,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.metal"
SOURCE_SHA256 = "6c3f93a9f55d60ec8e6bd14e1b3e88b37ec65185b424042f2fa6436b397aeaed"
SPLITK_HEADERS = {
    **HEADER_HASHES,
    "kernels/steel_gemm_splitk.h": (
        "2ada2a7ea548f6c4b0e7962b4a2f163e8a212620ecc46e078a06647df12b0635"
    ),
    "params.h": "6711e72e32310eafb088895d5a6d20fcfca814158f5b38eb72b6e69d9d773e43",
    "transforms.h": "73487e29beab5af9478bc931c49b5e8d66937c2fa2651499867f47ac5529afc2",
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMM_SPLITK"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_GEMM_SPLITK_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_GEMM_SPLITK_SHARD_COUNT"
# Seven is coprime with the four alignments, transpositions and shapes and
# with the three types, so every shard mixes all of them.
CI_SHARD_COUNT = 7
REAL_TYPES = ("float16", "bfloat16", "float32")
ENTRIES = tuple(entry for entry in model.entries() if entry.dtype in REAL_TYPES)
ACCUM_ENTRIES = tuple(
    entry for entry in model.accum_entries() if entry.dtype in REAL_TYPES
)
ACCUMULATOR = "float32"
# (M, N, K, padding, extraWorkgroups) per (MN_aligned, K_aligned). Padding
# widens A's and B's leading dimensions; extra workgroups exercise the
# early return. Unaligned shapes give partial tiles in M and N, K tails, and
# partitions with no whole tile; 17x40x100 splits into four partitions.
CASES = {
    (False, False): ((4, 5, 7, 0, 0), (33, 35, 70, 3, 1), (17, 40, 100, 1, 0)),
    (False, True): ((4, 5, 32, 0, 1), (33, 35, 64, 3, 1), (17, 40, 96, 1, 0)),
    (True, False): ((1, 1, 7, 0, 0), (2, 2, 70, 3, 1)),
    (True, True): ((1, 1, 16, 0, 0), (2, 2, 96, 3, 1)),
}
ACCUM_WORKGROUP = (4, 2, 1)
# (indexed storage, GLSL offset) for the accumulation kernels' 64-bit
# offsets; the pointer rebases are not subscripts, so they name no storage.
ACCUM_INDEX_EXPRESSIONS = (
    (None, "gid.x + gid.y * uint64(ldd)"),
    (None, "gid.x * uint64(fdc) + gid.y * uint64(ldc)"),
    ("C_split", "offset"),
    ("C_split", "C_split + offset"),
    ("D", "D + 0"),
    ("C", "C + 0"),
)
# (M, N, partitions); the accumulation kernels have no bounds check, so the
# grid covers exactly M x N threads.
ACCUM_CASES = ((2, 4, 2), (6, 8, 4), (34, 36, 3))
SCOPE = (
    "Every float32, float16 and bfloat16 steel_gemm_splitk entry and its "
    "accumulation kernels with matmul.cpp's partition plan; complex64 entries "
    "and the upstream suite are not covered."
)
TARGET_OPTIONS = {
    "directx": {
        "cooperative_matrix_software_lowering": True,
        "software_subgroup_width": 32,
        "relative_wave_shuffle_out_of_range": "self",
    },
    "opengl": {
        "cooperative_matrix_software_lowering": True,
        "software_subgroup_width": 32,
    },
}


def _shape(entry, case):
    """The case's shape, scaled to whole tiles for MN-aligned entries."""
    m, n, k, padding, extra = case
    if entry.mn_aligned:
        m, n = m * entry.bm, n * entry.bn
    return m, n, k, padding, extra


def _cases(entry):
    return tuple(
        _shape(entry, case) for case in CASES[entry.mn_aligned, entry.k_aligned]
    )


def _selected_entries():
    """This shard's ``gemm_splitk`` entries and accumulation kernels.

    The accumulation kernels are small, so shard 0 runs all of them.
    """
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES, ACCUM_ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"split-K shard {index} is outside [0, {count})")
    return ENTRIES[index::count], ACCUM_ENTRIES if index == 0 else ()


SELECTED_ENTRIES, SELECTED_ACCUM_ENTRIES = _selected_entries()


def _index_assertions(entry):
    """GLSL index ranges: MLX buffers stay below 2^31 elements.

    A loader reads ``reads`` adjacent elements past its source offset; the
    threadgroup views reach only the slots their loaders write.
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
        for access in sorted({f"load_{element}", "store_float"})
    ]
    workgroup = [
        {
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{tile}",
            "maximum": loader["maximum"],
        }
        for tile, loader in zip(("As", "Bs"), _loader_access_assertions(entry, SOURCE))
    ]
    return tuple(
        {"source": SOURCE, **item, "minimum": 0} for item in device + workgroup
    )


def _accum_index_assertions():
    """GLSL index ranges for the accumulation kernels' 64-bit offsets.

    ``C_split``, ``D`` and ``C`` are MLX buffers, so every element offset
    stays below 2^31; the loop's ``offset`` is ``i * partition_stride`` for
    ``i < k_partitions``, inside ``C_split``.
    """
    return tuple(
        {
            "source": SOURCE,
            "expression": expression,
            **({"indexed": indexed} if indexed else {}),
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for indexed, expression in ACCUM_INDEX_EXPRESSIONS
    )


def _config(root, output, target, name, workgroup):
    entry = model.BY_NAME.get(name)
    accesses = (
        _loader_access_assertions(entry, SOURCE)
        if entry is not None and target != "metal"
        else ()
    )
    indices = ()
    if target == "opengl":
        indices = (
            _index_assertions(entry) if entry is not None else _accum_index_assertions()
        )
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (name,)},
        workgroup_size_rules={SOURCE: workgroup},
        workgroup_access_assertions=accesses,
        index_range_assertions=indices,
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                # The accumulation kernels use no simdgroup operations.
                "target_options": TARGET_OPTIONS if entry is not None else {},
            }
        },
    )


def _layout(entry, m, n, k, padding):
    a_rows, a_columns = (k, m) if entry.transpose_a else (m, k)
    b_rows, b_columns = (n, k) if entry.transpose_b else (k, n)
    return a_rows, a_columns + padding, b_rows, b_columns + padding


def _case_data(entry, case):
    """Operands, GEMMSpiltKParams, geometry and the expected partitions."""
    m, n, k, padding, extra = case
    dtype, guard = entry.dtype, GUARDS[entry.dtype]
    a_rows, lda, b_rows, ldb = _layout(entry, m, n, k, padding)
    partitions, iterations, size = model.partition_plan(m, n, k, entry.bk)
    a = [guard] * (a_rows * lda + 8)
    b = [guard] * (b_rows * ldb + 8)
    a_logical = [
        [_operand(dtype, "A", row * k + inner) for inner in range(k)]
        for row in range(m)
    ]
    b_logical = [
        [_operand(dtype, "B", inner * n + col) for col in range(n)]
        for inner in range(k)
    ]
    for row in range(m):
        for inner in range(k):
            index = inner * lda + row if entry.transpose_a else row * lda + inner
            a[index] = a_logical[row][inner]
    for inner in range(k):
        for col in range(n):
            index = col * ldb + inner if entry.transpose_b else inner * ldb + col
            b[index] = b_logical[inner][col]
    expected = [GUARDS["float32"]] * (partitions * m * n + 8)
    ranges = model.partition_ranges(entry, k, partitions, size)
    for partition, (start, stop) in enumerate(ranges):
        for row in range(m):
            for col in range(n):
                total = sum(
                    a_logical[row][inner] * b_logical[inner][col]
                    for inner in range(start, stop)
                )
                assert _exact(ACCUMULATOR, total), (entry.name, total)
                expected[partition * m * n + row * n + col] = total
    params = dict(
        M=m,
        N=n,
        K=k,
        lda=lda,
        ldb=ldb,
        ldc=n,
        tiles_n=-(-n // entry.bn),
        tiles_m=-(-m // entry.bm),
        split_k_partitions=partitions,
        split_k_partition_stride=m * n,
        split_k_partition_size=size,
        swizzle_log=0,
        gemm_k_iterations_aligned=iterations,
    )
    geometry = {
        "workgroupCount": [
            params["tiles_n"] + extra,
            params["tiles_m"] + extra,
            partitions,
        ],
        "workgroupSize": list(entry.workgroup_size),
    }
    return (
        {"A": a, "B": b, "C": [GUARDS["float32"]] * len(expected)},
        params,
        expected,
        geometry,
    )


def _accum_operand(dtype, partition, index):
    numerator = (partition * 7 + index * 3) % 9 - 4
    return numerator / (2 if dtype == "bfloat16" else 4)


def _accum_data(entry, case):
    """C_split partitions, the axpby addend and the expected output."""
    m, n, partitions = case
    dtype, guard = entry.dtype, GUARDS[entry.dtype]
    stride = m * n
    split = [GUARDS["float32"]] * (partitions * stride + 8)
    for partition in range(partitions):
        for index in range(stride):
            split[partition * stride + index] = _accum_operand(dtype, partition, index)
    ldc, fdc, alpha, beta = 2 * n + 1, 2, 0.5, -0.25
    addend = [guard] * (m * ldc + 8)
    expected = [guard] * (stride + 8)
    for row in range(m):
        for col in range(n):
            index = row * n + col
            total = sum(split[p * stride + index] for p in range(partitions))
            value = total
            if entry.axpby:
                addend[row * ldc + col * fdc] = _operand(dtype, "C", index)
                value = alpha * total + beta * addend[row * ldc + col * fdc]
            assert _exact(dtype, value), (entry.name, value)
            expected[index] = value
    resources = {
        "C_split": ("float32", split),
        "D": (dtype, [guard] * len(expected)),
        "k_partitions": ("int32", [partitions]),
        "partition_stride": ("int32", [stride]),
        "ldd": ("int32", [n]),
    }
    if entry.axpby:
        resources.update(
            C=(dtype, addend),
            ldc=("int32", [ldc]),
            fdc=("int32", [fdc]),
            alpha=("float32", [alpha]),
            beta=("float32", [beta]),
        )
    geometry = {
        "workgroupCount": [n // ACCUM_WORKGROUP[0], m // ACCUM_WORKGROUP[1], 1],
        "workgroupSize": list(ACCUM_WORKGROUP),
    }
    return resources, expected, geometry


def _bindings(descriptor):
    return {
        item["provenance"]["sourceResource"]["parameter"]: item
        for item in descriptor["bindings"]
    }


def test_splitk_names_every_entry():
    assert len(model.entries()) == 4 * 4 * 4 * 4
    assert len(model.accum_entries()) == 8
    assert len(set(model.BY_NAME)) == 256 and len(set(model.ACCUM_BY_NAME)) == 8
    assert len(ENTRIES) == 192 and len(ACCUM_ENTRIES) == 6
    assert Counter(entry.dtype for entry in ENTRIES) == {
        dtype: 64 for dtype in REAL_TYPES
    }
    assert Counter((e.mn_aligned, e.k_aligned) for e in ENTRIES) == {
        alignment: 48 for alignment in model.ALIGNMENTS
    }


def test_splitk_host_names_resolve():
    """``matmul.cpp`` spells the kernel name the way the model does."""
    for entry in model.entries():
        assert entry.name.startswith(
            f"steel_gemm_splitk_{entry.transpose}_{entry.iname}_{entry.oname}_bm"
        )
    assert "steel_gemm_splitk_accum_float16_float32_axbpy" in model.ACCUM_BY_NAME


def test_splitk_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    accum = [e.name for _, shard in shards for e in shard]
    assert accum == [e.name for e in shards[0][1]] == [e.name for e in ACCUM_ENTRIES]
    shards = [shard for shard, _ in shards]
    assert sorted(e.name for shard in shards for e in shard) == sorted(
        e.name for e in ENTRIES
    )
    for shard in shards:
        assert {e.dtype for e in shard} == set(REAL_TYPES)
        assert {(e.mn_aligned, e.k_aligned) for e in shard} == set(model.ALIGNMENTS)
        assert {e.transpose for e in shard} == {t for t, _, _ in fused.TRANSPOSES}
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize(
    ("m", "n", "k", "plan"),
    [
        (4, 5, 7, (2, 0, 0)),
        (33, 35, 70, (2, 2, 32)),
        (17, 40, 100, (4, 1, 16)),
        (64, 64, 1024, (16, 4, 64)),
        (32, 32, 16384, (32, 32, 512)),
    ],
)
def test_splitk_partition_plan_follows_matmul(m, n, k, plan):
    assert model.partition_plan(m, n, k) == plan


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_splitk_cases_cover_every_partition(entry):
    for case in _cases(entry):
        m, n, k, _, _ = case
        if entry.mn_aligned:
            assert m % entry.bm == 0 and n % entry.bn == 0
        if entry.k_aligned:
            assert k % entry.bk == 0
        partitions, _, size = model.partition_plan(m, n, k, entry.bk)
        ranges = model.partition_ranges(entry, k, partitions, size)
        assert ranges[0][0] == 0 and ranges[-1][1] == k
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        resources, params, expected, geometry = _case_data(entry, case)
        assert geometry["workgroupCount"][2] == partitions
        assert expected.count(GUARDS["float32"]) == 8
        assert resources["A"].count(GUARDS[entry.dtype]) >= 8


@pytest.mark.parametrize("entry", model.entries(), ids=lambda entry: entry.name)
def test_splitk_tile_reads_cover_exactly_the_loader_slots(entry):
    loaders = entry.loaders
    assert len({loader.function for loader in loaders}) == 2
    reads = entry.tile_reads()
    for loader, slots in zip(loaders, reads):
        written = {slot for lane in loader.slots() for slot in lane}
        assert slots == written, (entry.name, loader.function)


@pytest.mark.parametrize("entry", ACCUM_ENTRIES, ids=lambda entry: entry.name)
@pytest.mark.parametrize("case", ACCUM_CASES)
def test_splitk_accum_cases_are_exact(entry, case):
    resources, expected, geometry = _accum_data(entry, case)
    m, n, _ = case
    assert geometry["workgroupCount"][0] * ACCUM_WORKGROUP[0] == n
    assert geometry["workgroupCount"][1] * ACCUM_WORKGROUP[1] == m
    assert expected[m * n :] == [GUARDS[entry.dtype]] * 8


@pytest.fixture(scope="module")
def splitk_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned split-K native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    tmp_path = tmp_path_factory.mktemp("splitk-original")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    git = ["git", "-C", str(root)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True, timeout=30)
    assert head.strip() == MLX_COMMIT
    status = subprocess.check_output(
        [
            *git,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "mlx/backend/metal/kernels",
        ],
        text=True,
        timeout=30,
    )
    assert not status.strip()
    source = root / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    for name, digest in SPLITK_HEADERS.items():
        header = root / "mlx/backend/metal/kernels/steel/gemm" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest, name
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


def _original_outputs(request, source, module, work, name):
    executor = _executor("metal")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        control = replace(
            native,
            artifact_path=source,
            module_path=module,
            entry_point=name,
            loaded_artifact=None,
            constants={},
        )
        actual = executor.runtime_adapter.runtime.dispatch(None, state, control)
        module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
        assert state.details["metalRuntime"]["librarySHA256"] == module_hash
        _write_json(
            work / "original-evidence.json",
            {
                "outputs": actual,
                "sourceCommit": MLX_COMMIT,
                "sourceSha256": SOURCE_SHA256,
                "moduleSha256": module_hash,
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


def _run_entry(tmp_path, splitk_source, name, workgroup, payloads):
    """Translate ``name`` and dispatch each ``(inputs, outputs, geometry)``."""
    root, original, target = splitk_source
    results = []
    with tempfile.TemporaryDirectory(prefix=".current-splitk-", dir=root) as directory:
        work = Path(directory)
        try:
            report = translate_project(
                _config(root, work.name, target, name, workgroup), format_output=False
            )
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            descriptor, package = _prepare_native_package(report, tmp_path)
            assert not descriptor.get("specializationConstants")
            bindings = _bindings(descriptor)
            for label, build in payloads:
                case_work = tmp_path / label
                case_work.mkdir()
                inputs, outputs, geometry = build(bindings, target)
                request = build_native_loader_dispatch_request(
                    descriptor,
                    package,
                    inputs,
                    outputs,
                    geometry,
                    expected_target=target,
                )
                assert not request.execution_plan.diagnostics
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
                    assert (
                        _original_outputs(
                            request, root / SOURCE, original, case_work, name
                        )
                        == record["outputs"]
                    )
                assert record["outputs"] == outputs
                results.append({"case": label, "equalExactReference": True})
        finally:
            shutil.copytree(work, tmp_path / "translation", dirs_exist_ok=True)
    _write_json(
        tmp_path / "parity.json",
        {
            "sourceCommit": MLX_COMMIT,
            "target": target,
            "entryPoint": name,
            "sourceSha256": SOURCE_SHA256,
            "cases": results,
            "equalOriginalMetal": True if target == "metal" else None,
            "scope": SCOPE,
        },
    )
    return results


def _params_value(layout, params):
    """GEMMSpiltKParams in the binding's reflected storage.

    The record holds only 32-bit ints, so targets may reflect it as an int32
    array in member order rather than as a packed record.
    """
    members = [member["name"] for member in layout["structMembers"]]
    assert members == list(params)
    if layout["elementType"] == "record":
        words = pack_storage_records(layout, [params])
        return {"dtype": "uint32", "shape": [len(words)], "values": words}
    assert layout["elementType"] == "int32" and layout["componentCount"] == 13
    assert [m["offsetBytes"] for m in layout["structMembers"]] == list(range(0, 52, 4))
    return {"dtype": "int32", "shape": [13], "values": list(params.values())}


def _splitk_payload(entry, case):
    def build(bindings, target):
        assert set(bindings) == {"A", "B", "C", "params"}
        layout = bindings["params"]["scalarLayout"]
        assert layout["elementStrideBytes"] == 52
        resources, params, expected, geometry = _case_data(entry, case)
        inputs = {
            bindings[name]["name"]: _storage(
                target, bindings[name], dtype, resources[name]
            )
            for name, dtype in (
                ("A", entry.dtype),
                ("B", entry.dtype),
                ("C", "float32"),
            )
        }
        inputs[bindings["params"]["name"]] = _params_value(layout, params)
        outputs = {
            bindings["C"]["name"]: _storage(target, bindings["C"], "float32", expected)
        }
        return inputs, outputs, geometry

    return "case-" + "-".join(map(str, case)), build


def _accum_payload(entry, case):
    def build(bindings, target):
        resources, expected, geometry = _accum_data(entry, case)
        assert set(bindings) == set(resources)
        inputs = {}
        for name, (dtype, values) in resources.items():
            if len(values) == 1 and name not in {"D", "C", "C_split"}:
                inputs[bindings[name]["name"]] = {
                    "dtype": dtype,
                    "shape": [1],
                    "values": values,
                }
            else:
                inputs[bindings[name]["name"]] = _storage(
                    target, bindings[name], dtype, values
                )
        outputs = {
            bindings["D"]["name"]: _storage(
                target, bindings["D"], entry.dtype, expected
            )
        }
        return inputs, outputs, geometry

    return "case-" + "-".join(map(str, case)), build


@pytest.mark.parametrize("entry", SELECTED_ENTRIES, ids=lambda entry: entry.name)
def test_current_gemm_splitk_executes(tmp_path, splitk_source, entry):
    payloads = [_splitk_payload(entry, case) for case in _cases(entry)]
    results = _run_entry(
        tmp_path, splitk_source, entry.name, entry.workgroup_size, payloads
    )
    assert len(results) == len(payloads)


@pytest.mark.parametrize("entry", SELECTED_ACCUM_ENTRIES, ids=lambda entry: entry.name)
def test_current_gemm_splitk_accum_executes(tmp_path, splitk_source, entry):
    payloads = [_accum_payload(entry, case) for case in ACCUM_CASES]
    results = _run_entry(tmp_path, splitk_source, entry.name, ACCUM_WORKGROUP, payloads)
    assert len(results) == len(payloads)
