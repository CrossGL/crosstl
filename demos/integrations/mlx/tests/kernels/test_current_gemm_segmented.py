"""Execute the pinned MLX segmented GEMM entries natively.

``steel_gemm_segmented.metal`` instantiates ``segmented_mm`` for three element
types, five tile shapes and four transpositions. This gate translates every
entry with four frozen function-constant configurations, which together take
both ``segments`` layouts and all four ``align_M`` and ``align_N`` pairs, and
dispatches it with the buffers ``matmul.cpp``'s ``segmented_mm`` binds. The
segments are empty, shorter than one K tile, exactly one tile, several tiles
with a tail, unaligned, and overlapping. Every output word, guards included,
must equal an exact reference; on Metal it must also equal the original
source.
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
from crosstl.project.runtime_verification import (
    NativeRuntimeConstantBinding,
    RuntimeExecutionState,
    RuntimeSpecializationConstant,
)
from crosstl.translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused
from demos.integrations.mlx.tests.kernels import gemm_segmented_model as model
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

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_segmented.metal"
SOURCE_SHA256 = "fe8a6ac43fd0383c94dff4cb0acf2d6e0b83b2ef692387a4e3307c824c6d14d6"
SEGMENTED_HEADERS = {
    **HEADER_HASHES,
    "kernels/steel_gemm_segmented.h": (
        "e1256eb925f97cd594ffd0382ecd22f5540637a7bbb902da357a80ece284a9ad"
    ),
    "../utils.h": "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a",
    "../../utils.h": "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf",
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMM_SEGMENTED"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_GEMM_SEGMENTED_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_GEMM_SEGMENTED_SHARD_COUNT"
# Seven is coprime with the four variants, three types, four transpositions
# and five shapes, so every shard mixes all of them.
CI_SHARD_COUNT = 7
CONSTANT_IDS = {"segments_contiguous": 199, "align_M": 200, "align_N": 201}
ENTRIES = model.entries()
# (segments_contiguous, align_M, align_N). ``matmul.cpp`` sets each alignment
# from the shape and the layout from the segments' strides; the four
# configurations take both layouts and every alignment pair once.
VARIANTS = {
    "pairs": (True, False, False),
    "boundaries-m": (False, True, False),
    "pairs-n": (True, False, True),
    "boundaries-mn": (False, True, True),
}
# ``single`` binds what ``segmented_mm`` binds for one segment over all of K;
# ``multi`` has five segments, padded leading dimensions and batch stride, and
# extra threadgroups that exercise the early return.
CASES = ("single", "multi")
SCOPE = (
    "Every float32, float16 and bfloat16 steel_gemm_segmented segmented_mm "
    "entry with four frozen constant configurations; the upstream suite is "
    "not covered."
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
BINDINGS = ("A", "B", "segments", "C", "params")


def _frozen_values(variant):
    return dict(zip(CONSTANT_IDS, VARIANTS[variant]))


PAIRS = tuple((entry, variant) for entry in ENTRIES for variant in VARIANTS)


def _selected_pairs():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return PAIRS
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"segmented GEMM shard {index} is outside [0, {count})")
    return PAIRS[index::count]


SELECTED_PAIRS = _selected_pairs()


def _pair_id(pair):
    entry, variant = pair
    return f"{entry.name}-{variant}"


def _shape(entry, variant, case):
    """``(M, N)`` as ``segmented_mm`` would select ``variant``'s alignment.

    Unaligned dimensions end in a partial tile; with both unaligned, the
    ``multi`` tiles take all four of the kernel's alignment branches.
    """
    _, align_m, align_n = VARIANTS[variant]
    if case == "single":
        return (entry.bm if align_m else 5), (entry.bn if align_n else 6)
    return (
        2 * entry.bm if align_m else entry.bm + 5,
        2 * entry.bn if align_n else entry.bn + 3,
    )


def _segment_words(entry, variant, case):
    """The ``segments`` buffer, each batch's inner range, and K."""
    contiguous = VARIANTS[variant][0]
    bk = entry.bk
    if case == "single":
        k = bk + 3
        ranges = [(0, k)]
    elif contiguous:
        k = 3 * bk + 4
        # Several tiles with a tail, empty, one tile, a tail alone that ends
        # at K, and several tiles overlapping the first segment.
        ranges = [(3, 2 * bk + 8), (7, 7), (1, bk + 1), (k - 9, k), (bk, 3 * bk + 1)]
    else:
        # Consecutive segments that skip the first two and last two columns.
        bounds = [2, 2 * bk + 7, 2 * bk + 7, 3 * bk + 7, 3 * bk + 16, 4 * bk + 19]
        k = bounds[-1] + 2
        ranges = list(zip(bounds, bounds[1:]))
    if contiguous:
        words = [word for segment in ranges for word in segment]
    else:
        words = [ranges[0][0], *(end for _, end in ranges)]
    return words, ranges, k


def _k_tiles(bk, start, end):
    """Whole K tiles and the tail ``segmented_mm``'s loop takes, in uint32."""
    tiles, k = 0, (start + bk) % 2**32
    while k <= end:
        tiles += 1
        k = (k + bk) % 2**32
    remain = bk - (k - end)
    return tiles, remain


def _index_assertions(entry):
    """GLSL index ranges: MLX buffers stay below 2^31 elements.

    A loader reads ``reads`` adjacent elements past its source offset;
    ``segments`` is read as ``uint``. The threadgroup views reach only the
    slots their loaders write.
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
        for access in sorted({f"load_{element}", f"store_{element}", "load_uint"})
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


def _config(root, output, target, entry, variant):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (entry.name,)},
        specialization_constants=_frozen_values(variant),
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: entry.workgroup_size},
        # segmented_mm binds the segments as an input; nothing writes them
        # while the kernel runs, which lets the subgroup proofs read the K
        # range, and so the K loop's trip count, as uniform.
        immutable_storage=[
            {"source": SOURCE, "entryPoint": entry.name, "parameter": "segments"}
        ],
        workgroup_access_assertions=(
            _loader_access_assertions(entry, SOURCE) if target != "metal" else ()
        ),
        index_range_assertions=(_index_assertions(entry) if target == "opengl" else ()),
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                "target_options": TARGET_OPTIONS,
            }
        },
    )


def _layout(entry, m, n, k, padding):
    a_rows, a_columns = (k, m) if entry.transpose_a else (m, k)
    b_rows, b_columns = (n, k) if entry.transpose_b else (k, n)
    lda, ldb, ldd = a_columns + padding, b_columns + padding, n + padding
    return {
        "lda": lda,
        "ldb": ldb,
        "ldd": ldd,
        "a_size": a_rows * lda,
        "b_size": b_rows * ldb,
        "batch_stride_d": m * ldd + padding,
    }


def _a_index(entry, layout, row, inner):
    if entry.transpose_a:
        return inner * layout["lda"] + row
    return row * layout["lda"] + inner


def _b_index(entry, layout, inner, column):
    if entry.transpose_b:
        return column * layout["ldb"] + inner
    return inner * layout["ldb"] + column


def _case_data(entry, variant, case):
    """Resources, the expected output and the dispatch geometry for ``case``."""
    m, n = _shape(entry, variant, case)
    words, ranges, k = _segment_words(entry, variant, case)
    padding, extra = (0, 0) if case == "single" else (3, 1)
    layout = _layout(entry, m, n, k, padding)
    dtype, guard = entry.dtype, GUARDS[entry.dtype]
    a = [guard] * (layout["a_size"] + 8)
    b = [guard] * (layout["b_size"] + 8)
    for row in range(m):
        for inner in range(k):
            # The inner // 5 term breaks bfloat16's period-5 operands, whose
            # whole-period sums would otherwise cancel across segments.
            a[_a_index(entry, layout, row, inner)] = _operand(
                dtype, "A", row * k + inner + inner // 5
            )
    for inner in range(k):
        for column in range(n):
            b[_b_index(entry, layout, inner, column)] = _operand(
                dtype, "B", inner * n + column
            )
    contiguous = VARIANTS[variant][0]
    batch_stride_d, ldd = layout["batch_stride_d"], layout["ldd"]
    expected = [guard] * (len(ranges) * batch_stride_d + 8)
    for z, segment in enumerate(model.segments(words, contiguous, len(ranges))):
        assert segment == ranges[z]
        values = model.product(
            m,
            n,
            segment,
            lambda row, inner: a[_a_index(entry, layout, row, inner)],
            lambda inner, column: b[_b_index(entry, layout, inner, column)],
        )
        for (row, column), total in values.items():
            # The float accumulator and the output cast round nothing.
            assert _exact(dtype, total), (entry.name, variant, case, total)
            expected[z * batch_stride_d + row * ldd + column] = total
    params = dict(
        M=m,
        N=n,
        K=k,
        lda=layout["lda"],
        ldb=layout["ldb"],
        ldd=ldd,
        tiles_n=-(-n // entry.bn),
        tiles_m=-(-m // entry.bm),
        batch_stride_a=0,
        batch_stride_b=0,
        batch_stride_d=batch_stride_d,
        swizzle_log=0,
        gemm_k_iterations_aligned=0,
        batch_ndim=0,
    )
    resources = dict(
        A=("element", a),
        B=("element", b),
        segments=("uint32", words),
        C=("element", [guard] * len(expected)),
    )
    geometry = {
        "workgroupCount": [
            params["tiles_n"] + extra,
            params["tiles_m"] + extra,
            len(ranges),
        ],
        "workgroupSize": list(entry.workgroup_size),
    }
    return resources, params, expected, geometry


def _dispatch_values(bindings, entry, variant, case, target):
    resources, params, expected, geometry = _case_data(entry, variant, case)
    inputs = {}
    for name, binding in bindings.items():
        if name == "params":
            words = pack_storage_records(binding["scalarLayout"], [params])
            value = {"dtype": "uint32", "shape": [len(words)], "values": words}
        else:
            dtype, values = resources[name]
            if dtype == "element":
                value = _storage(target, binding, entry.dtype, values)
            else:
                value = {"dtype": dtype, "shape": [len(values)], "values": values}
        inputs[binding["name"]] = value
    outputs = {
        bindings["C"]["name"]: _storage(target, bindings["C"], entry.dtype, expected)
    }
    return inputs, outputs, geometry


def test_segmented_model_names_every_entry():
    assert len(ENTRIES) == 3 * 5 * 4 == len(model.BY_NAME)
    assert Counter(entry.dtype for entry in ENTRIES) == {
        dtype: 20 for dtype, _, _ in model.TYPES
    }
    assert Counter(entry.transpose for entry in ENTRIES) == {
        name: 15 for name, _, _ in fused.TRANSPOSES
    }
    entry = model.BY_NAME[
        "steel_segmented_mm_tn_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2"
    ]
    assert (entry.transpose_a, entry.transpose_b) == (True, False)
    assert entry.workgroup_size == (32, 2, 2)
    # matmul.cpp's default tile is 64x64x16 with a 2x2 simdgroup grid.
    assert "steel_segmented_mm_nn_float32_float32_bm64_bn64_bk16_wm2_wn2" in (
        model.BY_NAME
    )


def test_segmented_variants_cover_every_constant_pair():
    values = [_frozen_values(variant) for variant in VARIANTS]
    assert {value["segments_contiguous"] for value in values} == {True, False}
    assert {(value["align_M"], value["align_N"]) for value in values} == {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }


def test_segmented_shards_partition_the_pairs(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_pairs())
    assert sorted(map(_pair_id, (p for shard in shards for p in shard))) == sorted(
        map(_pair_id, PAIRS)
    )
    assert {len(shard) for shard in shards} == {34, 35}
    for shard in shards:
        assert {variant for _, variant in shard} == set(VARIANTS)
        assert {entry.dtype for entry, _ in shard} == {d for d, _, _ in model.TYPES}
        assert {entry.transpose for entry, _ in shard} == {
            name for name, _, _ in fused.TRANSPOSES
        }
        assert {entry.bk for entry, _ in shard} == {16, 32}
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        _selected_pairs()
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_pairs()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_segmented_tile_reads_cover_exactly_the_loader_slots(entry):
    a_slots, b_slots = entry.tile_reads()
    loader_a, loader_b = entry.loaders
    for slots, loader in ((a_slots, loader_a), (b_slots, loader_b)):
        written = {slot for lane in loader.slots() for slot in lane}
        assert slots == written


@pytest.mark.parametrize("bk", [16, 32])
def test_segmented_segments_take_every_k_loop_path(bk):
    """The segments give the K loop zero, one and two whole tiles, with and
    without a tail, and each layout reads back the ranges it encodes."""
    entry = next(entry for entry in ENTRIES if entry.bk == bk)
    for variant, (contiguous, _, _) in VARIANTS.items():
        words, ranges, k = _segment_words(entry, variant, "multi")
        assert len(words) == (2 * len(ranges) if contiguous else len(ranges) + 1)
        assert model.segments(words, contiguous, len(ranges)) == ranges
        assert all(0 <= start <= end <= k for start, end in ranges)
        loops = [_k_tiles(bk, start, end) for start, end in ranges]
        for (start, end), (tiles, remain) in zip(ranges, loops):
            assert 0 <= remain < bk and tiles * bk + remain == end - start
        assert {tiles for tiles, _ in loops} == {0, 1, 2}
        assert {remain > 0 for _, remain in loops} == {True, False}
        assert any(start % bk for start, _ in ranges)
        if contiguous:
            # The last segment overlaps the first.
            assert ranges[-1][0] < ranges[0][1] and ranges[0][0] < ranges[-1][1]
        else:
            assert ranges[0][0] > 0 and ranges[-1][1] < k
        words, ranges, k = _segment_words(entry, variant, "single")
        assert ranges == [(0, k)] and _k_tiles(bk, 0, k) == (1, 3)


@pytest.mark.parametrize("transpose", [name for name, _, _ in fused.TRANSPOSES])
@pytest.mark.parametrize("dtype", [name for name, _, _ in model.TYPES])
def test_segmented_references_are_exact_and_guarded(dtype, transpose):
    """Every reference value is exact, every unwritten slot keeps its guard and
    distinct segments give distinct products."""
    guard = GUARDS[dtype]
    assert _exact(dtype, guard)
    for shape in ("bm32_bn64_bk16_wm1_wn2", "bm64_bn32_bk32_wm2_wn2"):
        entry = model.BY_NAME[f"steel_segmented_mm_{transpose}_{dtype}_{dtype}_{shape}"]
        for variant in VARIANTS:
            for case in CASES:
                resources, params, expected, geometry = _case_data(entry, variant, case)
                m, n = params["M"], params["N"]
                stride, ldd = params["batch_stride_d"], params["ldd"]
                _, ranges, _ = _segment_words(entry, variant, case)
                assert geometry["workgroupCount"][2] == len(ranges)
                assert geometry["workgroupSize"] == list(entry.workgroup_size)
                live = {
                    z * stride + row * ldd + column
                    for z in range(len(ranges))
                    for row in range(m)
                    for column in range(n)
                }
                assert all(
                    (expected[index] == guard) == (index not in live)
                    for index in range(len(expected))
                ), (variant, case)
                for name in ("A", "B"):
                    assert all(_exact(dtype, value) for value in resources[name][1])
                products = [
                    tuple(
                        expected[z * stride + row * ldd + column]
                        for row in range(m)
                        for column in range(n)
                    )
                    for z in range(len(ranges))
                ]
                assert len(set(products)) == len(products), (variant, case)
                for (start, end), values in zip(ranges, products):
                    assert (start == end) == all(value == 0 for value in values)


def test_segmented_transposed_layouts_hold_the_same_matrix():
    products = {}
    for name, _, _ in fused.TRANSPOSES:
        entry = model.BY_NAME[
            f"steel_segmented_mm_{name}_float32_float32_bm32_bn32_bk16_wm2_wn2"
        ]
        _, params, expected, _ = _case_data(entry, "pairs", "multi")
        products[name] = [
            expected[z * params["batch_stride_d"] + row * params["ldd"] + column]
            for z in range(5)
            for row in range(params["M"])
            for column in range(params["N"])
        ]
    assert len({tuple(values) for values in products.values()}) == 1
    assert len(set(products["nn"])) > 16


@pytest.fixture(scope="module")
def segmented_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned segmented GEMM native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    tmp_path = tmp_path_factory.mktemp("segmented-original")
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
    for name, digest in SEGMENTED_HEADERS.items():
        header = root / "mlx/backend/metal/kernels/steel/gemm" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest, name
    register_default_sources()
    discovery = SOURCE_REGISTRY.get("metal").discover_entry_points(
        source.read_text(encoding="utf-8"),
        file_path=str(source),
        include_paths=[str(root)],
    )
    assert discovery.status == ENTRY_DISCOVERY_AVAILABLE
    assert [entry.name for entry in discovery.entries] == [
        entry.name for entry in ENTRIES
    ]
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


def _original_outputs(request, source, module, work, specializations, name):
    executor = _executor("metal")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        constants = {
            constant: NativeRuntimeConstantBinding(
                name=constant,
                constant=RuntimeSpecializationConstant(
                    name=constant,
                    constant_id=identifier,
                    dtype="bool",
                    value=specializations[constant],
                    kind="function-constant",
                ),
                value=specializations[constant],
                source="original-source-specialization",
            )
            for constant, identifier in CONSTANT_IDS.items()
        }
        control = replace(
            native,
            artifact_path=source,
            module_path=module,
            entry_point=name,
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


@pytest.mark.parametrize("pair", SELECTED_PAIRS, ids=_pair_id)
def test_current_gemm_segmented_executes(tmp_path, segmented_source, pair):
    entry, variant = pair
    root, original, target = segmented_source
    specializations = _frozen_values(variant)
    results = []
    with tempfile.TemporaryDirectory(prefix=".current-segmented-", dir=root) as tmp:
        work = Path(tmp)
        try:
            report = translate_project(
                _config(root, work.name, target, entry, variant), format_output=False
            )
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert payload["project"]["indexRangeAssertions"] == (
                list(_index_assertions(entry)) if target == "opengl" else []
            )
            descriptor, package = _prepare_native_package(report, tmp_path)
            constants = descriptor["specializationConstants"]
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
            assert len(bindings) == len(descriptor["bindings"])
            assert set(bindings) == set(BINDINGS), sorted(bindings)
            assert bindings["params"]["scalarLayout"]["elementStrideBytes"] == 72
            for case in CASES:
                case_work = tmp_path / f"case-{case}"
                case_work.mkdir()
                inputs, outputs, geometry = _dispatch_values(
                    bindings, entry, variant, case, target
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
                    assert (
                        _original_outputs(
                            request,
                            root / SOURCE,
                            original,
                            case_work,
                            specializations,
                            entry.name,
                        )
                        == record["outputs"]
                    )
                assert not record["request"]["constants"]
                assert record["outputs"] == outputs
                m, n = _shape(entry, variant, case)
                _, ranges, k = _segment_words(entry, variant, case)
                results.append(
                    {
                        "case": case,
                        "shape": [m, n, k],
                        "segments": [list(segment) for segment in ranges],
                        "equalOriginalMetal": True if target == "metal" else None,
                        "equalExactReference": True,
                    }
                )
        finally:
            shutil.copytree(work, tmp_path / "translation", dirs_exist_ok=True)
    _write_json(
        tmp_path / "parity.json",
        {
            "sourceCommit": MLX_COMMIT,
            "target": target,
            "entryPoint": entry.name,
            "variant": variant,
            "specializations": specializations,
            "sourceSha256": SOURCE_SHA256,
            "cases": results,
            "complete": len(results) == len(CASES),
            "scope": SCOPE,
        },
    )
