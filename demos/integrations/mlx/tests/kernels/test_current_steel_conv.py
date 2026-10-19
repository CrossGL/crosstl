"""Execute the pinned MLX implicit-GEMM convolution entries natively.

``steel_conv.metal`` instantiates ``implicit_gemm_conv_2d`` for three element
types, six block shapes and six loader selections: the general-channel input
loader for small or large filters and the small-channel loaders for one to
four channels per group. This gate translates all 108 entries and dispatches
each with the buffers and parameters ``implicit_gemm_conv_2D_gpu`` binds, over
convolutions that meet the entry's loader preconditions: padded, strided,
dilated, flipped and grouped, with partial row and column tiles and surplus
threadgroups. Every output word, guards included, must equal an exact
reference; on Metal it must also equal the original source.
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
from crosstl.translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from demos.integrations.mlx.tests.kernels import steel_conv_model as model
from demos.integrations.mlx.tests.kernels.test_current_conv import _params_bytes
from demos.integrations.mlx.tests.kernels.test_current_gemm import (
    GATE_TOOLS,
    GUARDS,
    HELPER_ELEMENTS,
    MLX_COMMIT,
    ORIGINAL_FLAGS,
    PLATFORM_TARGETS,
    SIGNED_MAX,
    _exact,
    _storage,
    _write_json,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.metal"
SOURCE_SHA256 = "29ce84c7fbc964cc2257ad7a1f5792493f0af502a2e74223d473a26f83493928"
CONV_HEADERS = {
    "kernels/steel_conv.h": (
        "a34ae933e291169538ef9415323833f568d649b8ed6a5342c0eff40ced085d8c"
    ),
    "loaders/loader_channel_l.h": (
        "4377845f9553eed23ff1707cddfaabcf0056578c829f67616a1e68efcab94703"
    ),
    "loaders/loader_channel_n.h": (
        "6e6efd6b2b1af0686030f703fb25f3e022edf9079f81f3501d09548447f48294"
    ),
    "params.h": "354a832c5bf37791f4ad42ddc3ca9456a9c1ebbf3216650bb4a10c18110e03a8",
    "conv.h": "e469f51289eff2f632335c634bb5d346667a61f81e515f42159a1ebffd3dd4df",
    "loader.h": "a82850d760638c71a9ae8c3e426082bf6e2091822abae82583576aa129d1150f",
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_STEEL_CONV"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_STEEL_CONV_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_STEEL_CONV_SHARD_COUNT"
CI_SHARD_COUNT = 6
ENTRIES = model.entries()
BINDINGS = ("A", "B", "C", "params", "gemm_params")
PARAMS_BYTES = {"params": 176, "gemm_params": 40}
# Output channels per group for each case, by block width: one full tile,
# a partial second tile, and a partial tile per group.
OUTPUTS = {8: (8, 12, 5), 32: (32, 48, 16), 64: (64, 80, 16)}
SCOPE = (
    "Every float32, float16 and bfloat16 steel_conv implicit_gemm_conv_2d "
    "entry; steel_conv_general, steel_conv_3d and the upstream suite are not "
    "covered."
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
# Records the kernel reads through constant pointers, by GLSL helper name.
RECORD_HELPERS = (
    "ImplicitGemmConv2DParams",
    "MLXConvParams",
    "MLXConvParams<2>",
    "MLXConvParams_2",
    "MLXConvParams_int_2",
)


def _cases(entry):
    """Convolutions and surplus threadgroups that exercise ``entry``.

    Each convolution meets the entry's loader preconditions. The large-filter
    loader tests bounds at the unflipped tap, so its flipped case keeps every
    tap inside the input.
    """
    full, partial, grouped = OUTPUTS[entry.bn]
    channels = entry.channels or 16
    large = not entry.channels and not entry.small_filter
    first = model.Conv(
        batch=1,
        spatial=(3, 20) if large else (5, 6),
        kernel=(1, 17) if large else (3, 3),
        channels=channels,
        outputs=full,
        padding=(0, 1) if large else (1, 1),
    )
    second = model.Conv(
        batch=3,
        spatial=(9, 8),
        kernel=(2, 3),
        channels=entry.channels or 32,
        outputs=partial,
        stride=(2, 1),
        padding=(0, 0) if large else (1, 1),
        dilation=(1, 2),
        flip=True,
    )
    third = model.Conv(
        batch=1,
        spatial=(4, 5),
        kernel=(1, 1) if not entry.channels else (5, 4),
        channels=2 * channels,
        outputs=2 * grouped,
        groups=2,
        padding=(0, 0) if not entry.channels else (2, 1),
    )
    return ((first, 0), (second, 0), (third, 1))


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"steel_conv shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _tile_maxima(entry):
    a_slots, b_slots = entry.tile_reads()
    loader_a, loader_b = entry.loaders
    written_a = {slot for lane in loader_a.slots() for slot in lane}
    written_b = {slot for lane in loader_b.slots() for slot in lane}
    return max(a_slots | written_a), max(b_slots | written_b)


def _index_assertions(entry):
    """GLSL index ranges: MLX buffers stay below 2^31 elements.

    The two parameter records are single elements. The threadgroup tiles
    reach only the slots their loaders write and ``BlockMMA`` reads.
    """
    element = HELPER_ELEMENTS[entry.dtype]
    device = [
        {
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}_{scalar}",
            "maximum": SIGNED_MAX,
        }
        for access in ("load", "store")
        for scalar in sorted({element, "int", "int64_t", "uint"})
    ] + [
        {
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_load_{record}",
            "maximum": 0,
        }
        for record in RECORD_HELPERS
    ]
    workgroup = [
        {
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{tile}",
            "maximum": maximum,
        }
        for tile, maximum in zip(("As", "Bs"), _tile_maxima(entry))
    ]
    return tuple(
        {"source": SOURCE, **item, "minimum": 0} for item in device + workgroup
    )


def _config(root, output, target, entry):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (entry.name,)},
        workgroup_size_rules={SOURCE: entry.workgroup_size},
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


def _values(dtype, salt, count, step):
    """Deterministic multiples of ``step`` in [-1, 1], one in three zero.

    Inputs are quarters and weights halves, so every product is a multiple
    of 1/8 (bfloat16: halves, so 1/4) and the sparse signs keep each sum
    within the exact range of the output type.
    """
    if dtype == "bfloat16":
        step = max(step, 0.5)
    levels = int(1 / step)
    values = []
    for index in range(count):
        mixed = (index + salt * 0x9E3779B9) & 0xFFFFFFFF
        for _ in range(2):
            mixed = ((mixed ^ (mixed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
        mixed ^= mixed >> 16
        if mixed % 3 == 0:
            values.append(0.0)
            continue
        level = (mixed >> 8) % (2 * levels + 1) - levels
        values.append(level * step)
    return values


def _case_data(entry, conv, extra):
    """Resources, the expected output and the dispatch geometry."""
    assert entry.accepts(conv), (entry.name, conv)
    guard = GUARDS[entry.dtype]
    gemm = conv.gemm_params(entry)
    count_in = conv.batch * conv.spatial[0] * conv.spatial[1] * conv.channels
    count_wt = conv.outputs * conv.kernel[0] * conv.kernel[1] * conv.per_group
    inputs = _values(entry.dtype, 1, count_in, 0.25)
    weights = _values(entry.dtype, 2, count_wt, 0.5)
    reference = conv.reference(inputs, weights)
    for value in reference:
        # The float accumulator and the output cast round nothing.
        assert _exact(entry.dtype, value), (entry.name, conv, value)
    # Weight tiles past the last output channel read whole rows unchecked
    # (BN > 8), into storage that stays inside the binding.
    slack = entry.bn * gemm["K"]
    resources = dict(
        A=inputs + [guard] * 8,
        B=weights + [guard] * (slack + 8),
        C=[guard] * (len(reference) + 8),
    )
    expected = reference + [guard] * 8
    geometry = {
        "workgroupCount": [
            gemm["tiles_n"] + extra,
            gemm["tiles_m"] + extra,
            conv.groups,
        ],
        "workgroupSize": list(entry.workgroup_size),
    }
    return resources, conv.params(), gemm, expected, geometry


def _dispatch_values(bindings, entry, conv, extra, target):
    resources, params, gemm, expected, geometry = _case_data(entry, conv, extra)
    records = {"params": params, "gemm_params": gemm}
    inputs = {}
    for name, binding in bindings.items():
        if name in records:
            layout = binding["scalarLayout"]
            assert layout["elementStrideBytes"] == PARAMS_BYTES[name], layout
            if layout["elementType"] == "record":
                words = pack_storage_records(layout, [records[name]])
                value = {"dtype": "uint32", "shape": [len(words)], "values": words}
            else:
                # A record of int fields alone is reflected as its int words.
                assert layout["elementType"] == "int32", layout
                members = [member["name"] for member in layout["structMembers"]]
                assert members == list(records[name]), members
                words = [records[name][member] for member in members]
                value = {"dtype": "int32", "shape": [len(words)], "values": words}
        else:
            value = _storage(target, binding, entry.dtype, resources[name])
        inputs[binding["name"]] = value
    outputs = {
        bindings["C"]["name"]: _storage(target, bindings["C"], entry.dtype, expected)
    }
    return inputs, outputs, geometry


def test_steel_conv_model_names_every_entry():
    assert len(ENTRIES) == 3 * 6 * 6 == len(model.BY_NAME)
    assert Counter(entry.dtype for entry in ENTRIES) == {
        dtype: 36 for dtype, _, _ in model.TYPES
    }
    assert Counter((entry.channel_name, entry.filter_name) for entry in ENTRIES) == {
        (channel, name): 18 for channel, _, name, _ in model.FILTERS
    }
    entry = model.BY_NAME[
        "implicit_gemm_conv_2d_bfloat16_bm64_bn32_bk16_wm2_wn2_channel_3_filter_l"
    ]
    assert entry.workgroup_size == (32, 2, 2)
    assert entry.padding == 8 and entry.leading == 24


def test_steel_conv_host_selection():
    """``implicit_gemm_conv_2D_gpu`` selects every entry but two block groups.

    It widens a 64-row block to 64 columns unless the output fits 8, and
    picks 64 rows only for 64 or more channels per group. The gate executes
    the unselected entries too: their loaders take the same preconditions.
    """
    reached = set()
    for dtype, _, _ in model.TYPES:
        for batch in (1, 64):
            for outputs in (8, 32, 64):
                for channels in (64, 1, 2, 3, 4):
                    for kernel in ((3, 3), (1, 17)):
                        conv = model.Conv(batch, (16, 24), kernel, channels, outputs)
                        entry = conv.host_entry(dtype)
                        assert entry.accepts(conv)
                        reached.add(entry.name)
    unselected = {
        entry.name
        for entry in ENTRIES
        if entry.bm == 64 and (entry.bn == 32 or entry.channels)
    }
    assert len(unselected) == 3 * (6 + 2 * 4)
    assert reached == set(model.BY_NAME) - unselected
    small = model.Conv(1, (16, 8), (3, 3), 64, 32).host_entry("float32")
    assert small.name.endswith("bm32_bn32_bk16_wm2_wn2_channel_l_filter_s")


def test_steel_conv_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry.name for shard in shards for entry in shard) == sorted(
        model.BY_NAME
    )
    for shard in shards:
        assert len(shard) == 18
        assert {entry.dtype for entry in shard} == {d for d, _, _ in model.TYPES}
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        _selected_entries()
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_steel_conv_loaders_cover_exactly_the_tiles(entry):
    """Each loader writes every tile element once; ``BlockMMA`` reads only
    written elements, and nothing touches the row padding."""
    reads = entry.tile_reads()
    for slots, loader in zip(reads, entry.loaders):
        written = [slot for lane in loader.slots() for slot in lane]
        assert len(written) == len(set(written))
        assert set(written) == {
            row * loader.leading + column
            for row in range(loader.rows)
            for column in range(loader.columns)
        }
        assert slots == set(written)
    a_max, b_max = _tile_maxima(entry)
    assert a_max == (entry.bm - 1) * entry.leading + entry.bk - 1
    assert b_max == (entry.bn - 1) * entry.leading + entry.bk - 1
    workgroup = [item for item in _index_assertions(entry) if "indexed" in item]
    assert [(item["indexed"], item["maximum"]) for item in workgroup] == [
        ("crosstl_workgroup_As", a_max),
        ("crosstl_workgroup_Bs", b_max),
    ]


def test_steel_conv_gemm_params_follow_the_host():
    """The encoded jumps walk the input exactly as the filter taps do."""
    entry = model.BY_NAME[
        "implicit_gemm_conv_2d_float32_bm32_bn8_bk16_wm4_wn1_channel_l_filter_s"
    ]
    conv = model.Conv(2, (9, 8), (2, 3), 32, 12, stride=(2, 1), dilation=(1, 2))
    gemm = conv.gemm_params(entry)
    strides = conv.params()["in_strides"]
    assert strides == [9 * 8 * 32, 8 * 32, 32, 1]
    # Two taps along w, one step down a row, then the next channel block.
    assert gemm["inp_jump_w"] == 2 * 32
    assert gemm["inp_jump_w"] * 2 + gemm["inp_jump_h"] == 8 * 32
    walk = gemm["inp_jump_w"] * 2 + gemm["inp_jump_h"] + gemm["inp_jump_w"] * 2
    assert walk + gemm["inp_jump_c"] == entry.bk
    assert gemm["gemm_k_iterations"] == 2 * 3 * 2
    assert (gemm["M"], gemm["N"], gemm["K"]) == (2 * 4 * 4, 12, 2 * 3 * 32)
    assert (gemm["tiles_m"], gemm["tiles_n"]) == (1, 2)
    small = model.BY_NAME[
        entry.name.replace("channel_l_filter_s", "channel_3_filter_l")
    ]
    three = model.Conv(1, (5, 6), (3, 3), 3, 8, padding=(1, 1))
    # Three channels widen to four, so nine taps take 36 of 48 columns.
    assert three.gemm_params(small)["gemm_k_iterations"] == 3
    one = model.BY_NAME[entry.name.replace("channel_l_filter_s", "channel_1_filter_l")]
    assert (
        model.Conv(1, (5, 6), (3, 3), 1, 8).gemm_params(one)["gemm_k_iterations"] == 1
    )


def test_steel_conv_params_follow_the_mlx_struct_layout():
    params = model.Conv(3, (9, 8), (2, 3), 32, 12, flip=True).params()
    data = _params_bytes(params)
    assert len(data) == PARAMS_BYTES["params"]
    assert params["oS"] == [8, 6] and params["wt_strides"] == [6 * 32, 3 * 32, 32, 1]


def test_steel_conv_reference_flips_and_groups():
    """A one-hot input and weight place each tap where the kernel puts it."""
    conv = model.Conv(1, (3, 3), (2, 2), 2, 2, groups=2, padding=(1, 0), flip=True)
    assert conv.out_spatial == (4, 2)
    inputs = [0.0] * 18
    inputs[(1 * 3 + 1) * 2 + 1] = 1.0  # pixel (1, 1), group-1 channel
    weights = [0.0] * 8
    weights[1 * 4 + 0 * 2 + 1] = 1.0  # output 1, weight tap (0, 1)
    # Flipped, weight tap (0, 1) pairs with input tap (1, 0): output (1, 1).
    # Unflipped it pairs with input tap (0, 1): output (2, 0).
    for flip, pixel in ((True, (1, 1)), (False, (2, 0))):
        outputs = replace(conv, flip=flip).reference(inputs, weights)
        hits = [index for index, value in enumerate(outputs) if value]
        assert hits == [(pixel[0] * 2 + pixel[1]) * 2 + 1]
        assert outputs[hits[0]] == 1.0


@pytest.mark.parametrize("dtype", [name for name, _, _ in model.TYPES])
def test_steel_conv_references_are_exact_and_guarded(dtype):
    """Every case's reference is exact in the output type for each entry kind."""
    guard = GUARDS[dtype]
    assert _exact(dtype, guard)
    seen = set()
    for entry in ENTRIES:
        key = (entry.dtype, entry.bn, entry.channels, entry.small_filter)
        if entry.dtype != dtype or key in seen:
            continue
        seen.add(key)
        for conv, extra in _cases(entry):
            resources, params, gemm, expected, geometry = _case_data(entry, conv, extra)
            assert expected[-8:] == [guard] * 8
            assert len(resources["C"]) == len(expected)
            assert any(expected[:-8]) and guard not in expected[:-8]
            assert geometry["workgroupCount"][2] == conv.groups
            assert gemm["N"] * conv.groups == params["O"]
    assert len(seen) == 3 * 6


@pytest.fixture(scope="module")
def conv_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned steel_conv native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    tmp_path = tmp_path_factory.mktemp("steel-conv-original")
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
    for name, digest in CONV_HEADERS.items():
        header = root / "mlx/backend/metal/kernels/steel/conv" / name
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


def _validate(root):
    def validate(path, destination, target):
        return compile_metal(
            path, root, destination, flags=("-std=metal3.1", "-fno-fast-math")
        )

    return validate


@pytest.mark.parametrize("entry", SELECTED_ENTRIES, ids=lambda entry: entry.name)
def test_current_steel_conv_executes(tmp_path, conv_source, entry):
    root, original, target = conv_source
    results = []
    with tempfile.TemporaryDirectory(prefix=".current-conv-", dir=root) as directory:
        work = Path(directory)
        try:
            report = translate_project(
                _config(root, work.name, target, entry), format_output=False
            )
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert payload["project"]["indexRangeAssertions"] == (
                list(_index_assertions(entry)) if target == "opengl" else []
            )
            descriptor, package = _prepare_native_package(report, tmp_path)
            assert not descriptor["specializationConstants"]
            bindings = {
                item["provenance"]["sourceResource"]["parameter"]: item
                for item in descriptor["bindings"]
            }
            assert len(bindings) == len(descriptor["bindings"])
            assert set(bindings) == set(BINDINGS), sorted(bindings)
            for index, (conv, extra) in enumerate(_cases(entry)):
                case_work = tmp_path / f"case-{index}"
                case_work.mkdir()
                inputs, outputs, geometry = _dispatch_values(
                    bindings, entry, conv, extra, target
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
                _execute(
                    request,
                    outputs,
                    case_work,
                    validate=_validate(root) if target == "metal" else None,
                )
                generated = json.loads((case_work / "evidence.json").read_text())
                record = generated["records"]["generated"]
                if target == "metal":
                    assert (
                        _original_outputs(
                            request, root / SOURCE, original, case_work, entry.name
                        )
                        == record["outputs"]
                    )
                assert record["outputs"] == outputs
                results.append(
                    {
                        "conv": {
                            "batch": conv.batch,
                            "spatial": list(conv.spatial),
                            "kernel": list(conv.kernel),
                            "channels": conv.channels,
                            "outputs": conv.outputs,
                            "groups": conv.groups,
                            "stride": list(conv.stride),
                            "padding": list(conv.padding),
                            "dilation": list(conv.dilation),
                            "flip": conv.flip,
                        },
                        "extraWorkgroups": extra,
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
            "sourceSha256": SOURCE_SHA256,
            "cases": results,
            "complete": len(results) == len(_cases(entry)),
            "scope": SCOPE,
        },
    )
