"""Execute the pinned MLX general implicit-GEMM convolution entries natively.

``steel_conv_general.metal`` instantiates ``implicit_gemm_conv_2d_general``
for three element types and six block shapes, and ``conv.cpp`` specializes
each with ``align_C``: whole 16-channel blocks, or a final partial block.
This gate translates all 18 entries with both specializations and
dispatches each with the eight buffers ``implicit_gemm_conv_2D_general_gpu``
binds, over input-dilated, strided, dilated, padded and flipped
convolutions, residues without a filter tap, partial row and column tiles
and surplus threadgroups. Every output word, guards included, must equal an
exact reference; on Metal it must also equal the original source compiled
and specialized the same way.
"""

import hashlib
import json
import math
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
from crosstl.translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from demos.integrations.mlx.tests.kernels import steel_conv_general_model as general
from demos.integrations.mlx.tests.kernels.steel_conv_model import Conv
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
from demos.integrations.mlx.tests.kernels.test_current_steel_conv import (
    OUTPUTS,
    TARGET_OPTIONS,
    _tile_maxima,
    _validate,
    _values,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = general.SOURCE
SOURCE_HASH = "df95502fbc02ad2bdd31ac6e00c30ac8379dc07ab2038e915e02dbb813fb789b"
CONV_HEADERS = {
    "kernels/steel_conv_general.h": (
        "0128838502501415acf766e8b977711bcb3bdae7439eb014bdd693119d547a1d"
    ),
    "loaders/loader_general.h": (
        "9aabb2491e4f91029e3ae73c4201bb0937652152f7771428ae01ff4c75653a47"
    ),
    "params.h": "354a832c5bf37791f4ad42ddc3ca9456a9c1ebbf3216650bb4a10c18110e03a8",
    "conv.h": "e469f51289eff2f632335c634bb5d346667a61f81e515f42159a1ebffd3dd4df",
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_STEEL_CONV_GENERAL"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_STEEL_CONV_GENERAL_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_STEEL_CONV_GENERAL_SHARD_COUNT"
CI_SHARD_COUNT = 3
ENTRIES = general.entries()
BINDINGS = (
    "A",
    "B",
    "C",
    "params",
    "gemm_params",
    "jump_params",
    "base_h",
    "base_w",
)
RECORD_BYTES = {
    "params": 176,
    "gemm_params": 40,
    "jump_params": 32,
    "base_h": 8,
    "base_w": 8,
}
SCOPE = (
    "Every float32, float16 and bfloat16 implicit_gemm_conv_2d_general entry "
    "with align_C true and false; the upstream suite is not covered."
)
# Single records the kernel reads through constant pointers, by GLSL helper
# name; the residue tables are arrays.
RECORD_HELPERS = (
    "ImplicitGemmConv2DParams",
    "MLXConvParams",
    "MLXConvParams<2>",
    "MLXConvParams_2",
    "MLXConvParams_int_2",
    "Conv2DGeneralJumpParams",
)
TABLE_HELPERS = ("Conv2DGeneralBaseInfo",)


def _cases(entry, aligned):
    """One-group convolutions and surplus threadgroups for one specialization.

    ``align_C`` holds exactly when the channels fill whole blocks, as the host
    sets it. Each set has an undilated case, a flipped and input-dilated case
    with partial tiles, and one whose residues step several taps at a time.
    """
    full, partial, small = OUTPUTS[entry.bn]
    if aligned:
        return (
            (Conv(1, (5, 6), (3, 3), 16, full, padding=(1, 1)), 0),
            (
                Conv(
                    2,
                    (4, 5),
                    (2, 3),
                    32,
                    partial,
                    stride=(1, 2),
                    padding=(1, 1),
                    dilation=(2, 1),
                    flip=True,
                    input_dilation=(2, 2),
                ),
                1,
            ),
            (
                Conv(
                    1,
                    (3, 4),
                    (3, 2),
                    16,
                    small,
                    stride=(3, 2),
                    padding=(2, 0),
                    input_dilation=(2, 3),
                ),
                0,
            ),
        )
    return (
        (Conv(1, (5, 6), (3, 3), 5, full, padding=(1, 1)), 0),
        (
            Conv(
                3,
                (4, 4),
                (2, 2),
                21,
                partial,
                stride=(2, 1),
                padding=(0, 1),
                dilation=(1, 2),
                flip=True,
                input_dilation=(1, 2),
            ),
            1,
        ),
        # Odd output rows and columns have no tap on an input pixel.
        (Conv(1, (3, 3), (1, 1), 35, small, input_dilation=(2, 2)), 0),
    )


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"steel_conv_general shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED = tuple(
    (entry, aligned) for entry in _selected_entries() for aligned in (True, False)
)


def _index_assertions(entry):
    """GLSL index ranges: MLX buffers stay below 2^31 elements.

    The parameter records are single elements. The threadgroup tiles reach
    only the slots their loaders write and ``BlockMMA`` reads.
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
            "maximum": 0 if record in RECORD_HELPERS else SIGNED_MAX,
        }
        for record in RECORD_HELPERS + TABLE_HELPERS
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


def _config(root, output, target, entry, aligned):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (entry.name,)},
        specialization_constants={"align_C": aligned},
        freeze_specialization_constants=True,
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


def _case_data(entry, conv, extra):
    """Resources, the expected output and the dispatch geometry."""
    assert entry.accepts(conv), (entry.name, conv)
    guard = GUARDS[entry.dtype]
    host = general.launch(conv, entry)
    count_in = conv.batch * conv.spatial[0] * conv.spatial[1] * conv.channels
    count_wt = conv.outputs * host.gemm["K"]
    inputs = _values(entry.dtype, 3, count_in, 0.25)
    weights = _values(entry.dtype, 4, count_wt, 0.5)
    reference = conv.reference(inputs, weights)
    for value in reference:
        # The float accumulator and the output cast round nothing.
        assert _exact(entry.dtype, value), (entry.name, conv, value)
    resources = dict(
        A=inputs + [guard] * 8,
        B=weights + [guard] * 8,
        C=[guard] * (len(reference) + 8),
    )
    expected = reference + [guard] * 8
    geometry = {
        "workgroupCount": [
            host.grid[0] + extra,
            host.grid[1] + extra,
            host.grid[2],
        ],
        "workgroupSize": list(entry.workgroup_size),
    }
    records = {
        "params": [conv.params()],
        "gemm_params": [host.gemm],
        "jump_params": [host.jump],
        "base_h": list(host.base_h),
        "base_w": list(host.base_w),
    }
    return resources, records, expected, geometry, host


def _record_storage(binding, name, records):
    layout = binding["scalarLayout"]
    assert layout["elementStrideBytes"] == RECORD_BYTES[name], (name, layout)
    if layout["elementType"] == "record":
        words = pack_storage_records(layout, records)
        return {"dtype": "uint32", "shape": [len(words)], "values": words}
    # A record of int fields alone is reflected as its int words.
    assert layout["elementType"] == "int32", layout
    members = [member["name"] for member in layout["structMembers"]]
    assert all(members == list(record) for record in records), members
    words = [record[member] for record in records for member in members]
    return {"dtype": "int32", "shape": [len(words)], "values": words}


def _dispatch_values(bindings, entry, conv, extra, target):
    resources, records, expected, geometry, _ = _case_data(entry, conv, extra)
    inputs = {}
    for name, binding in bindings.items():
        if name in records:
            value = _record_storage(binding, name, records[name])
        else:
            value = _storage(target, binding, entry.dtype, resources[name])
        inputs[binding["name"]] = value
    outputs = {
        bindings["C"]["name"]: _storage(target, bindings["C"], entry.dtype, expected)
    }
    return inputs, outputs, geometry


def _simulate(entry, conv, host, inputs, weights):
    """The kernel's tap walk and stores, index for index, in Python.

    Each threadgroup ``z`` takes one output residue. Its loaders start at
    the residue's base taps, step ``f_wgt_jump`` taps and move to the next
    channel block after the last tap; ``align_C`` false finishes with
    ``load_safe(C % BK)``. Returns the output words keyed by index.
    """
    params, jump = conv.params(), host.jump
    taps_h, taps_w = params["wS"]
    stride_in, stride_wt, stride_out = (
        params["in_strides"],
        params["wt_strides"],
        params["out_strides"],
    )
    stores = {}
    for z in range(host.grid[2]):
        base_oh, base_ow = divmod(z, jump["f_out_jump_w"])
        bh, bw = host.base_h[base_oh], host.base_w[base_ow]
        positions = bh["weight_size"] * bw["weight_size"]
        if host.align_c:
            schedule = [general.BK] * (positions * host.gemm["gemm_k_iterations"])
        else:
            full = (host.gemm["gemm_k_iterations"] - 1) * positions
            schedule = [general.BK] * full + [conv.channels % general.BK] * positions
        for row in range(host.grid[1] * entry.bm):
            n, hw = divmod(row, jump["adj_out_hw"])
            oh = (hw // jump["adj_out_w"]) * jump["f_out_jump_h"] + base_oh
            ow = (hw % jump["adj_out_w"]) * jump["f_out_jump_w"] + base_ow
            if not (n < conv.batch and oh < conv.out_spatial[0]):
                continue
            if ow >= conv.out_spatial[1]:
                continue
            ih0 = oh * conv.stride[0] - conv.padding[0]
            iw0 = ow * conv.stride[1] - conv.padding[1]
            for o in range(conv.outputs):
                total = 0.0
                weight_h, weight_w, block = bh["weight_base"], bw["weight_base"], 0
                for width in schedule:
                    h = taps_h - weight_h - 1 if conv.flip else weight_h
                    w = taps_w - weight_w - 1 if conv.flip else weight_w
                    ih_dil = ih0 + h * conv.dilation[0]
                    iw_dil = iw0 + w * conv.dilation[1]
                    # Both are tested non-negative before the quotients matter.
                    ih = ih_dil // conv.input_dilation[0]
                    iw = iw_dil // conv.input_dilation[1]
                    if (
                        ih_dil >= 0
                        and ih < conv.spatial[0]
                        and iw_dil >= 0
                        and iw < conv.spatial[1]
                    ):
                        for c in range(block * general.BK, block * general.BK + width):
                            total += (
                                inputs[
                                    n * stride_in[0]
                                    + ih * stride_in[1]
                                    + iw * stride_in[2]
                                    + c
                                ]
                                * weights[
                                    o * stride_wt[0]
                                    + weight_h * stride_wt[1]
                                    + weight_w * stride_wt[2]
                                    + c
                                ]
                            )
                    weight_w += jump["f_wgt_jump_w"]
                    if weight_w < taps_w:
                        continue
                    weight_w = bw["weight_base"]
                    weight_h += jump["f_wgt_jump_h"]
                    if weight_h < taps_h:
                        continue
                    weight_h = bh["weight_base"]
                    block += 1
                index = n * stride_out[0] + oh * stride_out[1] + ow * stride_out[2] + o
                assert index not in stores, (entry.name, conv, index)
                stores[index] = total
    return stores


def test_steel_conv_general_model_names_every_entry():
    assert len(ENTRIES) == len(general.BY_NAME) == 3 * 6
    entry = general.BY_NAME[
        "implicit_gemm_conv_2d_general_bfloat16_bm64_bn8_bk16_wm4_wn1"
    ]
    assert entry.workgroup_size == (32, 1, 4)
    assert entry.leading == 24 and entry.loaders[1].vector == 1
    assert entry.source == SOURCE and entry.accepts(Conv(1, (3, 3), (1, 1), 5, 5))
    assert not entry.accepts(Conv(1, (3, 3), (1, 1), 4, 4, groups=2))
    assert not entry.accepts(Conv(1, (3, 3, 3), (1, 1, 1), 16, 16))


def test_steel_conv_general_host_selection():
    """The host picks 2x2 warps and 32 or 64 rows and columns: three blocks.

    It takes 64 rows only for 64 or more channels and 8192 or more residue
    rows, and 64 columns only with 64 rows.
    """
    reached = set()
    for dtype, _, _ in general.conv_model.TYPES:
        for batch, channels, outputs in (
            (1, 5, 8),
            (40, 64, 32),
            (40, 64, 64),
            (40, 21, 64),
            (1, 64, 64),
        ):
            conv = Conv(batch, (16, 16), (3, 3), channels, outputs, padding=(1, 1))
            entry = general.host_entry(conv, dtype)
            assert entry.accepts(conv)
            reached.add((entry.bm, entry.bn, entry.wm, entry.wn))
    assert reached == {(32, 32, 2, 2), (64, 32, 2, 2), (64, 64, 2, 2)}
    # Input dilation counts residue rows, not output rows.
    dilated = Conv(10, (16, 16), (3, 3), 64, 64, padding=(1, 1), input_dilation=(2, 2))
    assert math.prod(dilated.out_spatial) * 10 >= 8192
    assert general.host_entry(dilated, "float32").bm == 32


def test_steel_conv_general_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry.name for shard in shards for entry in shard) == sorted(
        general.BY_NAME
    )
    for shard in shards:
        assert len(shard) == 6
        assert {entry.dtype for entry in shard} == {
            dtype for dtype, _, _ in general.conv_model.TYPES
        }
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        _selected_entries()
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_steel_conv_general_launch_follows_the_host():
    """``implicit_gemm_conv_2D_general_gpu``'s tables for one dilated case."""
    entry = general.BY_NAME[
        "implicit_gemm_conv_2d_general_float32_bm32_bn32_bk16_wm2_wn2"
    ]
    conv = Conv(
        1, (3, 4), (3, 2), 16, 16, stride=(3, 2), padding=(2, 0), input_dilation=(2, 3)
    )
    assert conv.out_spatial == (3, 5)
    host = general.launch(conv, entry)
    # lcm(idil, str) / str output residues per axis, each a multiple of
    # lcm(idil, kdil) / kdil taps apart.
    assert host.jump == dict(
        f_wgt_jump_h=2,
        f_wgt_jump_w=3,
        f_out_jump_h=2,
        f_out_jump_w=3,
        adj_out_h=2,
        adj_out_w=2,
        adj_out_hw=4,
        adj_implicit_m=4,
    )
    # Row residue 0 starts at input row -2: tap 0 lands on dilated row -2;
    # residue 1 at row 1, so tap 1 lands on row 2.
    assert host.base_h == (
        {"weight_base": 0, "weight_size": 2},
        {"weight_base": 1, "weight_size": 1},
    )
    # Column residue j starts at 2j; taps land on multiples of 3.
    assert host.base_w == (
        {"weight_base": 0, "weight_size": 1},
        {"weight_base": 1, "weight_size": 1},
        {"weight_base": 2, "weight_size": 0},
    )
    assert host.grid == (1, 1, 6) and host.align_c
    assert host.gemm["K"] == 3 * 2 * 16 and host.gemm["gemm_k_iterations"] == 1
    assert (host.gemm["M"], host.gemm["N"]) == (15, 16)
    flipped = general.launch(replace(conv, flip=True), entry)
    # Flipped, weight row 0 is tap 2, which lands on dilated row 0.
    assert flipped.base_h[0] == {"weight_base": 0, "weight_size": 2}
    assert flipped.base_w[0] == {"weight_base": 1, "weight_size": 1}
    unaligned = general.launch(replace(conv, channels=21), entry)
    assert not unaligned.align_c and unaligned.gemm["gemm_k_iterations"] == 2
    assert unaligned.gemm["inp_jump_c"] == 16 - 2 * 4 * 21 - 1 * 21


def test_steel_conv_general_params_follow_the_mlx_struct_layout():
    conv = Conv(2, (4, 5), (2, 3), 32, 12, input_dilation=(2, 3), flip=True)
    params = conv.params()
    assert len(_params_bytes(params)) == RECORD_BYTES["params"]
    assert params["idil"] == [2, 3] and params["oS"] == [6, 11]
    entry = ENTRIES[0]
    host = general.launch(conv, entry)
    assert len(host.gemm) * 4 == RECORD_BYTES["gemm_params"]
    assert len(host.jump) * 4 == RECORD_BYTES["jump_params"]
    assert all(len(base) * 4 == RECORD_BYTES["base_h"] for base in host.base_h)


@pytest.mark.parametrize("aligned", (True, False), ids=("align_C", "unaligned_C"))
@pytest.mark.parametrize(
    "entry",
    [entry for entry in ENTRIES if entry.dtype == "float32"],
    ids=lambda entry: entry.name,
)
def test_steel_conv_general_walk_reaches_the_reference(entry, aligned):
    """The kernel's tap walk over the host tables stores each output once,
    and every stored word equals the exact reference."""
    for conv, extra in _cases(entry, aligned):
        resources, records, expected, geometry, host = _case_data(entry, conv, extra)
        assert host.align_c is aligned
        inputs = resources["A"][:-8]
        weights = resources["B"][:-8]
        stores = _simulate(entry, conv, host, inputs, weights)
        assert sorted(stores) == list(range(len(expected) - 8))
        assert [stores[index] for index in sorted(stores)] == expected[:-8]


@pytest.mark.parametrize("dtype", [name for name, _, _ in general.conv_model.TYPES])
def test_steel_conv_general_references_are_exact_and_guarded(dtype):
    guard = GUARDS[dtype]
    assert _exact(dtype, guard)
    seen = set()
    for entry in ENTRIES:
        if entry.dtype != dtype or entry.bn in seen:
            continue
        seen.add(entry.bn)
        for aligned in (True, False):
            for conv, extra in _cases(entry, aligned):
                resources, records, expected, geometry, host = _case_data(
                    entry, conv, extra
                )
                assert expected[-8:] == [guard] * 8
                assert len(resources["C"]) == len(expected)
                assert any(expected[:-8]) and guard not in expected[:-8]
                assert geometry["workgroupCount"][2] == len(host.base_h) * len(
                    host.base_w
                )
                assert records["gemm_params"][0]["N"] == conv.outputs
    assert seen == set(OUTPUTS)


@pytest.fixture(scope="module")
def general_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned steel_conv_general gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    tmp_path = tmp_path_factory.mktemp("steel-conv-general-original")
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
    for name, digest in CONV_HEADERS.items():
        header = root / "mlx/backend/metal/kernels/steel/conv" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest, name
    source = root / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    register_default_sources()
    discovery = SOURCE_REGISTRY.get("metal").discover_entry_points(
        source.read_text(encoding="utf-8"),
        file_path=str(source),
        include_paths=[str(root)],
    )
    assert discovery.status == ENTRY_DISCOVERY_AVAILABLE
    assert [entry.name for entry in discovery.entries] == list(general.BY_NAME)
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


def _original_outputs(request, source, module, work, name, aligned):
    executor = _executor("metal")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        constant = RuntimeSpecializationConstant(
            name="align_C",
            constant_id=general.ALIGN_C_ID,
            dtype="bool",
            value=aligned,
            kind="function-constant",
        )
        control = replace(
            native,
            artifact_path=source,
            module_path=module,
            entry_point=name,
            loaded_artifact=None,
            constants={
                "align_C": NativeRuntimeConstantBinding(
                    name="align_C",
                    constant=constant,
                    value=aligned,
                    source="original-source-specialization",
                )
            },
        )
        actual = executor.runtime_adapter.runtime.dispatch(None, state, control)
        module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
        assert state.details["metalRuntime"]["librarySHA256"] == module_hash
        _write_json(
            work / "original-evidence.json",
            {
                "outputs": actual,
                "sourceCommit": MLX_COMMIT,
                "sourceSha256": SOURCE_HASH,
                "moduleSha256": module_hash,
                "alignC": aligned,
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


def _conv_json(conv):
    return {
        "batch": conv.batch,
        "spatial": list(conv.spatial),
        "kernel": list(conv.kernel),
        "channels": conv.channels,
        "outputs": conv.outputs,
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "dilation": list(conv.dilation),
        "inputDilation": list(conv.input_dilation),
        "flip": conv.flip,
    }


@pytest.mark.parametrize(
    "entry, aligned",
    SELECTED,
    ids=[f"{e.name}-{'align_C' if a else 'unaligned_C'}" for e, a in SELECTED],
)
def test_current_steel_conv_general_executes(tmp_path, general_source, entry, aligned):
    root, original, target = general_source
    results = []
    with tempfile.TemporaryDirectory(prefix=".current-convg-", dir=root) as directory:
        work = Path(directory)
        try:
            report = translate_project(
                _config(root, work.name, target, entry, aligned), format_output=False
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
            assert [(item["name"], item["id"]) for item in constants] == [
                ("align_C", general.ALIGN_C_ID)
            ]
            assert constants[0]["frozen"] is True
            assert constants[0]["value"] is aligned
            bindings = {
                item["provenance"]["sourceResource"]["parameter"]: item
                for item in descriptor["bindings"]
            }
            assert len(bindings) == len(descriptor["bindings"])
            assert set(bindings) == set(BINDINGS), sorted(bindings)
            for index, (conv, extra) in enumerate(_cases(entry, aligned)):
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
                assert not request.adapter_contract.specialization_constants
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
                            request,
                            root / SOURCE,
                            original,
                            case_work,
                            entry.name,
                            aligned,
                        )
                        == record["outputs"]
                    )
                assert record["outputs"] == outputs
                results.append(
                    {
                        "conv": _conv_json(conv),
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
            "alignC": aligned,
            "source": SOURCE,
            "sourceSha256": SOURCE_HASH,
            "cases": results,
            "complete": len(results) == len(_cases(entry, aligned)),
            "scope": SCOPE,
        },
    )
