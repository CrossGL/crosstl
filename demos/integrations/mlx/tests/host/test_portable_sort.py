"""MLX merge sorts through the portable host: check gpu_merge_sort's plan,
dispatch every stage's package and audit the retained uploads, guards and
recomputed outputs."""

import ctypes
import json
import random
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    prepare,
    runtime,
    sort_dispatch,
    sort_evidence,
    sort_layout,
    sort_packages,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACT = ROOT / "demos/integrations/mlx/contracts/sort.current-tree.translation.json"
TARGETS = ("metal", "directx", "opengl")
NAN32 = 0x7FC00000


def f32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def words_of(dtype, count, seed=0):
    """Values with ties, both zero signs and NaNs, as storage words."""
    generator = random.Random(seed)
    pool = [-2.5, -0.0, 0.0, 1.0, 1.0, 3.25, 7.0]
    if dtype == "float32":
        return [
            NAN32 if generator.random() < 0.05 else f32(generator.choice(pool))
            for _ in range(count)
        ]
    if dtype == "complex64":
        return [
            f32(generator.choice(pool)) | f32(generator.choice(pool[:4])) << 32
            for _ in range(count)
        ]
    if dtype == "int32":
        return [generator.randrange(-5, 5) for _ in range(count)]
    raise AssertionError(dtype)


def row_major(shape):
    strides, size = [], 1
    for extent in reversed(shape):
        strides.append(size)
        size *= extent
    return strides[::-1]


def block_metadata(entry, shape, strides, axis):
    """The metadata single_block_sort binds for a view of ``shape``."""
    _, layout, _, _, _ = sort_layout.signature(entry)
    out_strides = row_major(shape)
    nc_shape = [e for i, e in enumerate(shape) if i != axis]
    in_nc = [s for i, s in enumerate(strides) if i != axis]
    out_nc = [s for i, s in enumerate(out_strides) if i != axis]
    metadata = {
        "size_sorted_axis": shape[axis],
        "in_stride_sorted_axis": strides[axis],
        "out_stride_sorted_axis": out_strides[axis],
    }
    if layout == "c":
        live = [i for i, e in enumerate(nc_shape) if e != 1]
        metadata["in_stride_segment_axis"] = min(
            (in_nc[i] for i in live), default=sort_layout.SIGNED_MAX
        )
        metadata["out_stride_segment_axis"] = min(
            (out_nc[i] for i in live), default=sort_layout.SIGNED_MAX
        )
    else:
        metadata.update(
            nc_dim=len(nc_shape),
            nc_shape=nc_shape or [0],
            in_nc_strides=in_nc or [0],
            out_nc_strides=out_nc or [0],
        )
    return metadata


def native_buffers(entry, metadata, values, counts):
    """The callback's buffers in parameter order, as the native host binds them."""
    types = sort_layout.operand_types(entry)
    written = sort_layout.outputs(entry)
    memory = {}
    for name, kind in types.items():
        if name in sort_layout.ARRAYS:
            memory[name] = (sort_layout.METADATA_TYPES[kind] * len(metadata[name]))(
                *metadata[name]
            )
        elif name not in sort_layout.ELEMENTS:
            memory[name] = (ctypes.c_int32 * 1)(metadata[name])
        elif name in written:
            memory[name] = (element_storage.CTYPES[kind] * counts[name])()
        else:
            memory[name] = (element_storage.CTYPES[kind] * len(values[name]))(
                *values[name]
            )
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name in written),
            )
            for name in memory
        )
    )
    return array, memory


class FakeCache:
    """The sort cache: one package per entry and width."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        entry, _ = sort_layout.split_variant(key)
        prefix = "" if self.target == "metal" else entry.rstrip("_") + "_"
        bindings = []
        for name, kind in sort_layout.operand_types(entry).items():
            value = element_storage.packet(kind, self.target, [0])
            if kind == "complex64":
                scalar = dict(element_storage.COMPLEX_LAYOUT)
            else:
                scalar = {
                    "elementType": value["dtype"],
                    "elementStrideBytes": element_storage.PHYSICAL_SIZES[
                        value["dtype"]
                    ],
                }
            element = name in sort_layout.ELEMENTS or name in sort_layout.ARRAYS
            member = "out_" if name == "out" else name if element else prefix + name
            bindings.append(
                {"name": name + "Binding", "scalarLayout": {**scalar, "memberName": member}}
            )
        return {"bindings": bindings, "artifact": {"id": key}}, Path("/packages")


def run(monkeypatch, tmp_path, target, entry, metadata, values, logical, fault=None):
    """Dispatch one stage through fake packages whose kernel is the model."""
    types = sort_layout.operand_types(entry)
    written = sort_layout.outputs(entry)
    counts = {}
    for name in written:
        if name == "block_partitions":
            counts[name] = (logical // metadata["size_sorted_axis"]) * (
                metadata["n_blocks"] + 1
            )
        else:
            counts[name] = logical
    _, n_blocks = sort_layout.launch(
        metadata["size_sorted_axis"], sort_layout.signature(entry)[3]
    )
    rows = logical // metadata["size_sorted_axis"]
    stage = sort_layout.signature(entry)[0]
    width = sort_layout.width(entry, n_blocks)
    groups = {"block": [1, rows, 1], "partition": [1, rows, 1]}.get(
        stage, [n_blocks, rows, 1]
    )
    launch = {"workgroupCount": groups, "workgroupSize": [width, 1, 1]}
    array, memory = native_buffers(entry, metadata, values, counts)
    host = SimpleNamespace(
        sorts=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=sort_dispatch.PREFIXES,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        assert request["execution"] == launch
        seen = {
            name: element_storage.unpack(
                types[name], target, inputs[name + "Binding"]["values"]
            )
            for name in sort_dispatch.inputs_of(entry)
        }
        assert seen == values
        model = sort_evidence.expected_outputs(entry, metadata, seen, counts, logical)
        outputs = {}
        for name in written:
            initial = element_storage.unpack(
                types[name], target, inputs[name + "Binding"]["values"]
            )
            assert initial == element_storage.guard_words(
                types[name], counts[name] + element_storage.GUARD_COUNT
            ) or types[name] == "bool_"
            result = model[name] + initial[counts[name] :]
            if fault == "guard":
                result[-1] ^= 1
            if fault == "value" and name == written[0]:
                result[0] ^= 1
            outputs[name + "Binding"] = element_storage.packet(
                types[name], target, result
            )
        return SimpleNamespace(status="ok", outputs=outputs, details={"request": {}})

    monkeypatch.setattr(
        sort_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(sort_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    sort_dispatch.dispatch(host, entry, array, len(array), logical, native)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return {name: list(memory[name]) for name in written}, event


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(sort_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        sort_evidence, "audit_native_execution", lambda e: checked.append(target)
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": event["entry"],
                "dispatch": {key: event[key] for key in sort_evidence.GEOMETRY},
            }
        },
    }
    retained = sort_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return retained


def keys(dtype, words):
    return [sort_evidence.decode(dtype, word) for word in words]


def reference(dtype, words):
    """Python's stable sort under MLX LessThan, as values and indices."""
    order = sort_evidence.stable_order(keys(dtype, words))
    return [words[i] for i in order], order


def test_entries_are_the_pinned_instantiations():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["provenance"]["commit"] == prepare.COMMIT
    assert contract["provenance"]["source"].endswith("sort.metal")
    assert set(sort_layout.ENTRIES) == set(contract["entries"])
    assert len(sort_layout.ENTRIES) == 264
    assert all(
        entry.startswith(sort_dispatch.PREFIXES) for entry in sort_layout.ENTRIES
    )


@pytest.mark.parametrize(
    "size, dtype, expected",
    [
        (1, "float32", (32, 1)),
        (128, "float32", (32, 1)),
        (129, "float32", (64, 1)),
        (257, "int32", (128, 1)),
        (1024, "int32", (256, 1)),
        (1025, "uint8", (512, 1)),
        (2048, "float16", (512, 1)),
        (2049, "float32", (512, 2)),
        (1025, "complex64", (256, 2)),
        (1024, "int64", (256, 1)),
        (32769, "float32", (512, 17)),
        (2**22, "float32", (512, 2048)),
    ],
)
def test_launch_mirrors_gpu_merge_sort(size, dtype, expected):
    assert sort_layout.launch(size, dtype) == expected


def test_entry_selection_and_variants():
    assert (
        sort_layout.entry_name("block", "float32", 10)
        == "c_block_sort_float32_float32_bn32_tn4"
    )
    assert (
        sort_layout.entry_name("block", "int32", 300, argsort=True, contiguous=False)
        == "ncarg_block_sort_int32_uint32_bn128_tn4"
    )
    assert (
        sort_layout.entry_name("merge", "complex64", 1025)
        == "merge_mbsort_complex64_uint32_bn256_tn4"
    )
    with pytest.raises(ValueError, match="stage"):
        sort_layout.entry_name("block", "float32", 4096)
    with pytest.raises(ValueError, match="stage"):
        sort_layout.entry_name("sort", "float32", 2048)
    partition = sort_layout.entry_name("partition", "float32", 32769)
    assert sort_layout.width(partition, 17) == 18
    assert sort_layout.width(partition, 2048) == 1024
    assert sort_layout.split_variant(partition + "@18") == (partition, (18, 1, 1))
    for key in (partition + "@2", partition + "@1025", "c_block_sort_bool__bool__bn32_tn4@32"):
        with pytest.raises(ValueError):
            sort_layout.split_variant(key)
    block = sort_layout.entry_name("block", "float32", 600)
    assert sort_layout.variant(block, 256) == block + "@256"
    with pytest.raises(ValueError, match="width"):
        sort_layout.variant(block, 512)
    assert sort_layout.merge_levels(17) == [2, 4, 8, 16, 32]
    assert sort_layout.merge_levels(2) == [2]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("argsort", (False, True))
@pytest.mark.parametrize(
    "dtype, shape, strides, axis, contiguous",
    [
        # A contiguous (6, 4, 10) array along its last and first axes.
        ("float32", (6, 4, 10), (40, 10, 1), 2, True),
        ("int32", (6, 4, 10), (40, 10, 1), 0, True),
        # Its (3, 4, 5) strided view along the middle axis, and flattened.
        ("complex64", (3, 4, 5), (80, 10, 2), 1, False),
        ("float32", (60,), (1,), 0, True),
        # A broadcast (16, 8) array along both axes, and a strided 1D view.
        ("int32", (16, 8), (0, 1), 0, True),
        ("int32", (16, 8), (0, 1), 1, False),
        ("int32", (4,), (2,), 0, False),
    ],
)
def test_single_block_dispatch_and_audit(
    monkeypatch, tmp_path, target, argsort, dtype, shape, strides, axis, contiguous
):
    entry = sort_layout.entry_name("block", dtype, shape[axis], argsort, contiguous)
    span = 1 + sum((e - 1) * s for e, s in zip(shape, strides))
    words = words_of(dtype, span, seed=len(shape) + axis)
    metadata = block_metadata(entry, shape, strides, axis)
    logical = 1
    for extent in shape:
        logical *= extent
    outputs, event = run(
        monkeypatch, tmp_path, target, entry, metadata, {"inp": words}, logical
    )
    # Every row of the view, in the output's row-major order, is sorted.
    out = outputs["out"]
    out_strides = row_major(shape)
    rows = logical // shape[axis]
    others = [i for i in range(len(shape)) if i != axis]
    for row in range(rows):
        index, in_base, out_base = row, 0, 0
        for i in reversed(others):
            index, position = divmod(index, shape[i])
            in_base += position * strides[i]
            out_base += position * out_strides[i]
        row_words = [words[in_base + k * strides[axis]] for k in range(shape[axis])]
        values, order = reference(dtype, row_words)
        got = [out[out_base + k * out_strides[axis]] for k in range(shape[axis])]
        expected = order if argsort else values
        if dtype == "complex64" and not argsort:
            expected = [ctypes.c_uint64(word).value for word in expected]
        assert got == expected
    assert audit(monkeypatch, event, target) is True


def test_multi_block_chain_sorts_a_row(monkeypatch, tmp_path):
    """sort_mbsort, then each partition and merge pass, is the stable sort."""
    dtype, size = "float32", 2 * 2048 + 900
    words = words_of(dtype, 2 * size - 1, seed=7)
    bn, n_blocks = sort_layout.launch(size, dtype)
    assert (bn, n_blocks) == (512, 3)
    metadata = {
        "size_sorted_axis": size,
        "stride_sorted_axis": 2,
        "nc_dim": 0,
        "nc_shape": [0],
        "nc_strides": [1],
    }
    entry = sort_layout.entry_name("sort", dtype, size)
    stage, event = run(
        monkeypatch, tmp_path, "directx", entry, metadata, {"inp": words}, size
    )
    assert audit(monkeypatch, event, "directx") is True
    vals, idxs = stage["out_vals"], stage["out_idxs"]
    for merge_tiles in sort_layout.merge_levels(n_blocks):
        common = {"size_sorted_axis": size, "merge_tiles": merge_tiles}
        partition = sort_layout.entry_name("partition", dtype, size)
        parts, event = run(
            monkeypatch,
            tmp_path,
            "directx",
            partition,
            {**common, "n_blocks": n_blocks},
            {"dev_vals": vals, "dev_idxs": idxs},
            size,
        )
        assert audit(monkeypatch, event, "directx") is True
        merge = sort_layout.entry_name("merge", dtype, size)
        merged, event = run(
            monkeypatch,
            tmp_path,
            "directx",
            merge,
            {**common, "num_tiles": n_blocks},
            {
                "block_partitions": parts["block_partitions"],
                "dev_vals_in": vals,
                "dev_idxs_in": idxs,
            },
            size,
        )
        assert audit(monkeypatch, event, "directx") is True
        vals, idxs = merged["dev_vals_out"], merged["dev_idxs_out"]
    values, order = reference(dtype, words[::2])
    assert vals == values and idxs == order


def test_partition_model_is_the_kernel_binary_search():
    keys_a = [0.0, 1.0, 1.0, 3.0]
    keys_b = [1.0, 2.0, float("nan")]
    for diagonal in range(len(keys_a) + len(keys_b) + 1):
        merged = sort_evidence.stable_order(keys_a + keys_b)[:diagonal]
        assert sort_evidence.merge_partition(keys_a, keys_b, diagonal) == sum(
            index < len(keys_a) for index in merged
        )


def test_less_than_orders_nan_last_and_complex_lexicographically():
    nan = float("nan")
    assert sort_evidence.less_than(1.0, nan) and not sort_evidence.less_than(nan, 1.0)
    assert not sort_evidence.less_than(nan, nan)
    assert not sort_evidence.less_than(-0.0, 0.0)
    assert sort_evidence.less_than((1.0, 2.0), (1.0, 3.0))
    assert sort_evidence.less_than((1.0, 9.0), (2.0, 0.0))
    assert sort_evidence.less_than((5.0, 0.0), (1.0, nan))
    words = [f32(3.0), NAN32, f32(2.0), f32(-0.0), f32(0.0)]
    assert reference("float32", words)[1] == [3, 4, 2, 0, 1]


@pytest.mark.parametrize("fault", ("guard", "value"))
def test_dispatch_refuses_guard_damage_and_audit_refuses_wrong_values(
    monkeypatch, tmp_path, fault
):
    entry = sort_layout.entry_name("block", "float32", 10)
    metadata = block_metadata(entry, (2, 10), (10, 1), 1)
    words = words_of("float32", 20, seed=3)
    if fault == "guard":
        with pytest.raises(RuntimeError, match="guard"):
            run(
                monkeypatch, tmp_path, "metal", entry, metadata, {"inp": words}, 20,
                fault=fault,
            )
        return
    _, event = run(
        monkeypatch, tmp_path, "metal", entry, metadata, {"inp": words}, 20,
        fault=fault,
    )
    with pytest.raises(ValueError, match="hash|stable"):
        audit(monkeypatch, event, "metal")
    event["outputHash"] = sort_dispatch.output_hash(entry, event["outputValues"])
    with pytest.raises(ValueError, match="stable LessThan"):
        audit(monkeypatch, event, "metal")


def test_compacted_records_keep_extents(monkeypatch, tmp_path):
    monkeypatch.setattr(sort_dispatch, "RETAINED_LIMIT", 64)
    monkeypatch.setattr(sort_evidence, "RETAINED_LIMIT", 64)
    entry = sort_layout.entry_name("block", "int32", 300)
    metadata = block_metadata(entry, (300,), (1,), 0)
    words = words_of("int32", 300, seed=5)
    outputs, event = run(
        monkeypatch, tmp_path, "opengl", entry, metadata, {"inp": words}, 300
    )
    assert outputs["out"] == sorted(words)
    assert "compactedValues" in event["operandValues"]["inp"]
    assert audit(monkeypatch, event, "opengl") is False
    tampered = json.loads(json.dumps(event))
    tampered["operandValues"]["inp"]["compactedValues"]["length"] = 299
    with pytest.raises(ValueError):
        audit(monkeypatch, tampered, "opengl")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda m: m.update(size_sorted_axis=2049), "axis length"),
        (lambda m: m.update(in_stride_sorted_axis=-1), "strides"),
        (lambda m: m.update(nc_shape=[5, 5]), "rows"),
        (lambda m: m.update(in_nc_strides=[80, 3]), "span"),
        (lambda m: m.update(nc_dim=1), "ranks"),
    ],
)
def test_plan_refuses_launches_outside_gpu_merge_sort(change, message):
    entry = sort_layout.entry_name("block", "complex64", 4, contiguous=False)
    metadata = block_metadata(entry, (3, 4, 5), (80, 10, 2), 1)
    change(metadata)
    counts = {"inp": 1 + 2 * 80 + 3 * 10 + 4 * 2, "out": 60}
    with pytest.raises(ValueError, match=message):
        sort_layout.plan(entry, metadata, counts, 60)


def test_plan_refuses_unrun_merge_passes_and_overlapping_partitions():
    entry = sort_layout.entry_name("partition", "float32", 32769)
    metadata = {"size_sorted_axis": 32769, "merge_tiles": 64, "n_blocks": 17}
    counts = {"block_partitions": 18, "dev_vals": 32769, "dev_idxs": 32769}
    with pytest.raises(ValueError, match="merge pass"):
        sort_layout.plan(entry, metadata, counts, 32769)
    metadata["merge_tiles"] = 32
    assert sort_layout.plan(entry, metadata, counts, 32769) == {
        "workgroupCount": [1, 1, 1],
        "workgroupSize": [18, 1, 1],
    }
    size = 1024 * 2048 + 1
    metadata = {"size_sorted_axis": size, "merge_tiles": 2, "n_blocks": 1025}
    counts = {"block_partitions": 2 * 1026, "dev_vals": 2 * size, "dev_idxs": 2 * size}
    with pytest.raises(ValueError, match="1023 tiles"):
        sort_layout.plan(entry, metadata, counts, 2 * size)


def test_package_configs_bound_opengl_views_and_offsets(tmp_path):
    for key in (
        "ncarg_block_sort_float32_uint32_bn64_tn4@64",
        "partition_mbsort_float32_uint32_bn512_tn4@18",
        "merge_mbsort_complex64_uint32_bn256_tn4@256",
    ):
        entry, workgroup = sort_layout.split_variant(key)
        configs = {
            target: sort_packages.project_config(tmp_path, "out", target, key)
            for target in TARGETS
        }
        for config in configs.values():
            assert list(config.include_patterns) == [sort_layout.SOURCE]
            (rules,) = config.entry_workgroup_size_rules.values()
            assert {k: tuple(map(int, v)) for k, v in rules.items()} == {
                entry: workgroup
            }
            assert not config.subgroup_width_rules
        for target in ("metal", "directx"):
            assert not configs[target].index_range_assertions
            assert not configs[target].workgroup_access_assertions
        opengl = configs["opengl"]
        assert [item.expression for item in opengl.index_range_assertions] == list(
            sort_layout.index_assertions(entry)
        )
        tile = sort_layout.signature(entry)[4] * 4
        views = [item.to_json() for item in opengl.workgroup_access_assertions]
        assert all(view["maximum"] == tile - 1 for view in views)
        assert len(views) == sum(
            len(parameters) for _, parameters in sort_layout.workgroup_views(entry)
        )
    keys = sort_packages.verification_entries()
    assert keys == sorted(set(keys))
    assert {
        "partition_mbsort_float32_uint32_bn512_tn4@1024",
        "partition_mbsort_float32_uint32_bn512_tn4@18",
        "partition_mbsort_float32_uint32_bn512_tn4@7",
        "c_block_sort_complex64_complex64_bn64_tn4@64",
        "ncarg_block_sort_int64_uint32_bn32_tn4@32",
        "merge_mbsort_int32_uint32_bn512_tn4@512",
    } <= set(keys)


def test_runtime_and_host_adapter_route_sorts():
    fake = SimpleNamespace(
        descriptors={}, copies=None, elementwise=None, aranges=None, arg_reductions=None
    )
    fake.softmaxes = None
    fake.sorts = object()
    assert runtime.HostRuntime._entry_available(fake, b"sort_mbsort_int8_uint32_bn512_tn4") == 1
    fake.sorts = None
    assert runtime.HostRuntime._entry_available(fake, b"c_block_sort_int8_int8_bn32_tn4") == 0
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in sort_layout.ENTRIES:" in source
    adapter = Path(runtime.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    for primitive in ("ArgSort", "Sort", "ArgPartition", "Partition"):
        assert f"void {primitive}::eval_gpu(" in adapter
        assert f'"{primitive}",' in Path(prepare.__file__).read_text(encoding="utf-8")
    for stage in ("sort_mbsort_", "partition_mbsort_", "merge_mbsort_"):
        assert f'"{stage}" + suffix' in adapter
