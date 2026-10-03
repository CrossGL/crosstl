"""Package and address contracts for the pinned slice-update kernels."""

import ctypes
import json
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    packages,
    prepare,
    runtime,
    slice_update_layout,
)


def buffers_for(dtype="float32", *, shape=(2, 3), strides=(6, 2), offset=0, count=12):
    logical = 1
    dense = []
    for size in reversed(shape):
        dense.insert(0, logical)
        logical *= size
    spec = {
        "updates": (dtype, [1] * logical),
        "out": (dtype, [1 if dtype == "bool_" else 2] * count),
        "update_shape": ("int32", list(shape)),
        "update_strides": ("int64", dense),
        "update_ndim": ("int32", [len(shape)]),
        "update_size": ("int64", [logical]),
        "output_strides": ("int64", list(strides)),
        "output_offset": ("int64", [offset]),
    }
    arrays = {
        name: (runtime.TYPES[kind] * len(values))(*values)
        for name, (kind, values) in spec.items()
    }
    buffers = {
        name: runtime.Buffer(
            name.encode(),
            kind.encode(),
            ctypes.cast(arrays[name], ctypes.c_void_p),
            len(values),
            2 if name == "out" else 0,
        )
        for name, (kind, values) in spec.items()
    }
    return buffers, arrays, logical


@pytest.mark.parametrize("dtype", packages.SLICE_UPDATE_TYPES)
@pytest.mark.parametrize(
    "shape,strides,offset,count,wanted",
    [
        ((2, 3), (3, 1), 0, 6, [0, 1, 2, 3, 4, 5]),
        ((2, 3), (6, 2), 0, 12, [0, 2, 4, 6, 8, 10]),
        ((2, 3), (6, -2), 4, 12, [4, 2, 0, 10, 8, 6]),
        ((3, 2), (1, 3), 0, 6, [0, 3, 1, 4, 2, 5]),
        ((1,), (1,), 3, 6, [3]),
    ],
)
def test_slice_update_layout_preserves_unique_signed_addresses(
    dtype, shape, strides, offset, count, wanted
):
    buffers, keepalive, logical = buffers_for(
        dtype, shape=shape, strides=strides, offset=offset, count=count
    )
    metadata = slice_update_layout.validate(buffers, logical, dtype=dtype)
    assert list(slice_update_layout.destination_indices(metadata)) == wanted
    assert metadata["destinationCount"] == count
    assert keepalive["updates"][0] == 1


@pytest.mark.parametrize(
    "change,error",
    [
        ("missing", "signature"),
        ("dtype", "type or direction"),
        ("direction", "type or direction"),
        ("null", "type or direction"),
        ("rank-zero", "rank must"),
        ("rank-limit", "rank must"),
        ("rank-value", "rank does not match"),
        ("size-value", "size does not match"),
        ("metadata-count", "metadata shape"),
        ("update-count", "metadata shape"),
        ("output-count", "output span"),
        ("output-limit", "output span"),
        ("shape-zero", "shape does not match"),
        ("shape-product", "shape does not match"),
        ("source-stride", "dense row-major"),
        ("offset-negative", "destination exceeds"),
        ("offset-high", "destination exceeds"),
        ("negative-stride", "destination exceeds"),
        ("stride-limit", "destination exceeds"),
        ("destination-alias", "destinations overlap"),
        ("source-alias", "source and destination overlap"),
    ],
)
def test_slice_update_layout_rejects_invalid_storage_before_dispatch(change, error):
    buffers, values, logical = buffers_for()
    if change == "missing":
        del buffers["updates"]
    elif change == "dtype":
        buffers["updates"].dtype = b"int32"
    elif change == "direction":
        buffers["out"].output = 1
    elif change == "null":
        buffers["updates"].data = None
    elif change.startswith("rank-"):
        if change == "rank-value":
            values["update_ndim"][0] = 3
        else:
            buffers["update_shape"].count = 0 if change == "rank-zero" else 65
    elif change == "size-value":
        values["update_size"][0] = 5
    elif change == "metadata-count":
        buffers["output_strides"].count = 1
    elif change == "update-count":
        buffers["updates"].count = 5
    elif change.startswith("output-"):
        buffers["out"].count = 5 if change == "output-count" else 65536
    elif change.startswith("shape-"):
        values["update_shape"][0] = 0 if change == "shape-zero" else 3
    elif change == "source-stride":
        values["update_strides"][0] = 4
    elif change.startswith("offset-"):
        values["output_offset"][0] = -1 if change == "offset-negative" else 3
    elif change == "negative-stride":
        values["output_strides"][1] = -2
    elif change == "stride-limit":
        values["output_strides"][0] = 65536
    elif change == "destination-alias":
        values["output_strides"][0] = 0
    elif change == "source-alias":
        buffers["out"].data = buffers["updates"].data
    else:
        raise AssertionError(change)
    with pytest.raises(ValueError, match=error):
        slice_update_layout.validate(buffers, logical, dtype="float32")


@pytest.mark.parametrize("logical", [0, -1, 65536, 6.0, True])
def test_slice_update_layout_requires_bounded_integer_size(logical):
    buffers, keepalive, _ = buffers_for()
    with pytest.raises(ValueError, match="size exceeds"):
        slice_update_layout.validate(buffers, logical, dtype="float32")
    assert keepalive


def test_slice_update_layout_accepts_maximum_count():
    buffers, keepalive, logical = buffers_for(
        "uint64", shape=(65535,), strides=(-1,), offset=65534, count=65535
    )
    metadata = slice_update_layout.validate(buffers, logical, dtype="uint64")
    assert list(slice_update_layout.destination_indices(metadata)) == list(
        range(65534, -1, -1)
    )
    assert keepalive


def test_slice_update_family_uses_unmodified_template_bodies():
    assert len(packages.SLICE_UPDATE_ENTRIES) == 24
    assert not set(packages.SLICE_UPDATE_ENTRIES) & set(packages.ENTRIES)
    assert packages.SLICE_UPDATE_SOURCE.count("#include") == 3
    assert "{" not in packages.SLICE_UPDATE_SOURCE
    for entry in packages.SLICE_UPDATE_ENTRIES:
        assert packages.SLICE_UPDATE_SOURCE.count(f'host_name("{entry}")') == 1
    assert packages.SLICE_UPDATE_SOURCE.count("false, true, false, 1, 0>") == 48


@pytest.fixture(scope="module", params=["metal", "directx", "opengl"])
def translated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("slice-update-" + request.param)
    kernels = root / "mlx/backend/metal/kernels"
    (kernels / "indexing").mkdir(parents=True)
    (kernels / "utils.h").write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
    )
    (kernels / "reduce_utils.h").write_text(
        "\n".join(
            f"template<typename T> struct {operation} {{}};"
            for operation in packages.SLICE_UPDATE_OPERATIONS
        )
    )
    # This fixture checks reflected signatures; native CI uses upstream bodies.
    (kernels / "indexing/scatter.h").write_text(
        """template<typename T, typename IdxT, typename Op,
bool OUT_ROW_CONTIG, bool UPD_ROW_CONTIG, bool UPD_SCALAR, int NWORK, int NDIM>
kernel void slice_update_op_impl(
const device T* updates [[buffer(0)]], device T* out [[buffer(1)]],
constant int* update_shape [[buffer(2)]], constant long* update_strides [[buffer(3)]],
constant int& update_ndim [[buffer(4)]], constant long& update_size [[buffer(5)]],
constant long* output_strides [[buffer(6)]], constant long& output_offset [[buffer(7)]],
uint3 gid [[thread_position_in_grid]], uint3 gsize [[threads_per_grid]]) {
    out[gid.x] = updates[gid.x];
}
"""
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(
            root, root / "packages", request.param, family="slice-update"
        )
    assert set(index["descriptors"]) == set(packages.SLICE_UPDATE_ENTRIES)
    assert index["family"] == "slice-update"
    assert (
        root / "packages/translation/slice-update.metal"
    ).read_text() == packages.SLICE_UPDATE_SOURCE
    return root / "packages", index


@pytest.fixture
def host(translated, tmp_path):
    directory, index = translated
    base = tmp_path / "base"
    base.mkdir()
    (base / "index.json").write_text(
        json.dumps(
            {
                "target": index["target"],
                "descriptors": {
                    entry: {"target": index["target"]} for entry in packages.ENTRIES
                },
            }
        )
    )
    return runtime.HostRuntime(base, tmp_path / "trace", slice_updates=directory)


@pytest.mark.parametrize("entry", packages.SLICE_UPDATE_ENTRIES)
@pytest.mark.parametrize("fault", [None, "metadata", "guard", "gap", "integer-range"])
def test_slice_update_callback_checks_layout_guards_and_gaps(
    host, monkeypatch, entry, fault
):
    dtype = packages.SLICE_UPDATE_ENTRIES[entry]
    supplied, memory, logical = buffers_for(dtype)
    buffers = (runtime.Buffer * len(supplied))(*supplied.values())
    before = list(memory["out"])
    expected = before.copy()
    written = [0, 2, 4, 6, 8, 10]
    for index in written:
        expected[index] = 0 if dtype == "bool_" else 7
    storage = runtime.physical_dtype(dtype, host.target)
    guard = (
        list(runtime.BOOLEAN_GUARD)
        if dtype == "bool_"
        else (
            [
                ctypes.c_float.from_buffer_copy(ctypes.c_uint32(value)).value
                for value in runtime.COPY_GUARD
            ]
            if dtype == "float32"
            else list(runtime.COPY_GUARD)
        )
    )
    values = (
        [
            ctypes.c_uint32.from_buffer_copy(ctypes.c_float(value)).value
            for value in expected
        ]
        + list(runtime.COPY_GUARD)
        if dtype == "float32"
        else expected + guard
    )
    if dtype == "bool_":
        values = [bool(value) if storage == "bool" else int(value) for value in values]
    if fault == "metadata":
        memory["output_offset"][0] = -1
    elif fault == "guard":
        values[-1] = (
            (not values[-1] if storage == "bool" else int(not values[-1]))
            if dtype == "bool_"
            else 0
        )
    elif fault == "gap":
        values[1] = not values[1] if storage == "bool" else 0
    elif fault == "integer-range":
        if dtype == "bool_":
            values[0] = 2
        elif dtype != "float32":
            values[0] = 2**64
        else:
            values.pop()
    calls = []
    binding = next(
        item["name"]
        for item in host.descriptors[entry]["bindings"]
        if item["access"] == "read_write"
    )

    def execute(request):
        calls.append(request)
        return SimpleNamespace(
            status="ok",
            details={},
            outputs={
                binding: {
                    "dtype": storage,
                    "shape": [len(before) + len(guard)],
                    "values": values,
                    **(
                        {"encoding": runtime.FLOAT32_BITS} if dtype == "float32" else {}
                    ),
                }
            },
        )

    monkeypatch.setattr(host.executor, "run", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(logical, 1, 1), (ctypes.c_uint32 * 3)(1, 1, 1)
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, len(buffers), logical, launch=launch)
        assert list(memory["out"]) == before and not host.trace.exists()
    else:
        host.dispatch(entry, buffers, len(buffers), logical, launch=launch)
        assert list(memory["out"]) == expected
        record = json.loads(host.trace.read_text())
        assert record["sliceUpdateValues"] == expected
        if dtype == "float32":
            assert record["sliceUpdateStorageWords"] == values[: len(before)]
            assert record["sliceUpdateGuardWords"] == list(runtime.COPY_GUARD)
        assert record["sliceUpdateMetadata"]["destinationStrides"] == [6, 2]
        assert record["workgroupCount"] == [logical, 1, 1]
    assert len(calls) == int(fault != "metadata")


def test_entry_lookup_tracks_available_packages_without_dispatch(host, monkeypatch):
    monkeypatch.setattr(
        host.executor, "run", lambda *args: pytest.fail("Entry query dispatched")
    )
    entry = "slice_update_sumfloat32"
    assert host.entry_available(entry.encode()) == 1
    del host.descriptors[entry]
    assert host.entry_available(entry.encode()) == 0
    assert host.entry_available(b"unknown") == host.entry_available(None) == 0
    assert host.entry_available(b"\xff") == 0
    assert host.dispatch_count == 0 and not host.trace.exists()
