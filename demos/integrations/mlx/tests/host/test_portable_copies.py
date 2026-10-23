"""Lazy casts and copies check their layouts, carry every real dtype exactly
and recompute each destination word from the retained source words."""

import ctypes
import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    copy_dispatch,
    copy_evidence,
    copy_layout,
    copy_packages,
    element_storage,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACT = ROOT / "demos/integrations/mlx/contracts/copy.current-tree.translation.json"
TARGETS = ("metal", "directx", "opengl")
NUMPY = {
    "bool_": np.bool_,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
}
WORDS = {"float16": np.uint16, "float32": np.uint32}


def test_entries_are_pinned_copy_instantiations():
    pinned = set(json.loads(CONTRACT.read_text(encoding="utf-8"))["entries"])
    assert set(copy_packages.ENTRIES) <= pinned
    assert len(copy_packages.ENTRIES) == 12 * 11 + 12
    assert copy_packages.signature("v_copyint8float16") == (
        "v_copy",
        "int8",
        "float16",
    )
    for entry in ("v_copyint8int8", "v_copycomplex64float32", "gg1_copyint8int8", 7):
        with pytest.raises(ValueError, match="Unknown copy entry"):
            copy_packages.signature(entry)
    opengl = copy_packages.options("opengl", "ggn2_dynamic_copyuint8uint8")
    assert opengl.count("[[project.index_range_assertions]]") == 4
    assert "index_range" not in copy_packages.options("opengl", "v_copyint8float32")
    assert "index_range" not in copy_packages.options(
        "directx", "ggn2_dynamic_copyuint8uint8"
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", element_storage.DTYPES)
def test_storage_packets_round_trip_every_real_dtype(target, dtype):
    words = element_storage.guard_words(dtype, 6)
    value = element_storage.packet(dtype, target, words)
    physical = value["dtype"]
    assert element_storage.unpack(dtype, target, value["values"]) == words
    if dtype == "bool_":
        assert physical == ("bool" if target == "metal" else "uint32")
        assert words == [1, 0, 1, 0, 1, 0]
    elif dtype in {"uint8", "int8"}:
        assert physical == (dtype if target == "metal" else dtype[:-1] + "32")
    elif dtype in {"uint16", "int16"}:
        assert physical == (dtype if target != "opengl" else dtype[:-2] + "32")
    layout = {
        "elementType": physical,
        "elementStrideBytes": element_storage.PHYSICAL_SIZES[physical],
    }
    assert element_storage.reflected_layout_matches(layout, value)


def test_storage_refuses_inexact_carriers():
    with pytest.raises(ValueError, match="canonical"):
        element_storage.packet("bool_", "directx", [2])
    with pytest.raises(ValueError, match="non-canonical"):
        element_storage.unpack("bool_", "metal", [1])
    with pytest.raises(ValueError, match="non-canonical"):
        element_storage.unpack("bool_", "opengl", [True])
    # A widened int8 word outside the int8 range was not normalized natively.
    with pytest.raises(ValueError, match="invalid scalar storage"):
        element_storage.unpack("int8", "opengl", [200])
    with pytest.raises(ValueError, match="invalid scalar storage"):
        element_storage.unpack("uint16", "opengl", [-1])
    with pytest.raises(ValueError, match="Unsupported real storage"):
        element_storage.packet("complex64", "metal", [0])


def float_words(values, dtype):
    return np.array(values, dtype=NUMPY[dtype]).view(WORDS[dtype]).tolist()


def test_casts_between_floats_round_to_nearest_even_with_gradual_underflow():
    rng = np.random.default_rng(7)
    singles = rng.integers(0, 1 << 32, 4096, dtype=np.uint64).astype(np.uint32)
    halves = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    with np.errstate(over="ignore", invalid="ignore"):
        narrowed = singles.view(np.float32).astype(np.float16).view(np.uint16)
        widened = halves.view(np.float16).astype(np.float32).view(np.uint32)
    for word, expected in zip(singles.tolist(), narrowed.tolist()):
        actual = copy_evidence.cast_word(word, "float32", "float16")
        if actual == copy_evidence.NAN:
            assert copy_evidence.is_nan(expected, "float16")
        else:
            assert actual == expected, hex(word)
    for word, expected in zip(halves.tolist(), widened.tolist()):
        actual = copy_evidence.cast_word(word, "float16", "float32")
        assert actual == expected or (
            actual == copy_evidence.NAN and copy_evidence.is_nan(expected, "float32")
        )
    # bfloat16 keeps binary32's exponent: nearest even on the low 16 bits.
    assert copy_evidence.cast_word(0x3F808000, "float32", "bfloat16") == 0x3F80
    assert copy_evidence.cast_word(0x3F818000, "float32", "bfloat16") == 0x3F82
    assert copy_evidence.cast_word(0x7F7FFFFF, "float32", "bfloat16") == 0x7F80
    assert copy_evidence.cast_word(0x3C00, "float16", "bfloat16") == 0x3F80
    assert copy_evidence.cast_word(0x0001, "float16", "bfloat16") == 0x3380
    with pytest.raises(ValueError, match="binary32 subnormals"):
        copy_evidence.cast_word(0x00000001, "float32", "bfloat16")
    with pytest.raises(ValueError, match="binary32 subnormals"):
        copy_evidence.cast_word(0x0001, "bfloat16", "float32")
    assert copy_evidence.cast_word(0x80000001, "float32", "float16") == 0x8000


@pytest.mark.parametrize("source", ("float16", "float32"))
@pytest.mark.parametrize(
    "destination", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64")
)
def test_casts_from_floats_truncate_in_range_values(source, destination):
    low, high = copy_evidence.integer_bounds(destination)
    values = [*(-130.75, -1.5, -0.0, 0.25, 3.9, 127.5, 255.99, 300.0)]
    for value in values:
        word = float_words([value], source)[0]
        stored = float(np.array([value], dtype=NUMPY[source])[0])
        if low <= math.trunc(stored) <= high:
            assert copy_evidence.cast_word(word, source, destination) == math.trunc(
                stored
            )
        else:
            with pytest.raises(ValueError, match="outside the integer range"):
                copy_evidence.cast_word(word, source, destination)
    for special in (math.nan, math.inf):
        with pytest.raises(ValueError, match="NaN or infinite"):
            copy_evidence.cast_word(float_words([special], source)[0], source, "int8")


def test_casts_to_bool_test_storage_and_integers_wrap():
    assert copy_evidence.cast_word(0x80000000, "float32", "bool_") == 0
    assert copy_evidence.cast_word(0x00000001, "float32", "bool_") == 1
    assert copy_evidence.cast_word(0x7FC00000, "float32", "bool_") == 1
    assert copy_evidence.cast_word(0x0001, "float16", "bool_") == 1
    assert copy_evidence.cast_word(0x8000, "bfloat16", "bool_") == 0
    assert copy_evidence.cast_word(-7, "int8", "bool_") == 1
    for source, destination in (
        ("int32", "int8"),
        ("int64", "uint16"),
        ("uint32", "int16"),
    ):
        values = [-(1 << 31), -129, -1, 0, 127, 128, 255, 65535, (1 << 31) - 1]
        values = [v for v in values if copy_evidence.integer_bounds(source)[0] <= v]
        expected = np.array(values, dtype=NUMPY[source]).astype(NUMPY[destination])
        assert [
            copy_evidence.cast_word(v, source, destination) for v in values
        ] == expected.tolist()
    assert copy_evidence.cast_word(1, "bool_", "float16") == 0x3C00
    assert copy_evidence.cast_word(1, "bool_", "int8") == 1


def test_casts_from_integers_round_once():
    for value in (0, 1, -3, (1 << 24) + 1, (1 << 62) + (1 << 38) + 1, -(1 << 63)):
        expected = struct.unpack("<I", struct.pack("<f", float(np.float32(value))))[0]
        if value == (1 << 62) + (1 << 38) + 1:
            # binary64 rounds this first; one rounding goes up instead.
            expected = copy_evidence.float_word(value, "float32")
            assert expected == 0x5E800001
        assert copy_evidence.cast_word(value, "int64", "float32") == expected
    assert copy_evidence.cast_word(65504, "int32", "float16") == 0x7BFF
    assert copy_evidence.cast_word(65520, "int32", "float16") == 0x7C00
    with pytest.raises(ValueError, match="exact in binary32"):
        copy_evidence.cast_word((1 << 24) + 1, "int32", "float16")
    assert copy_evidence.cast_word((1 << 64) - 1, "uint64", "float32") == 0x5F800000


class FakeCache:
    def __init__(self, entry, target):
        self.entry, self.target = entry, target

    def get(self, entry):
        assert entry == self.entry
        prefix = "" if self.target == "metal" else entry + "_"
        kernel, source, destination = copy_packages.signature(entry)
        names = (
            ("src", "dst", "size")
            if kernel == copy_packages.VECTOR
            else tuple(copy_layout.DTYPES)
        )
        bindings = []
        for name in names:
            dtype = (
                source
                if name == "src"
                else (
                    destination
                    if name == "dst"
                    else copy_layout.DTYPES.get(name, "uint32")
                )
            )
            value = (
                element_storage.packet(dtype, self.target, [])
                if name in {"src", "dst"}
                else {"dtype": dtype}
            )
            physical = value["dtype"]
            scalar = {
                "elementType": physical,
                "elementStrideBytes": element_storage.PHYSICAL_SIZES[physical],
            }
            scalar["memberName"] = prefix + name
            bindings.append({"name": name + "Binding", "scalarLayout": scalar})
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def native_buffers(kernel, source, destination, values, shape=None, strides=None):
    """A callback's buffers: a cast of ``values`` or a strided general copy."""
    size = len(values)
    memory = {
        "src": (element_storage.CTYPES[source] * size)(*values),
    }
    if kernel == copy_packages.VECTOR:
        memory["dst"] = (element_storage.CTYPES[destination] * size)()
        memory["size"] = (ctypes.c_uint32 * 1)(size)
        types = {"src": source, "dst": destination, "size": "uint32"}
        threads = size
    else:
        threads = math.prod(shape)
        memory["dst"] = (element_storage.CTYPES[destination] * threads)()
        dense = [math.prod(shape[axis + 1 :]) for axis in range(len(shape))]
        memory.update(
            src_shape=(ctypes.c_int32 * len(shape))(*shape),
            src_strides=(ctypes.c_int64 * len(shape))(*strides),
            dst_strides=(ctypes.c_int64 * len(shape))(*dense),
            ndim=(ctypes.c_int32 * 1)(len(shape)),
            src_offset=(ctypes.c_int64 * 1)(0),
            dst_offset=(ctypes.c_int64 * 1)(0),
        )
        types = {
            name: source if name in {"src", "dst"} else kind
            for name, kind in copy_layout.DTYPES.items()
        }
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == "dst"),
            )
            for name in memory
        )
    )
    return array, memory, threads


def run(
    monkeypatch, tmp_path, target, entry, values, shape=None, strides=None, fault=None
):
    kernel, source, destination = copy_packages.signature(entry)
    array, memory, threads = native_buffers(
        kernel, source, destination, values, shape, strides
    )
    host = SimpleNamespace(
        copies=FakeCache(entry, target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )
    seen = {}

    def execute(_, request):
        seen["request"] = request
        dst = request["inputs"]["dstBinding"]
        initial = element_storage.unpack(destination, target, dst["values"])
        words = element_storage.unpack(
            source, target, request["inputs"]["srcBinding"]["values"]
        )
        if kernel == copy_packages.VECTOR:
            result = [
                copy_evidence.cast_word(word, source, destination) for word in words
            ]
            result += initial[threads:]
        else:
            metadata = copy_layout.validate(
                {name: array[i] for i, name in enumerate(memory)}, threads, dtype=source
            )
            result = copy_evidence.expected_copy(metadata, words, initial)
        if fault == "guard":
            result[-1] = result[-2] ^ 1
        if fault == "range" and target != "metal":
            result[0] = 1 << 20
        return SimpleNamespace(
            status="ok",
            outputs={"dstBinding": element_storage.packet(destination, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        copy_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(copy_dispatch, "execute", execute)
    workgroups = (
        [threads, 1, 1]
        if kernel == copy_packages.VECTOR
        else [(shape[-1] + 1) // 2, shape[-2], math.prod(shape[:-2])]
    )
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*workgroups), (ctypes.c_uint32 * 3)(1, 1, 1)
    )
    copy_dispatch.dispatch(host, entry, array, len(memory), threads, launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["dst"]), event, host


@pytest.mark.parametrize("target", TARGETS)
def test_casts_dispatch_through_the_reflected_carriers(monkeypatch, tmp_path, target):
    values = [-128, -1, 0, 1, 127]
    words, event, host = run(monkeypatch, tmp_path, target, "v_copyint8float32", values)
    assert host.dispatch_count == 1
    assert words == float_words([float(v) for v in values], "float32")
    assert copy_evidence.audit_event(event, native=False) == 5
    words, event, _ = run(
        monkeypatch, tmp_path, target, "v_copyfloat16bool_", [0x0001, 0x8000, 0x7E00]
    )
    assert words == [1, 0, 1]
    assert copy_evidence.audit_event(event, native=False) == 3
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, target, "v_copyuint16int8", [1, 2], fault="guard")


def test_widened_results_outside_their_type_are_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="invalid scalar storage"):
        run(monkeypatch, tmp_path, "opengl", "v_copyint32int16", [1, 2], fault="range")


@pytest.mark.parametrize("target", TARGETS)
def test_general_copies_move_strided_narrow_words(monkeypatch, tmp_path, target):
    values = list(range(12))
    words, event, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "ggn2_dynamic_copyuint8uint8",
        values,
        shape=[4, 3],
        strides=[1, 4],
    )
    assert words == np.arange(12).reshape(3, 4).T.reshape(-1).tolist()
    assert copy_evidence.audit_event(event, native=False) == 12
    changed = dict(event, outputValues=list(event["outputValues"]))
    changed["outputValues"][5] ^= 1
    with pytest.raises(ValueError, match="result 5"):
        copy_evidence.audit_event(changed, native=False)
    uploaded = json.loads(json.dumps(event))
    uploaded["sourceValues"][0] = 9
    with pytest.raises(ValueError, match="upload differs"):
        copy_evidence.audit_event(uploaded, native=False)
    hashed = dict(event, outputHash="0" * 64)
    with pytest.raises(ValueError, match="output hash"):
        copy_evidence.audit_event(hashed, native=False)


def test_vector_layout_rejections():
    array, memory, threads = native_buffers("v_copy", "int8", "int16", [1, 2, 3])
    supplied = {name: array[i] for i, name in enumerate(memory)}
    execution = {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]}
    copy_dispatch.validate_vector(supplied, 3, execution, "int8", "int16")
    with pytest.raises(ValueError, match="launch does not match"):
        copy_dispatch.validate_vector(
            supplied, 3, {**execution, "workgroupCount": [4, 1, 1]}, "int8", "int16"
        )
    with pytest.raises(ValueError, match="type, extent"):
        copy_dispatch.validate_vector(supplied, 3, execution, "uint8", "int16")
    with pytest.raises(ValueError, match="host contract"):
        copy_dispatch.validate_vector(supplied, 65536, execution, "int8", "int16")
    memory["size"][0] = 2
    with pytest.raises(ValueError, match="size does not match"):
        copy_dispatch.validate_vector(supplied, 3, execution, "int8", "int16")


def test_audit_derives_the_cached_package_and_checks_native_modules(
    monkeypatch, tmp_path
):
    _, event, _ = run(monkeypatch, tmp_path, "directx", "v_copyint16uint8", [3, -1])
    package = tmp_path / "package"
    (package / "artifact.hlsl").parent.mkdir(parents=True)
    (package / "artifact.hlsl").write_bytes(b"kernel")
    artifact = {
        "packagePath": "artifact.hlsl",
        "sizeBytes": 6,
        "hash": {"algorithm": "sha256", "value": hashlib.sha256(b"kernel").hexdigest()},
    }
    cache = SimpleNamespace(get=lambda entry: ({"artifact": artifact}, package))
    event.update(artifact=artifact, packageRoot=str(package))
    assert copy_evidence.audit_event(event, cache, native=False) == 2
    with pytest.raises(ValueError, match="cached package"):
        copy_evidence.audit_event(
            dict(event, packageRoot="/other"), cache, native=False
        )
    seen = []
    monkeypatch.setattr(copy_evidence, "audit_native_execution", seen.append)
    event.update(
        workgroupCount=[2, 1, 1],
        workgroupSize=[1, 1, 1],
        details={
            "request": {
                "target": "directx",
                "entryPoint": "CSMain",
                "dispatch": {"workgroupCount": [2, 1, 1], "workgroupSize": [1, 1, 1]},
            }
        },
    )
    assert copy_evidence.audit_event(event, cache) == 2 and seen == [event]
    event["details"]["request"]["entryPoint"] = "main"
    with pytest.raises(ValueError, match="native request changed"):
        copy_evidence.audit_event(event, cache)


def test_runtime_prefers_installed_families_over_lazy_copies():
    host = SimpleNamespace(descriptors={"v_copyint32float32": {}}, copies=object())
    assert runtime.HostRuntime.lazy_copy(host, "v_copyint8float32")
    assert not runtime.HostRuntime.lazy_copy(host, "v_copyint32float32")
    assert not runtime.HostRuntime.lazy_copy(host, "v_Absint8int8")
    host.copies = None
    assert not runtime.HostRuntime.lazy_copy(host, "v_copyint8float32")
