"""Lazy aranges mirror the pinned instantiations, check their launches and
scalars, and recompute integer results exactly."""

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    arange_dispatch,
    arange_evidence,
    arange_packages,
    element_storage,
    packages,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")
NARROW = ("uint8", "uint16", "int8", "int16")


def test_entries_are_every_pinned_arange_but_the_installed_family():
    path = CONTRACTS / "arange.current-tree.translation.json"
    pinned = set(json.loads(path.read_text(encoding="utf-8"))["entries"])
    assert len(pinned) == 11
    assert set(arange_packages.ENTRIES) == pinned - set(packages.ARANGE_ENTRIES)
    assert sorted(arange_packages.ENTRIES.values()) == sorted(
        ("float16", "bfloat16", *NARROW)
    )
    assert arange_packages.signature("arangebfloat16") == "bfloat16"
    for entry in ("arangefloat32", "arangebool_", "arangecomplex64", None):
        with pytest.raises(ValueError, match="Unknown lazy arange entry"):
            arange_packages.signature(entry)
    assert arange_packages.RECIPE.source == packages.SOURCE
    assert arange_packages.options("opengl", "arangeint8") == (
        'binary32_fma_profile = "rne-flush"\n'
        'binary32_sqrt_profile = "flush-subnormals"\n'
        'binary32_rsqrt_profile = "flush-subnormals"\n'
    )


def native_buffers(entry, start, step, threads):
    """A callback's buffers: the two scalar constants and the output."""
    dtype = arange_packages.signature(entry)
    ctype = element_storage.CTYPES[dtype]
    memory = {
        "start": (ctype * 1)(start),
        "step": (ctype * 1)(step),
        "out": (ctype * threads)(),
    }
    array = (runtime.Buffer * 3)(
        *(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == "out"),
            )
            for name in memory
        )
    )
    return array, memory


class FakeCache:
    def __init__(self, target):
        self.target = target

    def get(self, entry):
        dtype = arange_packages.signature(entry)
        physical = element_storage.packet(dtype, self.target, [])["dtype"]
        prefix = "" if self.target == "metal" else entry + "_"
        bindings = [
            {
                "name": name + "Binding",
                "scalarLayout": {
                    "elementType": physical,
                    "elementStrideBytes": element_storage.PHYSICAL_SIZES[physical],
                    "memberName": prefix + {"out": "out_"}.get(name, name),
                },
            }
            for name in ("start", "step", "out")
        ]
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def run(monkeypatch, tmp_path, target, entry, start, step, threads, *, fault=None):
    dtype = arange_packages.signature(entry)
    array, memory = native_buffers(entry, start, step, threads)
    host = SimpleNamespace(
        aranges=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        scalars = {
            name: element_storage.unpack(
                dtype, target, request["inputs"][name + "Binding"]["values"]
            )[0]
            for name in ("start", "step")
        }
        initial = element_storage.unpack(
            dtype, target, request["inputs"]["outBinding"]["values"]
        )
        if dtype in element_storage.INEXACT:
            # The 16-bit floats echo their start; identity checks cover them.
            result = [scalars["start"]] * threads
        else:
            result = arange_evidence.expected_words(
                dtype, scalars["start"], scalars["step"], threads
            )
        result += initial[threads:]
        if fault == "guard":
            result[-1] ^= 1
        if fault == "result":
            result[0] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet(dtype, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        arange_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(arange_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(threads, 1, 1), (ctypes.c_uint32 * 3)(1, 1, 1)
    )
    arange_dispatch.dispatch(host, entry, array, 3, threads, launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", NARROW)
def test_narrow_aranges_wrap_like_numpy(monkeypatch, tmp_path, target, dtype):
    info = np.iinfo(dtype)
    start, step = info.max - 2, 1
    words, event, host = run(
        monkeypatch, tmp_path, target, f"arange{dtype}", start, step, 6
    )
    assert host.dispatch_count == 1
    expected = (np.arange(6, dtype=np.int64) * step + start).astype(dtype)
    assert words == expected.tolist()
    assert event["arangeMetadata"] == {"dtype": dtype}
    assert event["operandValues"] == {"start": [start], "step": [step]}
    assert arange_evidence.audit_event(event, native=False) == 6
    words, event, _ = run(
        monkeypatch, tmp_path, target, f"arange{dtype}", 3, -2 if info.min else 7, 5
    )
    assert arange_evidence.audit_event(event, native=False) == 5
    tampered = {**event, "outputValues": [words[0] ^ 1, *words[1:]]}
    with pytest.raises(ValueError, match="differs from its scalars"):
        arange_evidence.audit_event(tampered, native=False)
    tampered = {
        **event,
        "operandValues": {"start": [4], "step": event["operandValues"]["step"]},
    }
    with pytest.raises(ValueError, match="upload differs"):
        arange_evidence.audit_event(tampered, native=False)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_sixteen_bit_float_aranges_use_reflected_carriers(
    monkeypatch, tmp_path, target, dtype
):
    start = int(np.float16(1.5).view(np.uint16)) if dtype == "float16" else 0x3FC0
    words, event, _ = run(monkeypatch, tmp_path, target, f"arange{dtype}", start, 0, 3)
    assert words == [start] * 3
    assert not arange_evidence.recomputable(f"arange{dtype}")
    upload = event["inputs"]["startBinding"]
    assert upload == element_storage.packet(dtype, target, [start])


def test_dispatch_refuses_changed_guards_and_launches(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, "opengl", "arangeint8", 0, 1, 2, fault="guard")
    dtype = "int16"
    array, memory = native_buffers("arangeint16", 0, 1, 4)
    buffers = arange_dispatch.supplied_buffers(array, 3, "Arange")
    execution = {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]}
    assert arange_dispatch.validate("arangeint16", buffers, 4, execution) == {
        "dtype": dtype
    }
    for threads, launch, message in (
        (0, execution, "size exceeds"),
        (65536, execution, "size exceeds"),
        (4, {"workgroupCount": [4, 1, 1], "workgroupSize": [2, 1, 1]}, "launch"),
        (3, {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]}, "extent"),
    ):
        with pytest.raises(ValueError, match=message):
            arange_dispatch.validate("arangeint16", buffers, threads, launch)
    with pytest.raises(ValueError, match="type, extent"):
        arange_dispatch.validate("arangeuint16", buffers, 4, execution)
    with pytest.raises(ValueError, match="buffers do not match"):
        arange_dispatch.validate(
            "arangeint16", {"start": buffers["start"]}, 4, execution
        )
    host = SimpleNamespace(aranges=None)
    with pytest.raises(ValueError, match="pinned sources"):
        arange_dispatch.dispatch(host, "arangeint16", array, 3, 4, None)


def test_recompute_refuses_floats_and_out_of_range_scalars():
    assert arange_evidence.expected_words("uint8", 250, 3, 3) == [250, 253, 0]
    assert arange_evidence.expected_words("int16", -32768, -1, 2) == [-32768, 32767]
    with pytest.raises(ValueError, match="integer type"):
        arange_evidence.expected_words("float16", 0, 1, 1)
    with pytest.raises(ValueError, match="storage range"):
        arange_evidence.expected_words("int8", 128, 1, 1)


def test_float_events_check_package_and_native_identity(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        arange_evidence, "audit_native_execution", lambda event: seen.append(event)
    )
    package = tmp_path / "package"
    package.mkdir()
    (package / "artifact.hlsl").write_bytes(b"kernel")
    artifact = {
        "packagePath": "artifact.hlsl",
        "sizeBytes": 6,
        "hash": {"algorithm": "sha256", "value": hashlib.sha256(b"kernel").hexdigest()},
    }
    cache = SimpleNamespace(get=lambda entry: ({"artifact": artifact}, package))
    event = {
        "entry": "arangefloat16",
        "target": "directx",
        "threads": 4,
        "workgroupCount": [4, 1, 1],
        "workgroupSize": [1, 1, 1],
        "dispatchVersion": 3,
        "artifact": artifact,
        "packageRoot": str(package),
        "arangeMetadata": {"dtype": "float16"},
        "details": {
            "request": {
                "target": "directx",
                "entryPoint": "CSMain",
                "dispatch": {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
            }
        },
    }
    assert arange_evidence.audit_event(event, cache) == 4 and len(seen) == 1
    with pytest.raises(ValueError, match="metadata"):
        arange_evidence.audit_event(
            {**event, "arangeMetadata": {"dtype": "bfloat16"}}, cache
        )
    with pytest.raises(ValueError, match="native request changed"):
        arange_evidence.audit_event({**event, "workgroupCount": [2, 1, 1]}, cache)
    (package / "artifact.hlsl").write_bytes(b"change")
    with pytest.raises(ValueError, match="changed after translation"):
        arange_evidence.audit_event(event, cache)


def test_runtime_prefers_the_installed_family_over_lazy_aranges():
    host = SimpleNamespace(descriptors={"arangeint32": {}}, aranges=object())
    lazy = runtime.HostRuntime.lazy_arange
    assert lazy(host, "arangeint8") and lazy(host, "arangebfloat16")
    assert not lazy(host, "arangeint32") and not lazy(host, "arangefloat32")
    assert not lazy(host, "arangebool_")
    host.aranges = None
    assert not lazy(host, "arangeint8")


@pytest.mark.parametrize("dtype", (*NARROW, "int32", "uint32", "int64", "uint64"))
def test_backend_splits_integer_aranges_into_exact_launches(dtype):
    source = (ROOT / "demos/integrations/mlx/portable_host/backend.cpp").read_text()
    start = source.index("void dispatch_arange(")
    body = source[start : source.index("\n}\n", start)]
    # Each launch covers at most 65,535 elements from element ``first`` and
    # starts at start + first * step, wrapped to the element width.
    for text in (
        "for (uint64_t first = 0; first < count; first += 65535)",
        "if constexpr (std::is_integral_v<T>)",
        "first * uint64_t(static_cast<Unsigned>(step))",
        '{"out", dtype, out.data<T>() + first, size, 1}',
        "elementwise_launch(size)",
    ):
        assert text in body, text
    start = source.index("void Arange::eval_gpu(")
    body = source[start : source.index("\n}\n", start)]
    assert "issubdtype(out.dtype(), floating) && out.size() > 65535" in body
    # The wrapped base reproduces the single launch's wrapped products.
    kind = np.dtype(dtype)
    unsigned = np.dtype(kind.str.replace("i", "u"))
    first_value, step = np.array([-7, 3], dtype=np.int64).astype(kind)
    count, first = 3 * 65535 + 17, 65535 * 2
    with np.errstate(over="ignore"):
        whole = first_value + np.arange(count).astype(kind) * step
        base = (
            (
                np.uint64(first_value.astype(unsigned))
                + np.uint64(first) * np.uint64(step.astype(unsigned))
            )
            .astype(unsigned)
            .view(kind)
        )
        part = base + np.arange(count - first).astype(kind) * step
    assert np.array_equal(part, whole[first:])
