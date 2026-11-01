"""Softmax and logsumexp rows of the portable host: check every launch against
Softmax::eval_gpu's and LogSumExp::eval_gpu's plan, dispatch each width's
package and audit the retained uploads, guards and packages."""

import ctypes
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    prepare,
    runtime,
    softmax_dispatch,
    softmax_evidence,
    softmax_layout,
    softmax_packages,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")


def float_word(dtype, value):
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def word_value(dtype, word):
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", word))[0]
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", word))[0]
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def sample_words(dtype, count):
    return [float_word(dtype, ((index * 37) % 23) / 8 - 1) for index in range(count)]


def reference(entry, words, axis_size):
    """Softmax of each row, or its logsumexp, rounded to the element type."""
    operation, _, dtype, _ = softmax_layout.signature(entry)
    result = []
    for start in range(0, len(words), axis_size):
        row = [word_value(dtype, word) for word in words[start : start + axis_size]]
        top = max(row)
        exps = [math.exp(value - top) for value in row]
        total = sum(exps)
        if operation == "softmax":
            result += [float_word(dtype, value / total) for value in exps]
        else:
            result.append(float_word(dtype, top + math.log(total)))
    return result


def launch_for(entry, axis_size, rows):
    return {
        "workgroupCount": [rows, 1, 1],
        "workgroupSize": [softmax_layout.width(entry, axis_size), 1, 1],
    }


def output_count(entry, axis_size, rows):
    return rows * axis_size if softmax_layout.signature(entry)[0] == "softmax" else rows


def native_buffers(entry, words, outputs, axis_size):
    """A callback's three buffers: the rows, the output and ``axis_size``."""
    ctype = element_storage.CTYPES[softmax_layout.signature(entry)[2]]
    types = softmax_layout.operand_types(entry)
    memory = {
        "in": (ctype * len(words))(*words),
        "out": (ctype * outputs)(),
        "axis_size": (ctypes.c_int32 * 1)(axis_size),
    }
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == "out"),
            )
            for name in memory
        )
    )
    return array, memory


class FakeCache:
    """The softmax cache: one package per entry and width."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        entry, _ = softmax_layout.split_variant(key)
        # Direct3D and OpenGL prefix the scalars with the entry, once.
        prefix = "" if self.target == "metal" else entry.rstrip("_") + "_"
        bindings = []
        for name, kind in softmax_layout.operand_types(entry).items():
            value = element_storage.packet(kind, self.target, [])
            scalar = {
                "elementType": value["dtype"],
                "elementStrideBytes": element_storage.PHYSICAL_SIZES[value["dtype"]],
            }
            if self.target == "directx" and kind == "float16":
                scalar = {
                    "elementType": "uint16",
                    "elementStrideBytes": 2,
                    "storageEncoding": {
                        "encoding": "ieee754-binary16",
                        "logicalElementType": "float16",
                    },
                }
            member = {"in": "in_", "out": "out_"}.get(name, prefix + name)
            bindings.append(
                {
                    "name": name + "Binding",
                    "scalarLayout": {**scalar, "memberName": member},
                }
            )
        return {"bindings": bindings, "artifact": {"id": key}}, Path("/packages")


def run(monkeypatch, tmp_path, target, entry, words, axis_size, *, fault=None):
    _, _, dtype, _ = softmax_layout.signature(entry)
    rows = len(words) // axis_size
    outputs = output_count(entry, axis_size, rows)
    launch = launch_for(entry, axis_size, rows)
    array, memory = native_buffers(entry, words, outputs, axis_size)
    host = SimpleNamespace(
        softmaxes=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        values = element_storage.unpack(dtype, target, inputs["inBinding"]["values"])
        initial = element_storage.unpack(dtype, target, inputs["outBinding"]["values"])
        assert values == words and len(initial) == outputs + element_storage.GUARD_COUNT
        assert element_storage.unpack(
            "int32", target, inputs["axis_sizeBinding"]["values"]
        ) == [axis_size]
        assert request["execution"] == launch
        result = reference(entry, values, axis_size) + initial[outputs:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet(dtype, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        softmax_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(softmax_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    softmax_dispatch.dispatch(host, entry, array, len(array), len(words), native)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(softmax_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        softmax_evidence, "audit_native_execution", lambda e: checked.append(target)
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": event["entry"],
                "dispatch": {key: event[key] for key in softmax_evidence.GEOMETRY},
            }
        },
    }
    count = softmax_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return count


def test_entries_are_the_pinned_instantiations():
    for operation, source in softmax_layout.SOURCES.items():
        path = CONTRACTS / f"{operation}.current-tree.translation.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["provenance"]["commit"] == prepare.COMMIT
        assert contract["provenance"]["source"].endswith(Path(source).name)
        assert {
            entry
            for entry, (kind, _, _, _) in softmax_layout.ENTRIES.items()
            if kind == operation
        } == set(contract["entries"])
    assert len(softmax_layout.ENTRIES) == 16
    assert all(
        entry.startswith(softmax_layout.PREFIXES) for entry in softmax_layout.ENTRIES
    )


def test_kernels_and_widths_follow_upstream_and_key_each_variant():
    name = softmax_layout.entry_name
    assert name("softmax", "float32", 4096) == "block_softmax_float32"
    assert name("softmax", "float32", 4097, precise=True) == "looped_softmax_float32"
    assert (
        name("softmax", "float16", 1024, precise=True)
        == "block_softmax_precise_float16"
    )
    assert name("logsumexp", "bfloat16", 8193) == "looped_logsumexp_bfloat16"
    block = "block_softmax_float32"
    assert [softmax_layout.width(block, size) for size in (1, 128, 129, 4096)] == [
        32,
        32,
        64,
        1024,
    ]
    assert softmax_layout.width("looped_logsumexp_float16", 4097) == 1024
    for entry, size in ((block, 4097), ("looped_softmax_float32", 4096), (block, 0)):
        with pytest.raises(ValueError):
            softmax_layout.width(entry, size)
    assert softmax_layout.variant(block, 64) == "block_softmax_float32@64"
    assert softmax_layout.split_variant("looped_softmax_float32@1024") == (
        "looped_softmax_float32",
        (1024, 1, 1),
    )
    for key in ("block_softmax_float32@48", "looped_softmax_float32@512", "x@32", 7):
        with pytest.raises(ValueError):
            softmax_layout.split_variant(key)


def test_plan_refuses_launches_outside_upstream(monkeypatch):
    entry = "block_softmax_float32"
    plan = softmax_layout.plan
    assert plan(entry, 32, 4096, 4096, 4096) == launch_for(entry, 32, 128)
    assert plan("block_logsumexp_float32", 8, 32, 4, 32) == {
        "workgroupCount": [4, 1, 1],
        "workgroupSize": [32, 1, 1],
    }
    refused = (
        (entry, 32, 4000 + 1, 4001, 4001),  # partial row
        (entry, 32, 4096, 4096, 4095),  # logical size differs
        (entry, 32, 4096, 128, 4096),  # softmax output is per element
        ("block_logsumexp_float32", 32, 4096, 4096, 4096),  # one output per row
        (entry, 1, 65536, 65536, 65536),  # too many rows
        ("looped_softmax_float32", 4097, 4097 * 2, 4097 * 2, 4097 * 2 - 1),
    )
    for case in refused:
        with pytest.raises(ValueError):
            plan(*case)
    # The element bound keeps every offset signed; a wider bound would not.
    monkeypatch.setattr(softmax_layout, "MAX_ELEMENTS", 2**32)
    with pytest.raises(ValueError, match="signed 32-bit"):
        plan("looped_softmax_float32", 2**27, 2**31, 2**31, 2**31)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,axis_size,rows",
    (
        ("block_softmax_float32", 32, 6),
        ("block_softmax_float16", 129, 2),
        ("block_softmax_precise_bfloat16", 1024, 1),
        ("looped_softmax_float32", 4097, 1),
        ("block_logsumexp_float32", 8, 4),
        ("looped_logsumexp_float16", 4100, 2),
    ),
)
def test_dispatch_returns_and_audits_each_kernel(
    monkeypatch, tmp_path, target, entry, axis_size, rows
):
    dtype = softmax_layout.signature(entry)[2]
    words = sample_words(dtype, axis_size * rows)
    out, event, host = run(monkeypatch, tmp_path, target, entry, words, axis_size)
    assert out == reference(entry, words, axis_size)
    assert host.dispatch_count == 1
    width = softmax_layout.width(entry, axis_size)
    assert host.softmaxes.requests == [f"{entry}@{width}"]
    assert event["axisSize"] == axis_size
    assert event["workgroupCount"] == [rows, 1, 1]
    assert audit(monkeypatch, event, target) == len(out)


def test_dispatch_and_audit_refuse_changed_results(monkeypatch, tmp_path):
    entry = "block_softmax_float32"
    words = sample_words("float32", 64)
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, "opengl", entry, words, 32, fault="guard")
    out, event, _ = run(monkeypatch, tmp_path, "opengl", entry, words, 32)
    tampered = {**event, "outputValues": [out[0] ^ 1, *out[1:]]}
    with pytest.raises(ValueError, match="output hash"):
        audit(monkeypatch, tampered, "opengl")
    changed = {**event, "operandValues": [words[0] ^ 1, *words[1:]]}
    with pytest.raises(ValueError, match="upload differs"):
        audit(monkeypatch, changed, "opengl")
    moved = {**event, "workgroupSize": [64, 1, 1]}
    with pytest.raises(ValueError, match="upstream plan"):
        audit(monkeypatch, moved, "opengl")
    resized = {**event, "axisSize": 16}
    with pytest.raises(ValueError, match="upstream plan"):
        audit(monkeypatch, resized, "opengl")
    compacted = {**event, "operandValues": {"length": 64, "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="retain"):
        audit(monkeypatch, compacted, "opengl")


def test_dispatch_refuses_incomplete_buffers_and_subnormal_bfloat(
    monkeypatch, tmp_path
):
    host = SimpleNamespace(softmaxes=None, target="metal")
    with pytest.raises(ValueError, match="complete buffers"):
        softmax_dispatch.dispatch(
            host, "block_softmax_float32", [object()], 3, 1, object()
        )
    array, _ = native_buffers("block_softmax_float32", [0] * 64, 64, 32)
    buffers = {array[i].name.decode(): array[i] for i in range(len(array))}
    with pytest.raises(ValueError, match="buffer type"):
        softmax_layout.validate(
            "block_softmax_float16",
            buffers,
            64,
            launch_for("block_softmax_float16", 32, 2),
        )
    words = [1] + sample_words("bfloat16", 31)
    with pytest.raises(ValueError, match="crosstl#2000"):
        run(monkeypatch, tmp_path, "directx", "block_softmax_bfloat16", words, 32)


def test_package_configs_choose_exact_waves_or_software_subgroups(tmp_path):
    for key in ("block_softmax_precise_float16@256", "looped_logsumexp_float32@1024"):
        entry, workgroup = softmax_layout.split_variant(key)
        source = softmax_layout.source(entry)
        configs = {
            target: softmax_packages.project_config(tmp_path, "out", target, key)
            for target in TARGETS
        }
        for config in configs.values():
            assert list(config.include_patterns) == [source]
            (rules,) = config.entry_workgroup_size_rules.values()
            assert {name: tuple(map(int, size)) for name, size in rules.items()} == {
                entry: workgroup
            }
            options = config.source_options["metal"]["target_options"]
            assert options == softmax_packages.TARGET_OPTIONS
        assert {
            path: int(width)
            for path, width in configs["directx"].subgroup_width_rules.items()
        } == {source: 32}
        assert not configs["metal"].subgroup_width_rules
        assert not configs["opengl"].subgroup_width_rules
        assert [
            item.expression for item in configs["opengl"].index_range_assertions
        ] == list(softmax_layout.index_assertions())
        assert not configs["metal"].index_range_assertions
        assert not configs["directx"].index_range_assertions
    assert softmax_packages.TARGET_OPTIONS == {
        "opengl": {"software_subgroup_width": 32}
    }
    keys = softmax_packages.verification_entries()
    assert keys == sorted(set(keys))
    # eval_gpu sizes block rows at ceil(ceil(n / 4) / 32) * 32 threads: the
    # 2049-element -inf row needs 544 and the 1024-element precise reference
    # 256; rows over 4096 elements take the looped kernel at 1024 threads.
    assert {
        "looped_softmax_float32@1024",
        "block_softmax_float32@544",
        "block_softmax_float32@256",
    } <= set(keys)
    assert "block_softmax_float32@1024" not in keys
    for key in keys:
        softmax_layout.split_variant(key)


def test_runtime_and_host_adapter_route_softmax_and_logsumexp():
    fake = SimpleNamespace(
        descriptors={}, copies=None, elementwise=None, aranges=None, arg_reductions=None
    )
    fake.softmaxes = object()
    assert runtime.HostRuntime._entry_available(fake, b"looped_logsumexp_bfloat16") == 1
    fake.softmaxes = None
    assert runtime.HostRuntime._entry_available(fake, b"block_softmax_float32") == 0
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in softmax_layout.ENTRIES:" in source
    adapter = (
        Path(runtime.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    )
    assert "void Softmax::eval_gpu(" in adapter
    assert "void LogSumExp::eval_gpu(" in adapter
    adaptation = Path(prepare.__file__).read_text(encoding="utf-8")
    assert '"Softmax",' in adaptation and '"LogSumExp",' in adaptation
