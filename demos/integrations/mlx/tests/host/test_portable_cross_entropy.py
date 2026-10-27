"""Fused cross-entropy rows of the portable host: check every launch against
CrossEntropy and CrossEntropyVJP::eval_gpu's plan, refuse class indices
outside their rows, dispatch each width's package and audit the retained
uploads, guards and packages."""

import ctypes
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    cross_entropy_dispatch,
    cross_entropy_evidence,
    cross_entropy_layout,
    cross_entropy_packages,
    element_storage,
    prepare,
    runtime,
    verify_upstream_ops,
)
from demos.integrations.mlx.tests.host.test_portable_softmax import (
    float_word,
    sample_words,
    word_value,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACT = (
    ROOT
    / "demos/integrations/mlx/contracts/cross_entropy.current-tree.translation.json"
)
TARGETS = ("metal", "directx", "opengl")
layout = cross_entropy_layout


def reference(entry, values, axis_size):
    """The losses of each row, or their gradient, in the output's storage."""
    gradient, dtype = layout.signature(entry)
    logits = [word_value(dtype, word) for word in values["in"]]
    result = []
    for row, target in enumerate(values["targets"]):
        x = logits[row * axis_size : (row + 1) * axis_size]
        column = target + axis_size if target < 0 else target
        top = max(x)
        if not gradient:
            total = sum(math.exp(value - top) for value in x)
            result.append(float_word("float32", math.log(total) + top - x[column]))
            continue
        loss = word_value("float32", values["loss"][row])
        scale = word_value("float32", values["cotan"][row])
        for index, value in enumerate(x):
            p = math.exp(value - x[column] - loss)
            result.append(float_word(dtype, scale * (p - (index == column))))
    return result


def operands(entry, axis_size, targets):
    gradient, dtype = layout.signature(entry)
    rows = len(targets)
    values = {"in": sample_words(dtype, rows * axis_size), "targets": list(targets)}
    if gradient:
        values["loss"] = [float_word("float32", 0.5 + row / 4) for row in range(rows)]
        values["cotan"] = [float_word("float32", 1 - row / 3) for row in range(rows)]
    return values


def launch_for(axis_size, rows):
    return {
        "workgroupCount": [rows, 1, 1],
        "workgroupSize": [layout.width(axis_size), 1, 1],
    }


def counts_for(entry, axis_size, rows):
    gradient, _ = layout.signature(entry)
    counts = {"in": rows * axis_size, "targets": rows}
    if gradient:
        counts |= {"loss": rows, "cotan": rows}
    counts["out"] = rows * axis_size if gradient else rows
    return counts


def native_buffers(entry, values, axis_size):
    """A callback's buffers: the operands, the output and ``axis_size``."""
    types = layout.operand_types(entry)
    rows = len(values["targets"])
    outputs = counts_for(entry, axis_size, rows)["out"]
    memory = {}
    for name, kind in types.items():
        ctype = element_storage.CTYPES[kind]
        if name == "out":
            memory[name] = (ctype * outputs)()
        elif name == "axis_size":
            memory[name] = (ctypes.c_int32 * 1)(axis_size)
        else:
            memory[name] = (ctype * len(values[name]))(*values[name])
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
    """The cross-entropy cache: one package per entry and width."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        entry, _ = layout.split_variant(key)
        # Direct3D prefixes the scalar with the entry; "in" and "out" are keywords.
        prefix = "" if self.target != "directx" else entry + "_"
        bindings = []
        for name, kind in layout.operand_types(entry).items():
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


def run(monkeypatch, tmp_path, target, entry, values, axis_size, *, fault=None):
    gradient, dtype = layout.signature(entry)
    types = layout.operand_types(entry)
    rows = len(values["targets"])
    outputs = counts_for(entry, axis_size, rows)["out"]
    launch = launch_for(axis_size, rows)
    array, memory = native_buffers(entry, values, axis_size)
    host = SimpleNamespace(
        cross_entropies=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        uploaded = {
            name: element_storage.unpack(
                types[name], target, inputs[name + "Binding"]["values"]
            )
            for name in layout.inputs_of(entry)
        }
        assert uploaded == values
        initial = element_storage.unpack(
            types["out"], target, inputs["outBinding"]["values"]
        )
        assert len(initial) == outputs + element_storage.GUARD_COUNT
        assert element_storage.unpack(
            "int32", target, inputs["axis_sizeBinding"]["values"]
        ) == [axis_size]
        assert request["execution"] == launch
        result = reference(entry, values, axis_size) + initial[outputs:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={
                "outBinding": element_storage.packet(types["out"], target, result)
            },
            details={"request": {}},
        )

    monkeypatch.setattr(
        cross_entropy_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(cross_entropy_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    threads = len(values["in"])
    cross_entropy_dispatch.dispatch(host, entry, array, len(array), threads, native)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(cross_entropy_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        cross_entropy_evidence,
        "audit_native_execution",
        lambda e: checked.append(target),
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": event["entry"],
                "dispatch": {
                    key: event[key] for key in cross_entropy_evidence.GEOMETRY
                },
            }
        },
    }
    retained = cross_entropy_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return retained


def test_entries_are_the_pinned_instantiations():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["provenance"]["commit"] == prepare.COMMIT
    assert contract["provenance"]["source"].endswith(Path(layout.SOURCE).name)
    assert set(layout.ENTRIES) == set(contract["entries"])
    assert len(layout.ENTRIES) == 6
    assert all(entry.startswith(layout.PREFIX) for entry in layout.ENTRIES)
    assert layout.entry_name(True, "bfloat16") == "cross_entropy_vjp_bfloat16"
    with pytest.raises(ValueError):
        layout.entry_name(False, "float64")


def test_widths_follow_cross_entropy_dims_and_key_each_variant():
    # ceil(ceil(n / 4) / 32) * 32 threads, at least one simdgroup and at
    # most the pipelines' 1024.
    sizes = (1, 2, 128, 129, 1000, 4096, 4100, 8192)
    assert [layout.width(size) for size in sizes] == [
        32,
        32,
        32,
        64,
        256,
        1024,
        1024,
        1024,
    ]
    for size in (0, -1, 2**28, 3.0):
        with pytest.raises(ValueError):
            layout.width(size)
    assert layout.variant("cross_entropy_float16", 64) == "cross_entropy_float16@64"
    assert layout.split_variant("cross_entropy_vjp_float32@1024") == (
        "cross_entropy_vjp_float32",
        (1024, 1, 1),
    )
    for key in ("cross_entropy_float32@48", "cross_entropy_float32@2048", "x@32", 7):
        with pytest.raises(ValueError):
            layout.split_variant(key)


def test_plan_refuses_launches_outside_upstream(monkeypatch):
    entry, vjp = "cross_entropy_float32", "cross_entropy_vjp_float32"
    plan = layout.plan
    assert plan(entry, 32, counts_for(entry, 32, 128), 4096) == launch_for(32, 128)
    assert plan(vjp, 7, counts_for(vjp, 7, 3), 21) == launch_for(7, 3)
    forward = counts_for(entry, 32, 4)
    refused = (
        (entry, 32, {**forward, "in": 127}, 127),  # partial row
        (entry, 32, forward, 127),  # logical size differs
        (entry, 32, {**forward, "out": 128}, 128),  # one loss per row
        (entry, 32, {**forward, "targets": 3}, 128),  # one index per row
        (vjp, 32, {**counts_for(vjp, 32, 4), "out": 4}, 128),  # gradient per logit
        (vjp, 32, {**counts_for(vjp, 32, 4), "cotan": 3}, 128),
        (vjp, 32, forward, 128),  # the gradient binds losses and cotangents
        (entry, 1, counts_for(entry, 1, 65536), 65536),  # too many rows
    )
    for case in refused:
        with pytest.raises(ValueError):
            plan(*case)
    # The element bound keeps every offset signed; a wider bound would not.
    monkeypatch.setattr(layout, "MAX_ELEMENTS", 2**32)
    with pytest.raises(ValueError, match="signed 32-bit"):
        plan(entry, 2**27 + 1, counts_for(entry, 2**27 + 1, 16), (2**27 + 1) * 16)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,axis_size,targets",
    (
        ("cross_entropy_float32", 7, (0, 6, -1)),
        ("cross_entropy_float16", 129, (128, -129)),
        ("cross_entropy_bfloat16", 1000, (999,)),
        ("cross_entropy_vjp_float32", 32, (31, -32, 5)),
        ("cross_entropy_vjp_float16", 130, (0, 129)),
        ("cross_entropy_vjp_bfloat16", 9, (-1,)),
    ),
)
def test_dispatch_returns_and_audits_each_kernel(
    monkeypatch, tmp_path, target, entry, axis_size, targets
):
    values = operands(entry, axis_size, targets)
    out, event, host = run(monkeypatch, tmp_path, target, entry, values, axis_size)
    assert out == reference(entry, values, axis_size)
    assert host.dispatch_count == 1
    assert host.cross_entropies.requests == [f"{entry}@{layout.width(axis_size)}"]
    assert event["axisSize"] == axis_size
    assert event["workgroupCount"] == [len(targets), 1, 1]
    assert audit(monkeypatch, event, target)


def test_dispatch_and_audit_refuse_changed_results(monkeypatch, tmp_path):
    entry = "cross_entropy_float32"
    values = operands(entry, 32, (1, 2))
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, "opengl", entry, values, 32, fault="guard")
    out, event, _ = run(monkeypatch, tmp_path, "opengl", entry, values, 32)
    tampered = {**event, "outputValues": [out[0] ^ 1, *out[1:]]}
    with pytest.raises(ValueError, match="output hash"):
        audit(monkeypatch, tampered, "opengl")
    logits = event["operandValues"]["in"]
    changed = {
        **event,
        "operandValues": {**event["operandValues"], "in": [logits[0] ^ 1, *logits[1:]]},
    }
    with pytest.raises(ValueError, match="upload of in differs"):
        audit(monkeypatch, changed, "opengl")
    outside = {**event, "operandValues": {**event["operandValues"], "targets": [1, 32]}}
    with pytest.raises(ValueError, match="inside their rows"):
        audit(monkeypatch, outside, "opengl")
    moved = {**event, "workgroupSize": [64, 1, 1]}
    with pytest.raises(ValueError, match="upstream plan"):
        audit(monkeypatch, moved, "opengl")
    resized = {**event, "axisSize": 16}
    with pytest.raises(ValueError, match="match their rows"):
        audit(monkeypatch, resized, "opengl")
    digest = {"compactedValues": {"length": 2, "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="retention"):
        audit(monkeypatch, {**event, "outputValues": digest}, "opengl")
    with pytest.raises(ValueError, match="neither retained nor compacted"):
        audit(monkeypatch, {**event, "outputValues": {"length": 2}}, "opengl")


@pytest.mark.parametrize("targets", ((0, 32), (-33, 0), (2**31 - 1, 0)))
def test_dispatch_refuses_class_indices_outside_their_rows(
    monkeypatch, tmp_path, targets
):
    # MLX reads whatever lies past the row; the host refuses before translating.
    values = operands("cross_entropy_vjp_float32", 32, targets)
    with pytest.raises(ValueError, match="inside their rows"):
        run(monkeypatch, tmp_path, "metal", "cross_entropy_vjp_float32", values, 32)
    assert not (tmp_path / "dispatch.jsonl").exists()


def test_dispatch_refuses_incomplete_buffers_and_subnormal_bfloat(
    monkeypatch, tmp_path
):
    host = SimpleNamespace(cross_entropies=None, target="metal")
    with pytest.raises(ValueError, match="complete buffers"):
        cross_entropy_dispatch.dispatch(
            host, "cross_entropy_float32", [object()], 4, 1, object()
        )
    values = operands("cross_entropy_float32", 32, (0, 1))
    array, _ = native_buffers("cross_entropy_float32", values, 32)
    buffers = {array[i].name.decode(): array[i] for i in range(len(array))}
    with pytest.raises(ValueError, match="buffer type"):
        layout.validate("cross_entropy_float16", buffers, 64, launch_for(32, 2))
    values = operands("cross_entropy_bfloat16", 32, (0,))
    values["in"][3] = 1
    with pytest.raises(ValueError, match="crosstl#2000"):
        run(monkeypatch, tmp_path, "directx", "cross_entropy_bfloat16", values, 32)


def test_large_launches_keep_extents_and_digests(monkeypatch, tmp_path):
    entry = "cross_entropy_vjp_float32"
    monkeypatch.setattr(cross_entropy_dispatch, "RETAINED_LIMIT", 16)
    monkeypatch.setattr(cross_entropy_evidence, "RETAINED_LIMIT", 16)
    values = operands(entry, 300, (0, 299))
    out, event, _ = run(monkeypatch, tmp_path, "directx", entry, values, 300)
    assert out == reference(entry, values, 300)
    assert "compactedValues" in event["operandValues"]["in"]
    assert not audit(monkeypatch, event, "directx")


def test_package_configs_choose_exact_waves_or_software_subgroups(tmp_path):
    for key in ("cross_entropy_float16@256", "cross_entropy_vjp_bfloat16@1024"):
        entry, workgroup = layout.split_variant(key)
        gradient, _ = layout.signature(entry)
        configs = {
            target: cross_entropy_packages.project_config(tmp_path, "out", target, key)
            for target in TARGETS
        }
        for config in configs.values():
            assert list(config.include_patterns) == [layout.SOURCE]
            (rules,) = config.entry_workgroup_size_rules.values()
            assert {name: tuple(map(int, size)) for name, size in rules.items()} == {
                entry: workgroup
            }
            options = config.source_options["metal"]["target_options"]
            # Only the forward kernel reduces across simdgroups.
            assert options == (
                {} if gradient else cross_entropy_packages.TARGET_OPTIONS
            )
            assert not config.specialization_constants
        waves = {
            path: int(width)
            for path, width in configs["directx"].subgroup_width_rules.items()
        }
        assert waves == ({} if gradient else {layout.SOURCE: 32})
        assert not configs["metal"].subgroup_width_rules
        assert not configs["opengl"].subgroup_width_rules
        assert [
            item.expression for item in configs["opengl"].index_range_assertions
        ] == list(layout.index_assertions(entry))
        assert not configs["metal"].index_range_assertions
        assert not configs["directx"].index_range_assertions
    keys = cross_entropy_packages.verification_entries()
    assert keys == sorted(set(keys))
    # 8192 logits clamp to 1024 threads; 1000 need 256 and 512 need 128.
    assert {
        "cross_entropy_bfloat16@1024",
        "cross_entropy_float16@256",
        "cross_entropy_vjp_float32@128",
        "cross_entropy_vjp_float32@1024",
    } <= set(keys)
    assert "cross_entropy_vjp_float16@32" not in keys
    for key in keys:
        layout.split_variant(key)


def test_runtime_and_host_adapter_route_cross_entropy():
    fake = SimpleNamespace(
        descriptors={},
        copies=None,
        elementwise=None,
        aranges=None,
        arg_reductions=None,
        softmaxes=None,
        sorts=None,
        searchsorted=None,
        norms=None,
    )
    fake.cross_entropies = object()
    assert runtime.HostRuntime._entry_available(fake, b"cross_entropy_vjp_float16") == 1
    fake.cross_entropies = None
    assert runtime.HostRuntime._entry_available(fake, b"cross_entropy_float32") == 0
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in cross_entropy_layout.ENTRIES:" in source
    here = Path(runtime.__file__).parent
    adapter = (here / "cross_entropy.inc").read_text(encoding="utf-8")
    for primitive in ("CrossEntropy", "CrossEntropyVJP"):
        assert f"void {primitive}::eval_gpu(" in adapter
    # The GPU keeps the fused kernel and the CPU MLX's composed fallback, as
    # the Metal backend does.
    assert (
        "bool CrossEntropy::use_fallback(Stream s) {\n  return s.device == Device::cpu;"
        in adapter
    )
    assert "std::clamp<uint64_t>((needed + 31) / 32 * 32, 32, 1024)" in adapter
    assert prepare.CROSS_ENTROPY_PRIMITIVES == ("CrossEntropy", "CrossEntropyVJP")
    assert "CrossEntropy" in prepare.FALLBACK_PRIMITIVES
    assert "CrossEntropyVJP" not in prepare.FALLBACK_PRIMITIVES
    assert set(prepare.CROSS_ENTROPY_PRIMITIVES) <= set(prepare.FAST_PRIMITIVES)
    backend = (here / "backend.cpp").read_text(encoding="utf-8")
    # Only the FFT and vector attention ports follow it.
    assert backend.rstrip().endswith(
        '#include "mlx/backend/no_gpu/crosstl_cross_entropy.inc"\n'
        "// And the port of FFT, whose inputs take the strided copy.\n"
        '#include "mlx/backend/no_gpu/crosstl_fft.inc"\n'
        "// And the port of the vector attention kernels, whose views take the copy.\n"
        '#include "mlx/backend/no_gpu/crosstl_sdpa.inc"'
    )
    # The upstream op audit owns the cross-entropy dispatches.
    assert layout.PREFIX in verify_upstream_ops.RETAINED_VALUES
    assert {
        "test_fast.TestFast.test_cross_entropy",
        "test_losses.TestLosses.test_cross_entropy",
    } <= set(verify_upstream_ops.GROUPS["modules"])
