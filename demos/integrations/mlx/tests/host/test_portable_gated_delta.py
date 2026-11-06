"""Gated-delta recurrences of the portable host: check every launch against
gated_delta_update.cpp's plan, translate each pipeline with its frozen
constant, dispatch through the reflected bindings and audit the guards,
digests and packages."""

import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    gated_delta_dispatch,
    gated_delta_evidence,
    gated_delta_layout,
    gated_delta_packages,
    prepare,
    runtime,
    upstream_module_groups,
    verify_upstream_ops,
)

TARGETS = ("metal", "directx", "opengl")
HOST = Path(gated_delta_layout.__file__).parent
layout = gated_delta_layout
GUARDS = element_storage.GUARDS
GUARD_COUNT = element_storage.GUARD_COUNT


def test_entries_follow_the_pinned_instantiations():
    kinds = {}
    for entry, info in layout.ENTRIES.items():
        kinds.setdefault(info["kind"], []).append(entry)
    # Three types by six head shapes: four checkpoint intervals of the
    # sequential kernel, each with and without its save pass, the chunk
    # kernel and four intervals of the backward.
    assert {kind: len(entries) for kind, entries in kinds.items()} == {
        "seq": 144,
        "fused": 18,
        "vjp": 72,
    }
    assert kinds["seq"][:2] == [
        "seq_gated_delta_float_128_128_24_24_1",
        "seq_gated_delta_float_128_128_24_24_1_save",
    ]
    assert kinds["fused"][-1] == "gated_delta_fused_chunk_float16_t_128_128_16_64_8"
    assert kinds["vjp"][-1] == "seq_gated_delta_vjp_float16_t_128_128_16_64_16"
    save = "seq_gated_delta_bfloat16_t_128_128_16_48_4_save"
    assert layout.source_entry(save) == "seq_gated_delta_bfloat16_t_128_128_16_48_4"
    assert layout.constants(save) == {"save_state": True}
    assert layout.constants(save.removesuffix("_save")) == {"save_state": False}
    assert layout.constants("seq_gated_delta_vjp_float_128_128_16_16_1") == {}
    assert layout.source("seq_gated_delta_vjp_float_128_128_16_16_1").endswith(
        "gated_delta_update_vjp.metal"
    )
    assert (
        layout.entry_name("fused", "float16", 128, 128, 16, 32, 8)
        == "gated_delta_fused_chunk_float16_t_128_128_16_32_8"
    )
    for bad in (
        "seq_gated_delta_float_128_128_8_32_1",
        "gated_delta_fused_chunk_float_128_128_32_32_16",
        None,
    ):
        with pytest.raises(ValueError):
            layout.signature(bad)
    with pytest.raises(ValueError):
        layout.entry_name("seq", "int32", 128, 128, 32, 32, 1)


def test_parameters_directions_and_types_follow_the_kernels():
    entry = "seq_gated_delta_bfloat16_t_128_128_32_32_1"
    assert layout.parameters(entry) == layout.FORWARD
    assert layout.inactive(entry) == ("state_cache",)
    assert layout.parameter_type(entry, "q") == "bfloat16"
    assert layout.parameter_type(entry, "state_in") == "float32"
    assert layout.parameter_type(entry, "T") == "int32"
    assert layout.parameter_type(entry, "state_cache") == "float32"
    # eval_gpu fills the inference output; the save pass writes scratch.
    assert layout.directions(entry)["y"] == layout.INOUT
    assert layout.directions(entry + "_save")["y"] == 1
    assert layout.outputs(entry + "_save") == ("y", "state_out", "state_cache")
    assert layout.inactive(entry + "_save") == ()
    backward = "seq_gated_delta_vjp_float16_t_128_128_16_48_1"
    assert layout.parameters(backward) == layout.BACKWARD
    assert set(layout.outputs(backward)) == set(layout.GRADIENTS)
    assert {
        name for name, value in layout.directions(backward).items() if value
    } == set(layout.GRADIENTS)
    assert layout.parameter_type(backward, "cot_o") == "float16"
    assert layout.parameter_type(backward, "cot_h") == "float32"
    assert layout.parameter_type(backward, "dq") == "float32"
    with pytest.raises(ValueError):
        layout.parameter_type(backward, "state_in")


def test_plans_follow_eval_gpu_and_refuse_other_launches():
    entry = "seq_gated_delta_float_128_128_16_48_1_save"
    counts = layout.extents(entry, 2, 33)
    assert counts["q"] == 2 * 33 * 16 * 128
    assert counts["v"] == counts["y"] == 2 * 33 * 48 * 128
    assert counts["g"] == 2 * 33 * 48
    assert counts["state_cache"] == 2 * 48 * 33 * 128 * 128
    execution = layout.launch(entry, 2)
    # dispatch_threads((32, Dv, B * Hv), (32, 4, 1)).
    assert execution == {"workgroupCount": [1, 32, 96], "workgroupSize": [32, 4, 1]}
    threads = layout.threads(execution)
    assert threads == 32 * 128 * 96
    assert layout.plan(entry, counts, 33, threads, execution) == {
        "batch": 2,
        "steps": 33,
    }
    fused = "gated_delta_fused_chunk_float_128_128_32_32_8"
    # Four SIMD groups of eight columns: (32, Dv / 8, B * Hv).
    assert layout.launch(fused, 1)["workgroupCount"] == [1, 4, 32]
    sparse = "seq_gated_delta_float_128_128_32_32_4_save"
    assert layout.extents(sparse, 1, 9)["state_cache"] == 3 * 32 * 128 * 128
    assert (
        layout.extents("seq_gated_delta_vjp_float_128_128_32_32_8", 1, 9)["state_cache"]
        == 2 * 32 * 128 * 128
    )
    changed = (
        ({**counts, "q": counts["q"] - 1}, 33, threads, execution),
        ({**counts, "state_in": counts["state_in"] + 1}, 33, threads, execution),
        (counts, 32, threads, execution),
        (counts, 0, threads, execution),
        (counts, 33, threads + 1, execution),
        (counts, 33, threads, layout.launch(entry, 1)),
        (
            {key: value for key, value in counts.items() if key != "y"},
            33,
            threads,
            execution,
        ),
    )
    for arguments in changed:
        with pytest.raises(ValueError):
            layout.plan(entry, *arguments)
    # One launch axis carries at most 65535 workgroups.
    big = layout.extents("seq_gated_delta_float_128_128_32_32_1", 2048, 1)
    with pytest.raises(ValueError):
        layout.plan(
            "seq_gated_delta_float_128_128_32_32_1",
            big,
            1,
            0,
            layout.launch("seq_gated_delta_float_128_128_32_32_1", 2048),
        )


def test_package_configs_follow_the_kernel_gates(tmp_path):
    for target in TARGETS:
        for entry in (
            "seq_gated_delta_float_128_128_32_32_1",
            "seq_gated_delta_float_128_128_32_32_1_save",
            "gated_delta_fused_chunk_bfloat16_t_128_128_16_32_8",
            "seq_gated_delta_vjp_float_128_128_16_48_1",
        ):
            info = layout.signature(entry)
            config = gated_delta_packages.project_config(
                tmp_path, "work", target, entry
            )
            source = info["source"]
            assert list(config.include_patterns) == [source]
            assert dict(config.entry_points) == {source: (info["kernel"],)}
            assert dict(config.specialization_constants) == layout.constants(entry)
            assert config.freeze_specialization_constants
            metal = config.source_options["metal"]
            assert metal["preserve_resource_origins"] is True
            options = metal["target_options"]
            software = target in {"opengl", "directx"}
            assert bool(options) is software
            if software:
                assert options[target]["software_subgroup_width"] == 32
                assert options[target].get(
                    "cooperative_matrix_software_lowering", False
                ) is (info["kind"] == "fused")
                assert ("relative_wave_shuffle_out_of_range" in options[target]) is (
                    target == "directx"
                )
            assert ("cooperative_matrix_fragment_mapping" in metal) is (
                info["kind"] == "fused"
            )
            preconditions = [
                (item.entry_point, item.parameter, item.maximum)
                for item in config.metadata_preconditions
            ]
            assert preconditions == (
                [(info["kernel"], "T", layout.T_MAXIMUM)]
                if info["kind"] == "fused" and software
                else []
            )
            assertions = [
                getattr(item, "expression", None) or item["expression"]
                for item in config.index_range_assertions
            ]
            assert assertions == (
                ["object + offset"]
                if info["kind"] == "vjp" and target == "opengl"
                else []
            )
    entries = gated_delta_packages.verification_entries()
    assert entries == sorted(set(entries)) and len(entries) == 13
    assert {layout.signature(entry)["steps"] for entry in entries} <= {1, 8}
    with pytest.raises(ValueError):
        gated_delta_packages.GatedDeltaPackageCache(tmp_path, tmp_path, "vulkan")


class FakeCache:
    """A gated-delta cache whose descriptors reflect every declared buffer."""

    def __init__(self, target, *, drop_inactive=False):
        self.target = target
        self.drop_inactive = drop_inactive
        self.requests = []

    def get(self, entry):
        self.requests.append(entry)
        bindings = []
        declared = layout.parameters(entry) + (
            () if self.drop_inactive else layout.inactive(entry)
        )
        for name in declared:
            kind = layout.parameter_type(entry, name)
            if name in layout.ATOMIC_GRADIENTS:
                scalar = {
                    "componentCount": 1,
                    "elementStrideBytes": 4,
                    "elementType": "uint32" if self.target == "opengl" else "float32",
                    "structMembers": [{"name": "val", "offsetBytes": 0}],
                }
            else:
                value = element_storage.packet(kind, self.target, [])
                scalar = {
                    "elementType": value["dtype"],
                    "elementStrideBytes": element_storage.PHYSICAL_SIZES[
                        value["dtype"]
                    ],
                }
            bindings.append(
                {
                    "name": name + "Binding",
                    "access": (
                        "read_write"
                        if name in layout.outputs(entry) + layout.inactive(entry)
                        else "read"
                    ),
                    "scalarLayout": scalar,
                    "provenance": {"sourceResource": {"parameter": name}},
                }
            )
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def sample(dtype, count, seed):
    """Finite, normal storage words of ``dtype``."""
    if dtype == "float32":
        return [
            0x3F000000 + (977 * index + seed) % 0x00800000 for index in range(count)
        ]
    if dtype == "float16":
        return [0x3800 + (37 * index + seed) % 0x0400 for index in range(count)]
    return [0x3F00 + (13 * index + seed) % 0x0080 for index in range(count)]


def result_words(entry, name, count):
    """A stand-in kernel result: finite words of the output's type."""
    return sample(layout.parameter_type(entry, name), count, 101 + len(name))


def native_buffers(entry, batch, steps):
    counts = layout.extents(entry, batch, steps)
    directions = layout.directions(entry)
    memory, values = {}, {}
    for seed, name in enumerate(layout.parameters(entry)):
        kind = layout.parameter_type(entry, name)
        ctype = element_storage.CTYPES[kind]
        if name == "T":
            memory[name] = (ctypes.c_int32 * 1)(steps)
            continue
        words = (
            [0] * counts[name]
            if directions[name] == layout.INOUT
            else sample(kind, counts[name], seed)
        )
        values[name] = words
        memory[name] = (ctype * counts[name])(*words)
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                layout.parameter_type(entry, name).encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                directions[name],
            )
            for name in layout.parameters(entry)
        )
    )
    return array, memory, values


def run(monkeypatch, tmp_path, target, entry, batch, steps, *, fault=None, cache=None):
    launch = layout.launch(entry, batch)
    array, memory, values = native_buffers(entry, batch, steps)
    host = SimpleNamespace(
        gated_deltas=cache or FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=False,
        dispatch_count=0,
    )
    seen = {}

    def execute(_, request):
        inputs = request["inputs"]
        seen.update(inputs)
        assert request["execution"] == launch
        outputs = {}
        for name in layout.outputs(entry):
            binding = name + "Binding"
            initial = inputs[binding]
            kind = layout.parameter_type(entry, name)
            if initial["dtype"] == "uint32":
                words = list(initial["values"])
            else:
                words = element_storage.unpack(kind, target, initial["values"])
            count = len(words) - GUARD_COUNT
            words[:count] = result_words(entry, name, count)
            if fault == "guard" and name == layout.outputs(entry)[-1]:
                words[-1] ^= 1
            outputs[binding] = (
                {**initial, "values": words}
                if initial["dtype"] == "uint32"
                else element_storage.packet(kind, target, words)
            )
        for name in layout.inactive(entry):
            # A writable binding whose function constant is off reads back
            # as bound, unless the fault writes one element.
            binding = name + "Binding"
            if binding in request["outputs"]:
                words = list(inputs[binding]["values"])
                if fault == "inactive":
                    words[0] ^= 1
                outputs[binding] = {**inputs[binding], "values": words}
        return SimpleNamespace(status="ok", outputs=outputs, details={"request": {}})

    monkeypatch.setattr(
        gated_delta_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(gated_delta_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    gated_delta_dispatch.dispatch(
        host, entry, array, len(array), layout.threads(launch), native
    )
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return memory, values, seen, event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(gated_delta_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        gated_delta_evidence,
        "audit_native_execution",
        lambda e: checked.append(target),
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": (
                    {"opengl": "main", "directx": "CSMain"}.get(
                        target, layout.source_entry(event["entry"])
                    )
                ),
                "dispatch": {key: event[key] for key in gated_delta_evidence.GEOMETRY},
            }
        },
    }
    assert gated_delta_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    ("entry", "batch", "steps"),
    (
        ("seq_gated_delta_float_128_128_32_32_1", 1, 1),
        ("seq_gated_delta_bfloat16_t_128_128_16_32_1_save", 1, 2),
        ("gated_delta_fused_chunk_float16_t_128_128_32_32_8", 1, 9),
        ("seq_gated_delta_vjp_float_128_128_16_16_1", 2, 1),
    ),
)
def test_dispatch_binds_guards_and_audits_each_kernel(
    monkeypatch, tmp_path, target, entry, batch, steps
):
    memory, values, seen, event, host = run(
        monkeypatch, tmp_path, target, entry, batch, steps
    )
    assert host.dispatch_count == 1 and host.gated_deltas.requests == [entry]
    directions = layout.directions(entry)
    for name in layout.parameters(entry):
        kind = layout.parameter_type(entry, name)
        bound = seen[name + "Binding"]
        if name == "T":
            assert bound["values"] == [steps]
            continue
        words = (
            list(bound["values"])
            if bound["dtype"] == "uint32"
            else element_storage.unpack(kind, target, bound["values"])
        )
        count = len(values[name])
        if directions[name] == 1:
            # Fresh outputs start as guards.
            assert words == [GUARDS[kind]] * (count + GUARD_COUNT)
        elif directions[name] == layout.INOUT:
            # Initialized outputs start as the host's zeros.
            assert words[:count] == [0] * count
            assert words[count:] == [GUARDS[kind]] * GUARD_COUNT
        else:
            assert words[:count] == values[name]
        if name in layout.outputs(entry):
            assert list(memory[name]) == result_words(entry, name, count)
        else:
            assert list(memory[name]) == values[name]
    for name in layout.inactive(entry):
        assert set(seen[name + "Binding"]["values"]) == {0x3FC00000}
    if target == "opengl" and layout.signature(entry)["kind"] == "vjp":
        assert seen["dqBinding"]["dtype"] == "uint32"
    assert set(event["outputHashes"]) == set(layout.outputs(entry))
    assert event["gatedDeltaMetadata"] == {"batch": batch, "steps": steps}
    audit(monkeypatch, event, target)


def test_dispatch_accepts_a_target_that_drops_the_inactive_cache(monkeypatch, tmp_path):
    entry = "seq_gated_delta_float_128_128_32_32_1"
    _, _, seen, _, _ = run(
        monkeypatch,
        tmp_path,
        "directx",
        entry,
        1,
        3,
        cache=FakeCache("directx", drop_inactive=True),
    )
    assert "state_cacheBinding" not in seen


def test_dispatch_reads_back_and_refuses_a_written_inactive_cache(
    monkeypatch, tmp_path
):
    entry = "seq_gated_delta_float_128_128_32_32_1"
    with pytest.raises(RuntimeError, match="inactive state_cacheBinding"):
        run(monkeypatch, tmp_path, "metal", entry, 1, 1, fault="inactive")
    assert not (tmp_path / "dispatch.jsonl").exists()


def test_dispatch_and_audit_refuse_changed_launches_and_results(monkeypatch, tmp_path):
    entry = "seq_gated_delta_float_128_128_32_32_1_save"
    with pytest.raises(RuntimeError, match="guard"):
        run(monkeypatch, tmp_path, "metal", entry, 1, 2, fault="guard")
    _, _, _, event, _ = run(monkeypatch, tmp_path, "metal", entry, 1, 2)
    for changed in (
        {**event, "outputHashes": {**event["outputHashes"], "y": "0"}},
        {**event, "inputHashes": {}},
        {**event, "gatedDeltaMetadata": {"batch": 1, "steps": 3}},
        {**event, "workgroupCount": [1, 16, 32]},
        {**event, "threads": event["threads"] + 1},
        {**event, "counts": {**event["counts"], "state_cache": 1}},
    ):
        with pytest.raises(Exception):
            audit(monkeypatch, changed, "metal")
    array, memory, _ = native_buffers(entry, 1, 2)
    host = SimpleNamespace(gated_deltas=FakeCache("metal"), target="metal")
    wrong = runtime.Launch(
        (ctypes.c_uint32 * 3)(1, 1, 1),
        (ctypes.c_uint32 * 3)(32, 4, 1),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    with pytest.raises(ValueError, match="upstream plan"):
        gated_delta_dispatch.dispatch(host, entry, array, len(array), 128, wrong)
    with pytest.raises(ValueError, match="pinned packages"):
        gated_delta_dispatch.dispatch(host, entry, array, len(array) - 1, 128, wrong)
    # An output that names another direction is refused.
    array[6].output = layout.INOUT
    with pytest.raises(ValueError, match="direction"):
        gated_delta_dispatch.dispatch(host, entry, array, len(array), 128, wrong)
    assert memory["T"][0] == 2


def test_runtime_host_port_and_inventory_route_gated_delta():
    fake = SimpleNamespace(descriptors={}, gated_deltas=object())
    for entry in (
        "seq_gated_delta_float_128_128_16_64_1_save",
        "gated_delta_fused_chunk_float_128_128_24_24_8",
        "seq_gated_delta_vjp_bfloat16_t_128_128_16_16_1",
    ):
        assert runtime.HostRuntime._entry_available(fake, entry.encode()) == 1
    fake.gated_deltas = None
    assert (
        runtime.HostRuntime._entry_available(
            fake, b"seq_gated_delta_float_128_128_32_32_1"
        )
        == 0
    )
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if gated_delta_dispatch.is_entry(entry):" in source
    port = (HOST / "gated_delta.inc").read_text(encoding="utf-8")
    for line in (
        "void GatedDeltaUpdate::eval_gpu(",
        "void GatedDeltaUpdateVJP::eval_gpu(",
        "constexpr bool gated_delta_nax = false;",
        'env::get_var("GATED_DELTA_CHUNK", T > 8 ? (gated_delta_nax ? 16 : 8) : 1);',
        'if (env::get_var("GATED_DELTA_VJP_FALLBACK", 0) != 0) {',
        "const bool stage_fp32 = (q.dtype() != float32);",
        '"seq_gated_delta_" + suffix + ckpt_suffix + "_save"',
        "{1, uint32_t(columns / 4), uint32_t(rows)}, {32, 4, 1}, {0, 0, 0}};",
    ):
        assert line in port
    # The save pass runs before the backward that replays its cache.
    assert port.index('ckpt_suffix + "_save"') < port.index(
        '"seq_gated_delta_vjp_" + suffix'
    )
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert '#include "mlx/backend/no_gpu/crosstl_gated_delta.inc"' in backend
    prepared = Path(prepare.__file__).read_text(encoding="utf-8")
    assert (
        '"crosstl_gated_delta.inc": (HERE / "gated_delta.inc").read_bytes()' in prepared
    )
    assert set(prepare.GATED_DELTA_PRIMITIVES) <= set(prepare.FAST_PRIMITIVES)
    assert len(prepare.GATED_DELTA_DEFINITIONS) == 2
    tests = upstream_module_groups.GATED_DELTA_TESTS
    assert verify_upstream_ops.GROUPS["gated-delta"] == tests
    # The backward's composed reference runs in a job of its own.
    vjp = upstream_module_groups.GATED_DELTA_VJP_TESTS
    assert verify_upstream_ops.GROUPS["gated-delta-vjp"] == vjp
    tests += vjp
    assert set(upstream_module_groups.METAL_PATH_TESTS) - set(
        upstream_module_groups.CUSTOM_KERNEL_TESTS
    ) == {
        "test_conv_transpose.TestConvTranspose.test_conv_transpose_unfold_tiling",
        "test_fast_gated_delta.TestGatedDelta.test_gated_delta_vjp",
        "test_ops.TestOps.test_hadamard_m_only",
    }
    assert set(upstream_module_groups.METAL_PATH_TESTS) <= set(
        tests
        + upstream_module_groups.HADAMARD_TESTS
        + verify_upstream_ops.CONV_TESTS
        + upstream_module_groups.CUSTOM_KERNEL_TESTS
    )
    assert verify_upstream_ops.METAL_PATH_TESTS <= verify_upstream_ops.GPU_ONLY_TESTS


def test_metal_path_modules_import_with_the_host_gpu(monkeypatch, tmp_path):
    (tmp_path / "metal_gated_probe.py").write_text(
        "import unittest\n"
        "import metal_probe_core as mx\n"
        "class Case(unittest.TestCase):\n"
        "    @unittest.skipIf(not mx.metal.is_available(), 'Metal is not available')\n"
        "    def test_kernel(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    original = lambda: False  # noqa: E731
    mx = SimpleNamespace(
        gpu="gpu",
        is_available=lambda device: device == "gpu",
        metal=SimpleNamespace(is_available=original),
    )
    monkeypatch.setitem(sys.modules, "metal_probe_core", mx)
    monkeypatch.syspath_prepend(str(tmp_path))
    name = "metal_gated_probe.Case.test_kernel"
    monkeypatch.setattr(verify_upstream_ops, "METAL_PATH_TESTS", frozenset((name,)))
    monkeypatch.delitem(sys.modules, "metal_gated_probe", raising=False)
    # The CPU worker leaves the module to the test loader.
    verify_upstream_ops.import_gated_modules(mx, [name], native=False)
    assert "metal_gated_probe" not in sys.modules
    verify_upstream_ops.import_gated_modules(mx, [name], native=True)
    assert mx.metal.is_available is original
    assert not getattr(
        sys.modules["metal_gated_probe"].Case.test_kernel, "__unittest_skip__", False
    )
    # A module imported earlier is refused rather than run skipped.
    with pytest.raises(RuntimeError, match="imported before"):
        verify_upstream_ops.import_gated_modules(mx, [name], native=True)
    monkeypatch.delitem(sys.modules, "metal_gated_probe")
    mx.is_available = lambda device: False
    with pytest.raises(RuntimeError, match="native host"):
        verify_upstream_ops.import_gated_modules(mx, [name], native=True)
