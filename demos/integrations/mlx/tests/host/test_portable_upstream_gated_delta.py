"""Host contracts of the translated gated-delta recurrence path."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import gated_delta_dispatch as dispatch
from demos.integrations.mlx.portable_host import gated_delta_layout as layout
from demos.integrations.mlx.portable_host import (
    gated_delta_packages,
    gated_delta_workloads,
    prepare,
    runtime,
    verify_upstream_gated_delta,
)
from demos.integrations.mlx.tests.host.test_portable_matmul import (
    CTYPES,
    physical_layout,
)

HOST = Path(layout.__file__).parent
SEQ = "seq_gated_delta_float_128_128_32_32_1"
SAVE = "seq_gated_delta_bfloat16_t_128_128_16_48_1_save"
FUSED = "gated_delta_fused_chunk_float16_t_128_128_16_32_8"
VJP = "seq_gated_delta_vjp_float_128_128_16_32_1"


def test_signature_names_every_kernel_the_host_launches():
    seq = layout.signature(SEQ)
    assert seq["kind"] == "sequential" and seq["sourceEntry"] == SEQ
    assert seq["constants"] == {"save_state": False}
    assert seq["source"].endswith("gated_delta_update.metal")
    save = layout.signature(SAVE)
    assert save["sourceEntry"] == SAVE[: -len("_save")]
    assert save["dtype"] == "bfloat16" and save["constants"] == {"save_state": True}
    fused = layout.signature(FUSED)
    assert fused["kind"] == "fused" and fused["chunk"] == 8
    assert fused["constants"] == {"save_state": False}
    vjp = layout.signature(VJP)
    assert vjp["kind"] == "vjp" and vjp["constants"] == {}
    assert vjp["source"].endswith("gated_delta_update_vjp.metal")
    assert layout.specialization_constants(SAVE) == {"200": True}
    assert layout.specialization_constants(VJP) == {}
    assert layout.entry_name("sequential", "bfloat16", 16, 48, save=True) == SAVE
    assert layout.entry_name("fused", "float16", 16, 32, 8) == FUSED
    assert layout.entry_name("vjp", "float32", 16, 32) == VJP


@pytest.mark.parametrize(
    "entry",
    [
        "seq_gated_delta_float_128_128_8_32_1",
        "seq_gated_delta_float_128_128_32_32_2",
        "seq_gated_delta_float_64_128_32_32_1",
        "seq_gated_delta_float_128_128_32_32_01",
        "seq_gated_delta_vjp_float_128_128_32_32_1_save",
        "gated_delta_fused_chunk_float_128_128_32_32_16",
        "seq_gated_delta_double_128_128_32_32_1",
    ],
)
def test_signature_rejects_entries_upstream_never_instantiates(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_parameters_follow_the_kernel_buffer_order():
    assert layout.parameters(SEQ) == layout.FORWARD
    assert layout.parameters(SAVE) == (*layout.FORWARD, "state_cache")
    assert layout.parameters(FUSED) == layout.FORWARD
    assert layout.parameters(VJP)[:9] == (
        "q",
        "k",
        "v",
        "g",
        "b",
        "cot_o",
        "cot_h",
        "state_cache",
        "T",
    )
    assert layout.outputs(SAVE) == ("y", "state_out", "state_cache")
    assert layout.initialized(SEQ) == ("y",)
    assert layout.initialized(SAVE) == ()
    assert layout.initialized(VJP) == layout.outputs(VJP)
    # gated_delta_update_vjp.h: dq, dk, dg and db are mlx_atomic<float>.
    assert layout.atomics(VJP) == ("dq", "dk", "dg", "db")
    assert layout.atomics(SEQ) == layout.atomics(FUSED) == ()
    assert layout.parameter_type(SAVE, "q") == "bfloat16"
    assert layout.parameter_type(SAVE, "state_cache") == "float32"
    assert layout.parameter_type(VJP, "cot_o") == "float32"
    assert layout.parameter_type(VJP, "T") == "int32"


def test_execution_mirrors_the_metal_grids():
    assert layout.execution(SEQ, 2) == {
        "workgroupCount": [1, 32, 64],
        "workgroupSize": [32, 4, 1],
    }
    assert layout.execution(FUSED, 3)["workgroupCount"] == [1, 4, 96]
    assert layout.execution(VJP, 1)["workgroupCount"] == [1, 32, 32]


def _plan(entry, batch, steps, **changes):
    counts = layout.spans(entry, batch, steps)
    launch = layout.execution(entry, batch)
    primary = "dv" if layout.signature(entry)["kind"] == "vjp" else "y"
    threads = counts[primary]
    counts.update(changes.pop("counts", {}))
    launch.update(changes.pop("launch", {}))
    values = {"T": changes.pop("value", steps)}
    threads = changes.pop("threads", threads)
    return layout.plan_launch(entry, counts, values, threads, launch)


def test_plan_derives_each_span_from_batch_and_steps():
    plan = _plan(SAVE, 2, 3)
    assert plan["batch"] == 2 and plan["steps"] == 3 and plan["saveState"]
    elements = plan["elements"]
    assert elements["q"] == 2 * 3 * 16 * 128
    assert elements["y"] == 2 * 3 * 48 * 128
    assert elements["g"] == 2 * 3 * 48
    assert elements["state_cache"] == 2 * 48 * 128 * 128 * 3
    vjp = _plan(VJP, 1, 33)["elements"]
    assert vjp["state_cache"] == 32 * 128 * 128 * 33
    assert vjp["dq"] == 33 * 16 * 128 and vjp["dh"] == 32 * 128 * 128


@pytest.mark.parametrize(
    "change",
    [
        {"value": 0},
        {"value": layout.T_MAXIMUM + 1},
        {"counts": {"q": 1}},
        {"threads": 1},
        {"launch": {"workgroupCount": [1, 32, 47]}},
        {"launch": {"workgroupCount": [2, 32, 48]}},
        {"launch": {"workgroupSize": [32, 1, 1]}},
        {"launch": {"threadGridSize": [32, 128, 48]}},
    ],
)
def test_plan_rejects_launches_that_differ_from_the_kernel(change):
    with pytest.raises(ValueError):
        _plan(SAVE, 1, 3, **change)


def test_plan_bounds_every_span_to_signed_offsets():
    # A 70,000-step backward caches more than 2^31 - 1 state elements.
    with pytest.raises(ValueError):
        _plan(VJP, 2, 70000)


def test_package_config_carries_the_proven_recipe(tmp_path):
    config = gated_delta_packages.project_config(tmp_path, "out", "opengl", FUSED)
    source = layout.SOURCES["forward"]
    assert list(config.include_patterns) == [source]
    assert {key: list(value) for key, value in config.entry_points.items()} == {
        source: [FUSED]
    }
    assert dict(config.specialization_constants) == {"200": False}
    assert config.freeze_specialization_constants is True
    metal = config.source_options["metal"]
    assert metal["cooperative_matrix_fragment_mapping"] == layout.FRAGMENT_MAPPING
    assert metal["target_options"] == {
        "opengl": {
            "software_subgroup_width": 32,
            "cooperative_matrix_software_lowering": True,
        }
    }
    (precondition,) = (item.to_json() for item in config.metadata_preconditions)
    assert precondition["parameter"] == "T"
    assert precondition["maximum"] == layout.T_MAXIMUM
    save = gated_delta_packages.project_config(tmp_path, "out", "directx", SAVE)
    assert dict(save.specialization_constants) == {"200": True}
    assert save.source_options["metal"]["target_options"] == {
        "directx": {
            "software_subgroup_width": 32,
            "relative_wave_shuffle_out_of_range": "self",
        }
    }
    assert not save.metadata_preconditions
    vjp = gated_delta_packages.project_config(tmp_path, "out", "opengl", VJP)
    assert list(vjp.include_patterns) == [layout.SOURCES["vjp"]]
    assert not vjp.specialization_constants
    assert not vjp.freeze_specialization_constants
    (assertion,) = (item.to_json() for item in vjp.index_range_assertions)
    assert assertion["expression"] == "object + offset"
    metal = gated_delta_packages.project_config(tmp_path, "out", "metal", VJP)
    assert not metal.index_range_assertions
    assert metal.source_options["metal"]["target_options"] == {}
    cache = gated_delta_packages.GatedDeltaPackageCache(tmp_path, tmp_path, "metal")
    assert cache.source_entry(SAVE) == SAVE[: -len("_save")]
    assert cache.source_path(VJP) == layout.SOURCES["vjp"]


def native_buffers(entry, batch, steps):
    counts = layout.spans(entry, batch, steps)
    memory, buffers = {}, {}
    for name in layout.parameters(entry):
        dtype = layout.parameter_type(entry, name)
        if name == "T":
            memory[name] = (ctypes.c_int32 * 1)(steps)
            count = 1
        else:
            count = counts[name]
            initial = (
                range(1, count + 1) if dispatch._direction(entry, name) != 1 else ()
            )
            memory[name] = (CTYPES[dtype] * count)(*initial)
        buffers[name] = runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            count,
            dispatch._direction(entry, name),
        )
    return buffers, memory


@pytest.mark.parametrize("fault", [None, "dtype", "count", "direction", "alias", "T"])
def test_validate_checks_storage_direction_and_aliases(fault):
    buffers, memory = native_buffers(SAVE, 1, 2)
    launch = layout.execution(SAVE, 1)
    threads = layout.spans(SAVE, 1, 2)["y"]
    if fault == "dtype":
        buffers["q"].dtype = b"float16"
    elif fault == "count":
        buffers["g"].count -= 1
    elif fault == "direction":
        buffers["y"].output = 2
    elif fault == "alias":
        buffers["state_out"].data = buffers["state_in"].data
    elif fault == "T":
        memory["T"][0] = 3
    if fault is None:
        metadata, steps = dispatch.validate(SAVE, buffers, threads, launch)
        assert steps == 2 and metadata["batch"] == 1
    else:
        with pytest.raises(ValueError):
            dispatch.validate(SAVE, buffers, threads, launch)


def descriptor_for(entry, target, fault=None):
    bindings = []
    for name in layout.parameters(entry):
        if fault == "unbound" and name == "q":
            continue
        dtype = layout.parameter_type(entry, name)
        output = name in layout.outputs(entry)
        scalar = physical_layout(target, dtype)
        if name in layout.atomics(entry) and fault != "plain-atomic":
            # mlx_atomic<float>: a one-member struct, a uint word on OpenGL.
            storage = "uint32" if target == "opengl" else "float32"
            member = "uint" if target == "opengl" else "float"
            scalar = {
                "elementType": storage,
                "elementStrideBytes": 4,
                "componentCount": 1,
                "structMembers": [
                    {"name": "val", "offsetBytes": 0, "physicalType": member}
                ],
            }
        bindings.append(
            {
                "name": f"binding_{name}",
                "kind": "constant" if name == "T" else "buffer",
                "access": "read_write" if output and fault != "readonly" else "read",
                "provenance": {"sourceResource": {"parameter": name}},
                "scalarLayout": scalar,
            }
        )
    return {"artifact": {"hash": "h"}, "bindings": bindings}


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault", (None, "guard", "missing", "unbound", "readonly", "plain-atomic")
)
def test_vjp_dispatch_keeps_zero_fills_and_commits_after_validation(
    tmp_path, monkeypatch, target, fault
):
    batch, steps = 1, 1
    named, memory = native_buffers(VJP, batch, steps)
    for name in layout.outputs(VJP):
        ctypes.memset(memory[name], 0, ctypes.sizeof(memory[name]))
    descriptor = descriptor_for(VJP, target, fault)
    host = SimpleNamespace(
        target=target,
        gated_deltas=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    seen = {}

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        seen.update(inputs=inputs, execution=execution, kwargs=kwargs)
        return {name: inputs[name] for name in outputs}

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        value = outputs["binding_dq"]
        # An atomic sum adds to the zero fill the dispatch passed in.
        assert set(value["values"][: len(value["values"]) - dispatch.GUARD_COUNT]) == {
            0
        }
        value["values"][0] = 0x3F800000
        if fault == "guard":
            value["values"][-1] = 0
        elif fault == "missing":
            outputs.clear()
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    monkeypatch.setattr(dispatch, "execute", execute)
    names = layout.parameters(VJP)
    supplied = (runtime.Buffer * len(names))(*(named[name] for name in names))
    execution = layout.execution(VJP, batch)
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    threads = layout.spans(VJP, batch, steps)["dv"]
    call = lambda: dispatch.dispatch(host, VJP, supplied, len(names), threads, launch)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            call()
        assert host.dispatch_count == 0 and not any(memory["dq"])
        return
    call()
    assert host.dispatch_count == 1
    assert memory["dq"][0] == 0x3F800000 and not any(memory["dq"][1:])
    assert seen["execution"] == execution
    for name in layout.outputs(VJP):
        value = seen["inputs"][f"binding_{name}"]
        count = len(value["values"])
        if name in layout.atomics(VJP):
            storage = "uint32" if target == "opengl" else "float32"
            assert value["dtype"] == storage and value["shape"] == [count, 1]
        else:
            assert value["dtype"] == "float32" and value["shape"] == [count]
    assert set(seen["kwargs"]["observe_inputs"]) == {
        f"binding_{name}" for name in names if name not in layout.outputs(VJP)
    }
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == VJP and event["steps"] == steps
    assert event["gatedDeltaMetadata"]["kind"] == "vjp"


@pytest.mark.parametrize("access", ("read", "read_write"))
@pytest.mark.parametrize("fault", (None, "written"))
def test_inference_dispatch_reads_back_metal_declared_state_cache(
    tmp_path, monkeypatch, access, fault
):
    batch, steps = 1, 2
    named, memory = native_buffers(SEQ, batch, steps)
    descriptor = descriptor_for(SEQ, "metal")
    descriptor["bindings"].append(
        {
            "name": "binding_state_cache",
            "kind": "buffer",
            "access": access,
            "provenance": {"sourceResource": {"parameter": "state_cache"}},
            "scalarLayout": physical_layout("metal", "float32"),
        }
    )
    host = SimpleNamespace(
        target="metal",
        gated_deltas=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    seen = {}

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        seen.update(inputs=inputs, outputs=outputs, kwargs=kwargs)
        return {name: inputs[name] for name in outputs}

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        outputs["binding_y"]["values"][0] = 0x3F800000
        if fault and "binding_state_cache" in outputs:
            outputs["binding_state_cache"]["values"][0] = 0
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    monkeypatch.setattr(dispatch, "execute", execute)
    names = layout.parameters(SEQ)
    supplied = (runtime.Buffer * len(names))(*(named[name] for name in names))
    execution = layout.execution(SEQ, batch)
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    threads = layout.spans(SEQ, batch, steps)["y"]
    call = lambda: dispatch.dispatch(host, SEQ, supplied, len(names), threads, launch)
    before = list(memory["y"])
    if fault and access == "read_write":
        with pytest.raises(RuntimeError, match="unused state_cache"):
            call()
        assert host.dispatch_count == 0 and list(memory["y"]) == before
        return
    call()
    assert host.dispatch_count == 1 and memory["y"][0] == 0x3F800000
    cache = seen["inputs"]["binding_state_cache"]["values"]
    assert cache == [dispatch.POISON["float32"]] * dispatch.GUARD_COUNT
    writable = access == "read_write"
    assert ("binding_state_cache" in seen["outputs"]) == writable
    assert ("binding_state_cache" in seen["kwargs"]["observe_inputs"]) != writable


def test_dispatch_requires_the_cache_and_geometry():
    host = SimpleNamespace(target="metal", gated_deltas=None, dispatch_count=0)
    with pytest.raises(ValueError, match="pinned packages"):
        dispatch.dispatch(host, SEQ, [object()], 9, 1, None)


def test_runtime_routes_gated_delta_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors = {}
    for name in ("scans", "divmods", "hadamards", "sdpas", "gated_deltas"):
        setattr(host, name, None)
    assert host._entry_available(SEQ.encode()) == 0
    host.gated_deltas = object()
    assert host._entry_available(SEQ.encode()) == 1
    assert host._entry_available(b"seq_gated_delta_float_128_128_8_8_1") == 0
    calls = []
    monkeypatch.setattr(
        runtime.gated_delta_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    host.dispatch(FUSED, [], 9, 1, launch=None)
    assert calls == [FUSED]


def test_backend_mirrors_gated_delta_update_cpp():
    port = (HOST / "gated_delta.inc").read_text(encoding="utf-8")
    for text in (
        "void GatedDeltaUpdate::eval_gpu(",
        "void GatedDeltaUpdateVJP::eval_gpu(",
        'env::get_var("GATED_DELTA_CHUNK", T > 8 ? 8 : 1);',
        "return C == 16 ? 8 : C;",
        'env::get_var("GATED_DELTA_VJP_FALLBACK", 0) != 0',
        '"gated_delta_fused_chunk_" + suffix + "_" + std::to_string(C);',
        '"seq_gated_delta_" + suffix + "_" + std::to_string(ckpt);',
        '"seq_gated_delta_" + suffix + ckpt_suffix + "_save"',
        '"seq_gated_delta_vjp_" + suffix + ckpt_suffix',
        "{1, gated_delta_groups(rows / 4), gated_delta_groups(int64_t(B) * Hv)}, {32, 4, 1}};",
        "const bool stage_fp32 = q.dtype() != float32;",
        "gated_delta_zero(out);",
        "gated_delta_zero(dh);",
    ):
        assert text in port, text
    for name in set(layout.BACKWARD) | set(layout.FORWARD) | {"state_cache"}:
        assert f'"{name}"' in port, name
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert backend.rstrip().endswith(
        '#include "mlx/backend/no_gpu/crosstl_gated_delta.inc"'
    )
    assert {"GatedDeltaUpdate", "GatedDeltaUpdateVJP"} <= set(prepare.FAST_PRIMITIVES)
    assert len(prepare.GATED_DELTA_DEFINITIONS) == 2


def test_adaptation_replaces_the_fallback_rules():
    primitives = "\n".join(
        (
            *prepare.ATTENTION_DEFINITIONS,
            *prepare.GATED_DELTA_DEFINITIONS,
            "NO_GPU_MULTI(GatedDeltaUpdate)",
            "NO_GPU_MULTI(GatedDeltaUpdateVJP)",
        )
    )
    replaced = primitives
    for definition in prepare.GATED_DELTA_DEFINITIONS:
        replaced = prepare.replace_once(replaced, definition, "")
    assert "fast::GatedDeltaUpdate::use_fallback" not in replaced
    assert "fast::GatedDeltaUpdateVJP::use_fallback" not in replaced


def test_workloads_restore_the_environment(monkeypatch):
    monkeypatch.setenv("GATED_DELTA_CHUNK", "1")
    monkeypatch.delenv("GATED_DELTA_VJP_FALLBACK", raising=False)
    with gated_delta_workloads.scoped_env(
        GATED_DELTA_CHUNK="8", GATED_DELTA_VJP_FALLBACK="1"
    ):
        import os

        assert os.environ["GATED_DELTA_CHUNK"] == "8"
    import os

    assert os.environ["GATED_DELTA_CHUNK"] == "1"
    assert "GATED_DELTA_VJP_FALLBACK" not in os.environ


def test_workloads_cover_every_metal_only_path():
    cases = list(gated_delta_workloads.cases())
    assert len({case["id"] for case in cases}) == len(cases)
    kernels = {(case["kind"], case["chunk"]) for case in cases}
    assert kernels == {
        ("forward", "0"),
        ("forward", "8"),
        ("forward", "16"),
        ("vjp", None),
    }
    assert {case["dims"] for case in cases if case["kind"] == "vjp"} == set(
        gated_delta_workloads.DIMS
    )
    record = {
        "id": cases[0]["id"],
        "close": [True, True],
        "maximumErrors": [0.0, 0.0],
        "dispatchCount": 1,
    }
    records = [dict(record, id=case["id"]) for case in cases]
    gated_delta_workloads.validate(records, native=True)
    with pytest.raises(ValueError):
        gated_delta_workloads.validate(records, native=False)
    with pytest.raises(ValueError):
        gated_delta_workloads.validate(records[:-1], native=True)
    records[0] = dict(records[0], close=[True, False])
    with pytest.raises(ValueError):
        gated_delta_workloads.validate(records, native=True)


def test_upstream_dispatches_cover_both_tests_and_the_workloads():
    counts = verify_upstream_gated_delta.expected_dispatches()
    for entry in counts:
        layout.signature(entry)
    # test_gated_delta_dtypes checks dtypes of recurrences it never evaluates.
    assert "seq_gated_delta_bfloat16_t_128_128_32_32_1" not in counts
    # The 33-step backward forwards run the fused chunk at the default size.
    assert counts["gated_delta_fused_chunk_float_128_128_16_48_8"] == 4
    assert counts["seq_gated_delta_float_128_128_16_48_1_save"] == 2
    assert counts["seq_gated_delta_vjp_float_128_128_16_48_1"] == 2
    assert sum(counts.values()) == 38
    assert verify_upstream_gated_delta.package_entries() == sorted(counts)
    assert verify_upstream_gated_delta.UPSTREAM_TESTS == (
        "test_fast_gated_delta.TestGatedDelta.test_gated_delta_fallback_masked",
        "test_fast_gated_delta.TestGatedDelta.test_gated_delta_dtypes",
    )


def test_backward_fallback_column_sums_are_translated_ahead(monkeypatch, tmp_path):
    from demos.integrations.mlx.portable_host import reduction_packages

    columns = verify_upstream_gated_delta.COLUMN_SUMS
    assert "col_reduce_looped_1_32_32_reduce_sumfloat32" in columns
    assert set(columns) <= set(reduction_packages.ENTRY_GROUPS["column"])
    assert all(entry.endswith("_reduce_sumfloat32") for entry in columns)
    seen = {}

    def translate_ahead(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(
        verify_upstream_gated_delta.upstream_runner, "translate_ahead", translate_ahead
    )
    verify_upstream_gated_delta.prepare_packages(tmp_path, tmp_path, tmp_path, 1)
    assert seen["columns"] == columns
    assert seen["operators"] == ("selection",)
