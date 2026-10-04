"""Half arithmetic package selection, typed dispatch and exact result checks."""

import copy
import ctypes
import json
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import half_arithmetic_workloads as workloads
from demos.integrations.mlx.portable_host import (
    half_storage,
    packages,
    prepare,
    runtime,
)


@pytest.fixture(scope="module", params=("metal", "directx", "opengl"))
def translated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("half-arithmetic-" + request.param)
    binary = root / packages.BINARY_SOURCE
    binary.parent.mkdir(parents=True)
    binary.write_text(
        "\n".join(
            f"""template<typename T> kernel void {entry}_impl(device const T* a [[buffer(0)]],
device const T* b [[buffer(1)]], device {kind}* c [[buffer(2)]],
constant uint& size [[buffer(3)]], uint index [[thread_position_in_grid]]) {{
if (index < size) c[index] = {expression}; }}
template [[host_name("{entry}")]] [[kernel]]
decltype({entry}_impl<half>) {entry}_impl<half>;"""
            for entry in (
                *packages.HALF_BINARY_ENTRIES,
                *packages.HALF_COMPARISON_ENTRIES,
            )
            for kind, expression in [
                (
                    ("bool", "a[index] < b[index]")
                    if entry in packages.HALF_COMPARISON_ENTRIES
                    else ("T", "a[index] + b[index]")
                )
            ]
        )
    )
    (root / packages.UNARY_SOURCE).write_text(
        """template<typename T> kernel void absolute_impl(
device const T* in [[buffer(0)]], device T* out [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
if (index < size) out[index] = abs(in[index]); }
template [[host_name("v_Absfloat16float16")]] [[kernel]]
decltype(absolute_impl<half>) absolute_impl<half>;
"""
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kw: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(
            root, root / "arithmetic", request.param, family="half-arithmetic"
        )
    assert set(index["descriptors"]) == set(packages.HALF_ARITHMETIC_ENTRIES)
    return root / "arithmetic", index


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
    return runtime.HostRuntime(base, tmp_path / "trace", half_arithmetic=directory)


def buffers_for(entry):
    absolute = entry in packages.HALF_ABSOLUTE_ENTRIES
    boolean = entry in packages.HALF_COMPARISON_ENTRIES
    spec = {
        "in" if absolute else "a": ("float16", [0x3C00, 0xC000, 0x4200]),
        **({} if absolute else {"b": ("float16", [0x4000] * 3)}),
        "out" if absolute else "c": ("bool_" if boolean else "float16", [0] * 3),
        "size": ("uint32", [3]),
    }
    output = "out" if absolute else "c"
    memory = {
        name: (runtime.TYPES[kind] * len(values))(*values)
        for name, (kind, values) in spec.items()
    }
    buffers = (runtime.Buffer * len(spec))(
        *(
            runtime.Buffer(
                name.encode(),
                kind.encode(),
                ctypes.addressof(memory[name]),
                len(values),
                int(name == output),
            )
            for name, (kind, values) in spec.items()
        )
    )
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(3, 1, 1), (ctypes.c_uint32 * 3)(1, 1, 1)
    )
    return buffers, memory, output, launch


@pytest.mark.parametrize("entry", packages.HALF_ARITHMETIC_ENTRIES)
@pytest.mark.parametrize(
    "fault", (None, "dtype", "null", "guard", "encoding", "range", "shape")
)
def test_half_arithmetic_typed_dispatch_and_readback(host, monkeypatch, entry, fault):
    buffers, memory, output_name, launch = buffers_for(entry)
    boolean = entry in packages.HALF_COMPARISON_ENTRIES
    expected = [1, 0, 1] if boolean else [0x4200, 0, 0x4500]
    physical = (
        ([bool(value) for value in expected] if host.target == "metal" else expected)
        if boolean
        else half_storage.pack(expected, host.target)
    )
    guard = (
        (
            [index % 2 == 0 for index in range(32)]
            if host.target == "metal"
            else [int(index % 2 == 0) for index in range(32)]
        )
        if boolean
        else half_storage.pack(half_storage.GUARD, host.target)
    )
    encoding = None if boolean else half_storage.encoding(host.target)
    output = {
        "dtype": runtime.physical_dtype("bool_" if boolean else "float16", host.target),
        "shape": [35],
        "values": physical + guard,
    }
    if encoding:
        output["encoding"] = encoding
    if fault == "dtype":
        buffers[0].dtype = b"float32"
    elif fault == "null":
        buffers[0].data = None
    elif fault == "guard":
        output["values"][-1] = True if boolean else 0
    elif fault == "encoding":
        output["encoding"] = "invalid"
    elif fault == "range":
        output["values"][0] = (
            0x3F800001 if host.target == "opengl" and not boolean else 2**32
        )
    elif fault == "shape":
        output["shape"] = [34]
    calls = []

    def execute(host, request):
        calls.append(request)
        name = next(
            binding["name"]
            for binding in host.descriptors[entry]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(status="ok", outputs={name: output}, details={})

    monkeypatch.setattr(runtime.gather_dispatch, "execute", execute)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        assert list(memory[output_name]) == [0] * 3
        assert not host.trace.exists()
    else:
        host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        assert list(memory[output_name]) == expected
        event = json.loads(host.trace.read_text())
        assert event["halfStorage"]["logicalWords"] == expected
        assert event["halfStorage"]["guardValues"] == guard
        assert event["packageRoot"] == str(host.half_arithmetic_directory / "package")
    assert len(calls) == int(fault not in {"dtype", "null"})


@pytest.mark.parametrize("fault", ("family", "target", "missing", "extra"))
def test_half_arithmetic_family_rejects_wrong_inventory(host, tmp_path, fault):
    index = json.loads((host.half_arithmetic_directory / "index.json").read_text())
    if fault in {"family", "target"}:
        index[fault] = "invalid"
    elif fault == "missing":
        index["descriptors"].pop("vv_Addfloat16")
    else:
        index["descriptors"]["extra"] = copy.deepcopy(
            next(iter(index["descriptors"].values()))
        )
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="exact entry set"):
        runtime.HostRuntime(
            host.directory, tmp_path / "trace", half_arithmetic=directory
        )


def test_half_arithmetic_reference_inventory_and_mutations():
    import numpy as np

    records = []
    for case in workloads.cases():
        a, b = workloads.inputs(np, case)
        records.append(
            workloads.record(np, case, a, b, workloads.reference(np, case, a, b), 0)
        )
    assert len(records) == 120
    assert {item["entry"] for item in records} == set(packages.HALF_ARITHMETIC_ENTRIES)
    workloads.validate(records, [], native=False)
    for index, item in enumerate(records):
        if not item["resultWords"]:
            continue
        changed = copy.deepcopy(records)
        changed[index]["resultWords"][0] ^= 1
        with pytest.raises(ValueError, match="result differs"):
            workloads.validate(changed, [], native=False)
    with pytest.raises(ValueError, match="inventory"):
        workloads.validate(records[:-1], [], native=False)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_half_arithmetic_native_evidence_and_storage_layouts(target):
    import numpy as np

    records, trace = [], []
    for case in workloads.cases():
        a, b = workloads.inputs(np, case)
        expected = workloads.reference(np, case, a, b)
        sequence = workloads.sequence(case, a, b)
        records.append(workloads.record(np, case, a, b, expected, len(sequence)))
        if not sequence:
            continue
        if case["operation"] == "Abs" and case["layout"] == "transpose":
            assert sequence == [case["entry"]]
            stored = np.abs(a.ravel(order="K"))
        elif case["operation"] == "Abs" and case["layout"] == "broadcast":
            assert sequence == [case["entry"]]
            stored = np.abs(a[0])
            assert stored.size == 5
        else:
            stored = expected
        bits = workloads.words(np, stored)
        boolean = expected.dtype == np.bool_
        physical = list(map(bool, bits)) if boolean and target == "metal" else bits
        guard = [index % 2 == 0 for index in range(32)] if boolean else [0x3555] * 32
        encoding = None if boolean else "ieee754-binary16"
        dtype = ("bool" if target == "metal" else "uint32") if boolean else "float16"
        if not boolean and target == "opengl":
            physical = list(map(workloads.widen_reference, physical))
            guard = list(map(workloads.widen_reference, guard))
            encoding, dtype = "ieee754-binary32", "float32"
        trace.extend({"entry": entry} for entry in sequence[:-1])
        trace.append(
            {
                "entry": case["entry"],
                "target": target,
                "dispatchVersion": 3,
                "threads": stored.size,
                "workgroupCount": [stored.size, 1, 1],
                "workgroupSize": [1, 1, 1],
                "halfStorage": {
                    "logicalType": "bool_" if boolean else "float16",
                    "physicalType": dtype,
                    "encoding": encoding,
                    "values": physical,
                    "logicalWords": bits,
                    "guardValues": guard,
                },
            }
        )
    workloads.validate(records, trace, native=True)
    for key in ("values", "logicalWords", "guardValues"):
        changed = copy.deepcopy(trace)
        changed[0]["halfStorage"][key][0] ^= 1
        with pytest.raises(ValueError, match="readback or dispatch"):
            workloads.validate(records, changed, native=True)
    with pytest.raises(ValueError, match="sequence"):
        workloads.validate(records, trace[:-1], native=True)


def test_half_arithmetic_ci_requires_native_execution_and_unchanged_upstream_test():
    from pathlib import Path

    import yaml

    from demos.integrations.mlx.portable_host import verify_half_arithmetic

    assert verify_half_arithmetic.UPSTREAM_TESTS == (
        "test_random.TestRandom.test_broadcastable_scale_loc",
    )
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["half-host"]
    assert (
        job["if"] == "github.event_name != 'schedule'"
        and "continue-on-error" not in job
    )
    assert {item["target"] for item in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    steps = {step.get("name"): step for step in job["steps"]}
    for name in (
        "Translate half arithmetic packages",
        "Translate random prerequisites for half arithmetic",
        "Execute half arithmetic and upstream distribution test",
    ):
        assert "if" not in steps[name] and "continue-on-error" not in steps[name]
        assert "set -euo pipefail" in steps[name]["run"]
    proof = steps["Execute half arithmetic and upstream distribution test"]["run"]
    for option in ("--packages", "--half", "--arithmetic", "--random"):
        assert option in proof
    assert "portable_host.verify_half_arithmetic" in proof
    assert steps["Retain half execution evidence"]["if"] == "always()"
    assert (
        "demos/integrations/mlx/tests/host/test_portable_half_arithmetic.py"
        in steps["Validate half host contracts"]["run"]
    )
