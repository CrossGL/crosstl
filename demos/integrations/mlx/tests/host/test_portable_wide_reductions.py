"""Wide whole-array reductions mirror the pinned instantiations, check each
pass against upstream's plan and recompute integer passes exactly."""

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    reduction_layout,
    reduction_packages,
    runtime,
    wide_reduction_dispatch,
    wide_reduction_evidence,
    wide_reduction_packages,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")
INTEGERS = ("int64", "uint64")
INT64_MAX, UINT64_MAX = 2**63 - 1, 2**64 - 1


def test_entries_are_pinned_and_disjoint_from_the_installed_families():
    path = CONTRACTS / "reduce.current-tree.translation.json"
    pinned = set(json.loads(path.read_text(encoding="utf-8"))["entries"])
    entries = set(wide_reduction_packages.ENTRIES)
    assert len(entries) == 44 and entries <= pinned
    assert not entries & set(reduction_packages.ALL_ENTRIES)
    # instantiate_and_or's non-Boolean kernels write Booleans.
    assert {
        dtype
        for entry, dtype in wide_reduction_packages.ENTRIES.items()
        if entry.startswith(("all_reduce_and", "all_reduce_or"))
    } == {"int16", "int32", "int64", "float16", "bfloat16", "float32"}
    assert wide_reduction_packages.accumulator("or", "float32") == "bool_"
    assert {
        dtype
        for entry, dtype in wide_reduction_packages.ENTRIES.items()
        if entry.startswith(("all_reduce_min", "all_reduce_max"))
    } == {"int64", "uint64", "complex64", "int8", "int16", "uint8", "uint16"}
    assert wide_reduction_packages.signature("all_reduce_prodbfloat16") == (
        "prod",
        "bfloat16",
    )
    for entry in ("all_reduce_sumfloat32", "all_reduce_minfloat16", None):
        with pytest.raises(ValueError, match="Unknown wide"):
            wide_reduction_packages.signature(entry)


@pytest.mark.parametrize(
    "dtype, count, rows",
    [
        ("int64", 4096, 1),
        ("complex64", 4097, 128),
        ("uint64", 2**23, 128),
        ("int64", 2**23 + 1, 4096),
        ("float16", 2**25, 128),
        ("bfloat16", 2**25 + 1, 4096),
    ],
)
def test_contracts_follow_upstream_logical_bytes(dtype, count, rows):
    entry = f"all_reduce_sum{dtype}"
    plan = reduction_layout.stage(count, dtype)
    assert plan["workgroupCount"] == [1, rows, 1]
    execution = {key: plan[key] for key in ("workgroupCount", "workgroupSize")}
    contract = wide_reduction_packages.dispatch_contract(entry, count, execution)
    assert contract["rowSize"] == (count + rows - 1) // rows
    with pytest.raises(ValueError, match="upstream plan"):
        wide_reduction_packages.dispatch_contract(
            entry, count, {**execution, "workgroupSize": [64, 1, 1]}
        )


def test_cache_identity_covers_the_contract_and_this_recipe(monkeypatch, tmp_path):
    cache = wide_reduction_packages.cache(tmp_path, tmp_path / "cache", "opengl")
    monkeypatch.setattr(cache, "_require_source", lambda: None)
    execution = {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]}
    identity = cache._identity("all_reduce_sumcomplex64", 3, execution)
    assert identity["contract"] == {
        "entry": "all_reduce_sumcomplex64",
        "count": 3,
        "rowSize": 3,
        "execution": execution,
    }
    assert identity["target"] == "opengl" and len(identity["recipeHash"]) == 64
    with pytest.raises(ValueError, match="no row layout"):
        cache._identity("all_reduce_sumint64", 3, execution, layout={})


def native_buffers(entry, words, *, rows=1, row_size=None):
    """A callback's buffers for one pass: input, output and the metadata."""
    _, dtype = wide_reduction_packages.signature(entry)
    ctype = element_storage.CTYPES[dtype]
    memory = {
        "in": (ctype * len(words))(*words),
        "out": (ctype * rows)(),
        "in_size": (ctypes.c_uint64 * 1)(len(words)),
        "row_size": (ctypes.c_uint64 * 1)(row_size or len(words)),
    }
    types = wide_reduction_dispatch.operand_types(entry)
    array = (runtime.Buffer * 4)(
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
    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, entry, count, execution):
        self.requests.append((entry, count, execution))
        types = wide_reduction_dispatch.operand_types(entry)
        prefix = "" if self.target == "metal" else entry + "_"
        bindings = []
        for name in ("in", "out", "in_size", "row_size"):
            value = element_storage.packet(types[name], self.target, [])
            scalar = (
                dict(element_storage.COMPLEX_LAYOUT)
                if types[name] == "complex64"
                else {
                    "elementType": value["dtype"],
                    "elementStrideBytes": element_storage.PHYSICAL_SIZES[
                        value["dtype"]
                    ],
                }
            )
            if self.target == "directx" and types[name] == "float16":
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
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def run(monkeypatch, tmp_path, target, entry, words, *, fault=None):
    operation, dtype = wide_reduction_packages.signature(entry)
    plan = reduction_layout.stage(len(words), dtype)
    rows = plan["workgroupCount"][1]
    array, memory = native_buffers(entry, words, rows=rows, row_size=plan["rowSize"])
    host = SimpleNamespace(
        wide_reductions=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        values = element_storage.unpack(dtype, target, inputs["inBinding"]["values"])
        initial = element_storage.unpack(dtype, target, inputs["outBinding"]["values"])
        assert inputs["in_sizeBinding"]["values"] == [len(words)]
        if dtype in element_storage.INEXACT:
            # Inexact passes echo their first input; identity checks cover them.
            result = values[:rows]
        else:
            result = wide_reduction_evidence.expected_words(
                operation, dtype, values, rows, plan["rowSize"]
            )
        result += initial[rows:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet(dtype, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        wide_reduction_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(wide_reduction_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*plan["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*plan["workgroupSize"]),
    )
    wide_reduction_dispatch.dispatch(host, entry, array, 4, len(words), launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", INTEGERS)
def test_integer_passes_wrap_and_recompute(monkeypatch, tmp_path, target, dtype):
    high = INT64_MAX if dtype == "int64" else UINT64_MAX
    words = [high, 2, 3]
    # The sum passes the type's maximum and wraps modulo 2**64.
    wrapped = -(2**63) + 4 if dtype == "int64" else 4
    for operation, expected in (("sum", wrapped), ("max", high)):
        entry = f"all_reduce_{operation}{dtype}"
        out, event, host = run(monkeypatch, tmp_path, target, entry, words)
        assert out == [expected] and host.dispatch_count == 1
        assert host.wide_reductions.requests[0][1] == 3
        assert wide_reduction_evidence.audit_event(event, native=False) == 1
        tampered = {**event, "outputValues": [expected ^ 1]}
        with pytest.raises(ValueError, match="differs from its inputs"):
            wide_reduction_evidence.audit_event(tampered, native=False)
        tampered = {**event, "operandValues": [high, 2, 4]}
        with pytest.raises(ValueError, match="upload differs"):
            wide_reduction_evidence.audit_event(tampered, native=False)


@pytest.mark.parametrize("target", TARGETS)
def test_two_pass_rows_fold_their_own_spans(monkeypatch, tmp_path, target):
    words = list(range(-2048, 2049)) + [5]
    out, event, _ = run(monkeypatch, tmp_path, target, "all_reduce_sumint64", words)
    assert len(out) == 128 and sum(out) == sum(words)
    row_size = event["reductionMetadata"]["row_size"]
    assert row_size == 33 and out[0] == sum(words[:33])
    # Rows that start past the input keep the identity, as Op::init does.
    assert out[-1] == 0
    assert wide_reduction_evidence.audit_event(event, native=False) == 128


def test_recompute_identities_and_refusals():
    expected = wide_reduction_evidence.expected_words
    assert expected("prod", "int64", [2**62, 2], 1, 2) == [-(2**63)]
    assert expected("min", "uint64", [], 1, 4) == [UINT64_MAX]
    assert expected("max", "int64", [], 1, 4) == [-(2**63)]
    assert expected("prod", "uint64", [], 1, 4) == [1]
    with pytest.raises(ValueError, match="integer type"):
        expected("sum", "complex64", [0], 1, 1)
    with pytest.raises(ValueError, match="storage range"):
        expected("sum", "uint64", [-1], 1, 1)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "complex64"))
def test_inexact_passes_use_reflected_carriers(monkeypatch, tmp_path, target, dtype):
    words = {
        "float16": [0x3C00, 0x4000],
        "bfloat16": [0x3F80, 0x4000],
        "complex64": [0x400000003F800000, 0x3F80000040000000],
    }[dtype]
    entry = f"all_reduce_sum{dtype}"
    out, event, _ = run(monkeypatch, tmp_path, target, entry, words)
    assert out == words[:1]
    assert not wide_reduction_evidence.recomputable(entry)
    assert event["inputs"]["inBinding"] == element_storage.packet(dtype, target, words)


def test_dispatch_refuses_changed_guards_buffers_and_metadata(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="guard"):
        run(
            monkeypatch,
            tmp_path,
            "opengl",
            "all_reduce_sumint64",
            [1, 2],
            fault="guard",
        )
    # Off Metal, subnormal bfloat16 inputs stay refused, as in the base families.
    for target in ("directx", "opengl"):
        with pytest.raises(ValueError, match="crosstl#2000"):
            run(
                monkeypatch,
                tmp_path,
                target,
                "all_reduce_sumbfloat16",
                [0x3F80, 0x8001],
            )
    out, _, _ = run(monkeypatch, tmp_path, "metal", "all_reduce_sumbfloat16", [0x0001])
    assert out == [0x0001]
    entry = "all_reduce_minint64"
    execution = {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]}
    array, memory = native_buffers(entry, [4, 5, 6])
    buffers = wide_reduction_dispatch.supplied_buffers(array, 4, "Wide reduction")
    assert wide_reduction_dispatch.validate(entry, buffers, 3, execution) == {
        "operation": "min",
        "dtype": "int64",
        "rows": 1,
        "in_size": 3,
        "row_size": 3,
    }
    with pytest.raises(ValueError, match="type, extent"):
        wide_reduction_dispatch.validate("all_reduce_minuint64", buffers, 3, execution)
    with pytest.raises(ValueError, match="type, extent"):
        wide_reduction_dispatch.validate(entry, buffers, 2, execution)
    with pytest.raises(ValueError, match="upstream plan"):
        wide_reduction_dispatch.validate(
            entry, buffers, 3, {**execution, "workgroupCount": [1, 2, 1]}
        )
    memory["row_size"][0] = 4
    with pytest.raises(ValueError, match="metadata"):
        wide_reduction_dispatch.validate(entry, buffers, 3, execution)
    with pytest.raises(ValueError, match="buffers do not match"):
        wide_reduction_dispatch.validate(entry, {"in": buffers["in"]}, 3, execution)
    with pytest.raises(ValueError, match="pinned sources"):
        wide_reduction_dispatch.dispatch(
            SimpleNamespace(wide_reductions=None), entry, array, 4, 3, None
        )


def test_inexact_events_check_package_and_native_identity(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        wide_reduction_evidence, "audit_native_execution", lambda event: seen.append(1)
    )
    package = tmp_path / "package"
    package.mkdir()
    (package / "artifact.glsl").write_bytes(b"kernel")
    artifact = {
        "packagePath": "artifact.glsl",
        "sizeBytes": 6,
        "hash": {"algorithm": "sha256", "value": hashlib.sha256(b"kernel").hexdigest()},
    }
    requests = []
    cache = SimpleNamespace(
        get=lambda *key: requests.append(key) or ({"artifact": artifact}, package)
    )
    launch = {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]}
    event = {
        "entry": "all_reduce_sumcomplex64",
        "target": "opengl",
        "threads": 3,
        **launch,
        "dispatchVersion": 3,
        "artifact": artifact,
        "packageRoot": str(package),
        "reductionMetadata": {
            "operation": "sum",
            "dtype": "complex64",
            "rows": 1,
            "in_size": 3,
            "row_size": 3,
        },
        "details": {
            "request": {"target": "opengl", "entryPoint": "main", "dispatch": launch}
        },
    }
    assert wide_reduction_evidence.audit_event(event, cache) == 1 and seen == [1]
    assert requests == [("all_reduce_sumcomplex64", 3, launch)]
    with pytest.raises(ValueError, match="metadata"):
        wide_reduction_evidence.audit_event(
            {**event, "entry": "all_reduce_prodcomplex64"}, cache
        )
    with pytest.raises(ValueError, match="native request changed"):
        wide_reduction_evidence.audit_event({**event, "target": "directx"}, cache)
    (package / "artifact.glsl").write_bytes(b"change")
    with pytest.raises(ValueError, match="changed after translation"):
        wide_reduction_evidence.audit_event(event, cache)


def test_runtime_routes_only_wide_entries_with_pinned_sources():
    host = SimpleNamespace(wide_reductions=object())
    lazy = runtime.HostRuntime.lazy_wide_reduction
    assert lazy(host, "all_reduce_sumcomplex64") and lazy(host, "all_reduce_maxuint64")
    assert not lazy(host, "all_reduce_sumfloat32")
    assert not lazy(host, "all_reduce_minfloat16")
    assert not lazy(host, "row_reduce_simple_sumint64")
    host.wide_reductions = None
    assert not lazy(host, "all_reduce_sumint64")


def test_host_adapter_routes_wide_types_to_whole_array_or_small_row_plans():
    source = Path(runtime.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    assert (
        "const bool wide = narrow || truth_test ||\n      (eight_byte ? !logical"
        in source
    )
    assert "plan.shape.back() <= 64;" in source
    assert (
        "if ((eight_byte || narrow || truth_test) && plan.type != ContiguousAllReduce &&"
        in source
    )
    # Non-Boolean all and any write Booleans in both passes; floats test by
    # value and the integers and complex64 by their bits, as remap_reduce_types.
    assert 'const char* output = truth_test ? "bool_"' in source
    assert (
        "const char* partial = promoted || truth_test ? storage_type(out.dtype()) : dtype;"
        in source
    )
    body = source.split("const char* logical_type(", 1)[1].split("\n}\n", 1)[0]
    assert body.count('return "int64";') == 1 and "case mlx::core::complex64:" in body
    assert "case mlx::core::int8:" not in body and "float64" not in body


@pytest.mark.parametrize(
    "operation, dtype, output",
    [
        ("sum", "int8", "int32"),
        ("prod", "int16", "int32"),
        ("sum", "uint8", "int32"),
        ("prod", "uint16", "uint32"),
        ("min", "int8", "int8"),
        ("max", "uint16", "uint16"),
        ("sum", "int64", "int64"),
        ("max", "complex64", "complex64"),
        ("and", "float16", "bool_"),
        ("or", "int64", "bool_"),
    ],
)
def test_narrow_integers_accumulate_as_reduce_metal_instantiates(
    operation, dtype, output
):
    # instantiate_sum_prod writes uint8 sums as int32_t and uint16 sums as
    # uint32_t; every extremum and every other type writes its element type.
    entry = f"all_reduce_{operation}{dtype}"
    assert wide_reduction_packages.accumulator(operation, dtype) == output
    assert wide_reduction_dispatch.operand_types(entry)["out"] == output
    assert wide_reduction_dispatch.operand_types(entry)["in"] == dtype


def test_narrow_integer_passes_wrap_in_their_accumulator():
    # 300 int8 values of 100 sum past int8 and fit int32; uint8 products wrap
    # at 32 bits, and extrema stay in the element type.
    expected = wide_reduction_evidence.expected_words
    assert expected("sum", "int8", [100] * 300, 1, 300) == [30000]
    assert expected("sum", "int8", [-128] * 3, 1, 3) == [-384]
    assert expected("prod", "uint8", [255] * 5, 1, 5) == [
        (255**5 + 2**31) % 2**32 - 2**31
    ]
    assert expected("prod", "uint16", [65535] * 3, 1, 3) == [65535**3 % 2**32]
    assert expected("min", "int16", [5, -7, 3], 1, 3) == [-7]
    assert expected("max", "uint8", [], 1, 0) == [0]
    with pytest.raises(ValueError, match="storage range"):
        expected("sum", "int8", [128], 1, 1)


def test_logical_passes_test_floats_by_value_and_integers_by_bits():
    # -0.0 is false, NaN and subnormals true; an empty row is the identity.
    expected = wide_reduction_evidence.expected_words
    negative_zero = {"float32": 0x80000000, "float16": 0x8000, "bfloat16": 0x8000}
    for dtype, word in negative_zero.items():
        assert expected("or", dtype, [0, word, word], 1, 3) == [0]
        assert expected("and", dtype, [word, 1], 1, 2) == [0]
        assert expected("and", dtype, [word | 1, word >> 1], 1, 2) == [1]
    assert expected("and", "float32", [0x7FC00000, 0x3F800000], 1, 2) == [1]
    assert expected("or", "int16", [0, -32768, 0, 0], 2, 2) == [1, 0]
    assert expected("and", "int64", [-1, 2, 0], 2, 2) == [1, 0]
    assert expected("and", "int32", [], 1, 0) == [1]
    assert expected("or", "float16", [], 1, 0) == [0]
    assert wide_reduction_evidence.recomputable("all_reduce_orbfloat16")
    assert not wide_reduction_evidence.recomputable("all_reduce_sumbfloat16")
    with pytest.raises(ValueError, match="storage range"):
        expected("or", "float16", [0x10000], 1, 1)
