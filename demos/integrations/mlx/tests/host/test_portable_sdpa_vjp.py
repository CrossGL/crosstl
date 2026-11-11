import ctypes
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project.storage_record_layout import storage_record_layout
from demos.integrations.mlx.portable_host import (
    element_storage,
    lazy_package_evidence,
    prepare,
    runtime,
    sdpa_vjp_dispatch,
    sdpa_vjp_layout,
    sdpa_vjp_packages,
    upstream_matrix_packages,
)

HOST = Path(sdpa_vjp_layout.__file__).parent
# scaled_dot_product_attention_vjp.metal's SDPAVJPTileParams, as OpenGL
# reflects it.
TILE_LAYOUT = storage_record_layout(
    "SDPAVJPTileParams",
    [
        ("float" if name == "scale" else "int", name)
        for name in sdpa_vjp_layout.TILE_FIELDS
    ],
    storage_layout="std430",
)
F32 = element_storage.GUARDS["float32"]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _execution(count, size):
    return {"workgroupCount": list(count), "workgroupSize": list(size)}


def tile(bq, bk, qL, i0=0, j0=0, scale=0.125, diag_off=0, causal=0):
    return [bq, bk, qL, i0, j0, _bits(scale), diag_off, causal]


def test_pipeline_names_and_bindings_follow_the_kernels():
    odo = sdpa_vjp_layout.odo_entry("bfloat16", 72)
    ds = sdpa_vjp_layout.ds_entry("float16")
    add = sdpa_vjp_layout.reduce_entry("float32", True)
    assert (odo, ds, add) == (
        "sdpa_vjp_odo_bfloat16_t_72",
        "sdpa_vjp_ds_float16_t",
        "sdpa_vjp_reduce_add_float",
    )
    assert sdpa_vjp_layout.reduce_entry("float16", False) == (
        "sdpa_vjp_reduce_set_float16_t"
    )
    assert sdpa_vjp_layout.parameters(odo) == ("o", "cot_o", "odo", "qL")
    assert sdpa_vjp_layout.outputs(ds) == ("dS", "P")
    assert sdpa_vjp_layout.records(ds) == ("p",)
    assert sdpa_vjp_layout.directions(add) == {
        "src": 0,
        "acc": 2,
        "out": 2,
        "rows": 0,
        "dim": 0,
        "group": 0,
        "acc_rows": 0,
        "row_off": 0,
    }
    assert sdpa_vjp_layout.parameter_type(odo, "o") == "bfloat16"
    assert sdpa_vjp_layout.parameter_type(odo, "odo") == "float32"
    assert sdpa_vjp_layout.parameter_type(ds, "lse") == "float32"
    assert sdpa_vjp_layout.parameter_type(ds, "p") == "int64"
    assert sdpa_vjp_layout.parameter_type(add, "acc") == "float32"
    assert sdpa_vjp_layout.parameter_type(add, "rows") == "int32"
    assert sdpa_vjp_layout.workgroup_size(odo) == (32, 1, 1)
    assert sdpa_vjp_layout.workgroup_size(add) == (32, 8, 1)
    # instantiate_odo_shapes has no head dim 32, and names are canonical.
    for name in (
        "sdpa_vjp_odo_float_32",
        "sdpa_vjp_odo_float_064",
        "sdpa_vjp_ds_half",
        "sdpa_vjp_reduce_mul_float",
    ):
        with pytest.raises(ValueError):
            sdpa_vjp_layout.signature(name)
    assert all(
        lazy_package_evidence.family(entry) == "sdpa-vjp"
        for entry in sdpa_vjp_packages.verification_entries()
    )


def test_launch_plans_bound_every_buffer():
    odo = sdpa_vjp_layout.odo_entry("float32", 64)
    plan = sdpa_vjp_layout.plan_launch(
        odo,
        {"o": 6 * 40 * 64, "cot_o": 6 * 40 * 64, "odo": 6 * 40},
        {"qL": 40},
        {},
        6 * 40,
        _execution((1, 40, 6), (32, 1, 1)),
    )
    assert plan["grid"] == [32, 40, 6] and plan["heads"] == 6
    ds = sdpa_vjp_layout.ds_entry("float32")
    block = 4 * 24 * 40
    counts = {"S": block, "dP": block, "lse": 4 * 50, "odo": 4 * 50}
    counts |= {"dS": block, "P": block}
    words = {"p": tile(24, 40, 50, i0=26, j0=40, diag_off=-3, causal=1)}
    plan = sdpa_vjp_layout.plan_launch(
        ds, counts, {}, words, block, _execution((2, 3, 4), (32, 8, 1))
    )
    assert plan["diag_off"] == -3 and plan["scale"] == 0.125
    with pytest.raises(ValueError):
        # The query block runs past the query length.
        sdpa_vjp_layout.plan_launch(
            ds,
            counts,
            {},
            {"p": tile(24, 40, 50, i0=27)},
            block,
            _execution((2, 3, 4), (32, 8, 1)),
        )
    with pytest.raises(ValueError, match="cover"):
        sdpa_vjp_layout.plan_launch(
            ds, counts, {}, words, block, _execution((2, 4, 4), (32, 8, 1))
        )
    add = sdpa_vjp_layout.reduce_entry("float16", True)
    values = {"rows": 10, "dim": 64, "group": 3, "acc_rows": 30, "row_off": 20}
    counts = {"src": 2 * 3 * 10 * 64, "acc": 2 * 30 * 64, "out": 2 * 30 * 64}
    plan = sdpa_vjp_layout.plan_launch(
        add, counts, values, {}, 2 * 30 * 64, _execution((2, 2, 2), (32, 8, 1))
    )
    assert plan["rowOffset"] == 20 and plan["accumulate"] is True
    for fault in ({"row_off": 21}, {"dim": 16}):
        with pytest.raises(ValueError):
            sdpa_vjp_layout.plan_launch(
                add,
                counts,
                values | fault,
                {},
                2 * 30 * 64,
                _execution((2, 2, 2), (32, 8, 1)),
            )
    with pytest.raises(ValueError):
        # dispatch_threads grids launch as covering threadgroups.
        sdpa_vjp_layout.plan_launch(
            add,
            counts,
            values,
            {},
            2 * 30 * 64,
            {**_execution((2, 2, 2), (32, 8, 1)), "threadGridSize": [64, 10, 2]},
        )


def test_project_config_lowers_only_the_row_products_subgroups(tmp_path):
    for entry in sdpa_vjp_packages.verification_entries():
        kind = sdpa_vjp_layout.signature(entry)["kind"]
        for target in ("metal", "directx", "opengl"):
            config = sdpa_vjp_packages.project_config(tmp_path, "out", target, entry)
            assert config.entry_points == {sdpa_vjp_layout.SOURCE: (entry,)}
            assert bool(config.index_range_assertions) == (target == "opengl")
            options = config.source_options["metal"]["target_options"]
            assert bool(options) == (kind == "odo")
    odo = sdpa_vjp_layout.odo_entry("float32", 96)
    expressions = [
        item["expression"] for item in sdpa_vjp_layout.opengl_index_assertions(odo)
    ]
    assert expressions == ["row * 96", "o_i + d", "co_i + d", "row"]
    assert sdpa_vjp_packages.CACHE_DIRECTORY == "sdpa-vjp"
    assert "sdpa-vjp" in upstream_matrix_packages.ORDER


class FakeCache:
    def __init__(self, entry):
        self.entry = entry

    def get(self, entry):
        assert entry == self.entry
        bindings = [
            self._binding(entry, name) for name in sdpa_vjp_layout.parameters(entry)
        ]
        return {"bindings": bindings, "artifact": {"hash": "fake"}}, "/fake"

    @staticmethod
    def _binding(entry, name):
        if name in sdpa_vjp_layout.records(entry):
            scalar = TILE_LAYOUT
        else:
            dtype = sdpa_vjp_layout.parameter_type(entry, name)
            scalar = {"elementType": dtype, "elementStrideBytes": 4}
        return {
            "name": f"{name}Binding",
            "access": (
                "read_write" if name in sdpa_vjp_layout.outputs(entry) else "read"
            ),
            "kind": "buffer",
            "provenance": {"sourceResource": {"parameter": name}},
            "scalarLayout": scalar,
        }


def run(monkeypatch, tmp_path, entry, memory, execution, write=None):
    directions = sdpa_vjp_layout.directions(entry)
    types = {name: sdpa_vjp_layout.parameter_type(entry, name) for name in memory}
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                directions[name],
            )
            for name in memory
        )
    )
    host = SimpleNamespace(
        sdpa_vjps=FakeCache(entry),
        target="opengl",
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )
    seen = {}

    def execute(_, request):
        seen.update(request)
        outputs = {}
        for binding in request["outputs"]:
            name = binding.removesuffix("Binding")
            initial = request["inputs"][binding]
            words = element_storage.unpack(types[name], "opengl", initial["values"])
            for index, word in write(name, len(memory[name])).items():
                words[index] = word
            outputs[binding] = element_storage.packet(types[name], "opengl", words)
        return SimpleNamespace(status="ok", outputs=outputs, details={"request": {}})

    monkeypatch.setattr(
        sdpa_vjp_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
        },
    )
    monkeypatch.setattr(sdpa_vjp_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*execution["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*execution["workgroupSize"]),
    )
    output = sdpa_vjp_layout.outputs(entry)[-1]
    sdpa_vjp_dispatch.dispatch(
        host, entry, array, len(memory), len(memory[output]), launch
    )
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return event, host, seen


def test_block_records_pack_and_outputs_start_as_guards(monkeypatch, tmp_path):
    entry = sdpa_vjp_layout.ds_entry("float32")
    heads, bq, bk, qL = 2, 8, 32, 16
    block = heads * bq * bk
    memory = {
        "S": (ctypes.c_uint32 * block)(*[_bits(1.0)] * block),
        "dP": (ctypes.c_uint32 * block)(),
        "lse": (ctypes.c_uint32 * (heads * qL))(),
        "odo": (ctypes.c_uint32 * (heads * qL))(),
        "dS": (ctypes.c_uint32 * block)(),
        "P": (ctypes.c_uint32 * block)(),
        "p": (ctypes.c_int64 * 8)(*tile(bq, bk, qL, i0=8, diag_off=-2, causal=1)),
    }

    def everywhere(name, size):
        return dict.fromkeys(range(size), _bits(0.5))

    execution = _execution((1, 1, heads), (32, 8, 1))
    event, host, request = run(
        monkeypatch, tmp_path, entry, memory, execution, everywhere
    )
    assert host.dispatch_count == 1
    record = request["inputs"]["pBinding"]["values"]
    assert record == [bq, bk, qL, 8, 0, _bits(0.125), 2**32 - 2, 1]
    assert request["inputs"]["dSBinding"]["values"][0] == F32
    assert (
        request["inputs"]["SBinding"]["values"][block:]
        == [0x3FC00000] * element_storage.GUARD_COUNT
    )
    assert set(request["outputs"]) == {"dSBinding", "PBinding"}
    assert list(memory["P"]) == [_bits(0.5)] * block
    assert event["sdpaVjpMetadata"]["causal"] == 1
    assert set(event["outputHashes"]) == {"dS", "P"}

    def past_the_end(name, size):
        return {size: 0} if name == "P" else everywhere(name, size)

    with pytest.raises(RuntimeError, match="P guard"):
        run(monkeypatch, tmp_path, entry, memory, execution, past_the_end)
    # An output that overlaps an input is refused before any launch.
    memory["dS"] = (ctypes.c_uint32 * block).from_address(ctypes.addressof(memory["S"]))
    with pytest.raises(ValueError, match="overlaps"):
        run(monkeypatch, tmp_path, entry, memory, execution, everywhere)


def test_reductions_keep_every_row_outside_their_window(monkeypatch, tmp_path):
    entry = sdpa_vjp_layout.reduce_entry("float32", True)
    heads, group, rows, dim, acc_rows, row_off = 2, 2, 3, 32, 8, 4
    total = heads * acc_rows * dim
    memory = {
        "src": (ctypes.c_uint32 * (heads * group * rows * dim))(),
        "acc": (ctypes.c_uint32 * total)(*range(total)),
        "out": (ctypes.c_uint32 * total)(*range(total, 2 * total)),
        "rows": (ctypes.c_int32 * 1)(rows),
        "dim": (ctypes.c_int32 * 1)(dim),
        "group": (ctypes.c_int32 * 1)(group),
        "acc_rows": (ctypes.c_int32 * 1)(acc_rows),
        "row_off": (ctypes.c_int32 * 1)(row_off),
    }
    window = [
        index
        for head in range(heads)
        for index in range(
            (head * acc_rows + row_off) * dim, (head * acc_rows + row_off + rows) * dim
        )
    ]

    def in_window(name, size):
        return dict.fromkeys(window, _bits(2.0))

    execution = _execution((1, 1, heads), (32, 8, 1))
    acc_before = list(memory["acc"])
    event, host, request = run(
        monkeypatch, tmp_path, entry, memory, execution, in_window
    )
    # Both start as the host's storage and keep a guard tail.
    assert request["inputs"]["accBinding"]["values"][:total] == acc_before
    assert (
        request["inputs"]["accBinding"]["values"][total:]
        == [F32] * element_storage.GUARD_COUNT
    )
    assert [memory["acc"][index] for index in window] == [_bits(2.0)] * len(window)
    assert memory["acc"][0] == 0 and memory["out"][total - 1] == 2 * total - 1
    assert event["sdpaVjpMetadata"]["rowOffset"] == row_off
    assert set(event["outputHashes"]) == {"acc", "out"}

    def outside(name, size):
        return {**in_window(name, size), window[0] - 1: 0}

    with pytest.raises(RuntimeError, match="outside its window"):
        run(monkeypatch, tmp_path, entry, memory, execution, outside)


def test_the_host_ports_and_routes_the_fused_vjp():
    source = (HOST / "runtime.py").read_text()
    assert "sdpa_vjp_packages.SdpaVjpPackageCache(" in source
    assert "if sdpa_vjp_dispatch.is_entry(entry):" in source
    port = (HOST / "sdpa.inc").read_text()
    for definition in (
        "bool ScaledDotProductAttentionVJP::use_fallback(",
        "void ScaledDotProductAttentionVJP::eval_gpu(",
        "void sdpa_vjp_blocked_host(",
        '"MLX_SDPA_VJP_BLOCK"',
        '"sdpa_vjp_odo_"',
        '"sdpa_vjp_reduce_add_"',
    ):
        assert definition in port
    assert "ScaledDotProductAttentionVJP" in prepare.FAST_PRIMITIVES
    assert any(
        "ScaledDotProductAttentionVJP::use_fallback" in definition
        for definition in prepare.ATTENTION_DEFINITIONS
    )
