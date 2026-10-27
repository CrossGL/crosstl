"""The portable host's port of scaled_dot_product_attention.cpp's vector kernels."""

import ctypes
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    lazy_package_evidence,
    prepare,
    runtime,
    sdpa_dispatch,
    sdpa_layout,
    sdpa_packages,
)

HOST = Path(sdpa_layout.__file__).parent
PLAIN = {
    "has_mask": False,
    "bool_mask": False,
    "float_mask": False,
    "query_transposed": False,
    "do_causal": False,
    "has_sinks": False,
}


def test_hash_names_freeze_the_function_constants_and_threadgroups():
    entry = "sdpa_vector_bfloat16_t_96_64_floatmask_qt_c_sinks"
    info = sdpa_layout.signature(entry)
    assert info["source"] == "sdpa_vector_bfloat16_t_96_64"
    assert (info["dtype"], info["D"], info["V"]) == ("bfloat16", 96, 64)
    assert info["constants"] == {
        "has_mask": True,
        "bool_mask": False,
        "float_mask": True,
        "query_transposed": True,
        "do_causal": True,
        "has_sinks": True,
    }
    assert info["workgroupSize"] == (1024, 1, 1)
    assert sdpa_layout.vector_entry("bfloat16", 96, 64, info["constants"]) == entry
    first = sdpa_layout.pass1_entry("float16", 128, 128, PLAIN, 64, 4, 2)
    assert (
        first == "sdpa_vector_2pass_1_float16_t_128_128_nomask_qnt_nc_nosinks_64@32x4x2"
    )
    info = sdpa_layout.signature(first)
    assert info["source"] == "sdpa_vector_2pass_1_float16_t_128_128"
    assert info["constants"] == {**PLAIN, "blocks": 64}
    assert info["workgroupSize"] == (32, 4, 2)
    grouped = sdpa_layout.pass1_entry("float32", 128, 128, PLAIN, 32, 16, 1, True)
    info = sdpa_layout.signature(grouped)
    assert info["source"] == "sdpa_vector_2pass_1_gqa_16_float_128_128"
    assert (info["kind"], info["heads"]) == ("pass1_gqa", 2)
    info = sdpa_layout.signature("sdpa_vector_2pass_2_float_512")
    assert (info["kind"], info["constants"], info["workgroupSize"]) == (
        "pass2",
        {},
        (1024, 1, 1),
    )


@pytest.mark.parametrize(
    "entry",
    [
        "sdpa_vector_float_72_72_nomask_qnt_nc_nosinks",
        "sdpa_vector_float_064_64_nomask_qnt_nc_nosinks",
        "sdpa_vector_float32_64_64_nomask_qnt_nc_nosinks",
        "sdpa_vector_2pass_1_float_64_64_nomask_qnt_nc_nosinks_48@32x4x1",
        "sdpa_vector_2pass_1_float_64_64_nomask_qnt_c_nosinks_32@32x4x1",
        "sdpa_vector_2pass_1_float_64_64_nomask_qnt_nc_nosinks_32@32x8x8",
        "sdpa_vector_2pass_1_float_64_64_nomask_qnt_nc_nosinks_32",
        "sdpa_vector_2pass_1_gqa_8_float_64_64_boolmask_qnt_nc_nosinks_32@32x8x1",
        "sdpa_vector_2pass_1_gqa_4_float_128_128_nomask_qnt_nc_nosinks_32@32x4x1",
        "sdpa_vector_2pass_1_gqa_8_float_64_64_nomask_qnt_nc_nosinks_32@32x4x1",
        "sdpa_vector_2pass_2_float_72",
        "sdpa_full_self_attention",
    ],
)
def test_other_names_are_refused(entry):
    with pytest.raises(ValueError):
        sdpa_layout.signature(entry)
    assert not sdpa_dispatch.is_entry(entry)


def _vector_values(**overrides):
    values = {
        "gqa_factor": 2,
        "N": 10,
        "k_head_stride": 640,
        "k_seq_stride": 64,
        "v_head_stride": 1280,
        "v_seq_stride": 128,
        "scale": 0.125,
    }
    values.update(overrides)
    return values


def _execution(count, size):
    return {"workgroupCount": list(count), "workgroupSize": list(size)}


def test_launches_follow_the_kernels_addressing():
    constants = {**PLAIN, **sdpa_layout.MASKS["boolmask"], "has_sinks": True}
    entry = sdpa_layout.vector_entry("float32", 64, 64, constants)
    # Two batches of three query heads per KV head pair, two positions, ten
    # keys; the mask is one row per head and position.
    values = _vector_values(
        gqa_factor=3,
        mask_kv_seq_stride=1,
        mask_q_seq_stride=10,
        mask_head_stride=20,
        num_q_heads=6,
    )
    heads, length = 12, 2
    spans = {
        "queries": heads * length * 64,
        "out": heads * length * 64,
        "keys": 3 * 640 + 9 * 64 + 64,
        "values": 3 * 1280 + 9 * 128 + 64,
        "bmask": 11 * 20 + 9 + 10 + 1,
        "sinks": 6,
    }
    execution = _execution((heads, length, 1), (1024, 1, 1))
    metadata = sdpa_layout.plan_launch(entry, spans, values, spans["out"], execution)
    assert metadata["elements"] == spans and metadata["keys"] == 10
    for change in (
        {"N": 1},
        {"gqa_factor": 5},
        {"num_q_heads": 4},
        {"scale": 1},
    ):
        with pytest.raises(ValueError):
            sdpa_layout.plan_launch(
                entry, spans, {**values, **change}, spans["out"], execution
            )
    with pytest.raises(ValueError, match="addressing"):
        sdpa_layout.plan_launch(
            entry, {**spans, "keys": spans["keys"] + 1}, values, spans["out"], execution
        )
    with pytest.raises(ValueError):
        sdpa_layout.plan_launch(
            entry, spans, values, spans["out"], _execution((heads, 9, 1), (1024, 1, 1))
        )
    # A causal launch needs more than one query position.
    causal = sdpa_layout.vector_entry("float32", 64, 64, {**PLAIN, "do_causal": True})
    plain_spans = {name: spans[name] for name in ("queries", "out", "keys", "values")}
    with pytest.raises(ValueError):
        sdpa_layout.plan_launch(
            causal,
            {**plain_spans, "queries": 768, "out": 768},
            _vector_values(gqa_factor=3),
            768,
            _execution((heads, 1, 1), (1024, 1, 1)),
        )


def test_two_pass_launches_follow_the_partials_layout():
    entry = sdpa_layout.pass1_entry("float16", 128, 128, PLAIN, 64, 4, 1)
    values = _vector_values(
        k_head_stride=8192 * 128,
        k_seq_stride=128,
        v_head_stride=8192 * 128,
        v_seq_stride=128,
        N=8192,
    )
    del values["gqa_factor"]
    heads = 1 * 8 * 4
    spans = {
        "queries": heads * 128,
        "out": heads * 64 * 128,
        "sums": heads * 64,
        "maxs": heads * 64,
        "keys": 7 * 8192 * 128 + 8191 * 128 + 128,
        "values": 7 * 8192 * 128 + 8191 * 128 + 128,
    }
    execution = _execution((8, 1, 64), (32, 4, 1))
    metadata = sdpa_layout.plan_launch(entry, spans, values, spans["out"], execution)
    assert (metadata["heads"], metadata["blocks"]) == (32, 64)
    with pytest.raises(ValueError):
        sdpa_layout.plan_launch(
            entry, spans, values, spans["out"], _execution((8, 1, 32), (32, 4, 1))
        )
    with pytest.raises(ValueError):
        # The key count stays inside the range OpenGL's translation assumes.
        sdpa_layout.plan_launch(
            entry,
            spans,
            {**values, "N": sdpa_layout.MAX_KEYS + 1},
            spans["out"],
            execution,
        )
    combine = "sdpa_vector_2pass_2_float16_t_128"
    spans = {
        "partials": heads * 64 * 128,
        "sums": heads * 64,
        "maxs": heads * 64,
        "out": heads * 128,
    }
    metadata = sdpa_layout.plan_launch(
        combine,
        spans,
        {"blocks": 64},
        spans["out"],
        _execution((32, 1, 1), (1024, 1, 1)),
    )
    assert metadata["blocks"] == 64
    for blocks in (16, 48):
        with pytest.raises(ValueError):
            sdpa_layout.plan_launch(
                combine,
                spans,
                {"blocks": blocks},
                spans["out"],
                _execution((32, 1, 1), (1024, 1, 1)),
            )


def test_grouped_first_passes_read_one_query_per_head():
    entry = sdpa_layout.pass1_entry("float32", 64, 64, PLAIN, 32, 8, 1, True)
    values = _vector_values(
        N=8192, k_head_stride=8192 * 64, v_head_stride=8192 * 64, v_seq_stride=64
    )
    del values["gqa_factor"]
    heads = 2 * 4 * 8
    spans = {
        "queries": heads * 64,
        "out": heads * 32 * 64,
        "sums": heads * 32,
        "maxs": heads * 32,
        "keys": 7 * 8192 * 64 + 8191 * 64 + 64,
        "values": 7 * 8192 * 64 + 8191 * 64 + 64,
    }
    metadata = sdpa_layout.plan_launch(
        entry, spans, values, spans["out"], _execution((4, 2, 32), (32, 8, 1))
    )
    assert metadata["kind"] == "pass1_gqa" and metadata["batch"] == 2


def test_project_config_freezes_constants_and_proves_opengl_offsets(tmp_path):
    entry = sdpa_layout.pass1_entry("float32", 64, 64, PLAIN, 32, 4, 2)
    for target in ("metal", "directx", "opengl"):
        config = sdpa_packages.project_config(tmp_path, "out", target, entry)
        assert config.freeze_specialization_constants is True
        assert dict(config.specialization_constants) == {**PLAIN, "blocks": 32}
        assert config.entry_points == {
            sdpa_layout.SOURCE: ("sdpa_vector_2pass_1_float_64_64",)
        }
        assert bool(config.index_range_assertions) == (target == "opengl")
        assert bool(config.metadata_preconditions) == (target == "opengl")
    expressions = {
        record["expression"] for record in sdpa_layout.opengl_index_assertions(entry)
    }
    assert {"keys + i", "values + i"} <= expressions
    grouped = sdpa_layout.pass1_entry("float32", 64, 64, PLAIN, 32, 8, 1, True)
    expressions = {
        record["expression"] for record in sdpa_layout.opengl_index_assertions(grouped)
    }
    assert {"kp + k_seq_stride", "k_seq_stride", "vp + i"} <= expressions
    assert sdpa_layout.parameter_ranges(grouped) == {}
    combine = sdpa_packages.project_config(
        tmp_path, "out", "opengl", "sdpa_vector_2pass_2_float_64"
    )
    assert combine.freeze_specialization_constants is False
    assert not combine.index_range_assertions
    entries = sdpa_packages.verification_entries()
    assert len(entries) == len(set(entries)) == 30
    for entry in entries:
        sdpa_layout.signature(entry)


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class FakeCache:
    """Reflects every declared parameter, active or not, as its own binding."""

    def __init__(self, entry):
        self.entry = entry

    def get(self, entry):
        names = list(sdpa_layout.parameters(entry))
        for name in sdpa_dispatch.DECLARED:
            if name not in names and sdpa_layout.signature(entry)["kind"] != "pass2":
                names.append(name)
        bindings = []
        for name in names:
            if name in sdpa_layout.parameters(entry):
                dtype = sdpa_layout.parameter_type(entry, name)
            else:
                dtype = sdpa_dispatch._declared_type(entry, name)
            physical = {"bool_": "uint32"}.get(dtype, dtype)
            size = {"uint64": 8, "uint32": 4, "int32": 4, "float32": 4}[physical]
            bindings.append(
                {
                    "name": f"{name}Binding",
                    "access": (
                        "read_write" if name in sdpa_layout.outputs(entry) else "read"
                    ),
                    "kind": "uniform" if name in sdpa_layout.scalars(entry) else "",
                    "provenance": {"sourceResource": {"parameter": name}},
                    "scalarLayout": {
                        "elementType": physical,
                        "elementStrideBytes": size,
                    },
                }
            )
        return {"bindings": bindings, "artifact": {"hash": "fake"}}, "/fake"


def run(monkeypatch, tmp_path, entry, memory, execution, fault=None):
    types = {name: sdpa_layout.parameter_type(entry, name) for name in memory}
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name in sdpa_layout.outputs(entry)),
            )
            for name in memory
        )
    )
    host = SimpleNamespace(
        sdpas=FakeCache(entry),
        target="opengl",
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )
    seen = {}

    def execute(_, request):
        seen.update(request)
        outputs = {}
        for name in sdpa_layout.outputs(entry):
            binding = f"{name}Binding"
            initial = request["inputs"][binding]
            words = element_storage.unpack(types[name], "opengl", initial["values"])
            size = len(memory[name])
            words[:size] = [_bits(0.5)] * size
            if fault == name:
                words[size] ^= 1
            outputs[binding] = element_storage.packet(types[name], "opengl", words)
        return SimpleNamespace(status="ok", outputs=outputs, details={"request": {}})

    monkeypatch.setattr(
        sdpa_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
        },
    )
    monkeypatch.setattr(sdpa_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*execution["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*execution["workgroupSize"]),
    )
    threads = len(memory["out"])
    sdpa_dispatch.dispatch(host, entry, array, len(memory), threads, launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return event, host, seen


def _scalar(ctype, value):
    return (ctype * 1)(value)


def test_inactive_bindings_hold_poison_and_outputs_keep_their_guards(
    monkeypatch, tmp_path
):
    entry = sdpa_layout.pass1_entry("float32", 64, 64, PLAIN, 32, 2, 1)
    heads, blocks = 2, 32
    memory = {
        "queries": (ctypes.c_uint32 * (heads * 64))(*[_bits(1.0)] * (heads * 64)),
        "keys": (ctypes.c_uint32 * (63 * 64 + 64))(),
        "values": (ctypes.c_uint32 * (63 * 64 + 64))(),
        "out": (ctypes.c_uint32 * (heads * blocks * 64))(),
        "sums": (ctypes.c_uint32 * (heads * blocks))(),
        "maxs": (ctypes.c_uint32 * (heads * blocks))(),
        "N": _scalar(ctypes.c_int32, 64),
        "k_head_stride": _scalar(ctypes.c_uint64, 4096),
        "k_seq_stride": _scalar(ctypes.c_uint64, 64),
        "v_head_stride": _scalar(ctypes.c_uint64, 4096),
        "v_seq_stride": _scalar(ctypes.c_uint64, 64),
        "scale": _scalar(ctypes.c_uint32, _bits(0.125)),
    }
    execution = _execution((1, 1, blocks), (32, 2, 1))
    event, host, request = run(monkeypatch, tmp_path, entry, memory, execution)
    assert host.dispatch_count == 1
    inputs = request["inputs"]
    # A mask or sink the function constants turn off is never read.
    assert inputs["bmaskBinding"]["values"] == [0] * element_storage.GUARD_COUNT
    assert (
        inputs["fmaskBinding"]["values"] == [0x3FC00000] * element_storage.GUARD_COUNT
    )
    assert inputs["mask_head_strideBinding"]["values"] == [0]
    assert inputs["k_seq_strideBinding"] == {
        "dtype": "uint64",
        "shape": [1],
        "values": [64],
    }
    assert inputs["queriesBinding"]["values"][heads * 64 :] == [0x3FC00000] * 32
    assert inputs["outBinding"]["values"][0] == element_storage.GUARDS["float32"]
    assert set(request["outputs"]) == {"outBinding", "sumsBinding", "maxsBinding"}
    assert list(memory["sums"]) == [_bits(0.5)] * (heads * blocks)
    assert event["sdpaMetadata"]["blocks"] == blocks
    assert event["scalars"]["scale"] == 0.125
    assert set(event["outputHashes"]) == {"out", "sums", "maxs"}
    for fault in ("out", "maxs"):
        with pytest.raises(RuntimeError, match=f"{fault} guard"):
            run(monkeypatch, tmp_path, entry, memory, execution, fault)


def test_the_host_routes_audits_and_prepares_the_vector_attention():
    source = (HOST / "runtime.py").read_text()
    assert "sdpa_packages.SdpaPackageCache(" in source
    assert "if sdpa_dispatch.is_entry(entry):" in source
    for entry in sdpa_packages.verification_entries():
        assert lazy_package_evidence.family(entry) == "sdpa"
    port = (HOST / "sdpa.inc").read_text()
    assert "bool ScaledDotProductAttention::use_fallback(" in port
    assert "void ScaledDotProductAttention::eval_gpu(" in port
    assert '"sdpa_vector_2pass_2_"' in port and '"MLX_SDPA_BLOCKS"' in port
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    original = prepare.replace_once
    try:
        prepare.replace_once = record
        sources = prepare.adapted_sources("", "", "")
    finally:
        prepare.replace_once = original
    assert "NO_GPU_MULTI(ScaledDotProductAttention)" in replaced
    assert "NO_GPU_MULTI(ScaledDotProductAttentionVJP)" in replaced
    assert all(definition in replaced for definition in prepare.ATTENTION_DEFINITIONS)
    assert sources["crosstl_sdpa.inc"] == (HOST / "sdpa.inc").read_bytes()
    assert (
        '#include "mlx/backend/no_gpu/crosstl_sdpa.inc"' in (
            HOST / "backend.cpp"
        ).read_text()
    )
