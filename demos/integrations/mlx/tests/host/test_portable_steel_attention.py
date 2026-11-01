"""The portable host's port of scaled_dot_product_attention.cpp's full attention."""

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
    runtime,
    steel_attention_dispatch,
    steel_attention_layout,
    steel_attention_packages,
    upstream_matrix_packages,
)

HOST = Path(steel_attention_layout.__file__).parent
NONE = dict.fromkeys(steel_attention_layout.FLAGS, False)
ALL = dict.fromkeys(steel_attention_layout.FLAGS, True)
# params.h's AttnParams and AttnMaskParams, as OpenGL reflects them.
PARAMS_LAYOUT = storage_record_layout(
    "AttnParams",
    [
        ("float" if name == "scale" else "int", name)
        for name in steel_attention_layout.PARAMS_FIELDS
    ]
    + [("int64_t", name, 3) for name in steel_attention_layout.STRIDE_FIELDS],
    storage_layout="std430",
)
MASK_LAYOUT = storage_record_layout(
    "AttnMaskParams", [("int64_t", "M_strides", 3)], storage_layout="std430"
)


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def params(B, H, qL, kL, D=64, V=64, bk=32, gqa=1, scale=0.125, q=None, k=None):
    """sdpa_full_self_attention_metal's AttnParams words for dense operands."""
    q = q or [H * qL * D, qL * D, D]
    k = k or [H // gqa * kL * D, kL * D, D]
    v = [H // gqa * kL * V, kL * V, V]
    o = [qL * H * V, V, H * V]
    nq, nk = -(-qL // 32), -(-kL // bk)
    head = [B, H, D, qL, kL, gqa, _bits(scale), nq, nk, qL // 32, kL // bk]
    head += [qL - qL // 32 * 32, kL - kL // bk * bk, kL - qL]
    return head + q + k + v + o


def test_hash_names_freeze_the_constants_and_threadgroups():
    entry = steel_attention_layout.entry_name("float16", 128, 128, "bool_", ALL)
    assert entry == (
        "steel_attention_float16_bq32_bk16_bd128_bv128_wm4_wn1_maskbool_"
        "_align_Q_t_align_K_t_has_mask_t_do_causal_t_has_sinks_t_save_lse_t"
    )
    info = steel_attention_layout.signature(entry)
    assert info["source"] == (
        "steel_attention_float16_bq32_bk16_bd128_bv128_wm4_wn1_maskbool_"
    )
    assert info["constants"] == ALL and info["workgroupSize"] == (32, 4, 1)
    assert steel_attention_layout.parameters(entry) == (
        "Q",
        "K",
        "V",
        "O",
        "params",
        "mask_params",
        "mask",
        "sinks",
        "lse",
    )
    assert steel_attention_layout.outputs(entry) == ("O", "lse")
    assert steel_attention_layout.parameter_type(entry, "mask") == "bool_"
    assert steel_attention_layout.parameter_type(entry, "lse") == "float32"
    # Without a mask the hash name spells the query type.
    plain = steel_attention_layout.entry_name("float32", 64, 64, "bool_", NONE)
    assert "_maskfloat32_" in plain
    assert steel_attention_layout.parameters(plain) == ("Q", "K", "V", "O", "params")
    # The host has no 'd' architecture: 32-row key blocks below head dim 128.
    assert steel_attention_layout.tile(96) == (32, 1)
    assert steel_attention_layout.tile(128) == (16, 1)
    assert steel_attention_layout.tile(256) == (16, 2)
    wide = steel_attention_layout.entry_name("float32", 256, 256, "float32", NONE)
    assert steel_attention_layout.workgroup_size(wide) == (32, 4, 2)
    for entry in steel_attention_packages.verification_entries():
        assert lazy_package_evidence.family(entry) == "steel-attention"


@pytest.mark.parametrize(
    "entry",
    [
        # No 32-row key blocks at head dim 128.
        "steel_attention_float32_bq32_bk32_bd128_bv128_wm4_wn1_maskfloat32"
        "_align_Q_t_align_K_t_has_mask_n_do_causal_n_has_sinks_n_save_lse_n",
        # A mask type other than the query's or Boolean.
        "steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskfloat16"
        "_align_Q_t_align_K_t_has_mask_t_do_causal_n_has_sinks_n_save_lse_n",
        # A Boolean mask name without has_mask.
        "steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskbool_"
        "_align_Q_t_align_K_t_has_mask_n_do_causal_n_has_sinks_n_save_lse_n",
        # Head dim 256 splits along WN = 2.
        "steel_attention_float32_bq32_bk16_bd256_bv256_wm4_wn1_maskfloat32"
        "_align_Q_t_align_K_t_has_mask_n_do_causal_n_has_sinks_n_save_lse_n",
        "steel_attention_float32_bq32_bk032_bd64_bv64_wm4_wn1_maskfloat32"
        "_align_Q_t_align_K_t_has_mask_n_do_causal_n_has_sinks_n_save_lse_n",
        "steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskfloat32",
    ],
)
def test_other_names_are_refused(entry):
    with pytest.raises(ValueError):
        steel_attention_layout.signature(entry)
    assert not steel_attention_dispatch.is_entry(entry)


def _execution(groups, size):
    return {"workgroupCount": list(groups), "workgroupSize": list(size)}


def test_launches_follow_the_kernels_addressing():
    flags = {**NONE, "align_Q": True, "has_mask": True, "has_sinks": True}
    flags["save_lse"] = True
    entry = steel_attention_layout.entry_name("float32", 64, 64, "bool_", flags)
    B, H, qL, kL = 2, 4, 64, 40
    words = {"params": params(B, H, qL, kL), "mask_params": [0, qL * kL, kL]}
    counts = {
        "Q": B * H * qL * 64,
        "K": B * H * kL * 64,
        "V": B * H * kL * 64,
        "O": B * H * qL * 64,
        "mask": (H - 1) * qL * kL + (qL - 1) * kL + kL,
        "sinks": H,
        "lse": B * H * qL,
    }
    execution = _execution((2, H, B), (32, 4, 1))
    plan = steel_attention_layout.plan_launch(
        entry, counts, words, counts["O"], execution
    )
    assert plan["elements"] == counts and plan["NQ"] == 2 and plan["NK"] == 2
    bad = []
    for change in (
        # A query block count that differs from ceil(qL / 32).
        lambda w, c, e: w["params"].__setitem__(7, 3),
        # The alignment the hash name freezes.
        lambda w, c, e: w["params"].__setitem__(3, 63),
        lambda w, c, e: e.__setitem__("workgroupCount", [2, H, 1]),
        lambda w, c, e: e.__setitem__("workgroupSize", [32, 4, 2]),
        # The output must be the (B, L, H, V) storage the host allocates.
        lambda w, c, e: w["params"].__setitem__(len(w["params"]) - 1, 64),
        lambda w, c, e: c.__setitem__("mask", c["mask"] + 1),
        lambda w, c, e: c.__setitem__("Q", c["Q"] - 1),
        lambda w, c, e: w["mask_params"].__setitem__(0, -1),
        lambda w, c, e: w["params"].__setitem__(6, 1 << 32),
        lambda w, c, e: w.pop("mask_params"),
    ):
        changed_words = {name: list(value) for name, value in words.items()}
        changed_counts, changed_execution = dict(counts), dict(execution)
        change(changed_words, changed_counts, changed_execution)
        with pytest.raises(ValueError):
            steel_attention_layout.plan_launch(
                entry, changed_counts, changed_words, counts["O"], changed_execution
            )
        bad.append(True)
    assert len(bad) == 10
    # A causal launch needs no more queries than keys.
    causal = steel_attention_layout.entry_name(
        "float32", 64, 64, "float32", {**NONE, "do_causal": True}
    )
    with pytest.raises(ValueError):
        steel_attention_layout.plan_launch(
            causal,
            {"Q": 33 * 64, "K": 20 * 64, "V": 20 * 64, "O": 33 * 64},
            {"params": params(1, 1, 33, 20)},
            33 * 64,
            _execution((2, 1, 1), (32, 4, 1)),
        )


def test_threadgroup_extents_bound_the_opengl_offsets():
    def extents(dtype, bd, bv=None):
        return steel_attention_layout.threadgroup_extents(
            steel_attention_layout.entry_name(dtype, bd, bv or bd, dtype, NONE)
        )

    # steel_attention.h pads each threadgroup row by 16 bytes.
    assert extents("float32", 64) == {"Q_smem": 32 * 68, "KV_smem": 36 * 64}
    assert extents("float16", 128) == {"Q_smem": 32 * 136, "KV_smem": 24 * 128}
    assert extents("bfloat16", 96, 64) == {"Q_smem": 32 * 104, "KV_smem": 40 * 96}
    # Four-byte queries stay in registers at head dim 256.
    assert extents("float32", 256) == {"Q_smem": 1, "KV_smem": 20 * 256}
    assert extents("float16", 256) == {"Q_smem": 32 * 264, "KV_smem": 24 * 256}
    entry = steel_attention_layout.entry_name("float32", 64, 64, "float32", NONE)
    assertions = steel_attention_layout.opengl_index_assertions(entry)
    assert [item.get("indexed") for item in assertions] == [
        "crosstl_workgroup_KV_smem",
        "crosstl_workgroup_Q_smem",
        None,
    ]
    assert [item["maximum"] for item in assertions] == [2303, 2175, 2**31 - 1]


def test_project_config_freezes_constants_and_lowers_the_matrices(tmp_path):
    entry = steel_attention_packages.verification_entries()[-3]
    for target in ("metal", "directx", "opengl"):
        config = steel_attention_packages.project_config(tmp_path, "out", target, entry)
        assert config.freeze_specialization_constants is True
        assert dict(config.specialization_constants) == ALL
        source = steel_attention_layout.source_entry(entry)
        assert config.entry_points == {steel_attention_layout.SOURCE: (source,)}
        assert bool(config.index_range_assertions) == (target == "opengl")
        metal = config.source_options["metal"]
        assert metal["cooperative_matrix_fragment_mapping"] == "tile_4x4_row_pair"
        for name in ("directx", "opengl"):
            options = metal["target_options"][name]
            assert options["cooperative_matrix_software_lowering"] is True
            assert options["software_subgroup_width"] == 32
    assert steel_attention_packages.CACHE_DIRECTORY == "steel-attention"
    assert "steel-attention" in upstream_matrix_packages.ORDER


class FakeCache:
    def __init__(self, entry):
        self.entry = entry

    def get(self, entry):
        assert entry == self.entry
        bindings = []
        for name in ("Q", "K", "V", "O", "params", "mask_params", "mask", "sinks"):
            bindings.append(self._binding(entry, name))
        bindings.append(self._binding(entry, "lse"))
        return {"bindings": bindings, "artifact": {"hash": "fake"}}, "/fake"

    @staticmethod
    def _binding(entry, name):
        info = steel_attention_layout.signature(entry)
        if name == "params":
            scalar = PARAMS_LAYOUT
        elif name == "mask_params":
            scalar = MASK_LAYOUT
        else:
            dtype = {"mask": info["maskType"], "lse": "float32"}.get(
                name, info["dtype"]
            )
            scalar = {
                "elementType": {"bool_": "bool"}.get(dtype, dtype),
                "elementStrideBytes": 4,
            }
        return {
            "name": f"{name}Binding",
            "access": "read_write" if name in ("O", "lse") else "read",
            "kind": "buffer",
            "provenance": {"sourceResource": {"parameter": name}},
            "scalarLayout": scalar,
        }


def run(monkeypatch, tmp_path, entry, memory, execution, fault=None):
    types = {
        name: steel_attention_layout.parameter_type(entry, name) for name in memory
    }
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name in steel_attention_layout.outputs(entry)),
            )
            for name in memory
        )
    )
    host = SimpleNamespace(
        steel_attentions=FakeCache(entry),
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
            dtype = types.get(name, "float32")
            initial = request["inputs"][binding]
            words = element_storage.unpack(dtype, "opengl", initial["values"])
            size = len(memory[name]) if name in memory else 0
            words[:size] = [_bits(0.5)] * size
            if fault == name:
                words[size] ^= 1
            outputs[binding] = element_storage.packet(dtype, "opengl", words)
        return SimpleNamespace(status="ok", outputs=outputs, details={"request": {}})

    monkeypatch.setattr(
        steel_attention_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
        },
    )
    monkeypatch.setattr(steel_attention_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*execution["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*execution["workgroupSize"]),
    )
    threads = len(memory["O"])
    steel_attention_dispatch.dispatch(host, entry, array, len(memory), threads, launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return event, host, seen


def test_records_pack_and_inactive_bindings_hold_poison(monkeypatch, tmp_path):
    flags = {**NONE, "align_Q": True, "align_K": True, "save_lse": True}
    entry = steel_attention_layout.entry_name("float32", 64, 64, "float32", flags)
    B, H, qL, kL = 1, 2, 32, 32
    words = params(B, H, qL, kL)
    memory = {
        "Q": (ctypes.c_uint32 * (H * qL * 64))(*[_bits(1.0)] * (H * qL * 64)),
        "K": (ctypes.c_uint32 * (H * kL * 64))(),
        "V": (ctypes.c_uint32 * (H * kL * 64))(),
        "O": (ctypes.c_uint32 * (H * qL * 64))(),
        "params": (ctypes.c_int64 * len(words))(*words),
        "lse": (ctypes.c_uint32 * (H * qL))(),
    }
    execution = _execution((1, H, B), (32, 4, 1))
    event, host, request = run(monkeypatch, tmp_path, entry, memory, execution)
    assert host.dispatch_count == 1
    inputs = request["inputs"]
    record = inputs["paramsBinding"]
    assert record["dtype"] == "uint32" and len(record["values"]) == 152 // 4
    # B, H, D, qL and kL, the scale's bits, then the int64 query strides.
    assert record["values"][:5] == [B, H, 64, qL, kL]
    assert record["values"][6] == _bits(0.125)
    assert record["values"][14:16] == [H * qL * 64, 0]
    # The mask record, mask and sinks the function constants turn off are
    # never read.
    assert inputs["mask_paramsBinding"]["values"] == [0] * 6
    poison = [0x3FC00000] * element_storage.GUARD_COUNT
    assert inputs["maskBinding"]["values"] == poison
    assert inputs["sinksBinding"]["values"] == poison
    assert inputs["QBinding"]["values"][H * qL * 64 :] == poison
    assert inputs["OBinding"]["values"][0] == element_storage.GUARDS["float32"]
    assert set(request["outputs"]) == {"OBinding", "lseBinding"}
    assert list(memory["lse"]) == [_bits(0.5)] * (H * qL)
    assert event["steelAttentionMetadata"]["NQ"] == 1
    assert event["records"]["params"] == words
    assert set(event["outputHashes"]) == {"O", "lse"}
    for fault in ("O", "lse"):
        with pytest.raises(RuntimeError, match=f"{fault} guard"):
            run(monkeypatch, tmp_path, entry, memory, execution, fault)
    # An output that overlaps an input is refused before any launch.
    memory["lse"] = (ctypes.c_uint32 * (H * qL)).from_address(
        ctypes.addressof(memory["Q"])
    )
    with pytest.raises(ValueError, match="overlaps"):
        run(monkeypatch, tmp_path, entry, memory, execution)


def test_an_unbound_logsumexp_is_requested_as_guards_and_must_stay_so(
    monkeypatch, tmp_path
):
    # The reflected lse binding stays writable without save_lse, and the
    # loader requires every writable binding as an output.
    flags = {**NONE, "align_Q": True, "align_K": True}
    entry = steel_attention_layout.entry_name("float32", 64, 64, "float32", flags)
    B, H, qL, kL = 1, 2, 32, 32
    words = params(B, H, qL, kL)
    memory = {
        "Q": (ctypes.c_uint32 * (H * qL * 64))(),
        "K": (ctypes.c_uint32 * (H * kL * 64))(),
        "V": (ctypes.c_uint32 * (H * kL * 64))(),
        "O": (ctypes.c_uint32 * (H * qL * 64))(),
        "params": (ctypes.c_int64 * len(words))(*words),
    }
    execution = _execution((1, H, B), (32, 4, 1))
    event, host, request = run(monkeypatch, tmp_path, entry, memory, execution)
    assert host.dispatch_count == 1
    assert set(request["outputs"]) == {"OBinding", "lseBinding"}
    guards = [element_storage.GUARDS["float32"]] * element_storage.GUARD_COUNT
    assert request["inputs"]["lseBinding"]["values"] == guards
    assert set(event["outputHashes"]) == {"O"}
    with pytest.raises(RuntimeError, match="unbound lse"):
        run(monkeypatch, tmp_path, entry, memory, execution, "lse")


def test_the_host_routes_and_prepares_the_full_attention():
    source = (HOST / "runtime.py").read_text()
    assert "steel_attention_packages.SteelAttentionPackageCache(" in source
    assert "if steel_attention_dispatch.is_entry(entry):" in source
    port = (HOST / "sdpa.inc").read_text()
    assert "void sdpa_full_attention_host(" in port
    assert '"steel_attention_"' in port and '"_align_Q_"' in port
    # The host allocates the output as scaled_dot_product_attention.cpp does
    # and refuses the blocked head-dim-512 attention.
    assert "{str_oB, str_oH, str_oL, str_oD}," in port
    assert 'blocked "\n          "head-dim-512 attention yet.' in port
    assert 'full-attention "\n        "kernels yet.' not in port
