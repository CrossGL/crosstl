"""GatherQMM's gather_qmm_rhs: sorted single-row products and their contracts."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import gather_qmm_rhs_layout as layout
from demos.integrations.mlx.portable_host import (
    gather_qmm_rhs_packages,
    lazy_package_evidence,
)
from demos.integrations.mlx.portable_host import matmul_dispatch as dispatch
from demos.integrations.mlx.portable_host import runtime
from demos.integrations.mlx.tests.host.test_portable_fp_quantized_matmul import (
    CTYPES,
    physical_layout,
)

HOST = Path(layout.__file__).parent


def sorted_launch(
    mode="affine",
    *,
    transpose=True,
    m=64,
    n=40,
    k=512,
    groups=4,
    offsets=(0, 16, 16, 40),
    global_scales=None,
    route=b"gather_qmm",
    dtype="float32",
):
    """gather_qmm.inc's binding of one gather_qmm_rhs launch."""
    group, bits = (64, 4) if mode == "affine" else layout.MODES[mode]
    affine = mode == "affine"
    words, scales = n * k * bits // 32, n * k // group
    data = {
        "w": ("uint32", [0] * (groups * words), 0),
        "scales": (dtype if affine else "uint8", [0] * (groups * scales), 0),
        "x": (dtype, [0] * (m * k), 0),
        "y": (dtype, [0] * (m * n), 1),
        "M": ("int32", [m], 0),
        "N": ("int32", [n], 0),
        "K": ("int32", [k], 0),
        "num_groups": ("int32", [groups], 0),
        "offsets": ("int32", list(offsets), 0),
    }
    if affine:
        data["biases"] = (dtype, [0] * (groups * scales), 0)
    if global_scales is not None:
        data["global_scale"] = ("float32", [0x3F800000] * global_scales, 0)
    entry = layout.select(
        mode, dtype, n, k, global_scales is not None, transpose, group, bits
    )
    memory, buffers = {}, {}
    for name, (dtype, values, output) in data.items():
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers[name] = runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            output,
        )
    if route is not None:
        memory["route"] = ctypes.create_string_buffer(route)
        buffers["route"] = runtime.Buffer(
            b"route", b"utf8", ctypes.addressof(memory["route"]), len(route), 0
        )
    execution = {
        "workgroupCount": [-(-n // 32), min(m, -(-m // 16) + groups - 1), 1],
        "workgroupSize": [32, 2, 1],
    }
    return entry, buffers, memory, execution


def planned(entry, buffers, execution):
    buffers = dict(buffers)
    route = dispatch._route(layout, buffers.pop("route", None), layout.LABEL)
    counts, parameters = layout.read_storage(entry, buffers)
    return layout.plan(entry, counts, parameters, counts["y"], execution, route)


@pytest.mark.parametrize(
    "entry,info",
    (
        (
            "affine_gather_qmm_rhs_nt_float_gs_64_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
            "_align_N_t_align_K_t",
            ("affine", True, "float32", 64, 4, False, True, True),
        ),
        (
            "nvfp4_gather_qmm_rhs_nn_float16_t_gs_16_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
            "_hgs_align_N_n_align_K_t",
            ("nvfp4", False, "float16", 16, 4, True, False, True),
        ),
        (
            "mxfp8_gather_qmm_rhs_nt_bfloat16_t_gs_32_b_8_bm_16_bn_32_bk_32_wm_1"
            "_wn_2_align_N_t_align_K_n",
            ("mxfp8", True, "bfloat16", 32, 8, False, True, False),
        ),
    ),
)
def test_signature_reads_the_instantiated_rhs_kernels_and_their_constants(entry, info):
    keys = ("mode", "transpose", "dtype", "groupSize", "bits", "globalScale")
    signature = layout.signature(entry)
    assert {key: signature[key] for key in keys} == dict(zip(keys, info[:6]))
    assert signature["constants"] == dict(zip(layout.CONSTANTS, info[6:]))
    base = entry[: entry.index("_align_N_")]
    assert layout.source_entry(entry) == signature["source"] == base
    assert layout.workgroup_size(entry) == (32, 2, 1)
    assert layout.source(entry) == (
        layout.AFFINE_SOURCE if info[0] == "affine" else layout.SOURCE
    )
    assert entry.startswith(layout.ENTRY_PREFIXES)
    assert lazy_package_evidence.family(entry) == "gather-quantized-mm-rhs"


@pytest.mark.parametrize(
    "entry",
    (
        # Only nvfp4 products carry global scales.
        "mxfp4_gather_qmm_rhs_nt_float_gs_32_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
        "_hgs_align_N_t_align_K_t",
        # quantized.cpp's hash name always names both constants.
        "affine_gather_qmm_rhs_nt_float_gs_64_b_4_bm_16_bn_32_bk_32_wm_1_wn_2",
        # gather_qmm_rhs runs one tile.
        "affine_gather_qmm_rhs_nt_float_gs_64_b_4_bm_32_bn_32_bk_32_wm_2_wn_2"
        "_align_N_t_align_K_t",
        "affine_gather_qmm_rhs_nt_float_gs_16_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
        "_align_N_t_align_K_t",
        "mxfp8_gather_qmm_rhs_nt_float_gs_32_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
        "_align_N_t_align_K_t",
    ),
)
def test_signature_refuses_entries_gather_qmm_rhs_never_routes(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_selection_follows_quantized_cpp_gather_qmm_rhs():
    assert layout.select("affine", "float16", 555, 512, False, True, 64, 4) == (
        "affine_gather_qmm_rhs_nt_float16_t_gs_64_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
        "_align_N_n_align_K_t"
    )
    assert layout.select("nvfp4", "bfloat16", 1040, 512, True, False) == (
        "nvfp4_gather_qmm_rhs_nn_bfloat16_t_gs_16_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
        "_hgs_align_N_n_align_K_t"
    )
    with pytest.raises(ValueError, match="group and bits"):
        layout.select("affine", "float32", 32, 256)
    # GatherQMM's test: one row, 16 batches, four per weight matrix.
    assert layout.routed(1, 16, 4, True)
    assert not layout.routed(1, 16, 5, True)
    assert not layout.routed(1, 15, 1, True)
    assert not layout.routed(2, 64, 4, True)
    assert not layout.routed(1, 64, 4, False)
    assert layout.row_tiles(64, 4) == 7 and layout.row_tiles(16, 4) == 4
    entries = gather_qmm_rhs_packages.verification_entries()
    assert entries == sorted(entries) and len(entries) == 17
    for entry in entries:
        layout.signature(entry)


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"transpose": False, "n": 512, "k": 544},
        {"mode": "mxfp4", "n": 64, "k": 64},
        {"mode": "nvfp4", "global_scales": 4, "n": 128, "k": 1040},
    ),
)
def test_plan_reads_the_group_offsets_and_bounds_their_matrices(kwargs):
    entry, buffers, _memory, execution = sorted_launch(**kwargs)
    metadata = planned(entry, buffers, execution)
    assert metadata["route"] == "gather_qmm"
    assert metadata["kernel"] == (
        "rhs_nt" if kwargs.get("transpose", True) else "rhs_nn"
    )
    # The second matrix owns no rows; the first, third and fourth do.
    assert metadata["owners"] == [0, 2, 3]
    values = metadata["parameters"]
    assert values["offsets"] == [0, 16, 16, 40]
    assert (values["M"], values["num_groups"]) == (64, 4)
    affine = kwargs.get("mode", "affine") == "affine"
    scaled = "global_scales" in kwargs
    assert ("global_scale" in values) == (not affine and not scaled)


@pytest.mark.parametrize(
    "kwargs,accepted",
    (
        ({"dtype": "bfloat16", "global_scales": 4}, True),
        # GatherQQMM's matrix kernels read the weights' global scales.
        ({"dtype": "bfloat16"}, False),
        ({"dtype": "float16", "global_scales": 4}, False),
        ({"dtype": "bfloat16", "global_scales": 4, "transpose": False}, False),
        ({"dtype": "bfloat16", "global_scales": 4, "k": 1040}, False),
    ),
)
def test_plan_holds_gather_qqmm_launches_to_its_routing(kwargs, accepted):
    entry, buffers, _memory, execution = sorted_launch(
        "nvfp4", n=128, route=b"gather_qqmm", **kwargs
    )
    if not accepted:
        with pytest.raises(ValueError, match="GatherQQMM"):
            planned(entry, buffers, execution)
        return
    metadata = planned(entry, buffers, execution)
    assert (metadata["route"], metadata["kernel"]) == ("gather_qqmm", "rhs_nt")


@pytest.mark.parametrize(
    "fault",
    (
        "unsorted",
        "past-rows",
        "offset-count",
        "weights",
        "biases",
        "global-scale",
        "grid",
        "entry",
        "rows-per-matrix",
        "route",
        "k-groups",
        "input",
    ),
)
def test_plan_refuses_launches_gather_qmm_rhs_never_makes(fault):
    kwargs = {"mode": "nvfp4", "global_scales": 4} if fault == "global-scale" else {}
    entry, buffers, memory, execution = sorted_launch(**kwargs)

    def shrink(name, count):
        buffers[name] = runtime.Buffer(
            name.encode(), buffers[name].dtype, buffers[name].data, count, 0
        )

    if fault == "unsorted":
        memory["offsets"][2] = 8
    elif fault == "past-rows":
        memory["offsets"][3] = 65
    elif fault == "offset-count":
        shrink("offsets", 3)
    elif fault == "weights":
        # The fourth matrix owns rows but its weights reach past the buffer.
        shrink("w", buffers["w"].count - 1)
    elif fault == "biases":
        shrink("biases", buffers["biases"].count - 1)
    elif fault == "global-scale":
        shrink("global_scale", 3)
    elif fault == "grid":
        execution["workgroupCount"][1] = 4
    elif fault == "entry":
        entry = entry.replace("_align_N_n_", "_align_N_t_")
    elif fault == "rows-per-matrix":
        # 64 rows over 17 matrices leave fewer than four per matrix.
        memory["num_groups"][0] = 17
        execution["workgroupCount"][1] = 20
    elif fault == "route":
        memory["route"] = ctypes.create_string_buffer(b"gather_qqmm")
        buffers["route"] = runtime.Buffer(
            b"route", b"utf8", ctypes.addressof(memory["route"]), 11, 0
        )
    elif fault == "k-groups":
        memory["K"][0] = 500
    elif fault == "input":
        shrink("x", 64 * 512 - 1)
    with pytest.raises(ValueError):
        planned(entry, buffers, execution)


def test_dispatch_binds_the_offsets_and_commits_after_validation(tmp_path, monkeypatch):
    entry, buffers, memory, execution = sorted_launch("mxfp4", n=64, k=64)
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in (*layout.parameter_names(entry), "global_scale"):
        kind = (
            "float32"
            if parameter == "global_scale"
            else layout.parameter_type(entry, parameter)
        )
        descriptor["bindings"].append(
            {
                "name": f"binding_{parameter}",
                "kind": "buffer",
                "access": "read_write" if parameter == "y" else "read",
                "provenance": {"sourceResource": {"parameter": parameter}},
                "scalarLayout": physical_layout("metal", kind),
            }
        )
    cache = SimpleNamespace(get=lambda name: (descriptor, tmp_path))
    host = SimpleNamespace(
        target="metal", trace=tmp_path / "trace.jsonl", dispatch_count=0
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append(inputs)
        return outputs

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        count = len(memory["y"])
        outputs["binding_y"]["values"][:count] = [0x40000000] * count
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    monkeypatch.setattr(dispatch, "execute", execute)
    supplied = (runtime.Buffer * len(buffers))(*buffers.values())
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    dispatch.dispatch_with(
        host, layout, cache, entry, supplied, len(buffers), len(memory["y"]), launch
    )
    assert list(memory["y"]) == [0x40000000] * len(memory["y"])
    ((inputs,),) = [requests]
    assert inputs["binding_offsets"]["values"] == [0, 16, 16, 40]
    assert inputs["binding_global_scale"]["values"] == [layout.INACTIVE_GLOBAL_SCALE]
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["matmulMetadata"]["owners"] == [0, 2, 3]
    assert set(event["inputHashes"]) == {"w", "scales", "x"}


def test_package_configs_freeze_the_alignment_constants(tmp_path):
    entry = layout.select("nvfp4", "float16", 1040, 512, True, False)
    for target in ("metal", "directx", "opengl"):
        config = gather_qmm_rhs_packages.project_config(tmp_path, "out", target, entry)
        base = layout.source_entry(entry)
        assert list(config.include_patterns) == [layout.SOURCE]
        assert config.entry_points == {layout.SOURCE: (base,)}
        assert dict(config.specialization_constants) == {
            "align_N": False,
            "align_K": True,
        }
        assert config.freeze_specialization_constants
        rules = config.entry_workgroup_size_rules[layout.SOURCE]
        assert [int(size) for size in rules[base]] == [32, 2, 1]
        assert [item.parameter for item in config.metadata_preconditions] == ["K"]
        loaders = {item.function for item in config.workgroup_access_assertions}
        assert loaders == (
            set()
            if target == "metal"
            else {"BlockLoader_float16_t_16_32_40_1_64__load_unsafe"}
        )
    cache = gather_qmm_rhs_packages.GatherQuantizedQmmRhsPackageCache(
        tmp_path, tmp_path / "cache", "metal"
    )
    assert cache.source_entry(entry) == layout.source_entry(entry)
    affine = layout.select("affine", "float32", 64, 512, False, True, 64, 4)
    assert cache.source_path(affine) == layout.AFFINE_SOURCE
    (assertion,) = layout.access_assertions(affine)
    assert (assertion["minimum"], assertion["maximum"]) == (0, 15 * 36 + 31)
