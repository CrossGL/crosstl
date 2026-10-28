"""conv.cpp's general implicit GEMM route on the portable host."""

import copy
import ctypes
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    conv_dispatch,
    conv_general_layout,
    conv_layout,
    conv_packages,
    runtime,
)
from demos.integrations.mlx.portable_host import verify_conv as verify
from demos.integrations.mlx.tests.kernels import steel_conv_general_model as model
from demos.integrations.mlx.tests.kernels import test_current_steel_conv_general as gate
from demos.integrations.mlx.tests.kernels.steel_conv_model import Conv
from demos.integrations.mlx.tests.kernels.test_current_conv import _params_bytes

general = conv_general_layout
layout = conv_layout
HOST = Path(layout.__file__).parent
RECORDS = {
    record["name"]: record
    for record in (verify.case_record(case) for case in verify.CASES)
    if record["route"] == general.ROUTE
}
ENTRY = "implicit_gemm_conv_2d_general_float32_bm32_bn32_bk16_wm2_wn2_alC_0"


def _launch(record):
    """The steel_conv_general launch conv.inc makes for ``record``."""
    (entry,) = record["convEntries"]
    p = copy.deepcopy(record["params"])
    gemm = general.gemm_params(p, layout.signature(entry)["model"])
    jump = general.jump_params(p)
    counts = {
        "A": math.prod(record["inShape"]),
        "B": math.prod(record["wtShape"]),
        "C": math.prod(record["outShape"]),
    }
    execution = {
        "workgroupCount": [
            gemm["tiles_n"],
            gemm["tiles_m"],
            jump["f_out_jump_h"] * jump["f_out_jump_w"],
        ],
        "workgroupSize": [32, 2, 2],
    }
    parameters = {
        "params": p,
        "gemm_params": gemm,
        "jump_params": jump,
        "base_h": general.base_info(p, 0),
        "base_w": general.base_info(p, 1),
    }
    return entry, counts, parameters, counts["C"], execution


def _model_cases():
    """The kernel gate's one-group convolutions, both alignments."""
    return [
        conv
        for entry in model.entries()
        if entry.dtype == "float32" and entry.bm == 32 and entry.bn in (32, 64)
        for aligned in (True, False)
        for conv, _ in gate._cases(entry, aligned)
    ] + [
        Conv(2, (16, 16), (3, 3), 21, 21),
        Conv(1, (64, 64), (3, 3), 64, 64, padding=(2, 2), input_dilation=(2, 2)),
        Conv(1, (9, 7), (4, 3), 16, 80, stride=(3, 2), padding=(3, 1), flip=True),
    ]


def test_cases_reach_the_general_route_with_both_alignments():
    assert set(RECORDS) == {
        "general-2d-strided-unaligned",
        "general-2d-transpose",
        "general-2d-dilated-copied",
    }
    infos = [layout.signature(record["launches"][0]) for record in RECORDS.values()]
    assert {info["constants"]["align_C"] for info in infos} == {False, True}
    assert {info["dtype"] for info in infos} == set(layout.DTYPES)
    assert general.ROUTE in layout.PORTED_ROUTES
    # A residue whose rows never land on an input pixel keeps no tap.
    dilated = RECORDS["general-2d-dilated-copied"]["params"]
    assert general.base_info(dilated, 0)[0] == {"weight_base": 2, "weight_size": 0}
    assert RECORDS["general-2d-dilated-copied"]["copies"] == 1


@pytest.mark.parametrize("conv", _model_cases(), ids=str)
def test_parameters_match_the_kernel_model(conv):
    p = conv.params()
    host = model.host_entry(conv, "float32")
    name = general.entry(p, "float32")
    assert name == f"{host.name}_alC_{int(model.align_c(conv))}"
    info = general.signature(name)
    assert info["model"] == host and info["source"] == host.name
    launch = model.launch(conv, host)
    assert general.gemm_params(p, host) == launch.gemm
    assert general.jump_params(p) == launch.jump
    assert tuple(general.base_info(p, 0)) == launch.base_h
    assert tuple(general.base_info(p, 1)) == launch.base_w
    assert info["constants"]["align_C"] == launch.align_c


def test_large_dilated_outputs_select_64_wide_tiles():
    p = Conv(
        1, (64, 64), (3, 3), 64, 64, padding=(2, 2), input_dilation=(2, 2)
    ).params()
    p["N"] = 2
    assert general.jump_params(p)["adj_implicit_m"] >= 8192
    assert general.entry(p, "float16").endswith("_bm64_bn64_bk16_wm2_wn2_alC_1")
    p["O"] = 48
    assert general.entry(p, "float16").endswith("_bm64_bn32_bk16_wm2_wn2_alC_1")
    p["C"] = 40
    assert general.entry(p, "bfloat16").endswith("_bm32_bn32_bk16_wm2_wn2_alC_0")


def test_signature_names_conv_cpp_pipelines_only():
    for dtype in layout.DTYPES:
        for bm in (32, 64):
            for bn in (32, 64):
                for align in (0, 1):
                    name = (
                        f"implicit_gemm_conv_2d_general_{dtype}_bm{bm}_bn{bn}"
                        f"_bk16_wm2_wn2_alC_{align}"
                    )
                    info = layout.signature(name)
                    assert info["kind"] == "general" and info["dtype"] == dtype
                    assert layout.source(name) == model.SOURCE
                    assert layout.source_entry(name) == name[: -len("_alC_0")]
                    assert layout.workgroup_size(name) == (32, 2, 2)
                    assert layout.host_parameters(name) == general.PARAMETERS
                    assert layout.output(name) == "C"
    for name in (
        "implicit_gemm_conv_2d_general_float32_bm32_bn32_bk16_wm2_wn2",
        "implicit_gemm_conv_2d_general_float32_bm32_bn32_bk16_wm2_wn2_alC_2",
        # conv.cpp never launches the eight-column tiles.
        "implicit_gemm_conv_2d_general_float32_bm32_bn8_bk16_wm4_wn1_alC_1",
        "implicit_gemm_conv_2d_general_int32_bm32_bn32_bk16_wm2_wn2_alC_1",
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            layout.signature(name)


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_plan_accepts_conv_cpp_launches(name):
    entry, counts, parameters, threads, execution = _launch(RECORDS[name])
    metadata = layout.plan(entry, counts, parameters, threads, execution)
    assert metadata["route"] == general.ROUTE and metadata["weightSlack"] == 0
    assert metadata["alignC"] == layout.signature(entry)["constants"]["align_C"]


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (lambda c, p, e: p["params"].update(idil=[1, 1], str=[1, 1]), "routing"),
        (lambda c, p, e: p["params"].update(groups=2), "invalid"),
        (lambda c, p, e: p["params"].update(in_strides=[0, 0, 0, 1]), "strides"),
        (lambda c, p, e: p["jump_params"].update(adj_out_w=1), "parameters differ"),
        (lambda c, p, e: p["base_w"][0].update(weight_size=9), "parameters differ"),
        (lambda c, p, e: p["base_h"].append(p["base_h"][0]), "parameters differ"),
        (lambda c, p, e: p["gemm_params"].update(swizzle_log=1), "parameters differ"),
        (lambda c, p, e: e["workgroupCount"].__setitem__(2, 1), "threadgroups"),
        (lambda c, p, e: e.update(workgroupSize=[32, 1, 4]), "threadgroups"),
        (lambda c, p, e: c.update(A=c["A"] - 1), "exceeds its buffer"),
        (lambda c, p, e: c.update(C=c["C"] + 1), "output differs"),
    ),
)
def test_plan_refuses_launches_conv_cpp_would_not_make(change, message):
    entry, counts, parameters, threads, execution = _launch(
        RECORDS["general-2d-dilated-copied"]
    )
    change(counts, parameters, execution)
    with pytest.raises(ValueError, match=message):
        layout.plan(entry, counts, parameters, threads, execution)


def test_plan_refuses_the_other_alignment():
    entry, counts, parameters, threads, execution = _launch(
        RECORDS["general-2d-strided-unaligned"]
    )
    with pytest.raises(ValueError, match="entry differs"):
        layout.plan(entry[:-1] + "1", counts, parameters, threads, execution)


def _native(entry, counts, parameters):
    memory, buffers = {}, {}
    ctypes_of = {"float32": ctypes.c_float, "float16": ctypes.c_uint16}
    for name in layout.host_parameters(entry):
        dtype = layout.parameter_type(entry, name)
        if name == "params":
            data = _params_bytes(parameters["params"])
            memory[name] = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            count = len(data)
        elif name in counts:
            count = counts[name]
            memory[name] = (ctypes_of.get(dtype, ctypes.c_uint16) * count)()
        else:
            value = parameters[name]
            rows = value if isinstance(value, list) else [value]
            words = [word for row in rows for word in row.values()]
            memory[name] = (ctypes.c_int32 * len(words))(*words)
            count = len(words)
        buffers[name] = runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            count,
            int(name == layout.output(entry)),
        )
    return buffers, memory


def test_storage_reads_the_jump_record_and_residue_tables():
    entry, counts, parameters, threads, execution = _launch(
        RECORDS["general-2d-transpose"]
    )
    supplied, memory = _native(entry, counts, parameters)
    read_counts, values = layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters
    assert len(values["base_h"]) == 2 and len(values["base_w"]) == 2
    layout.plan(entry, read_counts, values, threads, execution)
    for name, count in (("jump_params", 7), ("base_h", 3)):
        short = dict(supplied)
        short[name] = runtime.Buffer(
            name.encode(), b"int32", supplied[name].data, count, 0
        )
        with pytest.raises(ValueError, match="differ in size"):
            layout.read_storage(entry, short)
    del memory


def test_dispatch_packs_tables_as_their_reflected_records():
    entry, counts, parameters, *_ = _launch(RECORDS["general-2d-dilated-copied"])
    table = parameters["base_w"]
    words = conv_dispatch._record(
        {
            "scalarLayout": {
                "elementType": "int32",
                "elementStrideBytes": 8,
                "structMembers": [{"name": "weight_base"}, {"name": "weight_size"}],
            }
        },
        "base_w",
        table,
        2,
        "opengl",
    )
    assert words == {
        "dtype": "int32",
        "shape": [6],
        "values": [value for row in table for value in row.values()],
    }
    with pytest.raises(ValueError, match="record differs"):
        conv_dispatch._record(
            {
                "scalarLayout": {
                    "elementType": "int32",
                    "elementStrideBytes": 8,
                    "structMembers": [{"name": "weight_size"}, {"name": "weight_base"}],
                }
            },
            "base_w",
            table,
            2,
            "opengl",
        )
    jump = conv_dispatch._record(
        {
            "scalarLayout": {
                "elementType": "int32",
                "elementStrideBytes": 32,
                "structMembers": [{"name": field} for field in general.JUMP_FIELDS],
            }
        },
        "jump_params",
        parameters["jump_params"],
        2,
        "directx",
    )
    assert jump["values"] == list(parameters["jump_params"].values())


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_packages_follow_the_general_gate(tmp_path, target):
    entry = ENTRY
    source = layout.signature(entry)["model"]
    built = conv_packages.ConvPackageCache(Path("/pinned"), tmp_path, target)
    config = built.project_config("out", entry)
    expected = gate._config(Path("/pinned"), "out", target, source, False)
    assert (
        config.entry_points == expected.entry_points == {model.SOURCE: (source.name,)}
    )
    assert config.specialization_constants == {"align_C": False}
    assert config.freeze_specialization_constants is True
    assert config.index_range_assertions == expected.index_range_assertions
    assert config.source_options == expected.source_options
    (rules,) = config.entry_workgroup_size_rules.values()
    assert {name: tuple(map(int, size)) for name, size in rules.items()} == {
        source.name: (32, 2, 2)
    }
    assert built.source_entry(entry) == source.name
    assert built.source_path(entry) == model.SOURCE


def test_conv_inc_ports_the_general_route():
    port = (HOST / "conv.inc").read_text()
    assert 'conv_unported("implicit_gemm_conv_2D_general_gpu")' not in port
    assert "return conv_implicit_gemm_general(in, wt, out, conv_params);" in port
    body = port[port.index("void conv_implicit_gemm_general(") :]
    body = body[: body.index("\n}\n")]
    for name in general.PARAMETERS:
        assert f'{{"{name}", ' in body, name
    for text in (
        "std::lcm(conv_params.idil[0], conv_params.kdil[0]) / conv_params.kdil[0]",
        "std::lcm(conv_params.idil[1], conv_params.str[1]) / conv_params.str[1]",
        "adj_implicit_m >= 8192 && conv_params.C >= 64 ? 64 : 32",
        "(bm == 64 && implicit_N >= 64) ? 64 : 32",
        "const bool align_C = conv_params.C % bk == 0;",
        '"_alC_" + std::to_string(align_C)',
        "{32, uint32_t(wn), uint32_t(wm)}",
    ):
        assert text in body, text
