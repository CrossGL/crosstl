"""The portable host's port of fft.cpp's Stockham FFT launches."""

import ctypes
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    fft_dispatch,
    fft_layout,
    fft_packages,
    lazy_package_evidence,
    runtime,
)

HOST = Path(fft_layout.__file__).parent

# plan_fft and compute_elems_per_thread of the pinned fft.cpp, compiled and
# printed for these sizes: (stockham, rader, rader_n, bluestein_n,
# four_step, n1, n2, elems_per_thread).
CPP_PLANS = {
    8: ([0, 0, 0, 0, 0, 0, 1, 0, 1], [0] * 9, 1, -1, False, 0, 0, 4),
    17: ([0] * 9, [0, 0, 0, 0, 0, 0, 2, 0, 0], 17, -1, False, 0, 0, 4),
    47: ([0, 0, 0, 0, 0, 0, 3, 0, 1], [0] * 9, 1, 128, False, 0, 0, 4),
    100: ([0, 0, 0, 0, 0, 2, 1, 0, 0], [0] * 9, 1, -1, False, 0, 0, 5),
    1982: (
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 1, 0],
        991,
        -1,
        False,
        0,
        0,
        5,
    ),
    3159: ([1, 0, 0, 0, 0, 0, 0, 5, 0], [0] * 9, 1, -1, False, 0, 0, 13),
    4096: ([0, 0, 4, 0, 0, 0, 0, 0, 0], [0] * 9, 1, -1, False, 0, 0, 8),
    4097: ([], [0] * 9, 1, 16384, True, 0, 0, None),
    8192: ([], [0] * 9, 1, -1, True, 128, 64, None),
}


@pytest.mark.parametrize("n", sorted(CPP_PLANS))
def test_plans_match_the_pinned_fft_cpp(n):
    stockham, rader, rader_n, bluestein_n, four_step, n1, n2, elems = CPP_PLANS[n]
    plan = fft_layout.plan_fft(n)
    assert (
        plan["stockham"],
        plan["rader"],
        plan["rader_n"],
        plan["bluestein_n"],
        plan["four_step"],
        plan["n1"],
        plan["n2"],
    ) == (stockham, rader, rader_n, bluestein_n, four_step, n1, n2)
    if elems is not None:
        assert fft_layout.compute_elems_per_thread(plan) == elems


def test_the_largest_four_step_plan_grows_n2_and_larger_sizes_are_refused():
    plan = fft_layout.plan_fft(1 << 24)
    assert (plan["n1"], plan["n2"]) == (4096, 4096)
    with pytest.raises(ValueError, match="not supported for size"):
        fft_layout.plan_fft(1 << 25)


def test_hash_names_freeze_fft_cpp_constants_and_threadgroups():
    entry = "fft_mem_256_float2_float2_n100_inv_0"
    assert fft_layout.entry_name(100, False, "float2", "float2") == entry
    info = fft_layout.signature(entry)
    assert info["source"] == "fft_mem_256_float2_float2"
    assert info["workgroupSize"] == (1, 2, 20)
    assert info["constants"]["elems_per_thread_"] == 5
    assert info["constants"]["radix_5_steps_"] == 2
    assert info["constants"]["radix_4_steps_"] == 1
    assert info["constants"]["rader_m_"] == 100
    assert info["constants"]["is_power_of_2_"] is False
    assert set(info["constants"]) == set(fft_layout.CONSTANT_IDS)
    assert fft_layout.CONSTANT_IDS["radix_13_steps_"] == 4
    assert fft_layout.CONSTANT_IDS["rader_2_steps_"] == 21
    # 8-point transforms batch 32 to a threadgroup's 256 elements.
    small = fft_layout.signature("fft_mem_256_float_float2_n8_inv_0")
    assert (small["transform"], small["workgroupSize"]) == ("r2c", (1, 32, 2))
    large = fft_layout.signature("fft_mem_4096_float2_float_n4096_inv_1")
    assert (large["transform"], large["workgroupSize"]) == ("c2r", (1, 1, 512))


@pytest.mark.parametrize(
    "entry",
    [
        "fft_mem_512_float2_float2_n100_inv_0",  # not fft.cpp's memory size
        "fft_mem_256_float_float2_n8_inv_1",  # an inverse real-to-complex
        "fft_mem_256_float2_float_n8_inv_0",  # a forward complex-to-real
        "fft_mem_256_float_float_n8_inv_0",
        "fft_mem_256_float2_float2_n17_inv_0",  # a Rader plan
        "fft_mem_256_float2_float2_n47_inv_0",  # a Bluestein plan
        "fft_mem_4096_float2_float2_n8192_inv_0",  # a four-step plan
        "fft_mem_256_float2_float2_n1_inv_0",
        "fft_mem_256_float2_float2_n08_inv_0",
        "rader_fft_mem_256_float2_float2_n17_inv_0",
    ],
)
def test_other_names_are_refused(entry):
    with pytest.raises(ValueError):
        fft_layout.signature(entry)
    assert not fft_dispatch.is_entry(entry)


def test_launches_follow_fft_cpp_grids_and_extents():
    c2c = fft_layout.signature("fft_mem_256_float2_float2_n100_inv_1")
    assert fft_layout.launch(c2c, 5)["workgroupCount"] == [3, 1, 1]
    assert fft_layout.extents(c2c, 5) == {"in": 500, "out": 500}
    r2c = fft_layout.signature("fft_mem_256_float_float2_n8_inv_0")
    # 65 transforms in threadgroups of 32 take three groups; two real rows
    # share each complex transform, so two groups run.
    assert fft_layout.launch(r2c, 65)["workgroupCount"] == [2, 1, 1]
    assert fft_layout.extents(r2c, 65) == {"in": 520, "out": 325}
    c2r = fft_layout.signature("fft_mem_256_float2_float_n8_inv_1")
    assert fft_layout.extents(c2r, 3) == {"in": 15, "out": 24}
    # Inverse real transforms pair their rows too: 100 rows of 100 take 50
    # threadgroups of two, and two rows share each one.
    c2r = fft_layout.signature("fft_mem_256_float2_float_n100_inv_1")
    assert fft_layout.launch(c2r, 100)["workgroupCount"] == [25, 1, 1]
    assert fft_layout.launch(c2r, 3)["workgroupCount"] == [1, 1, 1]
    entry = "fft_mem_256_float2_float2_n100_inv_1"
    execution = fft_layout.launch(c2c, 5)
    metadata = fft_layout.plan_launch(
        entry, {"in": 500, "out": 500}, 100, 5, 500, execution
    )
    assert metadata["batchSize"] == 5 and metadata["transform"] == "c2c"
    for counts, n, total, threads, launch in (
        ({"in": 400, "out": 500}, 100, 5, 500, execution),
        ({"in": 500, "out": 500}, 80, 5, 500, execution),
        ({"in": 500, "out": 500}, 100, 5, 499, execution),
        (
            {"in": 500, "out": 500},
            100,
            5,
            500,
            dict(execution, workgroupCount=[5, 1, 1]),
        ),
    ):
        with pytest.raises(ValueError):
            fft_layout.plan_launch(entry, counts, n, total, threads, launch)


def test_project_config_freezes_constants_and_proves_opengl_subscripts(tmp_path):
    entry = "fft_mem_256_float_float2_n8_inv_0"
    for target in ("metal", "directx", "opengl"):
        config = fft_packages.project_config(tmp_path, "out", target, entry)
        assert config.freeze_specialization_constants is True
        assert dict(config.specialization_constants) == {
            **fft_layout.signature(entry)["constants"]
        }
        assert config.entry_points == {fft_layout.SOURCE: ("fft_mem_256_float_float2",)}
        assert bool(config.index_range_assertions) == (target == "opengl")
    records = fft_layout.opengl_index_assertions(entry)
    scoped = {
        (record["expression"], record.get("indexed")): record["maximum"]
        for record in records
    }
    assert scoped[("reference.offset + index", "crosstl_workgroup_shared_in")] == 255
    assert (
        scoped[("reference.offset + index", "crosstl_resource")]
        == fft_layout.SIGNED_MAX
    )
    assert scoped[("crosstl_resource_2 + reference.offset + index", None)] == (
        fft_layout.SIGNED_MAX
    )
    (access,) = fft_layout.workgroup_access_assertions(entry)
    assert (
        access["maximum"] == 255 and access["entry_point"] == "fft_mem_256_float_float2"
    )


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _complex_word(real, imag):
    return _float_bits(real) | _float_bits(imag) << 32


class FakeCache:
    """Reflects each parameter as fft.metal's float2, float and int bindings."""

    def __init__(self, target):
        self.target = target

    def get(self, entry):
        bindings = []
        for name in fft_layout.PARAMETERS:
            dtype = fft_layout.parameter_type(entry, name)
            scalar = (
                dict(fft_dispatch.VECTOR_LAYOUT)
                if dtype == element_storage.COMPLEX
                else {
                    "elementType": dtype,
                    "elementStrideBytes": 4,
                    "storageLayout": "hlsl-constant-buffer",
                }
            )
            bindings.append(
                {
                    "name": name + "Binding",
                    "kind": "constant-buffer" if name in fft_layout.SCALARS else "",
                    "access": "read_write" if name == "out" else "read",
                    "scalarLayout": scalar,
                    "provenance": {"sourceResource": {"parameter": name}},
                }
            )
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def run(monkeypatch, tmp_path, entry, total, source, fault=None, **overrides):
    info = fft_layout.signature(entry)
    spans = fft_layout.extents(info, total)
    source_type = ctypes.c_uint64 if info["input"] == "complex64" else ctypes.c_uint32
    output_type = ctypes.c_uint64 if info["output"] == "complex64" else ctypes.c_uint32
    memory = {
        "in": (source_type * spans["in"])(*source),
        "out": (output_type * spans["out"])(),
        "n": (ctypes.c_int32 * 1)(overrides.get("n", info["n"])),
        "batch_size": (ctypes.c_int32 * 1)(total),
    }
    types = {name: fft_layout.parameter_type(entry, name) for name in memory}
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
    host = SimpleNamespace(
        ffts=FakeCache("opengl"),
        target="opengl",
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )
    seen = {}

    def execute(_, request):
        seen.update(request)
        inputs = request["inputs"]
        size = spans["out"]
        # The transform writes every output element: each the input element
        # of the same index, keeping the guard tail.
        words = element_storage.unpack(
            info["input"], "opengl", inputs["inBinding"]["values"]
        )
        out = element_storage.unpack(
            info["output"], "opengl", inputs["outBinding"]["values"]
        )
        result = words[:size] + out[size:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={
                "outBinding": element_storage.packet(info["output"], "opengl", result)
            },
            details={"request": {}},
        )

    monkeypatch.setattr(
        fft_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(fft_dispatch, "execute", execute)
    execution = overrides.get("execution", fft_layout.launch(info, total))
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*execution["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*execution["workgroupSize"]),
    )
    fft_dispatch.dispatch(host, entry, array, 4, spans["out"], launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host, seen


def test_complex_operands_travel_as_float2_bits_with_guards(monkeypatch, tmp_path):
    entry = "fft_mem_256_float2_float2_n8_inv_0"
    source = [_complex_word(index + 0.5, -index) for index in range(16)]
    values, event, host, request = run(monkeypatch, tmp_path, entry, 2, source)
    assert values == source and host.dispatch_count == 1
    uploaded = request["inputs"]["inBinding"]
    assert uploaded["dtype"] == "float32" and uploaded["shape"] == [2 * (16 + 32)]
    assert uploaded["values"][:2] == [_float_bits(0.5), _float_bits(0.0)]
    # Poisoned elements follow the input; guards fill the fresh output.
    assert uploaded["values"][32:34] == [0x3FC00000, 0x3FC00000]
    initial = request["inputs"]["outBinding"]["values"]
    assert initial[:2] == [0x6A15BEEF, 0x3EAB3555]
    assert request["inputs"]["nBinding"]["values"] == [8]
    assert request["inputs"]["batch_sizeBinding"]["values"] == [2]
    assert event["fftMetadata"]["elements"] == {"in": 16, "out": 16}
    assert event["outputValues"] == source


def test_real_operands_travel_as_float32_bits(monkeypatch, tmp_path):
    entry = "fft_mem_256_float_float2_n8_inv_0"
    source = [_float_bits(index / 4) for index in range(24)]
    values, event, *_ = run(monkeypatch, tmp_path, entry, 3, source)
    # The fake transform copies the first 15 input words into the output.
    assert values == source[:15]
    assert event["workgroupCount"] == [1, 1, 1]


@pytest.mark.parametrize(
    "fault, overrides, message",
    [
        ("guard", {}, "output guard"),
        (None, {"n": 16}, "launch size"),
        (
            None,
            {"execution": {"workgroupCount": [2, 1, 1], "workgroupSize": [1, 32, 2]}},
            "dispatch",
        ),
    ],
)
def test_dispatch_refuses_changed_guards_sizes_and_grids(
    monkeypatch, tmp_path, fault, overrides, message
):
    source = [_complex_word(1.0, 2.0)] * 16
    with pytest.raises((ValueError, RuntimeError), match=message):
        run(
            monkeypatch,
            tmp_path,
            "fft_mem_256_float2_float2_n8_inv_0",
            2,
            source,
            fault,
            **overrides,
        )


def test_the_host_routes_and_audits_stockham_ffts():
    source = (HOST / "runtime.py").read_text()
    assert "fft_packages.FftPackageCache(" in source
    assert "if fft_dispatch.is_entry(entry):" in source
    assert lazy_package_evidence.family("fft_mem_256_float2_float2_n8_inv_0") == "fft"
    assert lazy_package_evidence.family("rader_fft_mem_256_float2_float2") is None
    port = (HOST / "fft.inc").read_text()
    for route in ("Rader plans", "fused Bluestein plans", "four-step plans"):
        assert f'fft_unported("{route}")' in port or route in port
    assert "void FFT::eval_gpu" in port
    prepare = (HOST / "prepare.py").read_text()
    assert '"crosstl_fft.inc": (HERE / "fft.inc").read_bytes()' in prepare
    assert (
        '#include "mlx/backend/no_gpu/crosstl_fft.inc"' in (
            HOST / "backend.cpp"
        ).read_text()
    )
