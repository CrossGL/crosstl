"""The portable host's port of fft.cpp's Stockham, Rader and Bluestein FFT launches."""

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
        "rader_fft_mem_256_float2_float2_n8_inv_0",  # a Stockham plan
        "rader_fft_mem_256_float2_float2_n47_inv_0",  # a Bluestein plan
        "rader_fft_mem_512_float2_float2_n17_inv_0",
        "bluestein_fft_mem_256_float2_float2_n47_inv_0",  # not bluestein_n
        "bluestein_fft_mem_256_float2_float2_n64_inv_0",  # no length plans at 64
        "bluestein_fft_mem_512_float2_float2_n128_inv_0",
        "bluestein_fft_mem_256_float2_float2_n128_inv_0_precomputed",
        "fft_mem_256_float2_float2_n128_inv_0_precomputed",
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
        entry, {"in": 500, "out": 500}, {"n": 100, "batch_size": 5}, 500, execution
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
            fft_layout.plan_launch(
                entry, counts, {"n": n, "batch_size": total}, threads, launch
            )
    with pytest.raises(ValueError):
        fft_layout.plan_launch(
            entry,
            {"in": 500, "out": 500},
            {"n": 100, "batch_size": 5, "rader_n": 1},
            500,
            execution,
        )


def test_rader_names_freeze_fft_cpp_constants_tables_and_threadgroups():
    entry = "rader_fft_mem_256_float2_float2_n17_inv_0"
    assert fft_layout.entry_name(17, False, "float2", "float2") == entry
    info = fft_layout.signature(entry)
    assert (info["kind"], info["raderN"], info["source"]) == (
        "rader",
        17,
        "rader_fft_mem_256_float2_float2",
    )
    # 15 transforms of 17 fill 255 of the threadgroup's 256 elements.
    assert info["workgroupSize"] == (1, 15, 5)
    assert info["constants"]["rader_4_steps_"] == 2
    assert info["constants"]["rader_m_"] == 1
    assert set(info["constants"]) == set(fft_layout.CONSTANT_IDS)
    # 1982 = 2 * 991: one radix-2 step around a Rader factor of 991.
    large = fft_layout.signature("rader_fft_mem_2048_float2_float_n1982_inv_1")
    assert (large["raderN"], large["constants"]["rader_m_"]) == (991, 2)
    assert large["constants"]["radix_2_steps_"] == 1
    assert fft_layout.parameters(entry) == (
        "in",
        "out",
        "raders_b_q",
        "raders_g_q",
        "raders_g_minus_q",
        "n",
        "batch_size",
        "rader_n",
    )
    assert fft_layout.parameter_type(entry, "raders_g_q") == "int16"
    assert fft_layout.parameter_type(entry, "raders_b_q") == "complex64"
    assert fft_layout.parameter_type(entry, "rader_n") == "int32"
    assert fft_layout.extents(info, 2) == {
        "in": 34,
        "out": 34,
        "raders_b_q": 16,
        "raders_g_q": 16,
        "raders_g_minus_q": 16,
    }
    # A partial threadgroup's cooperative copy touches 20 elements per row
    # of 17, clamped to the threadgroup's 255; real transforms stay exact.
    assert fft_layout.footprint(info, 1) == {"in": 20, "out": 20}
    assert fft_layout.footprint(info, 15) == {"in": 255, "out": 255}
    assert fft_layout.footprint(info, 16) == {"in": 275, "out": 275}
    stockham = fft_layout.signature("fft_mem_256_float2_float2_n100_inv_0")
    assert fft_layout.footprint(stockham, 3) == {"in": 300, "out": 300}
    real = fft_layout.signature("rader_fft_mem_256_float_float2_n17_inv_0")
    assert fft_layout.footprint(real, 1) == {"in": 17, "out": 9}
    metadata = fft_layout.plan_launch(
        entry,
        fft_layout.extents(info, 2),
        {"n": 17, "batch_size": 2, "rader_n": 17},
        34,
        fft_layout.launch(info, 2),
    )
    assert metadata["kind"] == "rader"
    for values in (
        {"n": 17, "batch_size": 2},
        {"n": 17, "batch_size": 2, "rader_n": 16},
    ):
        with pytest.raises(ValueError):
            fft_layout.plan_launch(
                entry,
                fft_layout.extents(info, 2),
                values,
                34,
                fft_layout.launch(info, 2),
            )


def test_rader_tables_follow_compute_raders_constants():
    # 3 generates the units modulo 17, and 6 is its inverse.
    assert fft_layout.primitive_root(17) == 3
    g_q, g_minus_q = fft_layout.rader_generator_tables(17)
    assert g_q[:5] == [1, 3, 9, 10, 13] and g_minus_q[:5] == [1, 6, 2, 12, 4]
    assert sorted(g_q) == sorted(g_minus_q) == list(range(1, 17))
    assert fft_layout.primitive_root(991) == 6
    spectrum = fft_layout.rader_kernel_spectrum(17)
    # The DFT of a unit-modulus sequence of 16 conserves its energy.
    assert sum(abs(value) ** 2 for value in spectrum) == pytest.approx(16 * 16)


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
    assert not any(record.get("indexed") == "crosstl_resource_3" for record in records)
    rader = {
        record.get("indexed")
        for record in fft_layout.opengl_index_assertions(
            "rader_fft_mem_256_float2_float2_n17_inv_0"
        )
    }
    assert {"crosstl_resource_3", "crosstl_resource_5"} <= rader
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

    def __init__(self, target, declared=None):
        self.target = target
        # The workgroup size the translated entry declares, when it differs.
        self.declared = declared

    def get(self, entry):
        bindings = []
        for name in fft_layout.parameters(entry):
            dtype = fft_layout.parameter_type(entry, name)
            scalar = (
                dict(fft_dispatch.VECTOR_LAYOUT)
                if dtype == element_storage.COMPLEX
                else {
                    # OpenGL widens int16 storage to int32 elements.
                    "elementType": "int32" if dtype == "int16" else dtype,
                    "elementStrideBytes": 4,
                    "storageLayout": "hlsl-constant-buffer",
                }
            )
            scalars = fft_layout.scalars(entry)
            bindings.append(
                {
                    "name": name + "Binding",
                    "kind": "constant-buffer" if name in scalars else "",
                    "access": "read_write" if name == "out" else "read",
                    "scalarLayout": scalar,
                    "provenance": {"sourceResource": {"parameter": name}},
                }
            )
        config = {} if self.declared is None else {"local_size": self.declared}
        return {
            "bindings": bindings,
            "artifact": {"id": entry},
            "entryPoint": {"name": "main", "executionConfig": config},
        }, Path("/packages")


def rader_tables(rader_n, fault=None):
    """compute_raders_constants' tables as the host binds them."""
    g_q, g_minus_q = fft_layout.rader_generator_tables(rader_n)
    if fault == "generator":
        g_q[1], g_q[2] = g_q[2], g_q[1]
    spectrum = [
        _complex_word(value.real, value.imag)
        for value in fft_layout.rader_kernel_spectrum(rader_n)
    ]
    if fault == "spectrum":
        spectrum[3] = _complex_word(0.0, 0.0)
    size = rader_n - 1
    return {
        "raders_b_q": (ctypes.c_uint64 * size)(*spectrum),
        "raders_g_q": (ctypes.c_int16 * size)(*g_q),
        "raders_g_minus_q": (ctypes.c_int16 * size)(*g_minus_q),
    }


def bluestein_tables(length, bluestein_n, fault=None):
    """compute_bluestein_constants' tables as the host binds them."""
    chirp, spectrum = fft_layout.bluestein_tables(length, bluestein_n)
    w_k = [_complex_word(value.real, value.imag) for value in chirp]
    w_q = [_complex_word(value.real, value.imag) for value in spectrum]
    if fault == "chirp":
        w_k[1] = _complex_word(chirp[1].real, -chirp[1].imag)
    if fault == "spectrum":
        w_q[5] = _complex_word(0.0, 0.0)
    return {
        "w_q": (ctypes.c_uint64 * bluestein_n)(*w_q),
        "w_k": (ctypes.c_uint64 * length)(*w_k),
    }


def run(monkeypatch, tmp_path, entry, total, source, fault=None, **overrides):
    info = fft_layout.signature(entry)
    length = overrides.get("transform_length", info["lengths"][0])
    spans = fft_layout.extents(info, total, length)
    source_type = ctypes.c_uint64 if info["input"] == "complex64" else ctypes.c_uint32
    output_type = ctypes.c_uint64 if info["output"] == "complex64" else ctypes.c_uint32
    memory = {
        "in": (source_type * spans["in"])(*source),
        "out": (output_type * spans["out"])(),
    }
    if info["kind"] == "rader":
        memory.update(rader_tables(info["raderN"], fault))
    if info["kind"] == "bluestein":
        memory.update(bluestein_tables(length, info["n"], fault))
        memory["length"] = (ctypes.c_int32 * 1)(overrides.get("length", length))
    memory["n"] = (ctypes.c_int32 * 1)(overrides.get("n", info["n"]))
    memory["batch_size"] = (ctypes.c_int32 * 1)(total)
    if info["kind"] == "rader":
        memory["rader_n"] = (ctypes.c_int32 * 1)(
            overrides.get("rader_n", info["raderN"])
        )
    types = {name: fft_layout.parameter_type(entry, name) for name in memory}
    array = (runtime.Buffer * len(memory))(
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
        ffts=FakeCache("opengl", overrides.get("declared")),
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
        if fault == "tail":
            # Past the transforms but inside the cooperative copy's reach.
            result[size] ^= 1
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
    fft_dispatch.dispatch(host, entry, array, len(memory), spans["out"], launch)
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


def test_rader_tables_travel_with_poisoned_tails_and_are_checked(monkeypatch, tmp_path):
    entry = "rader_fft_mem_256_float2_float2_n17_inv_1"
    source = [_complex_word(index, 0.25) for index in range(34)]
    values, event, host, request = run(monkeypatch, tmp_path, entry, 2, source)
    assert values == source and event["fftMetadata"]["kind"] == "rader"
    inputs = request["inputs"]
    g_q, _ = fft_layout.rader_generator_tables(17)
    # OpenGL carries each short in an int and follows the table with poison.
    assert inputs["raders_g_qBinding"]["values"][:16] == g_q
    assert inputs["raders_g_qBinding"]["values"][16:] == [0x3FC0] * 32
    assert inputs["raders_b_qBinding"]["shape"] == [2 * (16 + 32)]
    assert inputs["rader_nBinding"]["values"] == [17]
    assert event["fftMetadata"]["elements"]["raders_g_minus_q"] == 16
    for fault, message in (
        ("generator", "generator tables"),
        ("spectrum", "kernel spectrum"),
    ):
        with pytest.raises(ValueError, match=message):
            run(monkeypatch, tmp_path, entry, 2, source, fault)
    with pytest.raises(ValueError, match="launch size"):
        run(monkeypatch, tmp_path, entry, 2, source, rader_n=16)
    # The staged output spans the 40-element reach of two 17-element rows;
    # a write into its tail never reaches MLX.
    values, event, _, request = run(monkeypatch, tmp_path, entry, 2, source, "tail")
    assert values == source and event["fftMetadata"]["footprint"]["out"] == 40
    assert request["inputs"]["outBinding"]["shape"] == [2 * (40 + 32)]
    assert request["inputs"]["inBinding"]["values"][2 * 34 :] == [0x3FC00000] * (
        2 * (6 + 32)
    )
    with pytest.raises(RuntimeError, match="output guard"):
        run(monkeypatch, tmp_path, entry, 2, source, "guard")


def test_the_host_routes_and_audits_stockham_and_rader_ffts():
    source = (HOST / "runtime.py").read_text()
    assert "fft_packages.FftPackageCache(" in source
    assert "if fft_dispatch.is_entry(entry):" in source
    assert lazy_package_evidence.family("fft_mem_256_float2_float2_n8_inv_0") == "fft"
    assert (
        lazy_package_evidence.family("rader_fft_mem_256_float2_float2_n17_inv_0")
        == "fft"
    )
    assert (
        lazy_package_evidence.family("bluestein_fft_mem_256_float2_float2_n128_inv_0")
        == "fft"
    )
    port = (HOST / "fft.inc").read_text()
    assert 'fft_unported("Rader plans")' not in port
    assert 'fft_unported("fused Bluestein plans")' not in port
    assert '"rader_fft_mem_"' in port and "pocketfft::c2c(" in port
    assert '"bluestein_fft_mem_"' in port and "fft_bluestein_tables(" in port
    for route in ("Bluestein twiddle-table plans", "four-step plans"):
        assert f'fft_unported("{route}")' in port or route in port
    assert "void FFT::eval_gpu" in port
    prepare = (HOST / "prepare.py").read_text()
    assert '"crosstl_fft.inc": (HERE / "fft.inc").read_bytes()' in prepare
    assert (
        '#include "mlx/backend/no_gpu/crosstl_fft.inc"' in (
            HOST / "backend.cpp"
        ).read_text()
    )


def test_threadgroups_past_the_z_limit_dispatch_at_the_declared_linear_size(
    monkeypatch, tmp_path
):
    entry = "fft_mem_1024_float2_float2_n1024_inv_0"
    assert fft_layout.signature(entry)["workgroupSize"] == (1, 1, 128)
    source = [_complex_word(index, 1.0) for index in range(2048)]
    values, event, _host, request = run(
        monkeypatch, tmp_path, entry, 2, source, declared=[128, 1, 1]
    )
    assert values == source
    assert request["execution"] == {
        "workgroupCount": [2, 1, 1],
        "workgroupSize": [128, 1, 1],
    }
    assert event["workgroupSize"] == [128, 1, 1]
    assert event["launchWorkgroupSize"] == [1, 1, 128]
    with pytest.raises(ValueError, match="declares another workgroup"):
        run(monkeypatch, tmp_path, entry, 2, source, declared=[64, 2, 1])


def test_bluestein_names_carry_the_convolution_size_and_freeze_its_constants():
    entry = "bluestein_fft_mem_256_float2_float2_n128_inv_0"
    for length in (47, 59):
        assert fft_layout.entry_name(length, False, "float2", "float2") == entry
    info = fft_layout.signature(entry)
    assert (info["kind"], info["n"], info["lengths"]) == ("bluestein", 128, (47, 59))
    # 128-point convolutions batch two to a threadgroup's 256 elements.
    assert info["workgroupSize"] == (1, 2, 32)
    constants = info["constants"]
    assert constants["is_power_of_2_"] is True
    assert constants["elems_per_thread_"] == 4
    assert (constants["radix_4_steps_"], constants["radix_2_steps_"]) == (3, 1)
    assert not any(constants[name] for name in fft_layout.RADER_CONSTANTS)
    assert constants[fft_layout.TWIDDLE_TABLE_CONSTANT] is False
    # fft.cpp sets the twiddle-table constant for complex transforms only.
    real = fft_layout.signature("bluestein_fft_mem_256_float_float2_n128_inv_0")
    assert fft_layout.TWIDDLE_TABLE_CONSTANT not in real["constants"]
    large = fft_layout.signature(fft_layout.entry_name(1031, True, "float2", "float"))
    assert (large["source"], large["workgroupSize"]) == (
        "bluestein_fft_mem_4096_float2_float",
        (1, 1, 512),
    )
    assert 2039 in large["lengths"] and 1031 in large["lengths"]
    # Every length a size convolves plans the same steps and threads.
    for size in (128, 256, 512, 1024, 2048, 4096):
        lengths = fft_layout.bluestein_lengths(size)
        assert lengths
        assert {
            fft_layout.compute_elems_per_thread(fft_layout.plan_fft(length))
            for length in lengths
        } == {
            fft_layout.signature(
                fft_layout.entry_name(lengths[0], False, "float2", "float2")
            )["constants"]["elems_per_thread_"]
        }


def test_bluestein_launches_span_the_transform_length():
    info = fft_layout.signature("bluestein_fft_mem_256_float_float2_n128_inv_0")
    assert fft_layout.extents(info, 3, 47) == {
        "in": 141,
        "out": 72,
        "w_q": 128,
        "w_k": 47,
    }
    assert fft_layout.footprint(info, 3, 47) == {"in": 141, "out": 72}
    with pytest.raises(ValueError, match="length"):
        fft_layout.extents(info, 3, 48)
    execution = fft_layout.launch(info, 3)
    assert execution == {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 2, 32]}
    counts = {"in": 141, "out": 72, "w_q": 128, "w_k": 47}
    values = {"length": 47, "n": 128, "batch_size": 3}
    entry = "bluestein_fft_mem_256_float_float2_n128_inv_0"
    metadata = fft_layout.plan_launch(entry, counts, values, 72, execution)
    assert metadata["length"] == 47 and metadata["n"] == 128
    for changed in (
        {"length": 46},
        {"length": 64},
        {"n": 47},
        {"length": None},
    ):
        with pytest.raises(ValueError):
            fft_layout.plan_launch(entry, counts, {**values, **changed}, 72, execution)
    with pytest.raises(ValueError):
        fft_layout.plan_launch(entry, dict(counts, w_k=48), values, 72, execution)


def test_bluestein_tables_follow_compute_bluestein_constants():
    numpy = pytest.importorskip("numpy")
    for length, size in ((47, 128), (289, 1024), (2039, 4096)):
        chirp, spectrum = fft_layout.bluestein_tables(length, size)
        k = numpy.arange(-length + 1, length)
        padded = numpy.zeros(size, dtype=complex)
        padded[: 2 * length - 1] = numpy.exp(1j * numpy.pi * k**2 / length)
        assert numpy.allclose(spectrum, numpy.fft.fft(padded), atol=1e-9)
        assert numpy.allclose(
            chirp, numpy.exp(-1j * numpy.pi * numpy.arange(length) ** 2 / length)
        )


@pytest.mark.parametrize("entry", ["bluestein_fft_mem_256_float2_float2_n128_inv_0"])
def test_bluestein_tables_travel_and_are_checked(monkeypatch, tmp_path, entry):
    source = [_complex_word(index, -1.0) for index in range(2 * 47)]
    values, event, _host, request = run(monkeypatch, tmp_path, entry, 2, source)
    assert values == source
    assert request["inputs"]["lengthBinding"]["values"] == [47]
    assert request["inputs"]["nBinding"]["values"] == [128]
    assert len(request["inputs"]["w_qBinding"]["values"]) == 2 * (128 + 32)
    assert len(request["inputs"]["w_kBinding"]["values"]) == 2 * (47 + 32)
    assert event["fftMetadata"]["length"] == 47
    for fault, message in (("chirp", "w_k differs"), ("spectrum", "w_q differs")):
        with pytest.raises(ValueError, match=message):
            run(monkeypatch, tmp_path, entry, 2, source, fault)
    with pytest.raises(ValueError, match="length"):
        run(monkeypatch, tmp_path, entry, 2, source, length=46)
