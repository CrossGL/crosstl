"""Hadamard host routing, launch contracts, JIT sources and guarded commits."""

import ctypes
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import hadamard_dispatch as dispatch
from demos.integrations.mlx.portable_host import hadamard_layout as layout
from demos.integrations.mlx.portable_host import (
    hadamard_packages,
    hadamard_workloads,
    runtime,
    verify_upstream_hadamard,
)
from demos.integrations.mlx.tests.host.test_portable_matmul import (
    CTYPES,
    physical_layout,
)

HOST = Path(layout.__file__).parent
# The first rows of the pinned h12, so source rendering needs no MLX checkout.
H12 = """
+-++++++++++
--+-+++-++-+
-++-+-+++--+
-+-+-+-+-+-+
"""


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (2, (2, 1)),
        (12, (1, 12)),
        (20, (1, 20)),
        (28 * 256, (256, 28)),
        # decompose_hadamard tries 12 first: 3 * 2^14 is 12 * 2^12.
        (3 * 2**14, (2**12, 12)),
        (7 * 2**16, (2**14, 28)),
        (2**26, (2**26, 1)),
    ],
)
def test_decompose_mirrors_decompose_hadamard(size, expected):
    assert layout.decompose(size) == expected


@pytest.mark.parametrize("size", [0, 1, 3, 36, 44, 2**27, 12 * 2**27, "4", 4.0])
def test_decompose_rejects_sizes_the_kernels_cannot_transform(size):
    # 36 = 12 * 3 passes upstream's divisibility check but is not m * 2^k.
    with pytest.raises(ValueError):
        layout.decompose(size)


def test_plan_splits_power_of_two_factors_above_8192():
    assert layout.plan(8192)["stages"].keys() == {"n2"}
    plan = layout.plan(2**14)
    assert (plan["n1"], plan["n2"]) == (128, 128)
    assert plan["stages"]["n1"]["arguments"] == (128, 16, 4, 128)
    assert plan["stages"]["n2"]["arguments"] == (128, 16, 4)
    plan = layout.plan(28 * 2**15)
    assert (plan["n1"], plan["n2"], plan["m"]) == (128, 256, 28)
    assert plan["stages"]["m"]["arguments"] == (2**15, 28, 2)
    assert plan["stages"]["m"]["workgroupWidth"] == 256
    # n == 2 caps the m stage's read width at n; n == 1 reads one value.
    assert layout.plan(24)["stages"]["m"]["arguments"] == (2, 12, 2)
    assert layout.plan(12)["stages"]["m"]["arguments"] == (1, 12, 1)
    assert layout.plan(48)["stages"]["m"]["arguments"] == (4, 12, 4)
    assert layout.plan(2)["stages"]["n2"]["arguments"] == (2, 2, 2)


def test_entries_follow_hadamard_mn_contiguous_launch_order():
    assert layout.entries(28 * 2**14, "float16") == [
        "n1hadamard_458752_float16",
        "n2hadamard_458752_float16",
        "mhadamard_458752_float16",
    ]
    assert layout.entries(12, "bfloat16") == ["mhadamard_12_bfloat16"]
    with pytest.raises(ValueError):
        layout.entries(4, "int32")


@pytest.mark.parametrize(
    "entry",
    [
        "n1hadamard_8_float32",
        "mhadamard_8_float32",
        "n2hadamard_12_float32",
        "n2hadamard_08_float32",
        "n2hadamard_8_float64",
        "hadamard_8_float32",
        "n2hadamard_8_float32 ",
        None,
    ],
)
def test_signature_rejects_entries_upstream_never_launches(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_signature_derives_every_template_argument():
    info = layout.signature("n1hadamard_196608_bfloat16")
    assert info["templateArguments"] == ("bfloat16_t", 128, 16, 4, 128)
    assert info["workgroupSize"] == (8, 1, 1)
    info = layout.signature("mhadamard_196608_bfloat16")
    assert info["function"] == "hadamard_m"
    assert info["templateArguments"] == ("bfloat16_t", 16384, 12, 4)


def test_scales_match_the_stage_that_applies_them():
    assert layout.scale("n1hadamard_16384_float32", 0.5) == 1.0
    assert layout.scale("n2hadamard_16384_float32", 0.5) == 0.5
    assert layout.scale("n2hadamard_56_float32", 0.5) == 1.0
    assert layout.scale("mhadamard_56_float32", 0.5) == 0.5


def test_execution_mirrors_the_metal_grids():
    assert layout.execution("n2hadamard_256_float32", 4 * 256) == {
        "workgroupCount": [1, 4, 1],
        "workgroupSize": [16, 1, 1],
    }
    assert layout.execution("n1hadamard_16384_float32", 2 * 16384) == {
        "workgroupCount": [1, 2, 128],
        "workgroupSize": [8, 1, 1],
    }
    assert layout.execution("mhadamard_7168_float32", 3 * 7168) == {
        "workgroupCount": [1, 3, 1],
        "workgroupSize": [128, 1, 1],
    }
    assert layout.execution("mhadamard_12_float32", 15 * 12) == {
        "workgroupCount": [1, 15, 1],
        "workgroupSize": [1, 1, 1],
    }
    for elements in (0, 7, 2 * 65536 * 2, layout.MAX_ELEMENTS + 1):
        with pytest.raises(ValueError):
            layout.execution("n2hadamard_2_float32", elements)


@pytest.mark.parametrize("size", [2, 12, 256, 8192, 16384, 28 * 2**14, 2**26])
def test_batch_limit_keeps_every_stage_within_portable_limits(size):
    batch = layout.batch_limit(size)
    entries = layout.entries(size, "float32")
    for entry in entries:
        layout.execution(entry, batch * size)

    def accepted(entry, elements):
        try:
            layout.execution(entry, elements)
        except ValueError:
            return False
        return True

    # One more transform overflows some stage's groups or the element bound.
    assert not all(accepted(entry, (batch + 1) * size) for entry in entries)


def native_buffers(entry, elements, scale=1.0, *, alias=False):
    dtype = layout.signature(entry)["dtype"]
    memory = {
        "in": (CTYPES[dtype] * elements)(*range(1, elements + 1)),
        "out": (CTYPES[dtype] * elements)(),
        "scale": (ctypes.c_float * 1)(scale),
    }
    if alias:
        memory["out"] = memory["in"]
    buffers = {
        name: runtime.Buffer(
            name.encode(),
            ("float32" if name == "scale" else dtype).encode(),
            ctypes.addressof(memory[name]),
            1 if name == "scale" else elements,
            int(name == "out"),
        )
        for name in layout.PARAMETERS
    }
    return buffers, memory


@pytest.mark.parametrize(
    "fault",
    [None, "alias", "dtype", "count", "direction", "partial", "scale", "nan", "launch"],
)
def test_validate_checks_storage_direction_aliases_and_scale(fault):
    entry = "n2hadamard_56_float16"
    buffers, memory = native_buffers(entry, 112, alias=fault == "alias")
    launch = layout.execution(entry, 112)
    if fault == "dtype":
        buffers["in"].dtype = b"bfloat16"
    elif fault == "count":
        buffers["in"].count = 111
    elif fault == "direction":
        buffers["in"].output = 1
    elif fault == "partial":
        buffers["out"].data = buffers["in"].data + 2
    elif fault == "scale":
        memory["scale"][0] = 0.5
    elif fault == "nan":
        memory["scale"][0] = math.nan
    elif fault == "launch":
        launch = {"workgroupCount": [2, 1, 1], "workgroupSize": [2, 1, 1]}
    if fault in (None, "alias"):
        metadata = layout.validate(entry, buffers, 112, launch)
        assert metadata["transforms"] == 2 and metadata["scale"] == 1.0
    else:
        with pytest.raises(ValueError):
            layout.validate(entry, buffers, 112, launch)


def write_matrices(root, rows=H12):
    path = root / hadamard_packages.MATRICES
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            f'constexpr std::string_view h{m} = R"({rows})";' for m in layout.FACTORS
        ),
        encoding="utf-8",
    )


def test_codelet_matches_gen_hadamard_codelet(tmp_path):
    write_matrices(tmp_path)
    assert hadamard_packages.codelet(tmp_path, 1) == (
        "METAL_FUNC void hadamard_radix_m(thread float *x) {\n}\n"
    )
    lines = hadamard_packages.codelet(tmp_path, 12).splitlines()
    assert lines[0] == "METAL_FUNC void hadamard_radix_m(thread float *x) {"
    assert lines[1] == "  float tmp[12];"
    assert (
        lines[2]
        == "  tmp[0] =  + x[0] - x[1]"
        + "".join(f" + x[{i}]" for i in range(2, 12))
        + ";"
    )
    assert len(lines) == 2 + 4 + 2
    assert lines[-2] == "  for (int i = 0; i < 12; i++) { x[i] = tmp[i]; }"


def test_source_renders_the_jit_library_for_one_stage(tmp_path):
    write_matrices(tmp_path)
    text = hadamard_packages.source(tmp_path, "mhadamard_24_bfloat16")
    utils, rest = text.split("\n", 1)
    assert utils == '#include "mlx/backend/metal/kernels/utils.h"'
    codelet, definition = rest.split(
        '#include "mlx/backend/metal/kernels/hadamard.h"\n'
    )
    assert codelet == hadamard_packages.codelet(tmp_path, 12)
    assert definition == (
        '\ntemplate [[host_name("mhadamard_24_bfloat16")]] [[kernel]] '
        "decltype(hadamard_m<bfloat16_t, 2, 12, 2>) hadamard_m<bfloat16_t, 2, 12, 2>;\n"
    )
    text = hadamard_packages.source(tmp_path, "n1hadamard_16384_float32")
    assert text.endswith("hadamard_n<float, 128, 16, 4, 128>;\n")
    # Every stage of a size shares the same codelet, as in one JIT library.
    assert hadamard_packages.codelet(tmp_path, 1) in text


def test_missing_matrices_fail(tmp_path):
    (tmp_path / "mlx/backend/common").mkdir(parents=True)
    (tmp_path / hadamard_packages.MATRICES).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="h12"):
        hadamard_packages.codelet(tmp_path, 12)


def test_package_config_translates_the_generated_wrapper(tmp_path):
    write_matrices(tmp_path)
    cache = hadamard_packages.HadamardPackageCache(tmp_path, tmp_path / "c", "opengl")
    (tmp_path / ".work").mkdir()
    config = cache.project_config(".work/out", "n2hadamard_256_float16")
    wrapper = tmp_path / ".work" / hadamard_packages.WRAPPER
    assert wrapper.read_text() == hadamard_packages.source(
        tmp_path, "n2hadamard_256_float16"
    )
    relative = ".work/" + hadamard_packages.WRAPPER
    assert list(config.include_patterns) == [relative]
    assert {key: list(value) for key, value in config.entry_points.items()} == {
        relative: ["n2hadamard_256_float16"]
    }
    assert {
        key: list(value) for key, value in dict(config.workgroup_size_rules).items()
    } == {relative: ["16", "1", "1"]}
    with pytest.raises(ValueError):
        hadamard_packages.HadamardPackageCache(tmp_path, tmp_path, "vulkan")


def descriptor_for(target, dtype, fault=None):
    bindings = []
    for name in layout.PARAMETERS:
        if fault == "unbound" and name == "in":
            continue
        bindings.append(
            {
                "name": f"binding_{name}",
                "kind": "constant" if name == "scale" else "buffer",
                "access": (
                    "read_write" if name == "out" and fault != "readonly" else "read"
                ),
                "provenance": {"sourceResource": {"parameter": name}},
                "scalarLayout": physical_layout(
                    target, "float32" if name == "scale" else dtype
                ),
            }
        )
    bindings.append(
        {
            "name": "dispatch_info",
            "provenance": {"executionInput": "workgroupCount"},
            "scalarLayout": {},
        }
    )
    if fault == "duplicate":
        bindings.append(dict(bindings[0], name="again"))
    return {"artifact": {"hash": "h"}, "bindings": bindings}


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (None, "guard", "missing", "shape", "trace", "unbound", "readonly", "duplicate"),
)
@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
def test_dispatch_commits_the_output_only_after_validation(
    tmp_path, monkeypatch, target, fault, dtype
):
    entry, elements = f"n2hadamard_4_{dtype}", 8
    named, memory = native_buffers(entry, elements, 0.5)
    descriptor = descriptor_for(target, dtype, fault)
    host = SimpleNamespace(
        target=target,
        hadamards=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append((inputs, execution, kwargs))
        return outputs

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    one = {"float32": 0x3F800000, "float16": 0x3C00, "bfloat16": 0x3F80}[dtype]

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        value = outputs["binding_out"]
        value["values"][:elements] = dispatch.encode_binding(
            dtype, [one] * elements, target, physical_layout(target, dtype), "buffer"
        )["values"]
        if fault == "guard":
            value["values"][-1] = value["values"][0]
        elif fault == "missing":
            outputs.clear()
        elif fault == "shape":
            value["shape"] = [1]
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "execute", execute)
    if fault == "trace":
        host.trace.mkdir()
    supplied = (runtime.Buffer * 3)(*(named[name] for name in layout.PARAMETERS))
    execution = layout.execution(entry, elements)
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    call = lambda: dispatch.dispatch(host, entry, supplied, 3, elements, launch)
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory["out"])
        return
    call()
    assert list(memory["out"]) == [one] * elements
    assert host.dispatch_count == 1
    ((inputs, launched, kwargs),) = requests
    assert launched == execution
    assert kwargs["expected_target"] == target
    assert set(kwargs["observe_inputs"]) == {"binding_in", "binding_scale"}
    assert "dispatch_info" not in inputs
    assert len(inputs["binding_in"]["values"]) == elements + dispatch.GUARD_COUNT
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == entry
    assert event["hadamardMetadata"]["transforms"] == 2
    assert event["hadamardMetadata"]["scale"] == 0.5


def test_dispatch_requires_the_cache_and_geometry(tmp_path):
    host = SimpleNamespace(target="metal", hadamards=None, dispatch_count=0)
    with pytest.raises(ValueError, match="pinned packages"):
        dispatch.dispatch(host, "n2hadamard_4_float32", [object()], 3, 8, None)


def test_runtime_routes_hadamard_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.divmods, host.hadamards = {}, None, None, None
    entry = "mhadamard_56_float32"
    assert host._entry_available(entry.encode()) == 0
    host.hadamards = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"n1hadamard_56_float32") == 0
    calls = []
    monkeypatch.setattr(
        runtime.hadamard_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    host.dispatch(entry, [], 3, 1, launch=None)
    assert calls == [entry]


def test_backend_mirrors_hadamard_cpp():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    for text in (
        "void Hadamard::eval_gpu(",
        "auto [n, m] = decompose_hadamard(size);",
        "for (n2 = 2; n2 * n2 < n; n2 *= 2) {",
        "const int read_width_m = (n == 1) ? 1 : ((n == 2 || m == 28) ? 2 : 4);",
        "const uint32_t width_m = std::min(n / read_width_m, 256);",
        "const float scale_n2 = (m == 1) ? scale_ : 1.0f;",
        "{1, uint32_t(elements / n), uint32_t(n2)},",
        "{uint32_t(n1 / max_radix_1), 1, 1}};",
        "{1, uint32_t(elements / n2), 1}, {uint32_t(n2 / max_radix_2), 1, 1}};",
        "{1, uint32_t(elements / m / read_width_m / width_m), 1}, {width_m, 1, 1}};",
        'launch_stage("n2" + name, n1 > 1 ? out : x, scale_n2,',
        'launch_stage("m" + name, n > 1 ? out : x, scale_m,',
        "dispatch_copy(in, out);",
        "out.copy_shared_buffer(in);",
    ):
        assert text in backend, text
    for name in layout.PARAMETERS:
        assert f'{{"{name}", ' in backend, name
    prepare = (HOST / "prepare.py").read_text(encoding="utf-8")
    assert '"Hadamard",' in prepare


def test_reference_uses_the_kronecker_product_and_checks_the_matrix():
    rows = ["++", "+-"]
    matrices = {12: "\n" + "\n".join(row * 6 for row in rows * 6) + "\n"}
    values = np.arange(8.0).reshape(2, 4)
    expected = np.array([[6.0, -2.0, -4.0, 0.0], [22.0, -2.0, -4.0, 0.0]])
    assert np.array_equal(
        hadamard_workloads.reference(np, matrices, values, 1.0), expected
    )
    with pytest.raises(ValueError, match="Hadamard matrix"):
        hadamard_workloads.reference(np, matrices, np.ones((1, 12)), 1.0)


def test_workload_bounds_scale_with_the_storage_type():
    source = np.ones((2, 16), dtype=np.float32)
    bounds = {
        dtype: hadamard_workloads.bound_for(np, source, dtype, 0.25)
        for dtype in hadamard_workloads.DTYPES
    }
    assert bounds["float32"] < bounds["float16"] < bounds["bfloat16"]
    assert bounds["float32"] == pytest.approx((4 + 1 + 3) * 2**-24 * 0.25 * 16)


@pytest.mark.parametrize("size", (2, 16, 256, 12, 12 * 8, 20 * 4, 28 * 16))
def test_workload_reference_matches_the_explicit_matrix(monkeypatch, size):
    n, m = layout.decompose(size)
    state = np.random.RandomState(size)
    # Any codelet: the reference must apply kron(codelet, H_n) exactly. A
    # power-of-two size has no codelet stage.
    codelet = state.normal(size=(m, m)) if m > 1 else np.ones((1, 1))
    monkeypatch.setattr(
        hadamard_workloads, "codelet_matrix", lambda np, matrices, m: codelet
    )
    values = state.normal(size=(3, size)).astype(np.float32)
    matrix = np.kron(codelet, hadamard_workloads.sylvester(np, n))
    expected = np.einsum("ij,...j->...i", matrix, values.astype(np.float64)) * 0.5
    observed = hadamard_workloads.reference(np, {}, values, 0.5)
    assert observed.dtype == np.float64 and observed.shape == values.shape
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_workload_reference_scales_to_the_largest_transform(monkeypatch):
    # 28 * 2^14: kron(h28, H_n) would hold 1.7 TB. A unit input selects one
    # column, a codelet entry times (-1)^popcount(c & d) of Sylvester's H_n.
    n, m = 2**14, 28
    codelet = np.arange(1.0, m * m + 1).reshape(m, m)
    monkeypatch.setattr(
        hadamard_workloads, "codelet_matrix", lambda np, matrices, m: codelet
    )
    b, d = 5, 12345
    values = np.zeros((1, m * n), dtype=np.float32)
    values[0, b * n + d] = 1.0
    observed = hadamard_workloads.reference(np, {}, values, 1.0).reshape(m, n)
    signs = np.array([1.0 - 2.0 * (bin(c & d).count("1") % 2) for c in range(n)])
    np.testing.assert_array_equal(observed, np.outer(codelet[:, b], signs))


def test_workload_reference_rejects_codelets_that_are_not_hadamard():
    with pytest.raises(ValueError, match="h12"):
        hadamard_workloads.codelet_matrix(np, {12: H12}, 12)


def test_workloads_cover_m_only_strided_and_batched_launches():
    cases = list(hadamard_workloads.cases())
    assert len({case["id"] for case in cases}) == len(cases)
    entries = Counter()
    for case in cases:
        entries.update(hadamard_workloads.launches(case))
    assert {entry.split("_")[0] for entry in entries} == {
        "n1hadamard",
        "n2hadamard",
        "mhadamard",
    }
    assert {
        case["layout"] for case in cases if case["shape"][-1] in layout.FACTORS
    } == {"contiguous", "donated"}
    batched = next(case for case in cases if case["id"] == "batched-2")
    assert hadamard_workloads.launches(batched) == {"n2hadamard_2_float32": 2}
    records = [
        {"id": case["id"], "maximumError": 0.0, "bound": 1.0, "dispatchCount": 1}
        for case in cases
    ]
    hadamard_workloads.validate(records, native=True)
    with pytest.raises(ValueError):
        hadamard_workloads.validate(records, native=False)
    with pytest.raises(ValueError):
        hadamard_workloads.validate(records[1:], native=True)
    records[0]["maximumError"] = 2.0
    with pytest.raises(ValueError):
        hadamard_workloads.validate(records, native=True)


def test_upstream_dispatches_cover_both_tests_and_the_workloads():
    counts = verify_upstream_hadamard.expected_dispatches()
    entries = set(verify_upstream_hadamard.package_entries())
    assert set(counts) == entries
    assert hadamard_packages.verification_entries() == sorted(entries)
    # test_hadamard's largest CPU comparison runs all three stages.
    assert counts["n1hadamard_458752_float32"] == 1 + 1
    # Every bfloat16 transform comes from test_hadamard's float16 sweep.
    assert {entry for entry in entries if entry.endswith("bfloat16")} == {
        entry
        for size in [2**k for k in range(1, 14)] + [28 * 2**k for k in range(1, 9)]
        for entry in layout.entries(size, "bfloat16")
    } | {
        entry
        for case in hadamard_workloads.cases()
        if case["dtype"] == "bfloat16"
        for entry in layout.entries(case["shape"][-1], "bfloat16")
    }
    # grad_vmap's 2^2..2^7 transforms: two vectors and three n * n batches,
    # on top of the sweep's two dtypes of x[::2].
    assert counts["n2hadamard_128_float32"] == 2 + 3 + 2
