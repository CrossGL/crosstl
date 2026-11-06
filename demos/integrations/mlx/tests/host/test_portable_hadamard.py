"""Hadamard transforms of the portable host: check every stage against
Hadamard::eval_gpu's plan, render the library hadamard.cpp generates from
the pinned headers, dispatch each stage's package in slices and audit the
retained uploads, guards and packages."""

import ctypes
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from demos.integrations.mlx.portable_host import (
    element_storage,
    hadamard_dispatch,
    hadamard_evidence,
    hadamard_layout,
    hadamard_packages,
    prepare,
    runtime,
    upstream_module_groups,
    verify_upstream_ops,
)

TARGETS = ("metal", "directx", "opengl")
HOST = Path(hadamard_layout.__file__).parent
layout = hadamard_layout


def matrix_text():
    """A common/hadamard.h stand-in: one m-by-m sign matrix per factor."""
    parts = []
    for m in layout.FACTORS:
        rows = "".join(
            "".join("+-"[(row * column) % 3 == 1] for column in range(m)) + "\n"
            for row in range(m)
        )
        parts.append(f'constexpr std::string_view h{m} = R"(\n{rows})";\n')
    return "".join(parts)


def test_entries_follow_decompose_hadamard_and_the_eval_gpu_split():
    assert layout.decompose(2) == (2, 1)
    assert layout.decompose(12) == (1, 12)
    assert layout.decompose(3 * 2**14) == (2**12, 12)
    assert layout.decompose(5 * 2**14) == (2**12, 20)
    assert layout.decompose(7 * 2**16) == (2**14, 28)
    # decompose_hadamard leaves a factor of three after 12, and no factor
    # divides 2 * 3; neither is a transform the kernels compute.
    for size in (1, 3, 36, 240, 28 * 2**27):
        with pytest.raises(ValueError):
            layout.decompose(size)
    assert layout.split(8192) == (1, 8192)
    assert layout.split(2**14) == (128, 128)
    assert layout.split(2**15) == (128, 256)
    assert layout.split(2**26) == (8192, 8192)
    assert layout.stages(16, "float32") == ("n2hadamard_16_float32",)
    assert layout.stages(12, "float16") == ("mhadamard_12_float16",)
    assert layout.stages(7 * 2**16, "float32") == (
        "n1hadamard_458752_float32",
        "n2hadamard_458752_float32",
        "mhadamard_458752_float32",
    )
    # 26 powers of two with one or two passes, and 27 sizes for each
    # factor with an m pass after them, in three types.
    assert len(layout.ENTRIES) == 3 * (26 + 13 + 3 * (1 + 2 * 13 + 3 * 13))
    for entry in (
        "hadamard_16_float32",
        "n2hadamard_12_float32",
        "mhadamard_16_float32",
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)


@pytest.mark.parametrize(
    ("entry", "arguments", "workgroup", "unit", "groups", "depth"),
    (
        ("n2hadamard_16_float32", ("float", 16, 16, 4), (1, 1, 1), 16, 1, 1),
        ("n2hadamard_2_bfloat16", ("bfloat16_t", 2, 2, 2), (1, 1, 1), 2, 1, 1),
        ("n2hadamard_1024_float16", ("float16_t", 1024, 16, 4), (64, 1, 1), 1024, 1, 1),
        (
            "n1hadamard_16384_float32",
            ("float", 128, 16, 4, 128),
            (8, 1, 1),
            16384,
            1,
            128,
        ),
        ("mhadamard_448_float32", ("float", 16, 28, 2), (8, 1, 1), 448, 1, 1),
        ("mhadamard_12_float32", ("float", 1, 12, 1), (1, 1, 1), 12, 1, 1),
        ("mhadamard_49152_float32", ("float", 4096, 12, 4), (256, 1, 1), 49152, 4, 1),
    ),
)
def test_stage_plans_are_hadamard_mn_contiguous(
    entry, arguments, workgroup, unit, groups, depth
):
    plan = layout.stage_plan(entry)
    assert plan["arguments"] == arguments
    assert plan["workgroupSize"] == workgroup
    assert (plan["unitElements"], plan["unitGroups"], plan["depth"]) == (
        unit,
        groups,
        depth,
    )


def test_plans_slices_and_scales_refuse_launches_outside_upstream():
    one = layout.UNIT_SCALE
    assert layout.plan("n2hadamard_16_float32", 64, 0x3EAE147B) == {
        "workgroupCount": [1, 4, 1],
        "workgroupSize": [1, 1, 1],
    }
    assert layout.plan("n1hadamard_16384_float32", 2 * 16384, one) == {
        "workgroupCount": [1, 2, 128],
        "workgroupSize": [8, 1, 1],
    }
    assert layout.plan("mhadamard_49152_float32", 3 * 49152, 0x3E800000) == {
        "workgroupCount": [1, 12, 1],
        "workgroupSize": [256, 1, 1],
    }
    with pytest.raises(ValueError, match="whole transforms"):
        layout.plan("n2hadamard_16_float32", 24, one)
    with pytest.raises(ValueError, match="unit scale"):
        layout.plan("n1hadamard_16384_float32", 16384, 0x3E800000)
    with pytest.raises(ValueError, match="unit scale"):
        layout.plan("n2hadamard_448_float32", 448, 0x3E800000)
    with pytest.raises(ValueError, match="group limit"):
        layout.plan("n2hadamard_16_float32", 16 * 65536, one)
    with pytest.raises(ValueError, match="float32 word"):
        layout.plan("n2hadamard_16_float32", 16, 1.0)
    # Slices hold whole rows (or whole batches of an m pass) within the
    # group and element limits, and each passes the plan.
    for entry, total, launches in (
        ("n2hadamard_16_float32", 16 * 70000, 2),
        # Wide m passes reach the element limit first: 5461 batches each.
        ("mhadamard_49152_float32", 49152 * 6000, 2),
        ("mhadamard_12_float16", 12 * 65536, 2),
    ):
        pieces = layout.slices(entry, total)
        assert len(pieces) == launches
        assert sum(count for _, count in pieces) == total
        assert [first for first, _ in pieces] == [0, pieces[0][1]]
        for _, count in pieces:
            layout.plan(entry, count, 0x3E800000 if entry[0] == "m" else one)
    # A batch of 28 * 2**26 points needs more groups than one axis carries.
    with pytest.raises(ValueError, match="group limit"):
        layout.slices("mhadamard_1879048192_float32", 28 * 2**26)


def test_codelet_and_library_are_the_ones_hadamard_cpp_generates():
    text = matrix_text()
    assert layout.codelet(1, text) == (
        "METAL_FUNC void hadamard_radix_m(thread float *x) {\n}\n"
    )
    lines = layout.codelet(12, text).split("\n")
    assert lines[:2] == [
        "METAL_FUNC void hadamard_radix_m(thread float *x) {",
        "  float tmp[12];",
    ]
    assert lines[2] == "  tmp[0] = " + "".join(f" + x[{i}]" for i in range(12)) + ";"
    assert lines[3].startswith("  tmp[1] =  + x[0] - x[1] + x[2] + x[3] - x[4]")
    assert lines[-3:] == ["  for (int i = 0; i < 12; i++) { x[i] = tmp[i]; }", "}", ""]
    source = layout.render("mhadamard_448_bfloat16", text)
    assert source.startswith(
        '#include "mlx/backend/metal/kernels/utils.h"\n'
        "METAL_FUNC void hadamard_radix_m(thread float *x) {\n  float tmp[28];\n"
    )
    assert source.endswith(
        '#include "mlx/backend/metal/kernels/hadamard.h"\n\n'
        'template [[host_name("mhadamard_448_bfloat16")]] [[kernel]] '
        "decltype(hadamard_m<bfloat16_t, 16, 28, 2>) "
        "hadamard_m<bfloat16_t, 16, 28, 2>;\n"
    )
    # Every library carries its size's codelet, as hadamard.cpp's does.
    assert "float tmp[28];" in layout.render("n2hadamard_448_float32", text)
    assert "float tmp" not in layout.render("n2hadamard_16_float32", text)
    for broken in (
        text.replace("h20", "h21"),
        text.replace("+-", "+*", 1),
        text.replace('\n)";', ')";', 1),
    ):
        with pytest.raises(ValueError):
            layout.codelet(12, broken)


def test_package_configs_translate_one_stage_at_its_threadgroup(tmp_path):
    for entry in ("n1hadamard_16384_float32", "mhadamard_49152_float32"):
        for target in TARGETS:
            config = hadamard_packages.project_config(tmp_path, "work", target, entry)
            assert list(config.include_patterns) == ["work/kernel.metal"]
            assert dict(config.entry_points) == {"work/kernel.metal": (entry,)}
            assert tuple(config.workgroup_size) == layout.workgroup_size(entry)
            assert not config.index_range_assertions
            assert not config.specialization_constants
    entries = hadamard_packages.verification_entries()
    assert entries == sorted(set(entries))
    assert {
        "n2hadamard_8192_bfloat16",
        "mhadamard_7168_float16",
        "n1hadamard_65536_float32",
        "mhadamard_458752_float32",
    } <= set(entries)
    # test_hadamard sends bfloat16 only where it sends float16.
    assert "n1hadamard_65536_bfloat16" not in entries
    with pytest.raises(ValueError):
        hadamard_packages.HadamardPackageCache(tmp_path, tmp_path, "vulkan")


class FakeCache:
    """The Hadamard cache: one package per stage."""

    def __init__(self, target):
        self.target = target
        self.requests = []

    def get(self, entry):
        self.requests.append(entry)
        # Direct3D prefixes the scalar with the entry; "in" and "out" are keywords.
        prefix = "" if self.target != "directx" else entry + "_"
        bindings = []
        for name, kind in layout.operand_types(entry).items():
            value = element_storage.packet(kind, self.target, [])
            scalar = {
                "elementType": value["dtype"],
                "elementStrideBytes": element_storage.PHYSICAL_SIZES[value["dtype"]],
            }
            if self.target == "directx" and kind == "float16":
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


def sample(dtype, count):
    """Finite, normal storage words of ``dtype``."""
    if dtype == "float32":
        return [0x3F000000 + 977 * index % 0x00800000 for index in range(count)]
    if dtype == "float16":
        return [0x3800 + 37 * index % 0x0400 for index in range(count)]
    return [0x3F00 + 13 * index % 0x0080 for index in range(count)]


def kernel(words):
    """A stand-in stage result: the slice's words reversed."""
    return list(reversed(words))


def native_buffers(entry, values, scale, *, alias=False):
    kind = layout.signature(entry)[2]
    ctype = element_storage.CTYPES[kind]
    memory = {"in": (ctype * len(values))(*values)}
    memory["out"] = memory["in"] if alias else (ctype * len(values))()
    memory["scale"] = (ctypes.c_uint32 * 1)(scale)
    types = layout.operand_types(entry)
    array = (runtime.Buffer * 3)(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == "out"),
            )
            for name in ("in", "out", "scale")
        )
    )
    return array, memory


def run(
    monkeypatch, tmp_path, target, entry, values, scale, *, alias=False, fault=None
):
    kind = layout.signature(entry)[2]
    launch = layout.plan(entry, len(values), scale)
    array, memory = native_buffers(entry, values, scale, alias=alias)
    host = SimpleNamespace(
        hadamards=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        assert (
            element_storage.unpack(kind, target, inputs["inBinding"]["values"])
            == values
        )
        initial = element_storage.unpack(kind, target, inputs["outBinding"]["values"])
        assert initial == element_storage.guard_words(kind, len(values) + 32)
        assert element_storage.unpack(
            "float32", target, inputs["scaleBinding"]["values"]
        ) == [scale]
        assert request["execution"] == launch
        result = kernel(values) + initial[len(values) :]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet(kind, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        hadamard_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(hadamard_dispatch, "execute", execute)
    native = runtime.Launch(
        (ctypes.c_uint32 * 3)(*launch["workgroupCount"]),
        (ctypes.c_uint32 * 3)(*launch["workgroupSize"]),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    hadamard_dispatch.dispatch(
        host, entry, array, len(array), layout.threads(launch), native
    )
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


def audit(monkeypatch, event, target):
    """Audit an event against fake packages, skipping native module checks."""
    checked = []
    monkeypatch.setattr(hadamard_evidence, "require_unchanged", lambda *_: None)
    monkeypatch.setattr(
        hadamard_evidence, "audit_native_execution", lambda e: checked.append(target)
    )
    event = {
        **event,
        "details": {
            "request": {
                "target": target,
                "entryPoint": (
                    {"opengl": "main", "directx": "CSMain"}.get(target, event["entry"])
                ),
                "dispatch": {key: event[key] for key in hadamard_evidence.GEOMETRY},
            }
        },
    }
    retained = hadamard_evidence.audit_event(event, FakeCache(target))
    assert checked == [target]
    return retained


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    ("entry", "count", "scale"),
    (
        ("n2hadamard_16_float32", 48, 0x3EAE147B),
        ("n1hadamard_16384_float16", 16384, layout.UNIT_SCALE),
        ("mhadamard_448_bfloat16", 896, 0x3E800000),
        ("mhadamard_12_float32", 36, layout.UNIT_SCALE),
    ),
)
def test_dispatch_returns_and_audits_each_stage(
    monkeypatch, tmp_path, target, entry, count, scale
):
    values = sample(layout.signature(entry)[2], count)
    out, event, host = run(monkeypatch, tmp_path, target, entry, values, scale)
    assert out == kernel(values)
    assert host.dispatch_count == 1 and host.hadamards.requests == [entry]
    assert event["scaleWord"] == scale and event["operandValues"] == {"in": values}
    assert audit(monkeypatch, event, target)


def test_dispatch_reads_an_aliased_input_before_writing_it(monkeypatch, tmp_path):
    values = sample("float32", 64)
    out, _, _ = run(
        monkeypatch,
        tmp_path,
        "opengl",
        "n2hadamard_16_float32",
        values,
        layout.UNIT_SCALE,
        alias=True,
    )
    assert out == kernel(values)


def test_dispatch_and_audit_refuse_changed_launches_and_results(monkeypatch, tmp_path):
    entry, values = "n2hadamard_16_float32", sample("float32", 32)
    with pytest.raises(RuntimeError, match="guard"):
        run(
            monkeypatch,
            tmp_path,
            "metal",
            entry,
            values,
            layout.UNIT_SCALE,
            fault="guard",
        )
    _, event, _ = run(monkeypatch, tmp_path, "metal", entry, values, layout.UNIT_SCALE)
    for changed in (
        {**event, "outputHash": "0" * 64},
        {**event, "scaleWord": layout.UNIT_SCALE + 1},
        {**event, "workgroupCount": [1, 3, 1]},
        {**event, "threads": event["threads"] + 1},
        {**event, "operandValues": {"in": values[:16]}},
    ):
        with pytest.raises(Exception):
            audit(monkeypatch, changed, "metal")
    array, _ = native_buffers(entry, values, layout.UNIT_SCALE)
    host = SimpleNamespace(hadamards=FakeCache("metal"), target="metal")
    wrong = runtime.Launch(
        (ctypes.c_uint32 * 3)(1, 1, 1),
        (ctypes.c_uint32 * 3)(1, 1, 1),
        (ctypes.c_uint32 * 3)(0, 0, 0),
    )
    with pytest.raises(ValueError, match="upstream plan"):
        hadamard_dispatch.dispatch(host, entry, array, 3, 2, wrong)
    with pytest.raises(ValueError, match="complete buffers"):
        hadamard_dispatch.dispatch(host, entry, array, 2, 2, wrong)


def test_dispatch_refuses_subnormal_bfloat_off_metal(monkeypatch, tmp_path):
    values = [0x0001] + sample("bfloat16", 15)
    with pytest.raises(ValueError, match="2000"):
        run(
            monkeypatch,
            tmp_path,
            "directx",
            "n2hadamard_16_bfloat16",
            values,
            layout.UNIT_SCALE,
        )
    out, _, _ = run(
        monkeypatch,
        tmp_path,
        "metal",
        "n2hadamard_16_bfloat16",
        values,
        layout.UNIT_SCALE,
    )
    assert out == kernel(values)


def test_large_slices_keep_extents_and_digests(monkeypatch, tmp_path):
    monkeypatch.setattr(hadamard_dispatch, "RETAINED_LIMIT", 16)
    monkeypatch.setattr(hadamard_evidence, "RETAINED_LIMIT", 16)
    values = sample("float16", 448)
    out, event, _ = run(
        monkeypatch, tmp_path, "directx", "mhadamard_448_float16", values, 0x3E800000
    )
    assert out == kernel(values)
    assert "compactedValues" in event["operandValues"]["in"]
    assert not audit(monkeypatch, event, "directx")


def test_runtime_host_port_and_upstream_inventory_route_hadamard():
    fake = SimpleNamespace(descriptors={}, hadamards=object())
    assert runtime.HostRuntime._entry_available(fake, b"mhadamard_448_float16") == 1
    fake.hadamards = None
    assert runtime.HostRuntime._entry_available(fake, b"n2hadamard_16_float32") == 0
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "if entry in hadamard_packages.ENTRIES:" in source
    port = (HOST / "hadamard.inc").read_text(encoding="utf-8")
    assert (
        "void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {"
        in port
    )
    for line in (
        "const int read_width_m = (n == 1) ? 1 : ((n == 2 || m == 28) ? 2 : 4);",
        "const float scale_n2 = (m == 1) ? scale : 1.0f;",
        "for (n2 = 2; n2 * n2 < n; n2 *= 2) {",
        "constexpr uint64_t hadamard_max_groups = 65535;",
        "constexpr uint64_t hadamard_max_elements = (uint64_t(1) << 28) - 1;",
        "constexpr int hadamard_max_threads = 256;",
        "hadamard_max_groups / stage.unit_groups, hadamard_max_elements / stage.unit);",
        "dispatch_copy(in, out);",
    ):
        assert line in port
    # The n1, n2 and m passes run in that order, each from the last output.
    assert (
        port.index('"n1" + name') < port.index('"n2" + name') < port.index('"m" + name')
    )
    assert (layout.MAX_GROUPS, layout.MAX_ELEMENTS, layout.MAX_THREADS) == (
        65535,
        2**28 - 1,
        256,
    )
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert (
        '#include "mlx/backend/no_gpu/crosstl_hadamard.inc"\n'
        "// The normalization ports use this translation unit's dispatch helpers.\n"
        '#include "mlx/backend/no_gpu/crosstl_normalization.inc"'
    ) in backend
    prepared = Path(prepare.__file__).read_text(encoding="utf-8")
    assert '"crosstl_hadamard.inc": (HERE / "hadamard.inc").read_bytes()' in prepared
    assert '        "FFT",\n        "Hadamard",\n' in prepared
    assert set(layout.PREFIXES) <= set(verify_upstream_ops.RETAINED_VALUES)
    assert upstream_module_groups.HADAMARD_TESTS == (
        "test_ops.TestOps.test_hadamard",
        "test_ops.TestOps.test_hadamard_grad_vmap",
        "test_ops.TestOps.test_hadamard_m_only",
    )
    assert (
        "test_ops.TestOps.test_hadamard_m_only" in verify_upstream_ops.METAL_PATH_TESTS
    )
    assert (
        verify_upstream_ops.GROUPS["fft"][-3:] == upstream_module_groups.HADAMARD_TESTS
    )
    assert verify_upstream_ops.CI_CLEARED_TESTS == {"test_ops.TestOps.test_hadamard"}


def test_ci_cleared_modules_import_without_ci(monkeypatch, tmp_path):
    (tmp_path / "ci_gated_probe.py").write_text(
        "import os, unittest\n"
        "class Case(unittest.TestCase):\n"
        "    @unittest.skipIf('CI' in os.environ, 'too slow in CI')\n"
        "    def test_slow(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CI", "1")
    name = "ci_gated_probe.Case.test_slow"
    monkeypatch.setattr(verify_upstream_ops, "CI_CLEARED_TESTS", frozenset((name,)))
    monkeypatch.delitem(sys.modules, "ci_gated_probe", raising=False)
    verify_upstream_ops.import_ci_cleared([name])
    assert os.environ["CI"] == "1"
    assert not getattr(
        sys.modules["ci_gated_probe"].Case.test_slow, "__unittest_skip__", False
    )
    # A module imported earlier with CI set is refused rather than run skipped.
    monkeypatch.delitem(sys.modules, "ci_gated_probe")
    __import__("ci_gated_probe")
    with pytest.raises(RuntimeError, match="imported with CI set"):
        verify_upstream_ops.import_ci_cleared([name])
    monkeypatch.delitem(sys.modules, "ci_gated_probe")


def test_the_fft_jobs_translate_the_upstream_hadamard_stages_first():
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["upstream-ops-host"]
    names = [step.get("name") for step in job["steps"]]
    step = job["steps"][names.index("Translate the upstream Hadamard stages")]
    # Only the fft group runs the Hadamard tests.
    assert step["if"] == "matrix.group == 'fft'"
    assert [
        group
        for group, tests in verify_upstream_ops.GROUPS.items()
        if set(tests) & set(upstream_module_groups.HADAMARD_TESTS)
    ] == ["fft"]
    assert names.index(step["name"]) < names.index(
        "Execute unchanged upstream op tests"
    )
    run = step["run"]
    assert "set -euo pipefail" in run
    assert "portable_host.hadamard_packages" in run
    assert "--verification" in run
    tests = job["steps"][names.index("Execute unchanged upstream op tests")]["run"]
    packages = re.search(r"--packages (\S+)", run).group(1)
    assert f"--packages {packages} " in tests
    deadline = int(re.search(r"--timeout-seconds (\d+)", run).group(1))
    budget = float(re.search(r"--budget-seconds (\d+)", run).group(1))
    # Translations started inside the budget finish before the deadline.
    assert deadline <= 600 and 0 < budget <= deadline - 120


def test_warming_defers_the_stages_outside_its_budget(tmp_path):
    entries = hadamard_packages.verification_entries()
    with pytest.raises(ValueError, match="positive budget"):
        hadamard_packages.warm_entries(tmp_path, tmp_path, "metal", entries, 2, 0)
    with pytest.raises(ValueError, match="jobs"):
        hadamard_packages.warm_entries(tmp_path, tmp_path, "metal", entries, 0, 60)
    # A budget that has run out before the first submission starts nothing.
    records, deferred = hadamard_packages.warm_entries(
        tmp_path, tmp_path, "metal", entries, 2, 1e-9
    )
    assert records == [] and deferred == sorted(set(entries))
    assert not (tmp_path / hadamard_packages.CACHE).exists()
