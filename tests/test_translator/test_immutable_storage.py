"""Checked immutable read-only storage in software-subgroup uniformity proofs.

``const device uint* flags`` may alias a writable binding of the same dispatch,
so a value read from it at a workgroup-uniform index is not uniform by type.
A project contract ``{entryPoint, parameter}`` states that no writable binding
reaches the named parameter. Only then do the DirectX and OpenGL software
subgroup proofs treat its reads as immutable storage, which lets a kernel return
early on a per-workgroup flag before a barrier. The artifact records the
contract, reflection marks the binding, C wrappers refuse it, and every native
dispatch path rejects a writable view that overlaps the named storage.

Immutable storage may also move its view in place, as ``flags += group.y *
stride``. Offsets by uniform amounts outside loops keep the view uniform. A
kernel may pass the whole view to helpers that receive it as read-only storage
and themselves only read, offset or forward it.
"""

import copy
import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    inspect_runtime_package,
    translate_project,
)
from crosstl.project.immutable_storage import check_native_immutable_snapshots
from crosstl.project.native_loader_abi import (
    NativeLoaderABIError,
    generate_native_loader_execution_abi,
)
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.runtime_verification import (
    NativeRuntimeParityAdapter,
    RuntimeAllocationView,
    RuntimeExecutionState,
    RuntimeValue,
)
from crosstl.translator.immutable_storage import (
    parse_immutable_storage,
    parse_immutable_storage_header,
    with_immutable_storage_header,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_IMMUTABLE_STORAGE"
ENTRY = "masked_sums"
WIDTH = 32
THREADS = 128
GROUPS = (3, 2)
STRIDE = 4
CONTRACT = [{"entryPoint": ENTRY, "parameter": "flags"}]
FLAGS = [5, 0, 7, 0xDEAD, 0, 9, 3, 0xBEEF]
VALUES = [(index * 2654435761) % 1013 for index in range(GROUPS[0] * GROUPS[1] * 128)]
GUARD = [0x6D2B79F5, 0x1234567, 0x5A5A5A5]
OFFSET = "  flags += group.y * uint(stride);\n"

TEMPLATE = """#include <metal_stdlib>
using namespace metal;
%(helpers)s
[[kernel]] void masked_sums(
    const device uint* flags [[buffer(0)]],
    const device uint* values [[buffer(1)]],
    device uint* results [[buffer(2)]],
    constant int& stride [[buffer(3)]],
    const device uint* other [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
%(prefix)s  uint flag = %(read)s;
  uint base = (group.y * 3u + group.x) * 128u;
  if (flag == 0u) {
    results[base + lid] = 0u;
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  results[base + lid] = simd_sum(values[base + lid]) + flag + other[0];
}
"""


def _source(prefix=OFFSET, read="flags[group.x]", helpers=""):
    return TEMPLATE % {"prefix": prefix, "read": read, "helpers": helpers}


READ_HELPER = """
uint flag_at(const device uint* view, uint index) {
  return view[index];
}
"""
# The view reaches the read through two helpers.
FORWARDING_HELPERS = READ_HELPER + """
uint forwarded_flag(const device uint* view, uint index) {
  return flag_at(view, index);
}
"""
HELPERS = {
    "read": (READ_HELPER, "flag_at(flags, group.x)"),
    "forwarded": (FORWARDING_HELPERS, "forwarded_flag(flags, group.x)"),
}


SOURCE = _source()


def _expected():
    results = []
    for gy in range(GROUPS[1]):
        for gx in range(GROUPS[0]):
            flag = FLAGS[gy * STRIDE + gx]
            base = (gy * GROUPS[0] + gx) * THREADS
            for lane in range(THREADS):
                first = base + lane // WIDTH * WIDTH
                total = sum(VALUES[first : first + WIDTH])
                results.append(0 if flag == 0 else (total + flag + 1) % (1 << 32))
    return results + GUARD


def _translate(root, target, source, contract=CONTRACT):
    root.mkdir(parents=True, exist_ok=True)
    (root / "masked.metal").write_text(source, encoding="utf-8")
    options = {"preserve_resource_origins": True}
    if target != "metal":
        target_options = {"software_subgroup_width": WIDTH}
        if target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        options["target_options"] = {target: target_options}
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("masked.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(THREADS, 1, 1),
            immutable_storage=contract,
            source_options={"metal": options},
        ),
        format_output=False,
    )
    errors = [
        item for item in report.to_json()["diagnostics"] if item["severity"] == "error"
    ]
    return report, errors


def _package(root, target, source=SOURCE):
    report, errors = _translate(root, target, source)
    assert not errors, errors
    report.write_json(root / "report.json")
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "manifest.json").write_text(json.dumps(manifest))
    package = root / "package"
    result = build_runtime_package(root / "manifest.json", package)
    assert result["success"], result
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    return descriptor, package


def _payload(data):
    return {"dtype": "uint32", "shape": [len(data)], "values": data}


def _values(descriptor, *, allocations=None):
    expected = _expected()
    initial = [0xA5A5A5A5] * (len(expected) - len(GUARD)) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "flags": _payload(FLAGS),
            "values": _payload(VALUES),
            "results": _payload(initial),
            "other": _payload([1, 0, 0, 0]),
        },
    )
    (stride,) = (
        binding["name"]
        for binding in descriptor["bindings"]
        if binding["scalarLayout"].get("memberName", binding["name"]).endswith("stride")
    )
    inputs[stride] = {"dtype": "int32", "shape": [1], "values": [STRIDE]}
    outputs = _bound_values(descriptor, {"results": _payload(expected)})
    for name, view in (allocations or {}).items():
        (bound,) = _bound_values(descriptor, {name: None})
        inputs[bound] = RuntimeValue(name=bound, **inputs[bound], allocation=view)
        if bound in outputs:
            outputs[bound] = RuntimeValue(name=bound, **outputs[bound], allocation=view)
    return inputs, outputs


def _request(descriptor, package, target, *, allocations=None):
    inputs, outputs = _values(descriptor, allocations=allocations)
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [*GROUPS, 1], "workgroupSize": [THREADS, 1, 1]},
        observe_inputs=sorted(_bound_values(descriptor, {"flags": 0})),
        expected_target=target,
    )


def _flags_binding(descriptor):
    return next(
        item
        for item in descriptor["bindings"]
        if "immutableStorage" in item.get("provenance", {})
    )


def test_reference_skips_masked_groups():
    expected = _expected()
    assert expected[THREADS : 2 * THREADS] == [0] * THREADS
    assert expected[0] != 0 and expected[-len(GUARD) :] == GUARD


@pytest.mark.parametrize(
    "record",
    [
        {"entryPoint": ENTRY},
        {"parameter": "flags"},
        {"entryPoint": ENTRY, "parameter": "flags", "access": "read"},
        {"entryPoint": "masked-sums", "parameter": "flags"},
        {"entryPoint": ENTRY, "parameter": "flags", "source": " *"},
    ],
)
def test_contract_records_reject_unknown_or_malformed_fields(record):
    with pytest.raises(ValueError):
        parse_immutable_storage([record])


def test_contract_records_reject_duplicates():
    with pytest.raises(ValueError, match="more than once"):
        parse_immutable_storage(CONTRACT * 2)


def test_header_round_trips_and_rejects_tampering():
    record = {
        "sourceResource": {
            "schemaVersion": 1,
            "backend": "metal",
            "entryPoint": ENTRY,
            "parameter": "flags",
            "parameterIndex": 0,
        }
    }
    source = with_immutable_storage_header("void main() {}\n", [record])
    assert parse_immutable_storage_header(source) == [record]
    with pytest.raises(ValueError):
        with_immutable_storage_header(source, [record])
    with pytest.raises(ValueError):
        parse_immutable_storage_header(source.replace('"schemaVersion":1', '"x":1', 1))
    with pytest.raises(ValueError):
        parse_immutable_storage_header(source + "// crosstl-immutable-storage: {}\n")
    with pytest.raises(ValueError):
        with_immutable_storage_header("", [record, record])


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("prefix", ("", OFFSET), ids=("direct", "offset"))
def test_read_only_flags_are_refused_without_the_contract(tmp_path, target, prefix):
    report, errors = _translate(tmp_path, target, _source(prefix), contract=())
    assert report.to_json()["summary"]["translatedCount"] == 0
    assert [item["code"] for item in errors][:1] == [
        f"project.translate.{target}-software-subgroup-invalid"
    ], errors


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("prefix", ("", OFFSET), ids=("direct", "offset"))
def test_contract_proves_masked_returns_and_compiles(tmp_path, target, prefix):
    report, errors = _translate(tmp_path, target, _source(prefix))
    assert not errors, errors
    (artifact,) = report.to_json()["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text()
    (record,) = parse_immutable_storage_header(generated)
    assert record["sourceResource"]["parameter"] == "flags"
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("helper", sorted(HELPERS))
def test_contract_follows_views_into_read_only_helpers(tmp_path, target, helper):
    helpers, read = HELPERS[helper]
    source = _source("", read, helpers)
    report, errors = _translate(tmp_path / "plain", target, source, contract=())
    assert report.to_json()["summary"]["translatedCount"] == 0
    assert SUBGROUP_REFUSAL in errors[0]["code"], errors
    report, errors = _translate(tmp_path / "contract", target, source)
    assert not errors, errors
    (artifact,) = report.to_json()["artifacts"]
    generated = (tmp_path / "contract" / artifact["path"]).read_text()
    (record,) = parse_immutable_storage_header(generated)
    assert record["sourceResource"]["parameter"] == "flags"
    _compile(generated, target, tmp_path)


def test_metal_artifact_records_the_contract(tmp_path):
    report, errors = _translate(tmp_path, "metal", SOURCE)
    assert not errors, errors
    (artifact,) = report.to_json()["artifacts"]
    generated = (tmp_path / artifact["path"]).read_text()
    assert len(parse_immutable_storage_header(generated)) == 1
    _compile(generated, "metal", tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


SUBGROUP_REFUSAL = "software-subgroup-invalid"
CONTRACT_REFUSAL = "project.translate.immutable-storage-unsupported"


NARROW_TEMPLATE = """#include <metal_stdlib>
using namespace metal;

[[kernel]] void masked_sums(
    const device %(element)s* flags [[buffer(0)]],
    const device uint* values [[buffer(1)]],
    device uint* results [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  %(element)s flag = flags[group.x];
  uint base = group.x * 128u;
  if (flag == %(element)s(0)) {
    results[base + lid] = 0u;
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  results[base + lid] = simd_sum(values[base + lid]) + uint(flag);
}
"""


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("element", ("half", "bfloat"))
def test_contract_covers_16_bit_floating_storage(tmp_path, target, element):
    """16-bit floating elements are values, so their storage can be immutable."""
    source = NARROW_TEMPLATE % {"element": element}
    report, errors = _translate(tmp_path / "plain", target, source, contract=())
    assert report.to_json()["summary"]["translatedCount"] == 0
    assert SUBGROUP_REFUSAL in errors[0]["code"], errors
    report, errors = _translate(tmp_path / "contract", target, source)
    assert not errors, errors
    (artifact,) = report.to_json()["artifacts"]
    generated = (tmp_path / "contract" / artifact["path"]).read_text()
    (record,) = parse_immutable_storage_header(generated)
    assert record["sourceResource"]["parameter"] == "flags"
    _compile(generated, target, tmp_path)


UNSAFE_HELPERS = """
uint aliased_flag(const device uint* view, uint index) {
  const device uint* alias = view;
  return alias[index];
}

uint cleared_flag(const device uint* view, uint index) {
  ((device uint*)view)[index + 1u] = 0u;
  return view[index];
}

uint writable_flag(device uint* view, uint index) {
  return view[index];
}
"""


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "prefix,read,contract,expected",
    [
        # A lane-varying index reads different elements in one subgroup.
        ("", "flags[lid]", CONTRACT, SUBGROUP_REFUSAL),
        # A lane-varying offset moves each invocation's view differently.
        ("  flags += lid;\n", "flags[group.x]", CONTRACT, SUBGROUP_REFUSAL),
        # Offsets inside loops are not followed.
        (
            "  for (int i = 0; i < 2; i++) {\n    flags += uint(stride);\n  }\n",
            "flags[group.x]",
            CONTRACT,
            SUBGROUP_REFUSAL,
        ),
        # The contract covers only the parameters it names.
        (
            "",
            "other[group.x]",
            CONTRACT,
            SUBGROUP_REFUSAL,
        ),
        # Copies and casts could become writable aliases, in the entry or in
        # a helper that receives the view.
        (
            "  const device uint* alias = flags;\n",
            "alias[group.x]",
            CONTRACT,
            CONTRACT_REFUSAL,
        ),
        (
            "",
            "aliased_flag(flags, group.x)",
            CONTRACT,
            CONTRACT_REFUSAL,
        ),
        (
            "",
            "cleared_flag(flags, group.x)",
            CONTRACT,
            CONTRACT_REFUSAL,
        ),
        # A helper parameter that is not read-only storage could write it.
        (
            "",
            "writable_flag(flags, group.x)",
            CONTRACT,
            CONTRACT_REFUSAL,
        ),
        (
            "  ((device uint*)flags)[lid] = 0u;\n",
            "flags[group.x]",
            CONTRACT,
            CONTRACT_REFUSAL,
        ),
        # Writable storage cannot be immutable.
        (
            "",
            "results[group.x]",
            [{"entryPoint": ENTRY, "parameter": "results"}],
            CONTRACT_REFUSAL,
        ),
        # A contract must name a real storage parameter of the entry.
        (
            "",
            "flags[group.x]",
            [{"entryPoint": ENTRY, "parameter": "missing"}],
            CONTRACT_REFUSAL,
        ),
        (
            "",
            "flags[group.x]",
            [{"entryPoint": ENTRY, "parameter": "stride"}],
            CONTRACT_REFUSAL,
        ),
    ],
    ids=(
        "lane-index",
        "lane-offset",
        "loop-offset",
        "unnamed-parameter",
        "copied-alias",
        "helper-alias",
        "helper-cast-write",
        "helper-writable-parameter",
        "cast-write",
        "writable-parameter",
        "missing-parameter",
        "constant-scalar",
    ),
)
def test_contract_keeps_unsafe_reads_refused(
    tmp_path, target, prefix, read, contract, expected
):
    source = _source(prefix, read, UNSAFE_HELPERS)
    report, errors = _translate(tmp_path, target, source, contract)
    assert report.to_json()["summary"]["translatedCount"] == 0
    assert errors and expected in errors[0]["code"], errors


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_package_records_the_contract_and_c_wrappers_refuse_it(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    binding = _flags_binding(descriptor)
    assert binding["access"] == "read"
    assert binding["provenance"]["immutableStorage"]["sourceResource"] == (
        binding["provenance"]["sourceResource"]
    )
    assert inspect_runtime_package(package / "runtime-package.json")["success"]
    with pytest.raises(NativeLoaderABIError, match="immutable storage"):
        generate_native_loader_execution_abi(descriptor)
    request = _request(descriptor, package, target)
    assert not request.execution_plan.diagnostics


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_dispatch_rejects_writable_views_of_immutable_storage(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    whole = RuntimeAllocationView("arena", 0, None, 1 << 20)
    with pytest.raises(NativeLoaderDispatchError, match="writable binding"):
        _request(
            descriptor,
            package,
            target,
            allocations={"flags": whole, "results": whole},
        )
    flags = RuntimeAllocationView("arena", 0, 4 * len(FLAGS), 1 << 20)
    results = RuntimeAllocationView("arena", 16, 4 * len(_expected()), 1 << 20)
    with pytest.raises(NativeLoaderDispatchError, match="writable binding"):
        _request(
            descriptor,
            package,
            target,
            allocations={"flags": flags, "results": results},
        )
    # Disjoint ranges of one allocation, and shared read-only views, are safe.
    disjoint = RuntimeAllocationView("arena", 4096, 4 * len(_expected()), 1 << 20)
    shared = RuntimeAllocationView("inputs", 0, None, 1 << 20)
    request = _request(
        descriptor,
        package,
        target,
        allocations={"flags": flags, "results": disjoint},
    )
    assert not request.execution_plan.diagnostics
    _request(
        descriptor,
        package,
        target,
        allocations={"flags": shared, "other": shared},
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_native_buffers_reject_writable_views_of_immutable_storage(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    request = _request(descriptor, package, target)
    adapter = NativeRuntimeParityAdapter()
    adapter.target = target
    buffers = adapter._native_buffer_bindings(
        RuntimeExecutionState(request=request, plan=request.execution_plan)
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    (receipt,) = check_native_immutable_snapshots(artifact, target, buffers)
    assert receipt["binding"] == _flags_binding(descriptor)["name"]
    flags = buffers[receipt["binding"]]
    view = RuntimeAllocationView("arena", 0, None, 1 << 20)
    aliased = {
        name: (
            replace(item, allocation=view)
            if item is flags or item.expected_output is not None
            else item
        )
        for name, item in buffers.items()
    }
    with pytest.raises(ValueError, match="writable binding"):
        check_native_immutable_snapshots(artifact, target, aliased)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("fault", ("drop", "move"))
def test_descriptor_cannot_change_the_contract(tmp_path, target, fault):
    descriptor, package = _package(tmp_path, target)
    changed = copy.deepcopy(descriptor)
    binding = _flags_binding(changed)
    record = binding["provenance"].pop("immutableStorage")
    if fault == "move":
        other = next(
            item
            for item in changed["bindings"]
            if item is not binding and "sourceResource" in item.get("provenance", {})
        )
        other["provenance"]["immutableStorage"] = record
    with pytest.raises(NativeLoaderDispatchError, match="(?i)immutable storage"):
        _request(changed, package, target)


@pytest.mark.parametrize("prefix", ("", OFFSET), ids=("direct", "offset"))
def test_immutable_storage_executes_natively(tmp_path, prefix):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native immutable storage checks")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = _source(prefix)
    if not prefix:
        # Without an offset the second row of groups reads the first row's flags.
        source = source.replace(
            "flags[group.x]", "flags[group.y * uint(stride) + group.x]"
        )
    descriptor, package = _package(tmp_path, target, source)
    request = _request(descriptor, package, target)
    _, expected = _values(descriptor)
    work = tmp_path / "native"
    work.mkdir()
    _execute(
        request,
        expected,
        work,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )
    evidence = json.loads((work / "evidence.json").read_text())
    details = evidence["records"]["generated"]["details"]
    (receipt,) = details["immutableStorage"]
    assert receipt["binding"] == _flags_binding(descriptor)["name"]
    assert all(row["unchanged"] for row in details["inputObservations"].values())


def test_immutable_storage_is_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_immutable_storage.py" in step
    assert "if:" not in step and "continue-on-error" not in step
