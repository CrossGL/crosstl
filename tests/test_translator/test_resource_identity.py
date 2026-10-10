"""Resource provenance survives canonical text, target renaming and packages."""

import copy
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from crosstl import translate
from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    inspect_runtime_package,
    reflect_target_host_interface,
    translate_project,
)
from crosstl.translator.ast import AttributeNode, LiteralNode, PrimitiveType
from crosstl.translator.resource_identity import (
    MAX_RESOURCE_IDENTITIES,
    MAX_RESOURCE_IDENTITY_BYTES,
    RESOURCE_IDENTITY_PREFIX,
    apply_resource_identities,
    parse_resource_identities,
    resource_identity_marker,
    source_resource_identity,
)

SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Params { uint offset; uint count; };
kernel void gather_values(
    const device uint* input [[buffer(0)]],
    constant Params& params [[buffer(1)]],
    constant uint& bias [[buffer(2)]],
    device uint* output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.count) {
        output[index] = input[index + params.offset] + bias;
    }
}
"""
TARGETS = ("directx", "opengl", "metal")


def _node():
    return SimpleNamespace(
        attributes=[
            AttributeNode(
                "source_resource",
                [
                    LiteralNode("metal", PrimitiveType("string")),
                    LiteralNode("gather_values", PrimitiveType("string")),
                    LiteralNode("input", PrimitiveType("string")),
                    LiteralNode(0, PrimitiveType("int")),
                ],
            )
        ]
    )


def _assert_origins(resources, *, descriptor=False):
    origins = [
        (item["provenance"] if descriptor else item["metadata"]["provenance"])[
            "sourceResource"
        ]
        for item in resources
    ]
    assert sorted(origins, key=lambda item: item["parameterIndex"]) == [
        dict(
            schemaVersion=1,
            backend="metal",
            entryPoint="gather_values",
            parameter=name,
            parameterIndex=index,
        )
        for index, name in enumerate(("input", "params", "bias", "output"))
    ]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("saved_crossgl", (False, True))
@pytest.mark.parametrize("formatted", (False, True))
def test_translation_preserves_original_resource_names(
    tmp_path, target, saved_crossgl, formatted
):
    path = tmp_path / "gather.metal"
    path.write_text(SOURCE)
    options = {"preserve_resource_origins": True}
    baseline = translate(path, backend=target, format_output=formatted)
    if saved_crossgl:
        canonical = translate(path, format_output=False, source_options=options)
        assert canonical.count("@source_resource(") == 4
        path = tmp_path / "gather.cgl"
        path.write_text(canonical)
    generated = translate(
        path, backend=target, source_options=options, format_output=formatted
    )
    assert (
        "".join(
            line
            for line in generated.splitlines(True)
            if not line.lstrip().startswith(RESOURCE_IDENTITY_PREFIX)
        )
        == baseline
    )
    artifact = (
        tmp_path
        / {
            "directx": "gather.hlsl",
            "opengl": "gather.comp",
            "metal": "roundtrip.metal",
        }[target]
    )
    artifact.write_text(generated)
    reflected = reflect_target_host_interface(artifact, target=target, stage="compute")
    assert reflected["status"] == "ready", reflected
    _assert_origins(reflected["resources"])
    assert len(parse_resource_identities(generated)) == 4
    if target != "metal":
        assert any(
            item["name"]
            != item["metadata"]["provenance"]["sourceResource"]["parameter"]
            for item in reflected["resources"]
        )


def _package(tmp_path, target, source=SOURCE):
    (tmp_path / "gather.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("*.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
            source_options={"metal": {"preserve_resource_origins": True}},
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    assert report.to_json()["summary"]["translatedCount"] == 1, report.to_json()
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest))
    package = tmp_path / "package"
    result = build_runtime_package(tmp_path / "artifacts.json", package)
    assert result["success"], result
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    _assert_origins(loader["loadUnits"][0]["hostInterface"]["resources"])
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    _assert_origins(descriptor["bindings"], descriptor=True)
    return descriptor, package


@pytest.mark.parametrize("target", TARGETS)
def test_project_package_and_loader_retain_resource_origins(tmp_path, target):
    _package(tmp_path, target)


@pytest.mark.parametrize("fault", ("declared-origin", "missing-marker", "bad-marker"))
def test_package_rejects_changed_resource_identity(tmp_path, fault):
    descriptor, package = _package(tmp_path, "directx")
    path = package / "runtime-package.json"
    payload = json.loads(path.read_text())
    artifact = payload["artifacts"][0]
    if fault == "declared-origin":
        artifact["hostInterface"]["resources"][0]["metadata"]["provenance"][
            "sourceResource"
        ]["parameter"] = "wrong"
    else:
        shader = package / descriptor["artifact"]["packagePath"]
        source = shader.read_text()
        if fault == "missing-marker":
            source = "\n".join(
                line
                for line in source.splitlines()
                if not line.startswith(RESOURCE_IDENTITY_PREFIX)
            )
        else:
            source = source.replace('"schemaVersion":1', '"schemaVersion":999', 1)
        shader.write_text(source)
        artifact["hash"]["value"] = hashlib.sha256(shader.read_bytes()).hexdigest()
        artifact["sizeBytes"] = shader.stat().st_size
    path.write_text(json.dumps(payload))
    result = inspect_runtime_package(path)
    assert not result["success"], result
    assert any(
        item["code"] == "project.runtime-package-inspection.resource-identity-invalid"
        for item in result["diagnostics"]
    )
    assert not build_runtime_loader_manifest(path)["success"]


def test_resource_origins_select_native_inputs(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_RESOURCE_IDENTITY") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_RESOURCE_IDENTITY=1 for native resource identity checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = SOURCE.replace(
        "constant Params& params", "const device Params* params"
    ).replace("params.", "params[0].")
    descriptor, package = _package(tmp_path, target, source)
    values = {
        "input": [90, 5, 19, 31, 900],
        "params": [1, 3],
        "bias": [7],
        "output": [0x12345678] * 5,
    }
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        name = binding["provenance"]["sourceResource"]["parameter"]
        words = values[name]
        inputs[binding["name"]] = dict(dtype="uint32", shape=[len(words)], values=words)
        if name == "output":
            outputs[binding["name"]] = dict(
                dtype="uint32", shape=[5], values=[12, 26, 38, 0x12345678, 0x12345678]
            )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [5, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    _execute(
        request,
        outputs,
        tmp_path,
        validate=_validate,
        original_source=source if target == "metal" else None,
        original_entry="gather_values",
    )


def test_resource_identity_ci_is_required():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    assert 'CROSTL_REQUIRE_RESOURCE_IDENTITY: "1"' in workflow
    assert "tests/test_translator/test_resource_identity.py" in workflow


def test_formatter_preserves_resource_metadata_comments(monkeypatch):
    from crosstl.formatter import CodeFormatter, ShaderLanguage

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("crosstl.formatter.subprocess.run", run)
    formatter = CodeFormatter(clang_format_path="clang-format")
    source = resource_identity_marker(_node(), "input")
    assert formatter.format_code(source, language=ShaderLanguage.HLSL) == source
    style = json.loads(
        next(arg[len("-style=") :] for arg in commands[0] if arg.startswith("-style="))
    )
    assert style == {
        "BasedOnStyle": "Microsoft",
        "CommentPragmas": (
            "^ (IWYU pragma:|crosstl-resource-origin:|crosstl-frozen-specializations:)"
        ),
    }


def test_identity_is_opt_in(tmp_path):
    path = tmp_path / "gather.metal"
    path.write_text(SOURCE)
    assert "source_resource" not in translate(path, format_output=False)
    assert RESOURCE_IDENTITY_PREFIX not in translate(
        path, backend="opengl", format_output=False
    )


@pytest.mark.parametrize("value", (1, "true", None, []))
def test_identity_option_requires_boolean(tmp_path, value):
    path = tmp_path / "gather.metal"
    path.write_text(SOURCE)
    with pytest.raises(ValueError, match="preserve_resource_origins must be a boolean"):
        translate(path, source_options={"preserve_resource_origins": value})


@pytest.mark.parametrize(
    "fault",
    ("version", "unknown", "duplicate-key", "bad-index", "missing-target", "bad-json"),
)
def test_invalid_marker_fails_reflection(tmp_path, fault):
    record = parse_resource_identities(resource_identity_marker(_node(), "input"))[0]
    if fault == "version":
        record["schemaVersion"] = True
    elif fault == "unknown":
        record["bounds"] = [0, 100]
    elif fault == "bad-index":
        record["source"]["parameterIndex"] = True
    elif fault == "missing-target":
        record["target"]["name"] = "absent"
    payload = json.dumps(record)
    if fault == "duplicate-key":
        payload = payload.replace(
            '"schemaVersion": 1', '"schemaVersion": 1, "schemaVersion": 1'
        )
    elif fault == "bad-json":
        payload = "{"
    path = tmp_path / "invalid.hlsl"
    path.write_text(
        RESOURCE_IDENTITY_PREFIX
        + payload
        + "\nStructuredBuffer<uint> input : register(t0);\n[numthreads(1,1,1)] void main() {}\n"
    )
    record = reflect_target_host_interface(path, target="directx")
    assert record["status"] == "failed", record
    assert (
        record["diagnosticRecords"][0]["details"]["contract"]
        == "source-resource-identity"
    )


@pytest.mark.parametrize("wrapper", ("/*\n%s*/", 'const char* s = "%s";', "uint x; %s"))
def test_marker_must_be_standalone_comment(wrapper):
    marker = resource_identity_marker(_node(), "input")
    with pytest.raises(ValueError):
        parse_resource_identities(wrapper % marker)


def test_duplicate_or_missing_identity_is_atomic():
    marker = resource_identity_marker(_node(), "input")
    record = parse_resource_identities(marker)[0]
    resources = [dict(name="input"), dict(name="output")]
    changed = copy.deepcopy(record)
    changed["target"]["name"] = "output"
    for records in ([record, record], [record, changed]):
        with pytest.raises(ValueError, match="Duplicate or ambiguous"):
            apply_resource_identities(resources, records, [])
        assert resources == [dict(name="input"), dict(name="output")]
    changed["source"]["parameter"] = "output"
    changed["source"]["parameterIndex"] = 1
    changed["target"]["name"] = "absent"
    with pytest.raises(ValueError, match="missing or ambiguous"):
        apply_resource_identities(resources, [record, changed], [])
    assert resources == [dict(name="input"), dict(name="output")]


def test_attribute_rejects_ambiguous_origins():
    node = _node()
    node.attributes *= 2
    with pytest.raises(ValueError, match="one source_resource"):
        source_resource_identity(node)


@pytest.mark.parametrize("field", ("parameter", "parameterIndex"))
def test_source_parameter_names_and_indices_cannot_disagree(field):
    record = parse_resource_identities(resource_identity_marker(_node(), "input"))[0]
    changed = copy.deepcopy(record)
    changed["target"]["name"] = "output"
    changed["source"][field] = "output" if field == "parameter" else 1
    with pytest.raises(ValueError, match="Duplicate or ambiguous"):
        apply_resource_identities(
            [dict(name="input"), dict(name="output")], [record, changed], []
        )


def test_marker_metadata_is_bounded():
    marker = resource_identity_marker(_node(), "input")
    with pytest.raises(ValueError, match="Too many"):
        parse_resource_identities(marker * (MAX_RESOURCE_IDENTITIES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        parse_resource_identities(marker.rstrip() + " " * MAX_RESOURCE_IDENTITY_BYTES)


def test_metal_target_entry_is_required():
    record = parse_resource_identities(
        resource_identity_marker(_node(), "input", entry_point="compute_gather")
    )[0]
    with pytest.raises(ValueError, match="entry point is missing"):
        apply_resource_identities(
            [dict(name="input", metadata={"entryPoint": "compute_gather"})],
            [record],
            [],
        )


@pytest.mark.parametrize("target", TARGETS)
def test_specialized_entry_preserves_host_name(tmp_path, target):
    path = tmp_path / "template.metal"
    path.write_text("""#include <metal_stdlib>
using namespace metal;
template<typename T> kernel void fill(device T* values [[buffer(3)]]) { values[0] = T(7); }
template [[host_name("fill_float")]] [[kernel]] decltype(fill<float>) fill<float>;
""")
    generated = translate(
        path,
        backend=target,
        format_output=False,
        source_options={"preserve_resource_origins": True},
    )
    origins = parse_resource_identities(generated)
    assert len(origins) == 1
    assert origins[0]["source"] == dict(
        backend="metal", entryPoint="fill_float", parameter="values", parameterIndex=0
    )
