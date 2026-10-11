"""Frozen project variants reject overrides and discard only inactive branches."""

import copy
import hashlib
import json
import os
import shutil
import sys

import pytest

from crosstl import translate
from crosstl.formatter import CodeFormatter, ShaderLanguage
from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    inspect_runtime_package,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.native_loader_abi import (
    NativeLoaderABIError,
    generate_native_loader_execution_abi,
)
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.translator.ast import (
    BinaryOpNode,
    BlockNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    ParameterNode,
    PrimitiveType,
)
from crosstl.translator.frozen_specializations import (
    fold_frozen_specialization_branches,
    frozen_specialization_header,
    parse_frozen_specializations,
)

SOURCE = """#include <metal_stdlib>
using namespace metal;
constant bool use_index [[function_constant(10)]];
uint indexed(constant uint* values, constant ulong* indices) {
    ulong index = indices[0];
    return values[index];
}
kernel void read_value(constant uint* values [[buffer(0)]],
                       constant ulong* indices [[buffer(1)]],
                       device uint* output [[buffer(2)]]) {
    if (!use_index || (use_index == false)) {
        output[0] = 7;
    } else {
        output[0] = indexed(values, indices);
    }
}
"""
TARGETS = ("metal", "opengl", "directx")


def _translate(
    tmp_path,
    target,
    *,
    enabled=False,
    frozen=True,
    source=SOURCE,
    values=None,
    canonical=False,
    formatted=False,
):
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "read.metal"
    path.write_text(source)
    if canonical:
        canonical_source = translate(
            path,
            backend="cgl",
            format_output=False,
            source_options={"preserve_resource_origins": True},
        )
        path = tmp_path / "read.cgl"
        path.write_text(canonical_source)
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=(path.name,),
        targets=(target,),
        entry_points={path.name: "read_value"},
        workgroup_size=(1, 1, 1),
        specialization_constants={"use_index": enabled} if values is None else values,
        freeze_specialization_constants=frozen,
        source_options={"metal": {"preserve_resource_origins": True}},
    )
    report = translate_project(config, format_output=formatted)
    report.write_json(tmp_path / "report.json")
    return report.to_json()


def _package(tmp_path, target, **kwargs):
    report = _translate(tmp_path, target, **kwargs)
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    package = tmp_path / "package"
    result = build_runtime_package(tmp_path / "manifest.json", package)
    assert result["success"], result
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    return (
        build_native_loader_abi_descriptor(
            loader, load_unit_id=loader["loadUnits"][0]["id"]
        ),
        package,
    )


def _request(
    descriptor, package, *, enabled=False, override=None, extra_specializations=None
):
    explicit = dict(extra_specializations or {})
    if override is not None:
        explicit[10] = override
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        name = binding["provenance"]["sourceResource"]["parameter"]
        dtype, values = {
            "values": ("uint32", [43]),
            "indices": ("uint64", [0]),
            "output": ("uint32", [0, 12345]),
        }[name]
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(values)], values=values)
        if name == "output":
            outputs[binding["name"]] = dict(
                dtype=dtype, shape=[2], values=[43 if enabled else 7, 12345]
            )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        specialization_values=explicit,
    )
    return request, outputs


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("canonical", [False, True])
def test_frozen_variant_package_and_dispatch(tmp_path, target, canonical):
    descriptor, package = _package(
        tmp_path, target, canonical=canonical, formatted=True
    )
    constants = descriptor["specializationConstants"]
    assert len(constants) == 1
    assert constants[0]["frozen"] is True
    assert constants[0]["value"] is False
    source = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert parse_frozen_specializations(source)[0]["value"] is False
    assert "indexed(" not in source
    for override in (None, False):
        request, _ = _request(descriptor, package, override=override)
        assert not request.adapter_contract.specialization_constants
    with pytest.raises(NativeLoaderDispatchError, match="override"):
        _request(descriptor, package, override=True)
    with pytest.raises(
        NativeLoaderABIError, match="execution-frozen-specialization-unsupported"
    ):
        generate_native_loader_execution_abi(descriptor)
    assert validate_project_report(tmp_path / "report.json")["success"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert (
        report["artifacts"][0]["specializationMaterialization"]["status"] == "concrete"
    )


@pytest.mark.parametrize("target", TARGETS)
def test_formatter_preserves_frozen_header(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    source = (package / descriptor["artifact"]["packagePath"]).read_text()
    language = {
        "metal": ShaderLanguage.METAL,
        "opengl": ShaderLanguage.GLSL,
        "directx": ShaderLanguage.HLSL,
    }[target]
    formatted = CodeFormatter().format_code(source, language)
    assert parse_frozen_specializations(formatted) == parse_frozen_specializations(
        source
    )


@pytest.mark.parametrize("fault", ["removed", "value", "nonbool"])
def test_dispatch_rechecks_frozen_descriptor_against_artifact(tmp_path, fault):
    descriptor, package = _package(tmp_path, "opengl")
    if fault == "removed":
        descriptor["specializationConstants"][0].pop("frozen")
    else:
        descriptor["specializationConstants"][0]["value"] = (
            True if fault == "value" else 1
        )
    with pytest.raises(
        NativeLoaderDispatchError, match="frozen-specialization-invalid"
    ):
        _request(descriptor, package)


@pytest.mark.parametrize("target", ["metal", "opengl"])
def test_frozen_and_deferred_values_remain_distinct(tmp_path, target):
    source = SOURCE.replace(
        "constant bool use_index",
        "constant bool runtime_flag [[function_constant(11)]];\nconstant bool use_index",
    ).replace("output[0] = 7;", "output[0] = runtime_flag ? 9u : 7u;")
    descriptor, package = _package(tmp_path, target, source=source)
    constants = {item["id"]: item for item in descriptor["specializationConstants"]}
    assert constants[10]["frozen"] is True
    assert "frozen" not in constants[11]
    request, _ = _request(descriptor, package, extra_specializations={11: True})
    assert [
        item.constant_id for item in request.adapter_contract.specialization_constants
    ] == [11]
    assert request.adapter_contract.metadata["frozenSpecializations"][0]["id"] == 10


@pytest.mark.parametrize(
    "values,code",
    [
        ({"10": False}, None),
        (
            {"10": True, "use_index": False},
            "project.translate.specialization-value-conflict",
        ),
        ({"10": 0}, "project.translate.specialization-value-type-incompatible"),
    ],
)
def test_frozen_selector_validation(tmp_path, values, code):
    report = _translate(tmp_path, "opengl", values=values)
    if code is None:
        assert report["summary"]["translatedCount"] == 1
    else:
        assert any(item["code"] == code for item in report["diagnostics"])


def test_folding_preserves_shadowed_names_unknown_conditions_and_blocks():
    flag = IdentifierNode("flag")
    yes = BlockNode([IdentifierNode("yes")])
    no = BlockNode([IdentifierNode("no")])
    records = [dict(name="flag", id=1, dtype="bool", value=False, frozen=True)]
    for shadowed in (False, True):
        branch = IfNode(flag, copy.deepcopy(yes), copy.deepcopy(no))
        parameters = [ParameterNode("flag", PrimitiveType("bool"))] if shadowed else []
        function = FunctionNode(
            "run", PrimitiveType("void"), parameters, BlockNode([branch])
        )
        fold_frozen_specialization_branches(function, records)
        if shadowed:
            assert isinstance(function.body.statements[0], IfNode)
        else:
            assert isinstance(function.body.statements[0], BlockNode)
            assert function.body.statements[0].statements[0].name == "no"
    unknown = BinaryOpNode(
        IdentifierNode("runtime"), "&&", LiteralNode(False, PrimitiveType("bool"))
    )
    function = FunctionNode(
        "run", PrimitiveType("void"), [], BlockNode([IfNode(unknown, yes, no)])
    )
    fold_frozen_specialization_branches(function, records)
    assert isinstance(function.body.statements[0], IfNode)


def test_frozen_specialization_ci_is_required():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    assert 'CROSTL_REQUIRE_FROZEN_SPECIALIZATION: "1"' in workflow
    assert "tests/test_translator/test_frozen_specializations.py" in workflow


@pytest.mark.parametrize("enabled,frozen", [(False, False), (True, True)])
def test_unfrozen_or_live_wide_access_remains_rejected(tmp_path, enabled, frozen):
    report = _translate(tmp_path, "opengl", enabled=enabled, frozen=frozen)
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize(
    "fault",
    ["header", "declared-value", "declared-flag", "invalid-flag", "interface-value"],
)
def test_package_rejects_changed_frozen_contract(tmp_path, fault):
    descriptor, package = _package(tmp_path, "opengl")
    path = package / "runtime-package.json"
    payload = json.loads(path.read_text())
    artifact = payload["artifacts"][0]
    if fault == "header":
        source = package / descriptor["artifact"]["packagePath"]
        source.write_text(source.read_text().split("\n", 1)[1])
        artifact["hash"]["value"] = hashlib.sha256(source.read_bytes()).hexdigest()
        artifact["sizeBytes"] = source.stat().st_size
    elif fault == "declared-value":
        artifact["specializationConstants"][0]["value"] = True
    elif fault == "invalid-flag":
        artifact["specializationConstants"][0]["frozen"] = 1
    elif fault == "interface-value":
        artifact["hostInterface"]["specializationConstants"][0]["value"] = True
    else:
        artifact["specializationConstants"][0].pop("frozen")
    path.write_text(json.dumps(payload))
    result = inspect_runtime_package(path)
    assert not result["success"], result
    assert any(
        item["code"]
        == "project.runtime-package-inspection.frozen-specialization-invalid"
        for item in result["diagnostics"]
    )


@pytest.mark.parametrize("value", [None, 1, "true", []])
def test_freeze_setting_requires_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="must be a boolean"):
        ProjectConfig(root=tmp_path, freeze_specialization_constants=value)


def test_config_loads_freeze_option(tmp_path):
    (tmp_path / "crosstl.toml").write_text(
        '[project]\nfreeze_specialization_constants = true\n[project.specialization_constants]\n"10" = false\n'
    )
    config = load_project_config(tmp_path)
    assert config.freeze_specialization_constants is True
    assert config.specialization_constants == {"10": False}


@pytest.mark.parametrize(
    "fault", ["duplicate-key", "version", "empty", "nonbool", "duplicate-id", "comment"]
)
def test_frozen_header_rejects_ambiguous_metadata(fault):
    records = [dict(name="flag", id=1, dtype="bool", value=False, frozen=True)]
    source = frozen_specialization_header(records)
    if fault == "duplicate-key":
        source = source.replace('"value":false', '"value":false,"value":true')
    elif fault == "version":
        source = source.replace('"schemaVersion":1', '"schemaVersion":true')
    elif fault == "empty":
        source = (
            '// crosstl-frozen-specializations: {"schemaVersion":1,"constants":[]}\n'
        )
    elif fault == "nonbool":
        source = source.replace('"value":false', '"value":0')
    elif fault == "duplicate-id":
        payload = json.loads(source.split(": ", 1)[1])
        payload["constants"].append(copy.deepcopy(payload["constants"][0]))
        source = "// crosstl-frozen-specializations: " + json.dumps(payload)
    else:
        source = "/*\n" + source + "*/\n"
    with pytest.raises(ValueError):
        parse_frozen_specializations(source)


@pytest.mark.parametrize("enabled", [False, True])
def test_frozen_variant_dxc_compilation(tmp_path, enabled):
    if not shutil.which("dxc"):
        if (
            os.environ.get("CROSTL_REQUIRE_FROZEN_SPECIALIZATION") == "1"
            and sys.platform == "win32"
        ):
            pytest.fail("DXC is required for frozen specialization controls")
        pytest.skip("DXC is not installed")
    from tests.runtime_helpers import _validate_generated_artifact

    descriptor, package = _package(tmp_path, "directx", enabled=enabled)
    _validate_generated_artifact(
        package / descriptor["artifact"]["packagePath"], tmp_path, "directx"
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_frozen_specialization_native_execution(tmp_path, enabled):
    if os.environ.get("CROSTL_REQUIRE_FROZEN_SPECIALIZATION") != "1":
        pytest.skip("set CROSTL_REQUIRE_FROZEN_SPECIALIZATION=1 for native controls")
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = (
        SOURCE.replace("values[index]", "values[uint(index)]") if enabled else SOURCE
    )
    descriptor, package = _package(tmp_path, target, enabled=enabled, source=source)
    request, outputs = _request(descriptor, package, enabled=enabled)
    _execute(request, outputs, tmp_path, validate=_validate)
