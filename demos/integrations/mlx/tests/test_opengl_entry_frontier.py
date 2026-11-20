"""OpenGL frontier contracts with one artifact per kernel (a GLSL unit holds one entry per stage)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
HARNESS_PATH = ROOT / "demos" / "integrations" / "mlx" / "run_porting.py"

TOOLS = {
    "glslangValidator": "/tools/glslangValidator",
    "spirv-val": "/tools/spirv-val",
}


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_mlx_porting", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generated_glsl(constants):
    declarations = [
        f"layout(constant_id = {constant_id}) const bool {name} = false;"
        for name, constant_id in constants.items()
    ]
    return "\n".join(["#version 450 core", *declarations, "void main() {}", ""])


def _entries(module, source, outcomes):
    """Entry names and outcomes in discovery order: refused entries first, then translated."""
    stem = Path(source).stem
    names = []
    for outcome, count in sorted(outcomes.items()):
        if outcome == "translated":
            continue
        names.extend(
            (f"{stem}_{outcome.rsplit('.', 1)[-1]}_{i}", outcome) for i in range(count)
        )
    translated = module.MLX_OPENGL_DYNAMIC_WORKGROUP_TRANSLATED_ENTRIES.get(source)
    if translated is None:
        translated = tuple(
            f"{stem}_entry_{i}" for i in range(outcomes.get("translated", 0))
        )
    names.extend((name, "translated") for name in translated)
    return names


def _diagnostic(module, code, entry, source, artifact_path):
    frontier = module.entry_frontier
    if code == frontier.WORKGROUP_SIZE_REQUIRED_CODE:
        return {
            "checkKind": "execution-specialization",
            "code": code,
            "details": {
                "executionSpecialization": {
                    "reason": "consumed-source-value-unconfigured",
                    "sourceEntryPoints": [entry],
                },
                "sourcePath": source,
                "targetArtifact": artifact_path,
            },
            "message": frontier.WORKGROUP_SIZE_REQUIRED_MESSAGE,
            "missingCapabilities": ["execution.workgroup-size-specialization"],
            "severity": "error",
            "sourceBackend": "metal",
            "target": "opengl",
        }
    return {
        "code": code,
        "details": {
            "indexConversion": {
                "acceptedRange": {
                    "maximum": 2147483647,
                    "minimum": -2147483648,
                    "status": "statically-bounded",
                },
                "indexExpression": "gid * uint64(axis_size)",
                "indexedValue": "in_",
                "rangeStatus": "unproven",
                "reason": "pointer-offset-range-unproven",
                "sourceType": "uint64_t",
                "targetProfile": "OpenGL #version 450 core",
                "targetType": "int",
            },
            "sourcePath": source,
            "targetArtifact": artifact_path,
        },
        "message": (
            "OpenGL cannot preserve subscript index type 'uint64_t' for 'in_': "
            "pointer offset range unproven"
        ),
        "missingCapabilities": ["opengl.index-width-normalization"],
        "severity": "error",
        "sourceBackend": "metal",
        "target": "opengl",
    }


def _report(module, mlx_root, output_dir, sources, outcomes_by_source, assertions):
    """A per-entry project report and the files it emits."""
    units, artifacts, diagnostics, files = [], [], [], {}
    for index, source in enumerate(sources):
        entries = _entries(module, source, outcomes_by_source[source])
        unit = {
            "id": source,
            "path": source,
            "sourceBackend": "metal",
            "sourceHash": {"algorithm": "sha256", "value": f"source-{index}"},
            "sourceSizeBytes": 1000 + index,
            "entryDiscovery": {
                "diagnosticCount": 0,
                "entries": [{"name": name, "stage": "compute"} for name, _ in entries],
            },
        }
        units.append(unit)
        constants = module.MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS.get(source, {})
        for name, outcome in entries:
            path = (
                (output_dir / "opengl" / Path(source).with_suffix("") / f"{name}.glsl")
                .relative_to(mlx_root)
                .as_posix()
            )
            artifact = {
                "source": source,
                "sourceBackend": "metal",
                "sourceHash": unit["sourceHash"],
                "sourceSizeBytes": unit["sourceSizeBytes"],
                "target": "opengl",
                "path": path,
                "entryPoint": {"source": name, "target": "main"},
                "templateMaterialization": {
                    "status": "materialized",
                    "unsupported": [],
                    "specializationCount": 1,
                    "specializations": [{"hostName": name}],
                },
            }
            if outcome == "translated":
                text = _generated_glsl(constants)
                files[path] = text
                artifact["status"] = "translated"
                artifact["entryPoint"]["stage"] = "compute"
                artifact["generatedHash"] = {
                    "algorithm": "sha256",
                    "value": hashlib.sha256(text.encode()).hexdigest(),
                }
                artifact["generatedSizeBytes"] = len(text.encode())
                if constants:
                    artifact["specializationConstants"] = [
                        {"name": key, "id": value} for key, value in constants.items()
                    ]
                    artifact["specializationMaterialization"] = {
                        "mode": "deferred",
                        "targetSupportsDeferredSpecialization": True,
                    }
            else:
                diagnostic = _diagnostic(module, outcome, name, source, path)
                diagnostics.append(diagnostic)
                artifact["status"] = "failed"
                artifact["error"] = diagnostic["message"]
            artifacts.append(artifact)
    translated = sum(1 for artifact in artifacts if artifact["status"] == "translated")
    failed = len(artifacts) - translated
    by_code = {}
    for diagnostic in diagnostics:
        by_code[diagnostic["code"]] = by_code.get(diagnostic["code"], 0) + 1
    report = {
        "project": {
            "includePatterns": list(sources),
            "targets": ["opengl"],
            "workgroupSizeRules": {},
            "workgroupSizeRuleCount": 0,
            "indexRangeAssertions": [dict(assertion) for assertion in assertions],
            "indexRangeAssertionCount": len(assertions),
        },
        "summary": {
            "unitCount": len(sources),
            "artifactCount": len(artifacts),
            "translatedCount": translated,
            "failedCount": failed,
            "diagnosticCounts": {"error": failed, "note": 0, "warning": 0},
            "diagnosticsByCode": by_code,
            "artifactsByTarget": {
                "opengl": {
                    "artifactCount": len(artifacts),
                    "translatedCount": translated,
                    "failedCount": failed,
                }
            },
        },
        "units": units,
        "artifacts": artifacts,
        "diagnostics": diagnostics,
    }
    return report, files


class FakeFrontier:
    """Plays the two OpenGL frontier translations and the validators."""

    def __init__(self, module, tmp_path):
        self.module = module
        self.mlx_root = tmp_path / "mlx"
        self.work_dir = self.mlx_root / ".crosstl-mlx-porting"
        self.config_dir = self.work_dir / "configs"
        self.report_dir = self.work_dir / "reports"
        self.log_dir = self.work_dir / "logs"
        for path in (self.config_dir, self.report_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        clean_sources = module.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
        self.runs = {
            "opengl-frontier": _report(
                module,
                self.mlx_root,
                self.work_dir / "out-opengl-frontier",
                clean_sources,
                {
                    source: {
                        "translated": module.MLX_OPENGL_FRONTIER_ENTRY_POINT_COUNTS[
                            source
                        ]
                    }
                    for source in clean_sources
                },
                module.MLX_OPENGL_TRANSLATED_FRONTIER_INDEX_RANGE_ASSERTIONS,
            ),
            "opengl-workgroup-frontier": _report(
                module,
                self.mlx_root,
                self.work_dir / "out-opengl-workgroup-frontier",
                module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
                module.MLX_OPENGL_DYNAMIC_WORKGROUP_ENTRY_OUTCOMES,
                module.MLX_OPENGL_DYNAMIC_WORKGROUP_INDEX_RANGE_ASSERTIONS,
            ),
        }
        self.commands = []

    @property
    def paths(self):
        return (
            self.mlx_root,
            self.work_dir,
            self.config_dir,
            self.report_dir,
            self.log_dir,
        )

    def run_command(self, name, command, *, log_dir, **_kwargs):
        self.commands.append((name, list(command)))
        if name in self.runs:
            report, files = self.runs[name]
            for path, text in files.items():
                generated = self.mlx_root / path
                generated.parent.mkdir(parents=True, exist_ok=True)
                generated.write_text(text, encoding="utf-8")
            (self.report_dir / f"{name}.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
        elif "-o" in command:
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x03\x02\x23\x07")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return self.module.CommandResult(
            name,
            list(command),
            1 if name == "opengl-workgroup-frontier" else 0,
            stdout_path,
            stderr_path,
        )

    def check(self, monkeypatch, *, require_toolchain):
        monkeypatch.setattr(self.module, "_run_command", self.run_command)
        monkeypatch.setattr(self.module.shutil, "which", TOOLS.get)
        return self.module._check_opengl_frontier(
            *self.paths, "python", require_toolchain=require_toolchain
        )


@pytest.fixture
def harness():
    return _load_harness()


def test_pinned_entry_outcomes_cover_every_discovered_entry(harness):
    counts = harness.MLX_OPENGL_FRONTIER_ENTRY_POINT_COUNTS
    assert harness.MLX_OPENGL_TRANSLATED_FRONTIER_ARTIFACT_COUNT == 455
    assert harness.MLX_OPENGL_DYNAMIC_WORKGROUP_ARTIFACT_COUNT == 94
    assert harness.MLX_OPENGL_DYNAMIC_WORKGROUP_TRANSLATED_ARTIFACT_COUNT == 41
    for source, outcomes in harness.MLX_OPENGL_DYNAMIC_WORKGROUP_ENTRY_OUTCOMES.items():
        assert sum(outcomes.values()) == counts[source]
        assert outcomes.get("translated", 0) == len(
            harness.MLX_OPENGL_DYNAMIC_WORKGROUP_TRANSLATED_ENTRIES.get(source, ())
        )


def test_index_range_assertions_are_split_by_frontier(harness):
    translated = harness.MLX_OPENGL_TRANSLATED_FRONTIER_INDEX_RANGE_ASSERTIONS
    dynamic = harness.MLX_OPENGL_DYNAMIC_WORKGROUP_INDEX_RANGE_ASSERTIONS

    def key(assertion):
        return tuple(sorted(assertion.items()))

    assert sorted(map(key, (*translated, *dynamic))) == sorted(
        map(key, harness.MLX_OPENGL_INDEX_RANGE_ASSERTIONS)
    )
    assert {assertion["source"] for assertion in translated} <= set(
        harness.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
    )
    # The row kernels and sdpa_vector need their pointer rebases bounded; arg_reduce has none.
    assert {assertion["source"] for assertion in dynamic} == {
        harness.MLX_LOGSUMEXP_SOURCE,
        harness.MLX_RMS_NORM_SOURCE,
        harness.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
        harness.MLX_SOFTMAX_SOURCE,
    }
    assert len(dynamic) == 15
    assert {(a["minimum"], a["maximum"]) for a in dynamic} == {(0, (1 << 31) - 1)}
    # No dynamic-workgroup entry is refused for its index type any more.
    for outcomes in harness.MLX_OPENGL_DYNAMIC_WORKGROUP_ENTRY_OUTCOMES.values():
        assert set(outcomes) <= {
            "translated",
            harness.entry_frontier.WORKGROUP_SIZE_REQUIRED_CODE,
        }


def test_opengl_frontier_validates_every_entry_artifact(harness, tmp_path, monkeypatch):
    fake = FakeFrontier(harness, tmp_path)

    result = fake.check(monkeypatch, require_toolchain=True)

    sdpa = harness.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    blocked_translated = {
        harness.MLX_LOGSUMEXP_SOURCE: 3,
        harness.MLX_RMS_NORM_SOURCE: 6,
        sdpa: 27,
        harness.MLX_SOFTMAX_SOURCE: 5,
    }
    toolchain_sources = [
        *harness.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES,
        *blocked_translated,
    ]
    assert result["status"] == "passed-with-expected-workgroup-blockers"
    assert result["artifactGranularity"] == "entry-point"
    assert result["artifactCount"] == 455 + 94
    assert result["translatedArtifactCount"] == 455
    assert result["translatedEntryPointCounts"] == {
        source: harness.MLX_OPENGL_FRONTIER_ENTRY_POINT_COUNTS[source]
        for source in harness.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
    }
    assert result["workgroupFrontierArtifactCount"] == 94
    assert result["workgroupFrontierTranslatedArtifactCount"] == 41
    assert result["workgroupBlockedArtifactCount"] == 53
    assert result["expectedProjectErrorCount"] == 53
    assert result["toolchainSources"] == toolchain_sources
    assert result["toolchainValidatedSources"] == toolchain_sources
    assert result["toolchainArtifactCount"] == 496
    assert result["toolchainValidatedArtifactCount"] == 496
    assert result["nativeValidationStatus"] == "validated"
    assert {
        source: len(outputs)
        for source, outputs in result["nativeValidationOutputs"].items()
    } == {
        **{
            source: harness.MLX_OPENGL_FRONTIER_ENTRY_POINT_COUNTS[source]
            for source in harness.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
        },
        **blocked_translated,
    }
    assert result["specializationConstants"] == {
        source: harness.MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS[source]
        for source in (harness.MLX_ROPE_SOURCE, harness.MLX_RMS_NORM_SOURCE, sdpa)
    }
    for source, evidence in result["dynamicWorkgroupDispatchEvidence"].items():
        assert evidence["entryOutcomes"] == dict(
            sorted(harness.MLX_OPENGL_DYNAMIC_WORKGROUP_ENTRY_OUTCOMES[source].items())
        )
        assert evidence["translatedEntryPoints"] == list(
            harness.MLX_OPENGL_DYNAMIC_WORKGROUP_TRANSLATED_ENTRIES.get(source, ())
        )
        assert (
            evidence["sourceEntryPointIdentityStatus"]
            == "matched-materialized-host-names"
        )
        for key, value in harness.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[
            source
        ].items():
            assert evidence[key] == value
    names = [name for name, _command in fake.commands]
    assert names[:2] == ["opengl-frontier", "opengl-workgroup-frontier"]
    assert len(names) == 2 + 2 * 496
    glslang = fake.commands[2][1]
    assert glslang[:7] == [
        TOOLS["glslangValidator"],
        "--target-env",
        "opengl",
        "--target-env",
        "spirv1.3",
        "-S",
        "comp",
    ]
    assert fake.commands[3][1][:3] == [TOOLS["spirv-val"], "--target-env", "spv1.3"]
    assert fake.commands[3][1][3] == glslang[-1]
    config = (fake.config_dir / "opengl-frontier.toml").read_text(encoding="utf-8")
    assert config.count("[[project.index_range_assertions]]") == len(
        harness.MLX_OPENGL_TRANSLATED_FRONTIER_INDEX_RANGE_ASSERTIONS
    )
    blocked_config = (fake.config_dir / "opengl-workgroup-frontier.toml").read_text(
        encoding="utf-8"
    )
    assert blocked_config.count("[[project.index_range_assertions]]") == len(
        harness.MLX_OPENGL_DYNAMIC_WORKGROUP_INDEX_RANGE_ASSERTIONS
    )


def test_opengl_frontier_skips_toolchain_when_not_required(
    harness, tmp_path, monkeypatch
):
    fake = FakeFrontier(harness, tmp_path)
    monkeypatch.setattr(
        harness.shutil,
        "which",
        lambda name: pytest.fail(f"tool lookup should not run for {name}"),
    )
    monkeypatch.setattr(harness, "_run_command", fake.run_command)

    result = harness._check_opengl_frontier(
        *fake.paths, "python", require_toolchain=False
    )

    assert [name for name, _ in fake.commands] == [
        "opengl-frontier",
        "opengl-workgroup-frontier",
    ]
    assert result["toolchainValidatedSources"] == []
    assert result["toolchainValidatedArtifactCount"] == 0
    assert result["nativeValidationStatus"] == "not-required"
    assert result["nativeValidationOutputs"] == {}


def _drop_last_artifact(report, _files):
    report["artifacts"].pop()
    report["summary"]["artifactCount"] -= 1
    report["summary"]["translatedCount"] -= 1
    target = report["summary"]["artifactsByTarget"]["opengl"]
    target["artifactCount"] -= 1
    target["translatedCount"] -= 1


def _drop_artifact_keep_counts(report, _files):
    report["artifacts"].pop()


def _reorder_entries(report, _files):
    first, second = report["artifacts"][:2]
    report["artifacts"][:2] = [second, first]


def _repeat_entry(report, _files):
    report["artifacts"][1]["entryPoint"] = dict(report["artifacts"][0]["entryPoint"])


def _drift_generated_bytes(report, files):
    files[report["artifacts"][0]["path"]] += "// drift\n"


def _escape_unit_directory(report, files):
    artifact = report["artifacts"][0]
    moved = artifact["path"].replace("/binary_two/", "/ternary/")
    files[moved] = files.pop(artifact["path"])
    artifact["path"] = moved


def _report_a_warning(report, _files):
    report["diagnostics"] = [
        {
            "severity": "warning",
            "code": "project.translate.test-warning",
            "message": "w",
        }
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_drop_last_artifact, "accounting changed"),
        (_drop_artifact_keep_counts, "do not match the discovered entries"),
        (_reorder_entries, "do not match the discovered entries"),
        (_repeat_entry, "entry is missing or repeated"),
        (_drift_generated_bytes, "artifact identity does not match"),
        (_escape_unit_directory, "outside its unit directory"),
        (_report_a_warning, "reported diagnostics"),
    ),
)
def test_opengl_clean_frontier_joins_one_artifact_to_each_entry(
    harness, tmp_path, monkeypatch, mutate, message
):
    fake = FakeFrontier(harness, tmp_path)
    mutate(*fake.runs["opengl-frontier"])

    with pytest.raises(harness.PortingCheckError, match=message):
        fake.check(monkeypatch, require_toolchain=False)


def _blocked_artifact(report, code):
    return next(
        artifact
        for artifact in report["artifacts"]
        if artifact.get("error")
        and any(
            diagnostic["code"] == code
            and diagnostic["details"]["targetArtifact"] == artifact["path"]
            for diagnostic in report["diagnostics"]
        )
    )


def _translate_a_refused_entry(_module, report, files):
    artifact = _blocked_artifact(report, "project.translate.workgroup-size-required")
    report["diagnostics"] = [
        diagnostic
        for diagnostic in report["diagnostics"]
        if diagnostic["details"]["targetArtifact"] != artifact["path"]
    ]
    text = _generated_glsl({})
    files[artifact["path"]] = text
    artifact.pop("error")
    artifact["status"] = "translated"
    artifact["entryPoint"]["stage"] = "compute"
    artifact["generatedHash"] = {
        "algorithm": "sha256",
        "value": hashlib.sha256(text.encode()).hexdigest(),
    }
    artifact["generatedSizeBytes"] = len(text.encode())
    summary = report["summary"]
    summary["translatedCount"] += 1
    summary["failedCount"] -= 1
    summary["diagnosticCounts"]["error"] -= 1
    summary["diagnosticsByCode"]["project.translate.workgroup-size-required"] -= 1
    target = summary["artifactsByTarget"]["opengl"]
    target["translatedCount"] += 1
    target["failedCount"] -= 1


def _emit_a_refused_entry(_module, report, files):
    artifact = _blocked_artifact(report, "project.translate.workgroup-size-required")
    files[artifact["path"]] = "// refused\n"


def _drift_workgroup_reason(_module, report, _files):
    diagnostic = next(
        diagnostic
        for diagnostic in report["diagnostics"]
        if diagnostic["code"] == "project.translate.workgroup-size-required"
    )
    diagnostic["details"]["executionSpecialization"][
        "reason"
    ] = "aggregate-entry-size-unproven"


def _translated_artifact(report, source):
    return next(
        artifact
        for artifact in report["artifacts"]
        if artifact["source"] == source and artifact["status"] == "translated"
    )


def _refuse(module, report, files, artifact, code):
    diagnostic = _diagnostic(
        module,
        code,
        artifact["entryPoint"]["source"],
        artifact["source"],
        artifact["path"],
    )
    report["diagnostics"].append(diagnostic)
    files.pop(artifact["path"])
    for key in (
        "generatedHash",
        "generatedSizeBytes",
        "specializationConstants",
        "specializationMaterialization",
    ):
        artifact.pop(key, None)
    artifact["entryPoint"].pop("stage")
    artifact["status"] = "failed"
    artifact["error"] = diagnostic["message"]


def _refuse_a_translated_entry_for_its_index(module, report, files):
    """An index-range assertion that stops applying turns a translated entry back into a refusal."""
    artifact = _translated_artifact(report, module.MLX_LOGSUMEXP_SOURCE)
    _refuse(
        module,
        report,
        files,
        artifact,
        "project.translate.opengl-index-type-unsupported",
    )
    summary = report["summary"]
    summary["translatedCount"] -= 1
    summary["failedCount"] += 1
    summary["diagnosticCounts"]["error"] += 1
    by_code = summary["diagnosticsByCode"]
    code = "project.translate.opengl-index-type-unsupported"
    by_code[code] = by_code.get(code, 0) + 1
    target = summary["artifactsByTarget"]["opengl"]
    target["translatedCount"] -= 1
    target["failedCount"] += 1


def _move_refusal_between_sources(module, report, files):
    """Keep every total but translate an arg_reduce entry and refuse a logsumexp one."""
    refused = _blocked_artifact(report, "project.translate.workgroup-size-required")
    translated = _translated_artifact(report, module.MLX_LOGSUMEXP_SOURCE)
    _refuse(
        module,
        report,
        files,
        translated,
        "project.translate.workgroup-size-required",
    )
    report["diagnostics"] = [
        diagnostic
        for diagnostic in report["diagnostics"]
        if diagnostic["details"]["targetArtifact"] != refused["path"]
    ]
    text = _generated_glsl({})
    files[refused["path"]] = text
    refused.pop("error")
    refused["status"] = "translated"
    refused["entryPoint"]["stage"] = "compute"
    refused["generatedHash"] = {
        "algorithm": "sha256",
        "value": hashlib.sha256(text.encode()).hexdigest(),
    }
    refused["generatedSizeBytes"] = len(text.encode())


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_translate_a_refused_entry, "accounting changed"),
        (_move_refusal_between_sources, "entry outcomes changed for .*arg_reduce"),
        (_emit_a_refused_entry, "refused entry contract changed"),
        (_drift_workgroup_reason, "workgroup-size-required diagnostic changed"),
        (_refuse_a_translated_entry_for_its_index, "accounting changed"),
    ),
)
def test_opengl_workgroup_frontier_pins_each_entry_outcome(
    harness, tmp_path, monkeypatch, mutate, message
):
    fake = FakeFrontier(harness, tmp_path)
    mutate(harness, *fake.runs["opengl-workgroup-frontier"])

    with pytest.raises(harness.PortingCheckError, match=message):
        fake.check(monkeypatch, require_toolchain=False)


def test_opengl_entry_artifact_declares_only_reflected_constants(
    harness, tmp_path, monkeypatch
):
    fake = FakeFrontier(harness, tmp_path)
    report, _files = fake.runs["opengl-frontier"]
    rope = next(
        artifact
        for artifact in report["artifacts"]
        if artifact["source"] == harness.MLX_ROPE_SOURCE
    )
    rope["specializationConstants"].pop()

    with pytest.raises(
        harness.PortingCheckError, match="does not match its reflection"
    ):
        fake.check(monkeypatch, require_toolchain=False)


def test_opengl_entry_artifacts_reflect_every_source_constant(
    harness, tmp_path, monkeypatch
):
    fake = FakeFrontier(harness, tmp_path)
    report, files = fake.runs["opengl-frontier"]
    for artifact in report["artifacts"]:
        if artifact["source"] != harness.MLX_ROPE_SOURCE:
            continue
        artifact["specializationConstants"].pop()
        text = _generated_glsl(
            {item["name"]: item["id"] for item in artifact["specializationConstants"]}
        )
        files[artifact["path"]] = text
        artifact["generatedHash"]["value"] = hashlib.sha256(text.encode()).hexdigest()
        artifact["generatedSizeBytes"] = len(text.encode())

    with pytest.raises(
        harness.PortingCheckError, match="every specialization constant"
    ):
        fake.check(monkeypatch, require_toolchain=False)


@pytest.mark.parametrize("drift", ("count", "content"))
def test_opengl_frontier_requires_exact_index_range_assertion_report(
    harness, tmp_path, monkeypatch, drift
):
    fake = FakeFrontier(harness, tmp_path)
    project = fake.runs["opengl-frontier"][0]["project"]
    if drift == "count":
        project["indexRangeAssertionCount"] -= 1
    else:
        project["indexRangeAssertions"][0]["expression"] = "offset + j"

    with pytest.raises(
        harness.PortingCheckError, match="index-range assertion contract changed"
    ):
        fake.check(monkeypatch, require_toolchain=False)


def test_opengl_frontier_requires_glslang_success_for_each_entry(
    harness, tmp_path, monkeypatch
):
    fake = FakeFrontier(harness, tmp_path)
    run_command = fake.run_command

    def failing_run_command(name, command, *, log_dir, **kwargs):
        result = run_command(name, command, log_dir=log_dir, **kwargs)
        if name.startswith("validate-opengl-rope-"):
            return harness.CommandResult(
                name, list(command), 1, result.stdout_path, result.stderr_path
            )
        return result

    monkeypatch.setattr(fake, "run_command", failing_run_command)

    with pytest.raises(
        harness.PortingCheckError, match=r"OpenGL rope:\w+ native compilation failed"
    ):
        fake.check(monkeypatch, require_toolchain=True)
