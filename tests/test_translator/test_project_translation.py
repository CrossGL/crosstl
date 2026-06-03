import json
import subprocess
import sys
import textwrap
from pathlib import Path

from crosstl.project import load_project_config, scan_project, translate_project

ROOT = Path(__file__).resolve().parents[2]


SIMPLE_CROSSL = textwrap.dedent("""
    shader RepoShader {
        struct VertexInput {
            vec3 position;
        }

        struct VertexOutput {
            vec4 position;
        }

        vertex {
            VertexOutput main(VertexInput input) {
                VertexOutput output;
                output.position = vec4(input.position, 1.0);
                return output;
            }
        }
    }
    """).strip()


def test_scan_project_discovers_supported_sources_and_ignores_default_unsupported(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "main.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (shader_dir / "post.frag").write_text(
        "#version 450\nvoid main() { gl_FragColor = vec4(1.0); }\n",
        encoding="utf-8",
    )
    (shader_dir / "README.txt").write_text("not shader code", encoding="utf-8")

    scan = scan_project(repo)

    assert [unit.relative_path for unit in scan.units] == [
        "shaders/main.cgl",
        "shaders/post.frag",
    ]
    assert [unit.source_backend for unit in scan.units] == ["cgl", "opengl"]
    assert scan.skipped == []


def test_scan_project_reports_explicitly_included_unsupported_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            include = ["**/*"]
            exclude = []
            """).strip(),
        encoding="utf-8",
    )
    (repo / "kernel.txt").write_text("not shader code", encoding="utf-8")

    config = load_project_config(repo)
    scan = scan_project(config)

    assert scan.units == []
    assert scan.skipped == [{"path": "kernel.txt", "reason": "unsupported-extension"}]
    assert {diagnostic.code for diagnostic in scan.diagnostics} == {
        "project.scan.empty",
        "project.scan.unsupported-source",
    }


def test_project_config_loads_overrides_and_variant_metadata(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]
            include = ["**/*"]
            exclude = []
            targets = ["metal"]
            output_dir = "generated"
            include_dirs = ["gpu/include"]

            [project.sources]
            "gpu/*.shader" = "cgl"

            [project.defines]
            USE_FAST_PATH = "1"

            [project.variants.debug]
            USE_FAST_PATH = "0"
            """).strip(),
        encoding="utf-8",
    )

    config = load_project_config(repo)
    scan = scan_project(config)

    assert config.targets == ["metal"]
    assert config.output_dir == "generated"
    assert config.include_dirs == ["gpu/include"]
    assert config.defines == {"USE_FAST_PATH": "1"}
    assert config.variants == {"debug": {"USE_FAST_PATH": "0"}}
    assert [(unit.relative_path, unit.source_backend) for unit in scan.units] == [
        ("gpu/kernel.shader", "cgl")
    ]
    diagnostics = {diagnostic.code: diagnostic for diagnostic in scan.diagnostics}
    assert set(diagnostics) == {
        "project.config.include-dirs-not-applied",
        "project.config.variants-not-applied",
    }
    assert diagnostics[
        "project.config.include-dirs-not-applied"
    ].missing_capabilities == ["include.resolution"]
    assert diagnostics["project.config.variants-not-applied"].missing_capabilities == [
        "macro.variants"
    ]
    payload = scan.to_report(targets=config.targets).to_json()
    assert payload["diagnosticCounts"]["warning"] == 2
    assert {
        diagnostic["location"]["file"] for diagnostic in payload["diagnostics"]
    } == {"crosstl.toml"}


def test_translate_project_honors_source_backend_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]
            include = ["**/*"]
            targets = ["opengl"]
            output_dir = "translated"

            [project.sources]
            "gpu/*.shader" = "cgl"
            """).strip(),
        encoding="utf-8",
    )

    config = load_project_config(repo)
    report = translate_project(config)
    payload = report.to_json()

    output = repo / "translated" / "opengl" / "gpu" / "kernel.glsl"
    assert output.exists()
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["units"][0]["sourceOverride"] == "cgl"
    assert payload["artifacts"][0]["sourceBackend"] == "cgl"
    assert payload["artifacts"][0]["path"] == "translated/opengl/gpu/kernel.glsl"


def test_scan_project_reports_unsupported_source_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]
            include = ["**/*"]

            [project.sources]
            "gpu/*.shader" = "unknown-backend"
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert scan.units == []
    assert scan.skipped == [
        {
            "path": "gpu/kernel.shader",
            "reason": "unsupported-source-override",
            "sourceOverride": "unknown-backend",
        }
    ]
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "project.config.unsupported-source-override",
        "project.scan.empty",
    ]
    assert payload["diagnostics"][0]["severity"] == "error"
    assert payload["diagnostics"][0]["location"]["file"] == "crosstl.toml"
    assert payload["diagnostics"][0]["missingCapabilities"] == ["source.override"]
    assert "unknown-backend" in payload["diagnostics"][0]["message"]


def test_translate_project_preserves_relative_paths_and_reports_artifacts(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders" / "graphics"
    shader_dir.mkdir(parents=True)
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="translated")
    payload = report.to_json()

    output = repo / "translated" / "opengl" / "shaders" / "graphics" / "simple.glsl"
    assert output.exists()
    assert payload["kind"] == "crosstl-project-portability-report"
    assert payload["schemaVersion"] == 1
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["artifacts"][0]["source"] == "shaders/graphics/simple.cgl"
    assert payload["artifacts"][0]["target"] == "opengl"
    assert payload["artifacts"][0]["path"] == (
        "translated/opengl/shaders/graphics/simple.glsl"
    )
    assert payload["artifacts"][0]["sourceHash"]["algorithm"] == "sha256"
    assert payload["migration"]["nonGoals"] == [
        "automatic runtime API migration",
        "application build-system rewrites",
        "backend framework integration",
    ]


def test_translate_project_validation_records_artifacts_and_toolchains(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        validate=True,
    )
    payload = report.to_json()

    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "exists": True,
            "status": "ok",
        }
    ]
    assert payload["validation"]["toolchains"][0]["target"] == "opengl"
    assert payload["validation"]["toolchains"][0]["status"] in {
        "available",
        "unavailable",
    }
    assert payload["diagnosticCounts"]["error"] == 0


def test_translate_project_records_structured_diagnostics_for_failures(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["not-a-backend"], output_dir="out")
    payload = report.to_json()

    assert payload["summary"]["failedCount"] == 1
    assert payload["diagnosticCounts"]["error"] == 1
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["severity"] == "error"
    assert diagnostic["code"] == "project.translate.failed"
    assert diagnostic["target"] == "not-a-backend"
    assert diagnostic["location"]["file"] == "simple.cgl"
    assert payload["artifacts"][0]["status"] == "failed"
    assert payload["artifacts"][0]["error"]


def test_project_cli_translate_project_writes_report(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "translate-project",
            str(repo),
            "--target",
            "opengl",
            "--output-dir",
            "out",
            "--report",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "Wrote" in result.stdout
    assert payload["summary"]["translatedCount"] == 1
    assert (repo / "out" / "opengl" / "simple.glsl").exists()


def test_legacy_single_file_cli_still_works(tmp_path):
    shader = tmp_path / "simple.cgl"
    output = tmp_path / "simple.glsl"
    shader.write_text(SIMPLE_CROSSL, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            str(shader),
            "--backend",
            "opengl",
            "--output",
            str(output),
            "--no-format",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Successfully translated" in result.stdout
    assert output.exists()
