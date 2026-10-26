import json
import subprocess
from pathlib import Path

import pytest

from .corpus import checks, manifest, run
from .corpus.manifest import Unit, load, project_files, rebuild
from .corpus.worker import markers

COMPUTE_GLSL = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data { uint values[]; };
void main() { values[gl_GlobalInvocationID.x] += 1u; }
"""


def test_manifest_is_complete_and_scoped():
    data = load()
    assert set(data["projects"]) == {
        "apple-metal-samples",
        "candle",
        "llama-cpp",
        "mlx",
        "pytorch",
        "sascha-vulkan",
        "vulkan-samples",
    }
    languages = {entry["language"] for entry in data["slices"].values()}
    assert languages == set(manifest.LANGUAGES)
    for spec in data["projects"].values():
        assert len(spec["commit"]) == 40 and spec["licence"] and spec["licenceFiles"]
    total = 0
    for name, entry in data["slices"].items():
        paths = [unit["path"] for unit in entry["units"]]
        assert paths == sorted(set(paths)), name
        for unit in entry["units"]:
            assert len(unit["sha256"]) == 64 and set(unit["stages"]) <= set(
                manifest.STAGES
            )
            assert (
                not Path(unit["path"]).is_absolute()
                and ".." not in Path(unit["path"]).parts
            )
        total += len(paths)
    assert total == 1667
    origin = {unit["path"] for unit in data["slices"]["sascha-glsl"]["units"]}
    assert {
        unit["source"] for unit in data["slices"]["sascha-spirv"]["units"]
    } <= origin


def test_project_files_cover_licences_units_and_support():
    data = load()
    files = project_files(data, "candle")
    assert {"LICENSE-MIT", "LICENSE-APACHE"} <= set(files)
    units = {unit["path"] for unit in data["slices"]["candle-metal"]["units"]}
    assert units <= set(files)
    assert all(files[path] for path in units)


def _git(repo, *arguments):
    subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, capture_output=True
    )


@pytest.fixture
def pinned_project(tmp_path):
    """A local repository with one GLSL unit, an include and a licence, pinned at its first commit."""
    origin = tmp_path / "origin"
    (origin / "shaders" / "inc").mkdir(parents=True)
    (origin / "shaders" / "add.comp").write_text(
        '#include "inc/common.glsl"\n' + COMPUTE_GLSL
    )
    (origin / "shaders" / "inc" / "common.glsl").write_text("// shared\n")
    (origin / "shaders" / "unlisted.comp").write_text(COMPUTE_GLSL)
    (origin / "LICENSE").write_text("MIT\n")
    _git(tmp_path, "init", "-q", str(origin))
    _git(origin, "add", ".")
    _git(
        origin,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "commit",
        "-q",
        "-m",
        "pin",
    )
    _git(origin, "config", "uploadpack.allowFilter", "true")
    _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    commit = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    data = {
        "schemaVersion": 1,
        "projects": {
            "demo": {
                "repository": origin.as_uri(),
                "commit": commit,
                "licence": "MIT",
                "licenceFiles": ["LICENSE"],
            }
        },
        "slices": {
            "demo-glsl": {
                "project": "demo",
                "language": "glsl",
                "tag": "held-out",
                "support": {
                    "shaders/inc/common.glsl": manifest.sha256(
                        origin / "shaders" / "inc" / "common.glsl"
                    )
                },
                "units": [
                    {
                        "path": "shaders/add.comp",
                        "sha256": manifest.sha256(origin / "shaders" / "add.comp"),
                        "stages": ["compute"],
                        "entry": "main",
                        "includeDirs": ["shaders"],
                    }
                ],
            }
        },
    }
    return data, origin


def test_rebuild_checks_out_exactly_the_pinned_files(tmp_path, pinned_project):
    data, _ = pinned_project
    units, problems = rebuild(tmp_path / "cache", data)
    assert problems == []
    (unit,) = units
    assert unit.id == "demo-glsl:shaders/add.comp" and unit.stages == ("compute",)
    root = tmp_path / "cache" / "sources" / "demo"
    assert unit.file == root / "shaders" / "add.comp"
    assert unit.include_dirs == (str(root / "shaders"),)
    assert (root / "shaders" / "inc" / "common.glsl").is_file()
    assert not (root / "shaders" / "unlisted.comp").exists()
    # A second rebuild reuses the checkout and still verifies it.
    assert rebuild(tmp_path / "cache", data)[1] == []


def test_rebuild_fails_closed_on_changed_or_missing_files(tmp_path, pinned_project):
    data, _ = pinned_project
    entry = data["slices"]["demo-glsl"]
    entry["units"][0]["sha256"] = "0" * 64
    entry["support"]["shaders/inc/missing.glsl"] = "1" * 64
    _, problems = rebuild(tmp_path / "cache", data)
    assert "demo: changed shaders/add.comp" in problems
    assert "demo: missing shaders/inc/missing.glsl" in problems
    with pytest.raises(ValueError, match="Unknown corpus slices"):
        rebuild(tmp_path / "cache", data, ["nope"])


def test_sparse_patterns_are_anchored_and_escaped():
    assert manifest._sparse_pattern("a/b.frag") == "/a/b.frag"
    assert manifest._sparse_pattern("a b/[x]*.h") == "/a\\ b/\\[x]\\*.h"


def test_feasibility_follows_target_stage_support():
    assert run.feasibility(("compute",), "essl") == "infeasible"
    assert run.feasibility(("vertex", "fragment"), "essl") == "feasible"
    assert run.feasibility(("compute", "ray_intersection"), "metal") == "partial"
    assert run.feasibility(("geometry",), "metal") == "infeasible"
    assert run.feasibility(("mesh",), "glsl") == "infeasible"
    assert all(
        run.feasibility((stage,), "spirv") == "feasible" for stage in manifest.STAGES
    )


def test_classification_requires_acceptance_entry_and_no_fallback(tmp_path):
    artifact = tmp_path / "out.spvasm"
    artifact.write_text('OpCapability Shader\nOpEntryPoint GLCompute %main "main"\n')
    accepted = {"verdict": "accepted"}
    assert (
        run.classify(
            {"status": "translated", "markers": 0}, accepted, artifact, "spirv"
        )
        == "usable"
    )
    assert (
        run.classify(
            {"status": "translated", "markers": 2}, accepted, artifact, "spirv"
        )
        == "degraded"
    )
    artifact.write_text("OpCapability Shader\n")
    assert (
        run.classify(
            {"status": "translated", "markers": 0}, accepted, artifact, "spirv"
        )
        == "degraded"
    )
    for verdict, outcome in (
        ("rejected", "invalid"),
        ("timeout", "validator-timeout"),
        ("unavailable", "unvalidated"),
        ("harness-error", "harness-error"),
    ):
        assert (
            run.classify(
                {"status": "translated"}, {"verdict": verdict}, artifact, "spirv"
            )
            == outcome
        )
    for status in ("refused", "translator-error", "timeout", "worker-crash"):
        assert (
            run.classify({"status": status}, {"verdict": None}, artifact, "spirv")
            == status
        )


def test_fallback_markers_and_entry_points():
    assert markers("; WARNING: lowered to 0\nfoo // WARNING x\n") == 2
    assert markers("float placeholder = 0;") == 1
    assert markers("void main() {}") == 0
    assert checks.count_entries("// void main() {}\n", "glsl") == 0
    assert checks.count_entries("void main() {}", "essl") == 1
    assert checks.count_entries("kernel void add(device float* a) {}", "metal") == 1
    assert (
        checks.count_entries(
            "[[object, max_total_threads_per_threadgroup(1)]] void object_main() {}",
            "metal",
        )
        == 1
    )
    assert (
        checks.count_entries(
            '[shader("vertex")]\nfloat4 VSMain() : SV_Position', "hlsl"
        )
        == 2
    )
    assert checks.count_entries('OpEntryPoint Fragment %1 "main"', "spirv") == 1


def test_hlsl_entry_discovery():
    text = '[numthreads(8, 1, 1)]\nvoid CSMain(uint3 id : SV_DispatchThreadID) {}\n[shader("pixel")]\nfloat4 shade() : SV_Target { return 0; }'
    assert checks.hlsl_entries(text, ("compute",)) == [
        ("CSMain", "compute"),
        ("shade", "fragment"),
    ]
    assert checks.hlsl_entries(
        "float4 main() : SV_Target { return 0; }", ("fragment",)
    ) == [("main", "fragment")]
    assert (
        checks.hlsl_entries(
            "float4 main() : SV_Target { return 0; }", ("vertex", "fragment")
        )
        == []
    )


def test_glsl_environment_checks_ray_and_mesh_stages_with_vulkan_rules():
    # GL_EXT_ray_tracing and GL_EXT_mesh_shader exist only for SPIR-V generation; every other stage keeps the
    # OpenGL rules.
    assert checks.glsl_environment(["ray_generation"]) == ["--target-env", "vulkan1.2"]
    assert checks.glsl_environment(["ray_closest_hit", "ray_any_hit"]) == [
        "--target-env",
        "vulkan1.2",
    ]
    assert checks.glsl_environment(["mesh"]) == ["--target-env", "vulkan1.2"]
    assert checks.glsl_environment(["task"]) == ["--target-env", "vulkan1.2"]
    assert checks.glsl_environment(["fragment"]) == []
    assert checks.glsl_environment(["mesh", "fragment"]) == []


def test_glsl_specialization_macros_are_defined_after_version(tmp_path):
    artifact = tmp_path / "unit.glsl"
    text = (
        "// record\n#version 450 core\n"
        "/* CrossGL specialization metadata: name=has_map; constant_id=0;\n"
        " * required_override=true; encoding_default=false;\n"
        " * source_macro=CROSSGL_SPECIALIZE_has_map */\n"
        "void main() {}\n"
    )
    artifact.write_text(text)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = Path(checks.glsl_specialized_source(artifact, text, scratch))
    assert source.read_text().splitlines()[1:4] == [
        "#version 450 core",
        "#define CROSSGL_SPECIALIZE_has_map false",
        "#line 3",
    ]
    plain = "#version 450 core\nvoid main() {}\n"
    assert checks.glsl_specialized_source(artifact, plain, scratch) == artifact


def test_essl_artifacts_must_declare_glsl_es_300(tmp_path):
    desktop = tmp_path / "desktop.essl.glsl"
    desktop.write_text("#version 450\nvoid main() {}\n")
    result = checks.check_essl(desktop, ["vertex"])
    assert result["verdict"] == "rejected"
    assert "#version 300 es" in result["log"]

    es = tmp_path / "es.essl.glsl"
    es.write_text("#version 300 es\nvoid main() { gl_Position = vec4(0.0); }\n")
    if checks.tool("glslangValidator").available:
        assert checks.check_essl(es, ["vertex"])["verdict"] == "accepted"


def test_spirv_environment_follows_module_version():
    assert checks.spirv_environment("; SPIR-V\n; Version: 1.0\n") == "vulkan1.1"
    assert checks.spirv_environment("; Version: 1.5\n") == "vulkan1.2"
    assert checks.spirv_environment("; Version: 1.6\n") == "vulkan1.3"
    assert checks.spirv_environment("OpCapability Shader\n") == "vulkan1.1"


def _records(outcomes):
    data = load()
    units = data["slices"]["candle-metal"]["units"][: len(outcomes)]
    return [
        {
            "cell": f"candle-metal:{unit['path']}|spirv",
            "language": "metal",
            "target": "spirv",
            "feasibility": "feasible",
            "outcome": outcome,
            "seconds": 0.5,
            "message": "",
        }
        for unit, outcome in zip(units, outcomes)
    ]


def test_scoreboard_round_trip_and_regression_rules():
    data = load()
    baseline = run.scoreboard(
        _records(["usable", "refused", "invalid", "degraded"]), data
    )
    row = baseline["cells"]["candle-metal"]["spirv"]
    assert row.startswith("URID") and set(row[4:]) == {"-"}
    assert len(row) == len(data["slices"]["candle-metal"]["units"])
    assert baseline["pairs"]["metal->spirv"]["usableShare"] == 0.25
    failures, better = run.compare(
        baseline, _records(["usable", "refused", "refused", "refused"]), data
    )
    assert failures == [] and len(better) == 2
    failures, _ = run.compare(
        baseline, _records(["refused", "usable", "invalid", "timeout"]), data
    )
    assert [f.split(":")[0] for f in failures] == [
        "Regression candle-metal",
        "Regression candle-metal",
    ]
    failures, _ = run.compare(baseline, _records(["unvalidated"]), data)
    assert failures[0].startswith("Unchecked")
    failures, _ = run.compare(
        {**baseline, "manifestSHA256": "x"}, _records(["usable"]), data
    )
    assert failures == ["The scoreboard was recorded for a different corpus manifest"]
    merged = run.scoreboard(_records(["refused"]), data, baseline)
    assert merged["cells"]["candle-metal"]["spirv"].startswith("RRID")
    with pytest.raises(ValueError, match="does not match the manifest"):
        run.decode({"candle-metal": {"spirv": "U"}}, data)


def test_latency_counts_timeouts_at_the_limit_and_checks_budgets():
    assert run.percentile([], 0.95) is None
    assert run.percentile(list(range(1, 101)), 0.95) == 95
    assert run.percentile([1, 2, 3], 0.5) == 2
    records = [
        {
            "language": "metal",
            "target": "spirv",
            "feasibility": "feasible",
            "outcome": "usable",
            "seconds": 0.5,
        }
        for _ in range(19)
    ]
    records.append({**records[0], "outcome": "timeout", "seconds": None})
    summary = run.summarize(records, timeout=300)
    assert summary["latency"]["metal"]["maxSeconds"] == 300
    assert summary["latency"]["metal"]["p95Seconds"] == 0.5
    assert summary["pairs"]["metal->spirv"]["usableShare"] == 0.95
    assert run.check_budgets(summary, {"p95Seconds": {"metal": 1}}) == []
    summary["latency"]["metal"]["p95Seconds"] = 2
    assert run.check_budgets(summary, {"p95Seconds": {"metal": 1}})[0].startswith(
        "Latency budget exceeded"
    )
    assert run.check_budgets(summary, {"p95Seconds": {}})[0].startswith(
        "No wall latency budget"
    )
    # CPU time is charged the limit for the timed-out cell as well, and has its own budget.
    assert summary["latency"]["metal"]["p95CpuSeconds"] == 300
    assert run.check_budgets(summary, {"p95CpuSeconds": {"metal": 1}})[0].startswith(
        "Latency budget exceeded for metal: CPU p95"
    )


def test_committed_budgets_and_scoreboards_match_the_manifest():
    budgets = json.loads(run.BUDGETS.read_text())
    assert budgets["timeoutSeconds"] > 0
    assert set(budgets["p95Seconds"]) == set(manifest.LANGUAGES)
    data = load()
    for path in sorted(run.SCOREBOARDS.glob("*.json")):
        board = json.loads(path.read_text())
        assert board["manifestSHA256"] == manifest.manifest_digest(), path.name
        run.decode(board["cells"], data)


def test_worker_translates_and_times_one_unit(tmp_path):
    source = tmp_path / "add.comp"
    source.write_text(COMPUTE_GLSL)
    worker = run.Worker(timeout=120)
    try:
        result = worker.translate(
            {
                "id": "x",
                "source": str(source),
                "sourceBackend": "opengl",
                "backend": "vulkan",
                "artifact": str(tmp_path / "out" / "add.spvasm"),
            }
        )
        assert result["status"] == "translated" and result["seconds"] > 0
        assert "OpEntryPoint" in (tmp_path / "out" / "add.spvasm").read_text()
        missing = worker.translate(
            {
                "id": "y",
                "source": str(tmp_path / "absent.comp"),
                "sourceBackend": "opengl",
                "backend": "vulkan",
                "artifact": str(tmp_path / "out" / "absent.spvasm"),
            }
        )
        assert (
            missing["status"] in ("refused", "translator-error")
            and missing["id"] == "y"
        )
    finally:
        worker.stop()


def test_interrupted_translation_resumes_from_its_journal(tmp_path):
    source = tmp_path / "add.comp"
    source.write_text(COMPUTE_GLSL)

    def job(name):
        return {
            "id": name,
            "source": str(source),
            "sourceLanguage": "glsl",
            "sourceBackend": "opengl",
            "backend": "vulkan",
            "artifact": str(tmp_path / "out" / f"{name}.spvasm"),
        }

    kept = tmp_path / "out" / "kept.spvasm"
    kept.parent.mkdir()
    kept.write_text("; recorded earlier\n")
    journal = tmp_path / "translations.jsonl"
    # 'kept' has its artifact and is reused; 'lost' lost its artifact and the cut-short line is ignored.
    journal.write_text(
        json.dumps({"id": "kept", "status": "translated", "seconds": 0.5})
        + "\n"
        + json.dumps({"id": "lost", "status": "translated", "seconds": 0.5})
        + "\n"
        + '{"id": "cut'
    )
    results = run.translate_all([job("kept"), job("lost")], 1, 120, journal=journal)
    assert results["kept"] == {"status": "translated", "seconds": 0.5}
    assert (
        results["lost"]["status"] == "translated" and results["lost"]["seconds"] != 0.5
    )
    assert kept.read_text() == "; recorded earlier\n"
    assert set(run.load_translations(journal)) == {"kept", "lost"}


def test_unit_identity():
    unit = Unit("s", "glsl", "held-out", "a/b.comp", "0" * 64, ("compute",), Path("/x"))
    assert unit.id == "s:a/b.comp" and run.cell_id(unit, "spirv") == "s:a/b.comp|spirv"
