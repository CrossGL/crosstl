"""Checks for the ComputeSharp workflow triggers, plan, result and artifacts.

Every pull request, merge-queue group and push to main starts the workflow so
that its single result check can be required; ``ci_plan.py`` skips the port's
jobs when nothing the port depends on changed. The import test translates
actual generated ComputeSharp HLSL in a fresh interpreter and requires every
loaded CrossTL module to match a relevant path, so a new dependency of the
translation path cannot silently skip the port's CI.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

PLAN = load_module("computesharp_ci_plan", ROOT / "ci_plan.py")
REPO_ROOT = ROOT.parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "computesharp.yml"
RELEVANT_PATHS = {
    ".github/workflows/computesharp.yml",
    "crosstl/*.py",
    "crosstl/backend/**",
    "crosstl/translator/**",
    "demos/integrations/computesharp/**",
    "pyproject.toml",
    "setup.py",
    "tests/test_backend/test_directx/test_hlsl_cbuffer_layouts.py",
}
RECORD_PATHS = {
    "demos/integrations/computesharp/README.md",
    "demos/integrations/computesharp/results.json",
}
# The jobs that build, translate and execute the port, as opposed to the Linux
# contract job that checks the demo's tests and published records.
EXECUTION_JOBS = {
    "original-directx",
    "adapted-source-parity",
    "translated-metal",
    "translated-opengl",
    "qualification",
}
RESULT_CHECK = "ComputeSharp validation result"
MAX_RETENTION_DAYS = 14

# Shaders emitted by the ComputeSharp source generator at the pinned commit
# (a structured-buffer write and a texture read/write with typed formats).
GENERATED_HLSL = {
    "buffer": (
        """\
#define __GroupSize__get_X 64
#define __GroupSize__get_Y 1
#define __GroupSize__get_Z 1

cbuffer _ : register(b0)
{
    uint __x;
    uint __y;
    uint __z;
}

RWStructuredBuffer<float> __reserved__buffer : register(u0);

[NumThreads(__GroupSize__get_X, __GroupSize__get_Y, __GroupSize__get_Z)]
void Execute(uint3 ThreadIds : SV_DispatchThreadID)
{
    if (ThreadIds.x < __x && ThreadIds.y < __y && ThreadIds.z < __z)
    {
        __reserved__buffer[0] = 42;
    }
}
"""
    ),
    "texture": (
        """\
#define __GroupSize__get_X 8
#define __GroupSize__get_Y 8
#define __GroupSize__get_Z 1

cbuffer _ : register(b0)
{
    uint __x;
    uint __y;
    uint __z;
}

Texture2D<unorm float> source : register(t0);

RWTexture2D<unorm float> destination : register(u0);

[NumThreads(__GroupSize__get_X, __GroupSize__get_Y, __GroupSize__get_Z)]
void Execute(uint3 ThreadIds : SV_DispatchThreadID)
{
    if (ThreadIds.x < __x && ThreadIds.y < __y && ThreadIds.z < __z)
    {
        destination[ThreadIds.xy] = (source[ThreadIds.xy] + destination[ThreadIds.xy]) / 2;
    }
}
"""
    ),
}


def load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_planner_owns_the_relevant_paths():
    assert set(PLAN.RELEVANT_PATHS) == RELEVANT_PATHS
    assert len(PLAN.RELEVANT_PATHS) == len(RELEVANT_PATHS)
    assert set(PLAN.RECORD_PATHS) == RECORD_PATHS
    assert all(PLAN.is_relevant(path) for path in RECORD_PATHS)


# The demo's scripts write and read per-run ``results.json`` files under a
# run's work or output directory; none of them names the published summary.
PER_RUN_RESULTS = {
    "companions.py": {"directory / 'run' / 'results.json'"},
    "qualify.py": {"run / 'results.json'", "output / 'results.json'"},
    "run_tests.py": {"self.work / 'results.json'", "run.work / 'results.json'"},
}


def _record_names(text: str) -> list[str]:
    return [name for name in ("README.md", "results.json") if name in text]


def test_published_records_are_read_only_by_the_contract_tests():
    """No build, translation or execution input names the published records.

    The plan skips the execution jobs for a change confined to these files,
    which is sound only while the contract tests (``test_results.py``) are
    their only reader.
    """
    jobs = load_workflow()["jobs"]
    for name in EXECUTION_JOBS:
        for step in jobs[name]["steps"]:
            assert _record_names(json.dumps(step)) == [], (name, step.get("name"))
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"tests", "__pycache__"} or path.name == "ci_plan.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            # Path operands, not prose: docstrings and messages contain spaces.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _record_names(node.value) and not re.search(r"\s", node.value):
                    expression = ast.unparse(parents[node])
                    allowed = PER_RUN_RESULTS.get(relative.as_posix(), set())
                    assert expression in allowed, (relative, node.lineno)
    for pattern in ("*.cs", "*.csproj", "*.props", "*.targets", "*.sh", "*.ps1"):
        for path in ROOT.rglob(pattern):
            text = path.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
            assert _record_names(text) == [], path.relative_to(ROOT)


def test_every_pull_request_and_merge_group_starts_the_workflow():
    triggers = load_workflow()[True]
    assert set(triggers) == {"pull_request", "push", "merge_group", "workflow_dispatch"}
    assert triggers["pull_request"] == {"branches": ["main"]}
    assert triggers["push"] == {"branches": ["main"]}
    # Path filters would leave the required result check unreported.
    for event, policy in triggers.items():
        assert not set(policy or {}) & {"paths", "paths-ignore"}, event
    assert "schedule" not in triggers


def test_plan_job_selects_the_port_from_the_event_diff():
    plan = load_workflow()["jobs"]["plan"]
    assert plan["outputs"] == {
        "tested": "${{ steps.queue.outputs.tested }}",
        "execute": "${{ steps.plan.outputs.execute }}",
        "run": "${{ steps.plan.outputs.run }}",
    }
    assert plan["timeout-minutes"] <= 5
    checkout = next(
        step
        for step in plan["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0
    step = next(step for step in plan["steps"] if step.get("id") == "plan")
    assert step["run"] == (
        'python demos/integrations/computesharp/ci_plan.py --event "$COMPUTESHARP_CI_EVENT"'
    )
    assert step["env"] == {
        "COMPUTESHARP_CI_EVENT": "${{ github.event_name }}",
        "COMPUTESHARP_CI_BASE": (
            "${{ github.event.pull_request.base.sha || "
            "github.event.merge_group.base_sha || github.event.before }}"
        ),
        "COMPUTESHARP_CI_HEAD": (
            "${{ github.event.pull_request.head.sha || "
            "github.event.merge_group.head_sha || github.sha }}"
        ),
    }


def test_every_port_job_waits_for_the_plan():
    jobs = load_workflow()["jobs"]
    for name, job in jobs.items():
        if name in {"plan", "result"}:
            continue
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "plan" in needs, name
        output = "execute" if name in EXECUTION_JOBS else "run"
        assert f"needs.plan.outputs.{output} == 'true'" in job["if"], name
        other = "run" if output == "execute" else "execute"
        assert f"needs.plan.outputs.{other}" not in job["if"], name
    assert EXECUTION_JOBS | {"plan", "result", "discovery-contract"} == set(jobs)


def test_one_result_job_reports_every_job():
    jobs = load_workflow()["jobs"]
    result = jobs["result"]
    assert result["name"] == RESULT_CHECK
    assert result["if"] == "always()"
    assert set(result["needs"]) == set(jobs) - {"result"}
    assert result["timeout-minutes"] <= 5
    step = result["steps"][0]
    assert step["env"]["RESULTS"] == "${{ join(needs.*.result, ' ') }}"
    assert "success|skipped) ;;" in step["run"]
    assert "exit 1" in step["run"]


def test_superseded_pull_request_runs_are_replaced():
    policy = load_workflow()["concurrency"]
    assert policy == {
        "group": (
            "ci-${{ github.workflow }}-${{ github.event_name }}-"
            "${{ github.event.pull_request.number || github.run_id }}"
        ),
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


def test_jobs_are_bounded_and_cache_pip_downloads():
    for name, job in load_workflow()["jobs"].items():
        assert 0 < job["timeout-minutes"] <= 90, name
        installs = any("pip install" in step.get("run", "") for step in job["steps"])
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/setup-python@"):
                assert (step["with"].get("cache") == "pip") == installs, name


RECORDS = (
    "demos/integrations/computesharp/README.md",
    "demos/integrations/computesharp/results.json",
)


@pytest.mark.parametrize(
    "event,paths,run,execute",
    [
        ("schedule", (), True, True),
        ("workflow_dispatch", (), True, True),
        ("pull_request", (), False, False),
        ("pull_request", ("docs/source/project-porting.rst",), False, False),
        ("pull_request", ("demos/integrations/mlx/run_mlx_porting.py",), False, False),
        ("pull_request", ("crosstl/translator/codegen/GLSL_codegen.py",), True, True),
        ("merge_group", ("crosstl/project/pipeline.py",), False, False),
        ("merge_group", ("README.md", "crosstl/_crosstl.py"), True, True),
        ("push", ("demos/integrations/computesharp/results.json",), True, False),
        ("pull_request", RECORDS, True, False),
        ("pull_request", (*RECORDS, "docs/source/index.rst"), True, False),
        (
            "pull_request",
            (*RECORDS, "demos/integrations/computesharp/qualify.py"),
            True,
            True,
        ),
        (
            "pull_request",
            ("demos/integrations/computesharp/patches/README.md",),
            True,
            True,
        ),
        ("push", (".github/workflows/computesharp.yml",), True, True),
        ("push", (".github/workflows/demo-project-testing.yml",), False, False),
    ],
)
def test_validation_plan_follows_the_changed_paths(event, paths, run, execute):
    assert PLAN.validation_plan(event, paths) == {"execute": execute, "run": run}


def test_validation_plan_rejects_unknown_events():
    with pytest.raises(ValueError, match="Unsupported workflow event"):
        PLAN.validation_plan("pull_request_target", ())


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _commit(repo: Path, relative: str, message: str) -> str:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message, encoding="utf-8")
    _git(repo, "add", relative)
    _git(
        repo,
        "-c",
        "user.name=CI",
        "-c",
        "user.email=ci@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def test_plan_reads_the_diff_from_the_merge_base(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    root = _commit(repo, "README.md", "root")
    _git(repo, "checkout", "-q", "-b", "port")
    head = _commit(repo, "docs/notes.md", "docs only")
    _git(repo, "checkout", "-q", "main")
    # A translator change that reached the base after the branch point must
    # not select the port for the branch.
    base = _commit(repo, "crosstl/translator/parser.py", "base change")
    monkeypatch.chdir(repo)
    assert PLAN.changed_paths(base, head) == ("docs/notes.md",)
    nothing = {"execute": False, "run": False}
    everything = {"execute": True, "run": True}
    assert PLAN.plan_for_event("pull_request", base, head) == nothing
    moved = _commit(repo, "demos/integrations/computesharp/run_tests.py", "port")
    assert PLAN.plan_for_event("merge_group", root, moved) == everything
    cited = _commit(repo, "demos/integrations/computesharp/README.md", "cite")
    assert PLAN.plan_for_event("pull_request", moved, cited) == {
        "execute": False,
        "run": True,
    }
    # A push without a known old tip validates everything.
    assert PLAN.plan_for_event("push", "0" * 40, head) == everything
    assert PLAN.plan_for_event("push", "f" * 40, head) == everything
    with pytest.raises(ValueError, match="full Git commit hashes"):
        PLAN.changed_paths("main", head)

    output = tmp_path / "github-output"
    assert (
        PLAN.main(
            [
                "--event",
                "pull_request",
                "--base",
                base,
                "--head",
                head,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text(encoding="utf-8") == "execute=false\nrun=false\n"


def test_target_jobs_require_the_hlsl_cbuffer_layout_dispatches():
    """Each translated target job dispatches the general layout checks natively."""
    jobs = load_workflow()["jobs"]
    for name, extra in (
        ("translated-metal", ()),
        ("translated-opengl", ("moderngl==",)),
    ):
        steps = jobs[name]["steps"]
        install = next(step for step in steps if step.get("name") == "Install CrossTL")
        for package in ("pytest==", *extra):
            assert package in install["run"], (name, package)
        checks = [
            step
            for step in steps
            if "tests/test_backend/test_directx/test_hlsl_cbuffer_layouts.py"
            in step.get("run", "")
        ]
        assert len(checks) == 1, name
        assert checks[0]["env"]["CROSSTL_REQUIRE_HLSL_CBUFFER_EXECUTION"] == "1"
        assert checks[0]["if"] == "${{ success() }}"
        assert "--junitxml=evidence/" in checks[0]["run"]
        dependent_steps = [
            step
            for step in steps
            if step.get("name")
            in {
                "Run the companion test projects on Metal with negative controls",
                "Evaluate the HLSL intrinsics on Metal",
                "Dispatch the HLSL constant-buffer layout checks on Metal",
                "Run the companion test projects on OpenGL with negative controls",
                "Evaluate the HLSL intrinsics on OpenGL",
                "Dispatch the HLSL constant-buffer layout checks on OpenGL",
            }
        ]
        assert dependent_steps
        assert {step.get("if") for step in dependent_steps} == {"${{ success() }}"}
    assert PLAN.is_relevant(
        "tests/test_backend/test_directx/test_hlsl_cbuffer_layouts.py"
    )


def test_portable_runtime_keeps_the_pinned_dependency_and_audit_scope():
    """Portable test builds retain the immutable upstream package graph."""
    runtime = ET.parse(ROOT / "runtime" / "ComputeSharp.Portable.csproj")
    references = runtime.findall(".//PackageReference[@Include='SixLabors.ImageSharp']")
    assert [reference.attrib["Version"] for reference in references] == ["3.1.11"]

    # Portable.targets is passed globally to every portable build, so the runtime and
    # the companion projects all import the same advisory exceptions through it.
    targets_path = ROOT / "runtime" / "Portable.targets"
    targets = ET.parse(targets_path).getroot()
    imports = [element.attrib["Project"] for element in targets.iter("Import")]
    assert imports == ["$(MSBuildThisFileDirectory)../baseline-audit.props"]
    targets_body = targets_path.read_text(encoding="utf-8").split("-->", 1)[1]
    assert "CustomAfterMicrosoftCommonTargets" not in targets_body

    # Every advisory exception is documented, and the ImageSharp advisories fixed
    # only by license-gated releases (3.2.0, 4.1.x) stay suppressed for the corpus.
    audit = (ROOT / "baseline-audit.props").read_text(encoding="utf-8")
    suppressed = re.findall(r"advisories/(GHSA-[0-9a-z-]+)", audit)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert suppressed and all(advisory in readme for advisory in suppressed)
    assert "GHSA-v76p-62qx-wwq2" in suppressed

    dependabot = yaml.safe_load(
        (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    nuget = [
        update
        for update in dependabot["updates"]
        if update["package-ecosystem"] == "nuget"
        and "/demos/integrations/computesharp/**" in update.get("directories", [])
    ]
    assert len(nuget) == 1
    assert {"dependency-name": "SixLabors.ImageSharp"} in nuget[0].get("ignore", [])


def test_upstream_project_references_have_one_build_instance():
    """No upstream project is built twice with different global properties.

    The upstream test projects reach ComputeSharp.Core without ComputeSharpRoot, while the
    portable runtime and the intrinsic probe receive it as a global property. A reference
    that forwards it creates a second ComputeSharp.Core instance, which MSBuild builds in
    parallel into the same output directory (both write ComputeSharp.Core.deps.json).
    """
    upstream = "$(ComputeSharpRoot)/src/"
    projects = sorted(ROOT.rglob("*.csproj")) + sorted(ROOT.rglob("*.targets"))
    checked = []
    for path in projects:
        if {"bin", "obj"} & set(path.relative_to(ROOT).parts):
            continue
        for reference in ET.parse(path).getroot().iter("ProjectReference"):
            include = reference.attrib.get("Include", "")
            if not include.startswith(upstream):
                continue
            removed = reference.attrib.get("GlobalPropertiesToRemove", "").split(";")
            assert "ComputeSharpRoot" in removed, (path.name, include)
            assert "AdditionalProperties" not in reference.attrib, (path.name, include)
            checked.append((path.name, include[len(upstream) :]))
    assert (
        "ComputeSharp.Portable.csproj",
        "ComputeSharp.Core/ComputeSharp.Core.csproj",
    ) in checked
    assert (
        "IntrinsicProbe.csproj",
        "ComputeSharp.Core/ComputeSharp.Core.csproj",
    ) in checked


def test_relevant_paths_exclude_other_projects():
    for path in (
        "crosstl/project/pipeline.py",
        "crosstl/project/native_runtime_drivers.py",
        "demos/integrations/mlx/run_mlx_porting.py",
        ".github/workflows/mlx-project-porting.yml",
        "tests/test_translator/test_project_translation.py",
        "docs/source/project-porting.rst",
    ):
        assert not PLAN.is_relevant(path), path
    for path in (
        "crosstl/_crosstl.py",
        "crosstl/backend/DirectX/DirectxParser.py",
        "crosstl/translator/codegen/GLSL_codegen.py",
        "crosstl/translator/codegen/metal_codegen.py",
        "demos/integrations/computesharp/runtime/README.md",
    ):
        assert PLAN.is_relevant(path), path


def test_relevant_paths_cover_every_module_the_translation_loads(tmp_path):
    for name, source in GENERATED_HLSL.items():
        (tmp_path / f"{name}.hlsl").write_text(source, encoding="utf-8")
    script = textwrap.dedent("""
        import json
        import sys
        from pathlib import Path

        repo, demo, sources = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
        sys.path[:0] = [str(repo), demo]
        import crosstl
        import translate

        translated = 0
        for source in sorted(Path(sources).glob("*.hlsl")):
            for target in sorted(translate.TARGETS):
                crosstl.translate_with_metadata(
                    str(source),
                    backend=target,
                    format_output=False,
                    source_options=translate.SOURCE_OPTIONS[target],
                )
                translated += 1
        package = (repo / "crosstl").resolve()
        loaded = sorted(
            Path(module.__file__).resolve().relative_to(repo.resolve()).as_posix()
            for module in list(sys.modules.values())
            if getattr(module, "__file__", None)
            and Path(module.__file__).resolve().is_relative_to(package)
        )
        print(json.dumps({"translated": translated, "loaded": loaded}))
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(REPO_ROOT), str(ROOT), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["translated"] == len(GENERATED_HLSL) * 2
    loaded = report["loaded"]
    assert "crosstl/backend/DirectX/DirectxParser.py" in loaded
    assert "crosstl/translator/codegen/metal_codegen.py" in loaded
    assert "crosstl/translator/codegen/GLSL_codegen.py" in loaded
    uncovered = [path for path in loaded if not PLAN.is_relevant(path)]
    assert uncovered == []


def test_uploaded_artifacts_have_bounded_retention():
    text = WORKFLOW.read_text(encoding="utf-8")
    uploads = text.count("actions/upload-artifact@")
    retention = [int(days) for days in re.findall(r"retention-days: (\d+)", text)]
    assert uploads > 0
    assert len(retention) == uploads
    assert max(retention) <= MAX_RETENTION_DAYS


def test_path_matching_follows_github_glob_semantics():
    def matches(path: str, pattern: str) -> bool:
        return PLAN.path_regex(pattern).fullmatch(path) is not None

    assert matches("crosstl/_crosstl.py", "crosstl/*.py")
    assert not matches("crosstl/project/pipeline.py", "crosstl/*.py")
    assert matches("crosstl/backend/Metal/MetalParser.py", "crosstl/backend/**")
    assert not matches("crosstl/backends.py", "crosstl/backend/**")
    assert not matches("pyproject.toml.bak", "pyproject.toml")
