"""Checks for the ComputeSharp workflow triggers and retained artifacts.

The workflow is path-filtered to the demo and to the CrossTL modules that its
HLSL translation imports. The import test translates actual generated
ComputeSharp HLSL in a fresh interpreter and requires every loaded CrossTL
module to match a trigger path, so a new dependency of the translation path
cannot silently stop triggering the port's CI.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT  # noqa: E402

REPO_ROOT = ROOT.parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "project-demo-testing.yml"
TRIGGER_PATHS = {
    ".github/workflows/project-demo-testing.yml",
    "crosstl/*.py",
    "crosstl/backend/**",
    "crosstl/translator/**",
    "demos/integrations/computesharp/**",
    "pyproject.toml",
    "setup.py",
}
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


def event_paths(text: str, event: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index("on:")
    header = f"  {event}:"
    index = lines.index(header, start)
    paths: list[str] = []
    for line in lines[index + 1 :]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        if line.strip() == "paths:":
            paths = []
            continue
        item = re.fullmatch(r"\s+- \"([^\"]+)\"", line)
        if item:
            paths.append(item.group(1))
    return paths


def filter_regex(pattern: str) -> re.Pattern[str]:
    """GitHub path filters: ``*`` stays in one directory, ``**`` spans them."""
    parts = re.split(r"(\*\*|\*)", pattern)
    regex = "".join(
        ".*" if part == "**" else "[^/]*" if part == "*" else re.escape(part)
        for part in parts
    )
    return re.compile(regex)


def triggers(path: str, patterns: set[str]) -> bool:
    return any(filter_regex(pattern).fullmatch(path) for pattern in patterns)


def test_pull_requests_and_pushes_use_the_same_bounded_trigger_paths():
    text = WORKFLOW.read_text(encoding="utf-8")
    for event in ("pull_request", "push"):
        assert set(event_paths(text, event)) == TRIGGER_PATHS, event
    assert "schedule:" not in text


def test_trigger_paths_exclude_other_projects():
    for path in (
        "crosstl/project/pipeline.py",
        "crosstl/project/native_runtime_drivers.py",
        "demos/integrations/mlx/run_mlx_porting.py",
        ".github/workflows/mlx-project-porting.yml",
        "tests/test_translator/test_project_translation.py",
        "docs/source/project-porting.rst",
    ):
        assert not triggers(path, TRIGGER_PATHS), path
    for path in (
        "crosstl/_crosstl.py",
        "crosstl/backend/DirectX/DirectxParser.py",
        "crosstl/translator/codegen/GLSL_codegen.py",
        "crosstl/translator/codegen/metal_codegen.py",
        "demos/integrations/computesharp/runtime/README.md",
    ):
        assert triggers(path, TRIGGER_PATHS), path


def test_trigger_paths_cover_every_module_the_translation_loads(tmp_path):
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
    uncovered = [path for path in loaded if not triggers(path, TRIGGER_PATHS)]
    assert uncovered == []


def test_uploaded_artifacts_have_bounded_retention():
    text = WORKFLOW.read_text(encoding="utf-8")
    uploads = text.count("actions/upload-artifact@")
    retention = [int(days) for days in re.findall(r"retention-days: (\d+)", text)]
    assert uploads > 0
    assert len(retention) == uploads
    assert max(retention) <= MAX_RETENTION_DAYS


def test_filter_matching_follows_github_glob_semantics():
    assert triggers("crosstl/_crosstl.py", {"crosstl/*.py"})
    assert not triggers("crosstl/project/pipeline.py", {"crosstl/*.py"})
    assert triggers("crosstl/backend/Metal/MetalParser.py", {"crosstl/backend/**"})
    assert not triggers("crosstl/backends.py", {"crosstl/backend/**"})
