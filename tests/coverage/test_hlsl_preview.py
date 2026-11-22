"""HLSL probes that DXC's Shader Model 6.10 preview checks, and the printf probe that DXC checks as SPIR-V."""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.fake_commands import write_python_command
from tools.coverage import probes, runner, semantic, validators
from tools.coverage.features import hlsl
from tools.coverage.model import TRIANGLE_POSITIONS, Type


def by_name(features):
    return {feature.name: feature for feature in features}


# The release table declares shader model 6.10 entries under the names of an earlier draft.
RELEASE = """
namespace Intrinsics {
float<3> [[rn]] WorldRayOrigin();
uint   [[rn,min_sm=6.10]] GetGroupWaveIndex();
void [[ro,min_sm=6.10]] TriangleObjectPosition();
void [[]] printf(in string Format, ...);
} namespace

namespace DxHitObjectMethods {
    bool [[rn,class_prefix,min_sm=6.9]] IsHit();
    uint [[ro,class_prefix,min_sm=6.10]] ClusterID();
} namespace
"""

PREVIEW = """
namespace Intrinsics {
float<3> [[rn]] WorldRayOrigin();
uint   [[rn,min_sm=6.10]] GetGroupWaveIndex();
triangle_positions [[rn,min_sm=6.10]] TriangleObjectPositions();
uint [[rn]] PreviewOnlyBeforeSixTen();
} namespace

namespace DxHitObjectMethods {
    uint [[rn,class_prefix,min_sm=6.9]] IsHit();
    uint [[rn,class_prefix,min_sm=6.10]] GetClusterID();
    triangle_positions [[rn,class_prefix,min_sm=6.10]] TriangleObjectPositions();
} namespace
"""


def test_shader_model_6_10_entries_come_from_the_preview_table():
    features = by_name(hlsl.parse(RELEASE, frozenset(), PREVIEW))
    # The release compiler cannot compile the draft names, and the preview renamed them.
    assert "TriangleObjectPosition" not in features
    assert "DxHitObject.ClusterID" not in features
    for name in (
        "GetGroupWaveIndex",
        "TriangleObjectPositions",
        "DxHitObject.GetClusterID",
        "DxHitObject.TriangleObjectPositions",
    ):
        assert features[name].target == validators.HLSL_PREVIEW, name
        assert features[name].note == "", name
    positions = features["TriangleObjectPositions"]
    assert positions.signature.result == Type.other(TRIANGLE_POSITIONS)
    assert positions.stage == "closest-hit"
    # Entries the release compiler has stay the release table's, and the preview adds none of its own below 6.10.
    assert features["WorldRayOrigin"].target == ""
    assert features["DxHitObject.IsHit"].signature.result == Type("bool")
    assert "PreviewOnlyBeforeSixTen" not in features


def test_without_the_preview_table_shader_model_6_10_entries_are_listed_with_a_note():
    features = by_name(hlsl.parse(RELEASE))
    assert features["GetGroupWaveIndex"].signature is None
    assert features["GetGroupWaveIndex"].note == "needs shader model 6.10"
    assert features["GetGroupWaveIndex"].target == ""


def test_preview_probes_compile_at_shader_model_6_10_and_printf_as_spirv():
    assert validators.hlsl_profile("compute") == "cs_6_9"
    assert validators.hlsl_profile("compute", validators.HLSL_PREVIEW) == "cs_6_10"
    assert validators.hlsl_profile("closest-hit", validators.HLSL_PREVIEW) == "lib_6_10"
    assert validators.hlsl_profile("compute", validators.HLSL_SPIRV) == "cs_6_9"
    assert validators.toolchain("hlsl", validators.HLSL_PREVIEW) == ("dxc-preview",)
    assert validators.toolchain("hlsl", validators.HLSL_SPIRV) == ("dxc",)


def _commands(monkeypatch, target):
    commands = []

    def run(arguments, cwd, environment=None, stdout=True):
        commands.append(arguments)
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "command", lambda name: (name,))
    monkeypatch.setattr(validators, "run", run)
    validators.validate("hlsl", "[numthreads(1, 1, 1)]", "compute", "main", target)
    return commands


def test_dxc_runs_the_preview_compiler_and_the_spirv_target(monkeypatch):
    tail = ["-HV", "2021", "-enable-16bit-types", "probe.hlsl", "-Fo", "probe.bin"]
    assert _commands(monkeypatch, "") == [["dxc", "-T", "cs_6_9", "-E", "main", *tail]]
    assert _commands(monkeypatch, validators.HLSL_PREVIEW) == [
        ["dxc-preview", "-T", "cs_6_10", "-E", "main", *tail]
    ]
    environment = f"-fspv-target-env={validators.SPIRV_ENVIRONMENT}"
    assert _commands(monkeypatch, validators.HLSL_SPIRV) == [
        ["dxc", "-spirv", environment, "-T", "cs_6_9", "-E", "main", *tail]
    ]


def test_a_preview_probe_is_untested_without_the_preview_compiler(monkeypatch):
    feature = by_name(hlsl.parse(RELEASE, frozenset(), PREVIEW))["GetGroupWaveIndex"]
    monkeypatch.setattr(
        validators, "command", lambda name: () if name == "dxc-preview" else (name,)
    )
    result = runner.run_input(feature, "hlsl")
    assert (result.outcome, result.detail) == (
        "untested",
        "dxc-preview is not installed",
    )


def test_probes_read_a_triangle_through_its_first_vertex_and_print_an_empty_format():
    features = by_name(hlsl.parse(RELEASE, frozenset(), PREVIEW))
    native = probes.build(features["TriangleObjectPositions"], "hlsl").source
    assert "float3 r = TriangleObjectPositions().p0;" in native
    crossgl = probes.build(features["DxHitObject.TriangleObjectPositions"], "crossgl")
    assert "vec3 r = obj.TriangleObjectPositions().p0;" in crossgl.source
    printed = probes.build(features["printf"], "hlsl").source
    assert 'printf("");' in printed
    assert 'printf("");' in probes.build(features["printf"], "crossgl").source


def test_a_probe_checked_as_spirv_still_gets_a_semantic_id(monkeypatch):
    features = by_name(hlsl.parse(RELEASE, frozenset(), PREVIEW))
    monkeypatch.setattr(semantic.lowering, "can_lower", lambda language: False)
    # The SPIR-V target is the one the semantic step lowers to, so only the missing compiler stops it.
    assert semantic.lower(features["printf"]) == (
        "",
        "dxc or spirv-dis is not installed",
    )
    assert semantic.lower(features["GetGroupWaveIndex"]) == (
        "",
        f"probed for {validators.HLSL_PREVIEW}, not SPIR-V",
    )


_INSTALLER = (
    Path(__file__).resolve().parents[2]
    / ".github/actions/install-dxc-preview/action.yml"
)
_FAKE = """\
import os
import sys
from pathlib import Path

tool = Path(sys.argv[0]).stem
args = sys.argv[1:]
with Path(os.environ["TEST_LOG"]).open("a") as log:
    log.write(tool + "\\n")
if tool == "curl":
    Path(args[args.index("-fsSLo") + 1]).write_text("archive")
elif tool == "sha256sum":
    sys.exit(0 if "dea02d61c0548a6163a5fb45daa9263faeb5334e35d25a30b0293a32d089ceae" in sys.stdin.read() else 1)
elif tool == "tar":
    binary = Path(args[args.index("-C") + 1]) / "bin/dxc"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\\necho '" + os.environ["TEST_VERSION"] + "'\\n")
    binary.chmod(0o755)
"""


@pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash"), reason="Bash required"
)
@pytest.mark.parametrize(
    "version, installed",
    [
        ("libdxcompiler.so: 1.10(5373-c4d8f4f9)(1.10.0.5373)", True),
        # The release's library, loaded instead of the preview's.
        ("libdxcompiler.so: 1.9(5191-d355aa83)(1.9.0.5191)", False),
    ],
)
def test_the_preview_installer_checks_the_archive_and_the_compiler_version(
    tmp_path, version, installed
):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("curl", "tar", "sha256sum"):
        write_python_command(tools / name, _FAKE)
    temp = tmp_path / "runner temp"
    temp.mkdir()
    environment = tmp_path / "github-env"
    body = textwrap.dedent(_INSTALLER.read_text().split("run: |\n", 1)[1])
    result = subprocess.run(
        [shutil.which("bash"), "-c", body],
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "RUNNER_TEMP": str(temp),
            "GITHUB_ENV": str(environment),
            "TEST_LOG": str(tmp_path / "calls"),
            "TEST_VERSION": version,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is installed, result.stdout + result.stderr
    assert (tmp_path / "calls").read_text().split() == ["curl", "sha256sum", "tar"]
    if installed:
        binary = temp / "dxc-preview/bin/dxc"
        assert environment.read_text() == f"CROSSTL_COVERAGE_DXC_PREVIEW={binary}\n"
    else:
        assert not environment.exists()
