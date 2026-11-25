"""WGSL is checked by naga and then by Tint (tools/coverage/tint.py), and naga reads it without ``enable subgroups;``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.coverage import __main__ as cli
from tools.coverage import lowering, probes, tint, validators
from tools.coverage.features import wgsl

WEBGPU = Path(tint.__file__).with_name("webgpu")
SUBGROUPS = (
    "enable subgroups;\n"
    "diagnostic(off, subgroup_uniformity);\n"
    "@group(0) @binding(0) var<storage, read_write> outp: array<f32>;\n"
    "@compute @workgroup_size(64, 1, 1)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "    if (id.x % 3u != 0u) {\n"
    "        outp[id.x] = subgroupAdd(outp[id.x]);\n"
    "    }\n"
    "}\n"
)


def test_naga_reads_wgsl_with_the_subgroups_directive_blanked_out():
    read = validators.naga_source(SUBGROUPS)
    # Only the directive naga has not implemented goes; every other line keeps its text and its number.
    assert read.splitlines() == ["", *SUBGROUPS.splitlines()[1:]]
    assert validators.naga_source("enable f16;\n") == "enable f16;\n"
    assert (
        validators.naga_source("  enable subgroups ;  \nfn f() {}\n") == "\nfn f() {}\n"
    )


def test_wgsl_toolchain_is_naga_then_tint():
    assert validators.toolchain("wgsl") == ("naga", "tint")


def _recorder(tmp_path: Path, tool: str, status: int) -> tuple[str, ...]:
    """A stand-in for ``tool`` that copies the file it is given to ``tmp_path/<tool>.wgsl`` and exits ``status``."""
    script = tmp_path / f"{tool}.py"
    script.write_text(
        "import shutil, sys\n"
        f"shutil.copy(sys.argv[1], {str(tmp_path / (tool + '.wgsl'))!r})\n"
        f"print('probe.wgsl:1:1: error: {tool} rejected it', file=sys.stderr) if {status} else None\n"
        f"sys.exit({status})\n"
    )
    return (sys.executable, str(script))


@pytest.mark.parametrize("naga_status, tint_status", [(0, 0), (0, 1), (1, 0)])
def test_naga_checks_the_blanked_source_and_tint_the_source_as_written(
    monkeypatch, tmp_path, naga_status, tint_status
):
    commands = {
        "naga": _recorder(tmp_path, "naga", naga_status),
        "tint": _recorder(tmp_path, "tint", tint_status),
    }
    monkeypatch.setattr(validators, "command", lambda name: commands[name])
    verdict = validators.validate("wgsl", SUBGROUPS)
    assert verdict.accepted == (not naga_status and not tint_status)
    assert (tmp_path / "naga.wgsl").read_text() == validators.naga_source(SUBGROUPS)
    if naga_status:
        # Tint does not run on a module naga rejects.
        assert "naga rejected it" in verdict.log
        assert not (tmp_path / "tint.wgsl").exists()
    else:
        assert (tmp_path / "tint.wgsl").read_text() == SUBGROUPS
        assert ("tint rejected it" in verdict.log) == bool(tint_status)


def test_lowering_hands_naga_the_blanked_source(monkeypatch, tmp_path):
    commands = {
        "naga": _recorder(tmp_path, "naga", 1),
        "spirv-dis": ("false",),
    }
    monkeypatch.setattr(validators, "command", lambda name: commands[name])
    assembly, log = lowering.lower("wgsl", SUBGROUPS)
    assert not assembly and "naga rejected it" in log
    assert (tmp_path / "naga.wgsl").read_text() == validators.naga_source(SUBGROUPS)


def test_tint_needs_node_and_the_configured_webgpu_package(monkeypatch, tmp_path):
    monkeypatch.delenv(tint.VARIABLE, raising=False)
    assert tint.command() == () and tint.version() == ""
    package = tmp_path / "webgpu"
    package.mkdir()
    monkeypatch.setenv(tint.VARIABLE, str(package))
    # A directory that holds no package is not an installation.
    assert tint.command() == ()
    (package / "index.js").write_text("")
    (package / "package.json").write_text(json.dumps({"version": "0.6.2"}))
    monkeypatch.setattr(tint.shutil, "which", lambda name: None)
    assert tint.command() == ()
    monkeypatch.setattr(tint.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert tint.command() == ("/usr/bin/node", str(tint.SCRIPT), str(package))
    assert tint.version() == "tint (npm webgpu 0.6.2)"


def test_the_pinned_webgpu_package_is_locked_with_integrity_hashes():
    manifest = json.loads((WEBGPU / "package.json").read_text())
    assert manifest["dependencies"] == {"webgpu": "0.6.2"}
    lock = json.loads((WEBGPU / "package-lock.json").read_text())
    assert lock["packages"][""]["dependencies"] == manifest["dependencies"]
    installed = {k: v for k, v in lock["packages"].items() if k}
    assert installed["node_modules/webgpu"]["version"] == "0.6.2"
    assert all(entry["integrity"].startswith("sha512-") for entry in installed.values())
    assert tint.SCRIPT.is_file()


def test_first_error_passes_over_naga_warnings():
    log = (
        "[WARN  naga::front::wgsl::parse::directive] error: unknown `diagnostic(…)` rule name `subgroup_uniformity`\n"
        "Could not parse WGSL:\n"
        "error: no definition in scope for identifier: `subgroupElect`\n"
    )
    assert validators.first_error(log, "wgsl") == (
        "error: no definition in scope for identifier: `subgroupElect`"
    )


def test_tools_names_tint_when_only_tint_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        validators, "available", lambda language, target="": language != "wgsl"
    )
    monkeypatch.setattr(validators, "missing", lambda language, target="": "tint")
    monkeypatch.setattr(validators, "version", lambda tool: "v")
    assert cli.main(["tools"]) == 0
    row = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("wgsl")
    )
    assert row.endswith("see tools/coverage/tint.py")


WGSL_DEF = """
match fiu32_f16: f32 | i32 | u32 | f16

////////////////////////////////////////////////////////////////////////////////
// Builtin Functions                                                          //
////////////////////////////////////////////////////////////////////////////////
@must_use @const implicit(T: fiu32_f16)
  fn abs(T) -> T

//////////////////////////////////////////
// subgroups
//////////////////////////////////////////
@must_use @stage("compute") implicit(T: fiu32_f16)
  fn subgroupAdd(value: T) -> T

////////////////////////////////////////////////////////////////////////////////
// Value constructors                                                         //
////////////////////////////////////////////////////////////////////////////////
@must_use @const fn i32() -> i32
"""


def test_subgroup_probes_enable_the_extension_and_drop_the_uniformity_diagnostic():
    found = {f.name: f for f in wgsl.parse(WGSL_DEF, {"abs"})}
    assert found["subgroupAdd"].extension == "subgroups"
    assert found["abs"].extension == ""
    probe = probes.build(found["subgroupAdd"], "wgsl").source
    assert probe.startswith(
        "enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n@group(0)"
    )
    # A function outside the subgroups section enables nothing.
    assert "enable" not in probes.build(found["abs"], "wgsl").source


@pytest.mark.skipif(not validators.command("tint"), reason="Tint is not installed")
def test_tint_requires_the_subgroups_directive(tmp_path):
    (tmp_path / "enabled.wgsl").write_text(SUBGROUPS)
    (tmp_path / "blanked.wgsl").write_text(validators.naga_source(SUBGROUPS))
    accepted = validators.run(
        [*validators.command("tint"), "enabled.wgsl"], str(tmp_path)
    )
    assert accepted.accepted, accepted.log
    rejected = validators.run(
        [*validators.command("tint"), "blanked.wgsl"], str(tmp_path)
    )
    assert not rejected.accepted
    assert "without extension 'subgroups'" in validators.first_error(rejected.log)
