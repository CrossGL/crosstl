"""The installed-package check reads the README quickstart and recognizes each target's output."""

from pathlib import Path

import pytest

from tools import install_smoke

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_readme_quickstart_has_the_shape_the_check_runs():
    guide = install_smoke.quickstart(README)
    assert guide.kernel_name.endswith(".cgl")
    assert guide.kernel.lstrip().startswith("shader ")
    assert "import crosstl" in guide.python
    assert guide.commands
    assert all(
        command.split()[0] in ("crosstl", "python") for command in guide.commands
    )


def test_every_readme_command_output_is_checked_for_its_language():
    for command in install_smoke.quickstart(README).commands:
        output, backend = install_smoke.output_of(command.split())
        if output:
            assert backend in install_smoke.MARKERS, command


def test_the_readme_python_block_reads_only_files_the_check_writes():
    guide = install_smoke.quickstart(README)
    written = {guide.kernel_name, "blur.hlsl"}
    for name in ("saxpy.cgl", "blur.hlsl", "shader.hlsl"):
        if f'"{name}"' in guide.python:
            assert name in written, name


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["crosstl", "shader.metal", "-o", "shader.hlsl"], ("shader.hlsl", "directx")),
        (["crosstl", "k.cgl", "--output", "k.WGSL"], ("k.WGSL", "wgsl")),
        (
            ["crosstl", "translate", "k.cgl", "--backend", "vulkan", "--output", "-"],
            ("-", "vulkan"),
        ),
        (["crosstl", "k.cgl", "-b", "metal", "-o", "k.txt"], ("k.txt", "metal")),
        (["crosstl", "k.cgl", "-o", "k.cu"], ("k.cu", None)),
        (["crosstl", "k.cgl"], (None, None)),
    ],
)
def test_output_of_reads_the_output_and_its_target(argv, expected):
    assert install_smoke.output_of(argv) == expected


def test_wrong_language_rejects_crossgl_and_empty_output():
    hlsl = "[numthreads(64, 1, 1)] void CSMain() {}"
    crossgl = "shader Saxpy {\n    compute {\n    }\n}\n"
    assert install_smoke.wrong_language(hlsl, "directx") is None
    assert "numthreads" in install_smoke.wrong_language(crossgl, "directx")
    assert install_smoke.wrong_language(crossgl, "cgl") is None
    assert "CrossGL" in install_smoke.wrong_language(
        crossgl + "// numthreads", "directx"
    )
    assert "empty" in install_smoke.wrong_language("  \n", "metal")


def test_a_readme_without_a_quickstart_is_reported():
    with pytest.raises(ValueError, match="Quickstart"):
        install_smoke.quickstart("# CrossTL\n\n## Install\n")
    with pytest.raises(ValueError, match="kernel file"):
        install_smoke.quickstart(
            "## Quickstart\n\n```bash\ncrosstl a.cgl\n```\n\n## Next\n"
        )


GUIDE = (ROOT / "docs" / "source" / "quickstart.rst").read_text(encoding="utf-8")


def test_the_quickstart_guide_saves_its_sources_before_it_translates_them():
    steps = install_smoke.guide(GUIDE)
    saved = [step.name for step in steps if step.kind == "save"]
    assert saved == ["shader.metal", "saxpy.cgl"]
    seen: set[str] = set()
    for step in steps:
        if step.kind == "save":
            seen.add(step.name)
            continue
        for name in ("shader.metal", "saxpy.cgl"):
            if name in step.text:
                assert name in seen, (name, step.text)
    commands = [step.text for step in steps if step.kind == "command"]
    assert "crosstl shader.metal -o shader.hlsl" in commands
    assert "crosstl doctor" in commands
    # pip install crosstl is what the check has already done with the wheel it tests.
    assert not any(command.startswith("pip ") for command in commands)


def test_every_quickstart_guide_output_is_checked_for_its_language():
    for step in install_smoke.guide(GUIDE):
        if step.kind == "command":
            output, backend = install_smoke.output_of(step.text.split())
            if output and output != "-":
                assert backend in install_smoke.MARKERS, step.text
        elif step.kind == "python":
            for backend, _ in install_smoke.SAVED.findall(step.text):
                assert backend in install_smoke.MARKERS, step.text


def test_rst_steps_read_saved_files_python_blocks_and_commands():
    body = (
        "Save this kernel as ``k.metal``:\n\n"
        ".. code-block:: cpp\n\n"
        "   kernel void k() {}\n\n"
        "Then run:\n\n"
        ".. code-block:: bash\n\n"
        "   pip install crosstl\n"
        "   # a comment\n"
        "   crosstl k.metal \\\n"
        "     -o k.hlsl\n\n"
        ".. code-block:: python\n\n"
        "   import crosstl\n"
    )
    steps = install_smoke.rst_steps(body)
    assert [(step.kind, step.name) for step in steps] == [
        ("save", "k.metal"),
        ("command", ""),
        ("python", ""),
    ]
    assert steps[0].text == "kernel void k() {}\n"
    assert steps[1].text.split() == ["crosstl", "k.metal", "-o", "k.hlsl"]
    assert steps[2].text == "import crosstl\n"


def test_a_guide_that_lost_a_section_or_runs_another_tool_is_reported():
    with pytest.raises(ValueError, match="no section"):
        install_smoke.guide("Quickstart\n==========\n\nInstall\n-------\n")
    with pytest.raises(ValueError, match="neither crosstl nor python"):
        install_smoke.rst_steps(".. code-block:: bash\n\n   glslangValidator k.comp\n")


@pytest.mark.parametrize("version", ["3.9.0", "4.0.0rc1", "3.9.0.post1", "10.2"])
def test_a_release_check_installs_that_version_from_pypi(version):
    assert install_smoke.release_requirement(version) == f"crosstl=={version}"


@pytest.mark.parametrize("version", ["", ">=3.9", "3.9.0; echo", "latest", "v3.9.0"])
def test_a_release_check_refuses_anything_but_a_version_number(version):
    with pytest.raises(ValueError):
        install_smoke.release_requirement(version)


def _pip_arguments(monkeypatch, tmp_path, package):
    calls = []

    class Builder:
        def __init__(self, **kwargs):
            pass

        def create(self, env):
            pass

    monkeypatch.setattr(install_smoke.venv, "EnvBuilder", Builder)
    monkeypatch.setattr(
        install_smoke.subprocess, "run", lambda argv, check: calls.append(argv)
    )
    install_smoke.install(package, tmp_path / "env")
    (argv,) = calls
    return argv[argv.index("install") + 1 :]


def test_a_wheel_installs_from_its_file(monkeypatch, tmp_path):
    wheel = tmp_path / "crosstl-3.9.0-py3-none-any.whl"
    arguments = _pip_arguments(monkeypatch, tmp_path, wheel)
    assert arguments[-1] == str(wheel)
    assert "--index-url" not in arguments


def test_a_release_installs_from_pypi_without_the_cache(monkeypatch, tmp_path):
    arguments = _pip_arguments(monkeypatch, tmp_path, "crosstl==3.9.0")
    assert arguments[-1] == "crosstl==3.9.0"
    assert "--no-cache-dir" in arguments
    assert arguments[arguments.index("--index-url") + 1] == install_smoke.PYPI


@pytest.mark.parametrize(
    "argv",
    [[], ["dist/", "--release", "3.9.0"], ["--release", "3.9.0; echo"]],
)
def test_the_check_needs_one_wheel_or_one_release_version(argv):
    with pytest.raises(SystemExit) as stopped:
        install_smoke.main(argv)
    assert stopped.value.code == 2


def test_the_release_dispatch_checks_the_published_package_on_every_platform():
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "examples-test.yml").read_text(
            encoding="utf-8"
        )
    )
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["workflow_dispatch"]["inputs"]["release"]["default"] == ""
    assert "!inputs.release" in workflow["jobs"]["test-examples"]["if"]
    steps = workflow["jobs"]["installed-package"]["steps"]
    released = [step for step in steps if step.get("if") == "inputs.release"]
    checkout, check = released
    assert checkout["with"]["ref"] == "v${{ inputs.release }}"
    # The version reaches the shell through the environment, never spliced into the script.
    assert check["env"]["RELEASE"] == "${{ inputs.release }}"
    assert "${{" not in check["run"] and '--release "$RELEASE"' in check["run"]
    built = [step for step in steps if "wheel" in step.get("run", "")]
    assert built and all(step["if"] == "${{ !inputs.release }}" for step in built)
