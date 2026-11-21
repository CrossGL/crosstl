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
