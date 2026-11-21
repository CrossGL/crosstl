"""Check that a CrossTL wheel works as the README says, straight after ``pip install``.

CI runs this on Linux, macOS and Windows for every supported Python. It installs the wheel into a new virtual
environment, then works in an empty directory outside the checkout, as a new user would:

- the README quickstart: it saves the ``saxpy.cgl`` kernel the README shows, runs the README's Python block and
  every command in its command-line block, and checks that each output file holds the language its extension
  names;
- ``crosstl shader.metal -o shader.hlsl`` writes HLSL, and an output file name that names no language CrossTL
  writes is refused without writing a file;
- ``crosstl --help``, ``python -m crosstl --help`` and ``crosstl doctor`` run, and ``import crosstl`` loads the
  installed package rather than the checkout.

No native validator is installed: CrossTL must work without one. Usage::

    python tools/install_smoke.py dist/crosstl-3.8.0-py3-none-any.whl
    python tools/install_smoke.py dist/
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import venv
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A word each target's output contains, so a file that holds CrossGL or another language fails the check.
MARKERS = {
    "cgl": "shader ",
    "directx": "numthreads",
    "metal": "metal_stdlib",
    "opengl": "#version",
    "vulkan": "OpCapability",
    "wgsl": "@compute",
}
EXTENSIONS = {
    ".cgl": "cgl",
    ".comp": "opengl",
    ".glsl": "opengl",
    ".hlsl": "directx",
    ".metal": "metal",
    ".spvasm": "vulkan",
    ".wgsl": "wgsl",
}

# The README's Python block imports ``blur.hlsl``, a native shader it does not show.
BLUR_HLSL = """\
RWStructuredBuffer<float> image : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    image[i] = (image[i] + image[i + 1] + image[i + 2]) / 3.0;
}
"""

SHADER_METAL = """\
#include <metal_stdlib>
using namespace metal;

kernel void add(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
    a[i] = a[i] + b[i];
}
"""


@dataclass
class Quickstart:
    """The README quickstart: the kernel it saves, its Python block and its command lines."""

    kernel_name: str
    kernel: str
    python: str
    commands: list[str] = field(default_factory=list)


def _blocks(text: str) -> list[tuple[str, str]]:
    return [
        (match[1], match[2])
        for match in re.finditer(
            r"^```(\w*)\n(.*?)^```", text, re.MULTILINE | re.DOTALL
        )
    ]


def quickstart(readme: str) -> Quickstart:
    """Read the quickstart section of ``readme``. Raises ``ValueError`` when its shape changed."""
    section = re.search(
        r"^## Quickstart\n(.*?)(?=^## )", readme, re.MULTILINE | re.DOTALL
    )
    if not section:
        raise ValueError("README.md has no '## Quickstart' section")
    body = section[1]
    name = re.search(r"Save this compute kernel as `([^`]+)`", body)
    blocks = _blocks(body)
    languages = [language for language, _ in blocks]
    if (
        not name
        or languages[:1] != ["glsl"]
        or "python" not in languages
        or "bash" not in languages
    ):
        raise ValueError(
            "README quickstart must name the kernel file, then show it, a Python block and a bash block; "
            f"found {languages}"
        )
    python = next(code for language, code in blocks if language == "python")
    bash = next(code for language, code in blocks if language == "bash")
    commands = [
        line.strip()
        for line in bash.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return Quickstart(name[1], blocks[0][1], python, commands)


def output_of(argv: list[str]) -> tuple[str | None, str | None]:
    """The output file and the target a ``crosstl`` command line names: ``(path or '-', backend)``."""
    output = backend = None
    for index, token in enumerate(argv[:-1]):
        if token in ("-o", "--output"):
            output = argv[index + 1]
        elif token in ("-b", "--backend"):
            backend = argv[index + 1]
    if output and output != "-" and not backend:
        backend = EXTENSIONS.get(Path(output).suffix.lower())
    return output, backend


def wrong_language(text: str, backend: str | None) -> str | None:
    """Why ``text`` is not output for ``backend``, or ``None`` when it looks right."""
    if not text.strip():
        return "the output is empty"
    marker = MARKERS.get(backend or "")
    if marker and marker not in text:
        return f"the {backend} output has no {marker!r}"
    if backend != "cgl" and text.lstrip().startswith("shader "):
        return f"the {backend} output is CrossGL"
    return None


class Smoke:
    """Run commands against one installed environment and collect failures."""

    def __init__(self, env: Path, work: Path):
        scripts = env / ("Scripts" if os.name == "nt" else "bin")
        suffix = ".exe" if os.name == "nt" else ""
        self.env = env
        self.python = str(scripts / f"python{suffix}")
        self.console = str(scripts / f"crosstl{suffix}")
        self.work = work
        self.failures: list[str] = []
        self.environ = {
            k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")
        }
        self.environ["PATH"] = str(scripts) + os.pathsep + self.environ.get("PATH", "")

    def run(
        self, argv: list[str], expect_ok: bool = True
    ) -> subprocess.CompletedProcess:
        program = {"crosstl": self.console, "python": self.python}.get(argv[0], argv[0])
        result = subprocess.run(  # the caller judges the exit status
            [program, *argv[1:]],
            check=False,
            cwd=self.work,
            env=self.environ,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        shown = shlex.join(argv)
        if (result.returncode == 0) != expect_ok:
            tail = (result.stdout + result.stderr).strip().splitlines()[-5:]
            self.fail(f"{shown}: exit {result.returncode}: " + " | ".join(tail))
        else:
            print(f"ok   {shown}")
        return result

    def fail(self, message: str) -> None:
        print(f"FAIL {message}")
        self.failures.append(message)

    def check_output(
        self, argv: list[str], result: subprocess.CompletedProcess
    ) -> None:
        output, backend = output_of(argv)
        if result.returncode or not output:
            return
        if output == "-":
            text = result.stdout
        elif not (self.work / output).is_file():
            self.fail(f"{shlex.join(argv)}: wrote no {output}")
            return
        else:
            text = (self.work / output).read_text(encoding="utf-8")
        reason = wrong_language(text, backend)
        if reason:
            self.fail(f"{shlex.join(argv)}: {reason}")


def install(wheel: Path, env: Path) -> None:
    venv.EnvBuilder(with_pip=True, clear=True).create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            str(wheel),
        ],
        check=True,
    )


def smoke(wheel: Path, readme: Path, root: Path) -> list[str]:
    """Install ``wheel`` under ``root`` and run every check; return the failures."""
    env, work = root / "env", root / "work"
    work.mkdir(parents=True)
    install(wheel, env)
    check = Smoke(env, work)

    where = check.run(["python", "-c", "import crosstl; print(crosstl.__file__)"])
    if where.returncode == 0 and not Path(
        where.stdout.strip()
    ).resolve().is_relative_to(env.resolve()):
        check.fail(
            f"import crosstl loaded {where.stdout.strip()}, not the installed package"
        )

    guide = quickstart(readme.read_text(encoding="utf-8"))
    (work / guide.kernel_name).write_text(guide.kernel, encoding="utf-8")
    (work / "blur.hlsl").write_text(BLUR_HLSL, encoding="utf-8")
    (work / "quickstart.py").write_text(guide.python, encoding="utf-8")
    check.run(["python", "quickstart.py"])
    for line in guide.commands:
        argv = shlex.split(line)
        check.check_output(argv, check.run(argv))

    (work / "shader.metal").write_text(SHADER_METAL, encoding="utf-8")
    argv = ["crosstl", "shader.metal", "-o", "shader.hlsl"]
    check.check_output(argv, check.run(argv))
    refused = check.run(
        ["crosstl", "shader.metal", "-o", "shader.spv"], expect_ok=False
    )
    if refused.returncode and ".hlsl" not in refused.stdout + refused.stderr:
        check.fail(
            "crosstl shader.metal -o shader.spv: the refusal does not list the accepted extensions"
        )
    if (work / "shader.spv").exists():
        check.fail("crosstl shader.metal -o shader.spv: wrote shader.spv")

    for argv in (
        ["crosstl", "--help"],
        ["python", "-m", "crosstl", "--help"],
        ["crosstl", "doctor"],
    ):
        check.run(argv)
    return check.failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "wheel",
        type=Path,
        help="the CrossTL wheel to install, or the directory that holds it",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=ROOT / "README.md",
        help="the README whose quickstart runs",
    )
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve()
    if wheel.is_dir():
        wheels = sorted(wheel.glob("crosstl-*.whl"))
        if len(wheels) != 1:
            parser.error(f"{wheel} holds {len(wheels)} CrossTL wheels, not one")
        wheel = wheels[0]
    with tempfile.TemporaryDirectory(prefix="crosstl-smoke-") as root:
        failures = smoke(wheel, args.readme.resolve(), Path(root))
    print(
        f"{len(failures)} failure(s)"
        if failures
        else "installed package works as documented"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
