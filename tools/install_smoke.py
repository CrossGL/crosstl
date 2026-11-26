"""Check that a CrossTL wheel works as the README and the quickstart guide say, straight after ``pip install``.

CI runs this on Linux, macOS and Windows for every supported Python. It installs the wheel into a new virtual
environment, then works in an empty directory outside the checkout, as a new user would:

- the README quickstart: it saves the ``saxpy.cgl`` kernel the README shows, runs the README's Python block and
  every command in its command-line block, and checks that each output file holds the language its extension
  names;
- the documentation's quickstart (``docs/source/quickstart.rst``), in a directory of its own: every file its
  sections in :data:`GUIDE_SECTIONS` ask the reader to save, every Python block and every command, with the same
  check of each output, and of each file a Python block saves with ``save_shader``;
- ``crosstl shader.metal -o shader.hlsl`` writes HLSL, and an input or output file name that names no language
  CrossTL reads or writes is refused without writing a file;
- ``crosstl --help``, ``python -m crosstl --help`` and ``crosstl doctor`` run, and ``import crosstl`` loads the
  installed package rather than the checkout.

No native validator is installed: CrossTL must work without one. ``--release VERSION`` installs
``crosstl==VERSION`` from PyPI instead of a wheel, without pip's cache, and checks that the installed version is
VERSION, so a published release can be checked the way a new user installs it. Usage::

    python tools/install_smoke.py dist/crosstl-3.8.0-py3-none-any.whl
    python tools/install_smoke.py dist/
    python tools/install_smoke.py dist/ --readme README.md --guide docs/source/quickstart.rst
    python tools/install_smoke.py --release 3.9.0 --readme released/README.md \\
        --guide released/docs/source/quickstart.rst
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


# The sections of docs/source/quickstart.rst whose steps run. The others need a checkout or a repository to port.
GUIDE_SECTIONS = ("Install", "Translate a native shader", "Write CrossGL")
# The installation the check has already done with the wheel or the release it tests.
INSTALL = ["pip", "install", "crosstl"]
# The index ``--release`` installs from: PyPI itself, not a mirror a runner may be configured with.
PYPI = "https://pypi.org/simple/"
# A ``crosstl.translate(..., backend="B", save_shader="F")`` call in a Python block: F must hold language B.
SAVED = re.compile(r'backend="(\w+)",\s*save_shader="([^"]+)"')
UNDERLINE = re.compile(r"([=\-~^])\1+")


@dataclass
class Step:
    """One thing a guide asks its reader to do: save a file, run a Python block, or run a command."""

    kind: str  # "save", "python" or "command"
    text: str
    name: str = ""


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


def rst_sections(text: str) -> dict[str, str]:
    """The body of each titled section of a reStructuredText document, by title."""
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    body: list[str] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            line.strip()
            and not line.startswith(" ")
            and UNDERLINE.fullmatch(following)
            and len(following) >= len(line.rstrip())
        ):
            body = sections.setdefault(line.strip(), [])
            index += 2
            continue
        if body is not None:
            body.append(line)
        index += 1
    return {title: "\n".join(body) for title, body in sections.items()}


def _code_block(lines: list[str], start: int) -> tuple[str, int]:
    """The code of the directive body that starts at ``lines[start]``, dedented, and the index after it."""
    index = start
    while index < len(lines) and lines[index].strip().startswith(":"):
        index += 1  # directive options such as :caption:
    code: list[str] = []
    while index < len(lines) and (
        not lines[index].strip() or lines[index].startswith("   ")
    ):
        code.append(lines[index][3:])
        index += 1
    while code and not code[-1].strip():
        code.pop()
    while code and not code[0].strip():
        code.pop(0)
    return "\n".join(code) + "\n", index


def _commands(code: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", code)
    return [
        line.strip()
        for line in joined.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def rst_steps(body: str) -> list[Step]:
    """The steps of one section: a code block after "Save this ... as ``NAME``:" is saved as ``NAME``."""
    lines = body.splitlines()
    steps: list[Step] = []
    # The paragraph before the current line; a blank line ends it, the next text line starts another.
    paragraph: list[str] = []
    ended = False
    index = 0
    while index < len(lines):
        directive = re.match(r"\.\. code-block:: (\S+)\s*$", lines[index])
        if not directive:
            text = lines[index].strip()
            if text and ended:
                paragraph, ended = [], False
            if text:
                paragraph.append(text)
            else:
                ended = True
            index += 1
            continue
        code, index = _code_block(lines, index + 1)
        save = re.search(r"Save this .*? as ``([^`]+)``:$", " ".join(paragraph))
        paragraph, ended = [], False
        if save:
            steps.append(Step("save", code, save[1]))
        elif directive[1] == "python":
            steps.append(Step("python", code))
        elif directive[1] in ("bash", "console", "shell"):
            for command in _commands(code):
                if shlex.split(command) == INSTALL:
                    continue
                if shlex.split(command)[0] not in ("crosstl", "python"):
                    raise ValueError(
                        f"quickstart command {command!r} runs neither crosstl nor python"
                    )
                steps.append(Step("command", command))
    return steps


def guide(text: str) -> list[Step]:
    """The steps of the :data:`GUIDE_SECTIONS` of the quickstart guide. Raises ``ValueError`` when its shape changed."""
    sections = rst_sections(text)
    missing = [title for title in GUIDE_SECTIONS if title not in sections]
    if missing:
        raise ValueError(f"the quickstart guide has no section {missing}")
    steps = [step for title in GUIDE_SECTIONS for step in rst_steps(sections[title])]
    kinds = {step.kind for step in steps}
    if not {"save", "python", "command"} <= kinds:
        raise ValueError(
            "the quickstart guide must save a source, then run a Python block and a command on it; "
            f"found {sorted(kinds)}"
        )
    return steps


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

    def check_saved(self, code: str, script: str) -> None:
        """Check each file ``code`` saves through ``save_shader`` for the language of its backend."""
        for backend, name in SAVED.findall(code):
            path = self.work / name
            if not path.is_file():
                self.fail(f"{script}: wrote no {name}")
                continue
            reason = wrong_language(path.read_text(encoding="utf-8"), backend)
            if reason:
                self.fail(f"{script}: {name}: {reason}")

    def follow(self, steps: list[Step]) -> None:
        """Do what a guide's steps say, in order, in this check's directory."""
        for number, step in enumerate(steps, 1):
            if step.kind == "save":
                (self.work / step.name).write_text(step.text, encoding="utf-8")
            elif step.kind == "python":
                script = f"guide-{number}.py"
                (self.work / script).write_text(step.text, encoding="utf-8")
                if self.run(["python", script]).returncode == 0:
                    self.check_saved(step.text, script)
            else:
                argv = shlex.split(step.text)
                self.check_output(argv, self.run(argv))


def install(package: Path | str, env: Path) -> None:
    """Install ``package`` into a new environment at ``env``.

    ``package`` is a wheel, or a requirement such as ``crosstl==3.9.0`` that is installed from :data:`PYPI` without
    pip's cache, as a new user's first install would be.
    """
    venv.EnvBuilder(with_pip=True, clear=True).create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    source = (
        [str(package)]
        if isinstance(package, Path)
        else ["--no-cache-dir", "--index-url", PYPI, package]
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            *source,
        ],
        check=True,
    )


def release_requirement(version: str) -> str:
    """The requirement that installs release ``version`` of CrossTL, refusing anything but a version number."""
    if not re.fullmatch(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?", version):
        raise ValueError(f"{version!r} is not a CrossTL version number")
    return f"crosstl=={version}"


def smoke(
    package: Path | str,
    readme: Path,
    root: Path,
    guide_path: Path | None = None,
    version: str | None = None,
) -> list[str]:
    """Install ``package`` under ``root`` and run every check; return the failures.

    ``package`` is a wheel or a PyPI requirement (:func:`install`). With ``version``, the installed distribution
    must have that version.
    """
    env, work = root / "env", root / "work"
    steps = guide(guide_path.read_text(encoding="utf-8")) if guide_path else []
    work.mkdir(parents=True)
    install(package, env)
    check = Smoke(env, work)
    if version is not None:
        installed = check.run(
            [
                "python",
                "-c",
                "import importlib.metadata as m; print(m.version('crosstl'))",
            ]
        )
        if installed.returncode == 0 and installed.stdout.strip() != version:
            check.fail(
                f"pip installed crosstl {installed.stdout.strip()}, not {version}"
            )

    where = check.run(["python", "-c", "import crosstl; print(crosstl.__file__)"])
    if where.returncode == 0 and not Path(
        where.stdout.strip()
    ).resolve().is_relative_to(env.resolve()):
        check.fail(
            f"import crosstl loaded {where.stdout.strip()}, not the installed package"
        )

    readme_guide = quickstart(readme.read_text(encoding="utf-8"))
    (work / readme_guide.kernel_name).write_text(readme_guide.kernel, encoding="utf-8")
    (work / "blur.hlsl").write_text(BLUR_HLSL, encoding="utf-8")
    (work / "quickstart.py").write_text(readme_guide.python, encoding="utf-8")
    if check.run(["python", "quickstart.py"]).returncode == 0:
        check.check_saved(readme_guide.python, "quickstart.py")
    for line in readme_guide.commands:
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
    (work / "notes.txt").write_text(SHADER_METAL, encoding="utf-8")
    refused = check.run(["crosstl", "notes.txt", "-o", "notes.hlsl"], expect_ok=False)
    if refused.returncode and ".metal" not in refused.stdout + refused.stderr:
        check.fail(
            "crosstl notes.txt -o notes.hlsl: the refusal does not list the accepted extensions"
        )
    if (work / "notes.hlsl").exists():
        check.fail("crosstl notes.txt -o notes.hlsl: wrote notes.hlsl")

    for argv in (
        ["crosstl", "--help"],
        ["python", "-m", "crosstl", "--help"],
        ["crosstl", "doctor"],
    ):
        check.run(argv)

    if steps:
        guided = Smoke(env, root / "guide")
        guided.work.mkdir()
        guided.follow(steps)
        check.failures += guided.failures
    return check.failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "wheel",
        type=Path,
        nargs="?",
        help="the CrossTL wheel to install, or the directory that holds it",
    )
    parser.add_argument(
        "--release",
        metavar="VERSION",
        help="install crosstl==VERSION from PyPI instead of a wheel",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=ROOT / "README.md",
        help="the README whose quickstart runs",
    )
    parser.add_argument(
        "--guide",
        type=Path,
        default=ROOT / "docs" / "source" / "quickstart.rst",
        help="the quickstart guide whose steps run",
    )
    args = parser.parse_args(argv)
    if (args.wheel is None) == (args.release is None):
        parser.error("give either a wheel or --release VERSION")
    package: Path | str
    if args.release is not None:
        try:
            package = release_requirement(args.release)
        except ValueError as error:
            parser.error(str(error))
    else:
        package = args.wheel.resolve()
        if package.is_dir():
            wheels = sorted(package.glob("crosstl-*.whl"))
            if len(wheels) != 1:
                parser.error(f"{package} holds {len(wheels)} CrossTL wheels, not one")
            package = wheels[0]
    with tempfile.TemporaryDirectory(prefix="crosstl-smoke-") as root:
        failures = smoke(
            package,
            args.readme.resolve(),
            Path(root),
            args.guide.resolve(),
            args.release,
        )
    print(
        f"{len(failures)} failure(s)"
        if failures
        else "installed package works as documented"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
