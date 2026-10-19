"""Locate the native validators and runtimes of the conformance harness and report their versions.

Every tool is resolved in the same order: an explicit ``CROSSTL_CONFORMANCE_<TOOL>`` override, then ``PATH``, then
the Vulkan SDK (``VULKAN_SDK`` or the newest ``~/VulkanSDK/<version>``). A tool that cannot be found is reported
with the reason; callers record it as ``unavailable`` rather than skipping it silently.

``python -m tests.conformance.toolchains`` prints the inventory as JSON, or as the Markdown table recorded in
``TOOLCHAINS.md`` with ``--markdown``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .process import run

HERE = Path(__file__).resolve().parent
CACHE = Path(
    os.environ.get(
        "CROSSTL_CONFORMANCE_CACHE", Path.home() / ".cache" / "crosstl-conformance"
    )
)
MESA_IMAGE = os.environ.get(
    "CROSSTL_CONFORMANCE_MESA_IMAGE", "crosstl-conformance-mesa:1"
)
MESA_DOCKERFILE = HERE / "docker" / "mesa" / "Dockerfile"


def profile():
    """Name of the scoreboard profile for this machine; CI jobs set CROSSTL_CONFORMANCE_PROFILE explicitly."""
    explicit = os.environ.get("CROSSTL_CONFORMANCE_PROFILE")
    return explicit or f"{platform.system()}-{platform.machine()}".lower()


@dataclass(frozen=True)
class Tool:
    name: str
    command: tuple
    version: str | None
    reason: str = ""

    @property
    def available(self):
        return bool(self.command)

    def to_json(self):
        return asdict(self)


def _override(name):
    return os.environ.get("CROSSTL_CONFORMANCE_" + re.sub(r"\W", "_", name).upper())


@functools.lru_cache(maxsize=None)
def vulkan_sdk():
    """Root of the Vulkan SDK platform tree (the directory that holds ``bin``, ``lib`` and ``include``)."""
    explicit = os.environ.get("VULKAN_SDK")
    candidates = [Path(explicit)] if explicit else []
    home = Path.home() / "VulkanSDK"
    if home.is_dir():
        versions = sorted(
            (p for p in home.iterdir() if re.fullmatch(r"\d+(\.\d+)+", p.name)),
            key=lambda p: tuple(int(x) for x in p.name.split(".")),
            reverse=True,
        )
        candidates += [v / "macOS" if (v / "macOS").is_dir() else v for v in versions]
    return next((path for path in candidates if (path / "bin").is_dir()), None)


def _version(command, pattern=None):
    result = run([*command], timeout=30)
    if result.status == "unavailable":
        return None
    lines = [line.strip() for line in result.log.splitlines() if line.strip()]
    if pattern:
        lines = [line for line in lines if re.search(pattern, line)]
    return "; ".join(lines[:3]) or None


def _resolve_executable(name):
    explicit = _override(name)
    if explicit:
        return (
            (explicit,) if Path(explicit).is_file() or shutil.which(explicit) else None
        )
    found = shutil.which(name)
    if found:
        return (found,)
    sdk = vulkan_sdk()
    if sdk and (sdk / "bin" / name).is_file():
        return (str(sdk / "bin" / name),)
    return None


VERSION_FLAGS = {
    "dxc": (("--version",), None),
    "glslangValidator": (("--version",), r"Glslang Version|GLSL Version|ESSL Version"),
    "spirv-as": (("--version",), r"SPIRV-Tools"),
    "spirv-val": (("--version",), r"SPIRV-Tools"),
}


@functools.lru_cache(maxsize=None)
def tool(name):
    """Resolve one validator or helper tool."""
    if name == "metal":
        if platform.system() != "Darwin" or not shutil.which("xcrun"):
            return Tool(name, (), None, "xcrun is only available on macOS")
        command = ("xcrun", "-sdk", "macosx", "metal")
        version = _version([*command, "--version"], r"metal version")
        sdk = _version(["xcrun", "-sdk", "macosx", "--show-sdk-version"])
        if version is None:
            return Tool(name, (), None, "xcrun cannot locate the Metal compiler")
        return Tool(name, command, f"{version}; macOS SDK {sdk}")
    if name == "docker":
        found = shutil.which("docker")
        if not found:
            return Tool(name, (), None, "docker is not installed")
        version = _version([found, "version", "--format", "{{.Server.Version}}"])
        if not version or "error" in version.lower() or "cannot" in version.lower():
            return Tool(name, (), None, "the docker daemon is not reachable")
        return Tool(name, (found,), f"server {version}")
    command = _resolve_executable(name)
    if command is None:
        return Tool(
            name, (), None, f"{name} was not found on PATH or in the Vulkan SDK"
        )
    flags, pattern = VERSION_FLAGS.get(name, (("--version",), None))
    return Tool(name, command, _version([*command, *flags], pattern))


def vkrun():
    """The Vulkan compute runner for this host, compiled from ``vkrun.c`` into the cache when needed."""
    explicit = _override("vkrun")
    if explicit:
        return (
            Tool("vkrun", (explicit,), None)
            if Path(explicit).is_file()
            else Tool(
                "vkrun",
                (),
                None,
                f"CROSSTL_CONFORMANCE_VKRUN does not exist: {explicit}",
            )
        )
    source = HERE / "vkrun.c"
    sdk = vulkan_sdk()
    include = [f"-I{sdk / 'include'}"] if sdk else []
    library = [f"-L{sdk / 'lib'}", f"-Wl,-rpath,{sdk / 'lib'}"] if sdk else []
    digest = (
        __import__("hashlib")
        .sha256(source.read_bytes() + str(sdk).encode())
        .hexdigest()[:16]
    )
    binary = CACHE / f"vkrun-{platform.system().lower()}-{platform.machine()}-{digest}"
    if not binary.is_file():
        compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
        if compiler is None:
            return Tool("vkrun", (), None, "no C compiler is available to build vkrun")
        binary.parent.mkdir(parents=True, exist_ok=True)
        staging = binary.with_suffix(".tmp")
        result = run(
            [
                compiler,
                "-O2",
                "-std=c11",
                *include,
                source,
                *library,
                "-lvulkan",
                "-o",
                staging,
            ],
            timeout=120,
        )
        if result.status != "accepted":
            return Tool("vkrun", (), None, f"vkrun did not build: {result.log[-400:]}")
        staging.replace(binary)
    return Tool("vkrun", (str(binary),), f"built from vkrun.c ({digest})")


def mesa_image():
    """The pinned Mesa llvmpipe/lavapipe container used where the host has no Mesa drivers."""
    docker = tool("docker")
    if not docker.available:
        return Tool("mesa-image", (), None, docker.reason)
    result = run(
        [*docker.command, "image", "inspect", "--format", "{{.Id}}", MESA_IMAGE],
        timeout=30,
    )
    if result.status != "accepted":
        return Tool(
            "mesa-image",
            (),
            None,
            f"image {MESA_IMAGE} is absent; build it with: docker build -t {MESA_IMAGE} "
            f"-f {MESA_DOCKERFILE.relative_to(HERE.parents[1])} {HERE.relative_to(HERE.parents[1])}",
        )
    probe = run(
        [
            *docker.command,
            "run",
            "--rm",
            MESA_IMAGE,
            "sh",
            "-c",
            "dpkg-query -W -f='${Version}' mesa-vulkan-drivers; echo; uname -m",
        ],
        timeout=120,
    )
    version = " ".join(probe.log.split()) if probe.status == "accepted" else None
    return Tool(
        "mesa-image",
        (*docker.command, MESA_IMAGE),
        f"Mesa {version}" if version else None,
    )


def python_module(name):
    try:
        module = __import__(name)
    except (
        Exception
    ) as error:  # noqa: BLE001 - any import failure means the adapter cannot run
        return Tool(name, (), None, f"{type(error).__name__}: {error}")
    return Tool(name, (sys.executable,), getattr(module, "__version__", None))


VALIDATORS = ("metal", "dxc", "glslangValidator", "spirv-as", "spirv-val")


def inventory():
    """Every validator and runtime prerequisite with its version or the reason it is unavailable."""
    from .runtimes import RUNTIMES, provision

    tools = [tool(name) for name in VALIDATORS] + [
        tool("docker"),
        vkrun(),
        mesa_image(),
    ]
    sdk = vulkan_sdk()
    return {
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "vulkanSDK": str(sdk) if sdk else None,
            "profile": profile(),
        },
        "tools": [item.to_json() for item in tools],
        "runtimes": [provision(runtime).to_json() for runtime in RUNTIMES.values()],
    }


def markdown(data):
    lines = [
        f"Host: {data['host']['system']} {data['host']['release']} {data['host']['machine']}, "
        f"Python {data['host']['python']}, Vulkan SDK {data['host']['vulkanSDK'] or 'absent'}, "
        f"scoreboard profile `{data['host'].get('profile')}`.",
        "",
        "| Tool | Version | Status |",
        "|---|---|---|",
    ]
    for item in data["tools"]:
        status = "available" if item["command"] else "unavailable: " + item["reason"]
        lines.append(f"| `{item['name']}` | {item['version'] or '-'} | {status} |")
    lines += [
        "",
        "| Runtime | Target | Mode | Identity | Status |",
        "|---|---|---|---|---|",
    ]
    for item in data["runtimes"]:
        status = "available" if item["available"] else "unavailable: " + item["reason"]
        lines.append(
            f"| `{item['runtime']}` | {item['target']} | {item['mode'] or '-'} | "
            f"{item['identity'] or '-'} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    data = inventory()
    print(markdown(data) if args.markdown else json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
