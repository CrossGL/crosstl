"""Execution runtimes: which driver runs each target, how it is provisioned, and isolated batch execution.

A runtime is one (API, implementation) pair, for example Vulkan on Mesa lavapipe. Each is provisioned either on
the host or in the pinned Mesa container, and every execution checks that the driver that actually ran reports the
expected identity, so a misconfigured loader cannot substitute another implementation unnoticed.
"""

from __future__ import annotations

import functools
import glob
import json
import os
import platform
import shlex
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .process import run
from .toolchains import mesa_image, python_module, tool, vkrun, vulkan_sdk

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
MESA_ENV = (
    ("LIBGL_ALWAYS_SOFTWARE", "1"),
    ("GALLIUM_DRIVER", "llvmpipe"),
    ("EGL_PLATFORM", "surfaceless"),
    ("MESA_SHADER_CACHE_DISABLE", "true"),
)


@dataclass(frozen=True)
class Runtime:
    id: str
    target: str
    adapter: str
    expect: str
    description: str


RUNTIMES = {
    runtime.id: runtime
    for runtime in (
        Runtime(
            "metal-apple",
            "metal",
            "metal",
            "Apple",
            "Metal on the system Apple GPU through PyObjC",
        ),
        Runtime(
            "vulkan-moltenvk",
            "spirv",
            "vulkan",
            "MoltenVK",
            "MoltenVK from the Vulkan SDK, on Metal",
        ),
        Runtime(
            "vulkan-kosmickrisp",
            "spirv",
            "vulkan",
            "KosmicKrisp",
            "Mesa KosmicKrisp from the Vulkan SDK, on Metal",
        ),
        Runtime(
            "vulkan-lavapipe",
            "spirv",
            "vulkan",
            "llvmpipe",
            "Mesa lavapipe software Vulkan",
        ),
        Runtime(
            "opengl-llvmpipe",
            "glsl",
            "opengl",
            "llvmpipe",
            "Mesa llvmpipe OpenGL 4.5 core through EGL",
        ),
        Runtime(
            "d3d12-warp",
            "hlsl",
            "d3d12",
            "Microsoft Basic Render Driver",
            "Direct3D 12 WARP through compushady",
        ),
    )
}

PROBE_SPIRV = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""

PROBE_PYTHON = {
    "metal": (
        "import Metal\nd = Metal.MTLCreateSystemDefaultDevice()\nprint('device=' + (str(d.name()) if d else ''))"
    ),
    "opengl": (
        "import moderngl\nc = moderngl.create_standalone_context(require=430, backend='egl')\n"
        "print('device=' + c.info['GL_RENDERER'] + '; ' + c.info['GL_VERSION'])"
    ),
    "d3d12": (
        "from compushady import config, get_discovered_devices\nconfig.set_backend('d3d12')\n"
        "print('device=' + '; '.join(d.name for d in get_discovered_devices() if not d.is_hardware))"
    ),
}


@dataclass(frozen=True)
class Provision:
    runtime: str
    target: str
    available: bool
    mode: str | None = None
    reason: str = ""
    identity: str | None = None
    env: tuple = ()
    container: tuple = field(default=(), repr=False)

    def to_json(self):
        data = asdict(self)
        data["env"] = dict(self.env)
        return data


def _field(log, name):
    for line in log.splitlines():
        if line.startswith(name + "="):
            return line[len(name) + 1 :].strip()
    return None


def _container_command(container, root, *command):
    user = (
        ["--user", f"{os.getuid()}:{os.getgid()}"]
        if platform.system() == "Linux"
        else []
    )
    mounts = ["-v", f"{REPOSITORY}:{REPOSITORY}:ro"]
    if root is not None:
        mounts += ["-v", f"{root}:{root}"]
    return [
        *container[:-1],
        "run",
        "--rm",
        "--network",
        "none",
        *user,
        "-e",
        "HOME=/tmp",
        *mounts,
        "-w",
        str(REPOSITORY),
        container[-1],
        *command,
    ]


def _probe_vulkan(env, container=()):
    """Run an empty compute dispatch and return the identity the driver reports."""
    with tempfile.TemporaryDirectory(dir=_scratch()) as directory:
        directory = Path(directory)
        (directory / "probe.spvasm").write_text(PROBE_SPIRV)
        (directory / "probe.bin").write_bytes(bytes(16))
        if container:
            command = _container_command(
                container,
                directory,
                "sh",
                "-c",
                f"spirv-as {directory}/probe.spvasm -o {directory}/probe.spv && "
                f"vkrun {directory}/probe.spv main 1 1 1 0:inout:{directory}/probe.bin",
            )
            result = run(command, timeout=180)
        else:
            assembled = run(
                [
                    *tool("spirv-as").command,
                    directory / "probe.spvasm",
                    "-o",
                    directory / "probe.spv",
                ]
            )
            if assembled.status != "accepted":
                return (
                    None,
                    f"spirv-as cannot assemble the probe: {assembled.log[-200:]}",
                )
            result = run(
                [
                    *vkrun().command,
                    directory / "probe.spv",
                    "main",
                    1,
                    1,
                    1,
                    f"0:inout:{directory / 'probe.bin'}",
                ],
                env={**os.environ, **dict(env)},
                timeout=60,
            )
    if result.status != "accepted":
        return None, f"probe dispatch {result.status}: {result.log[-300:]}"
    return (
        f"{_field(result.log, 'driver')} {_field(result.log, 'driver_info')} (Vulkan {_field(result.log, 'api')}) "
        f"on {_field(result.log, 'device')}"
    ), ""


def _probe_python(adapter, env=(), container=()):
    if container:
        result = run(
            _container_command(container, None, "python3", "-c", PROBE_PYTHON[adapter]),
            timeout=180,
        )
    else:
        result = run(
            [sys.executable, "-c", PROBE_PYTHON[adapter]],
            env={**os.environ, **dict(env)},
            timeout=120,
        )
    identity = _field(result.log, "device") if result.status == "accepted" else None
    return identity or None, (
        "" if identity else f"probe {result.status}: {result.log[-300:]}"
    )


def _scratch():
    path = Path(
        os.environ.get(
            "CROSSTL_CONFORMANCE_CACHE", Path.home() / ".cache" / "crosstl-conformance"
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sdk_icd(name):
    sdk = vulkan_sdk()
    path = sdk / "share" / "vulkan" / "icd.d" / name if sdk else None
    return path if path and path.is_file() else None


def _host_lavapipe_icd():
    if platform.system() != "Linux":
        return None
    found = sorted(glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json"))
    return Path(found[0]) if found else None


def _checked(runtime, mode, identity, reason, env=(), container=()):
    if identity is None:
        return Provision(runtime.id, runtime.target, False, mode, reason)
    if runtime.expect not in identity:
        return Provision(
            runtime.id,
            runtime.target,
            False,
            mode,
            f"the driver reports {identity!r}, not {runtime.expect!r}",
            identity,
        )
    return Provision(
        runtime.id,
        runtime.target,
        True,
        mode,
        "",
        identity,
        tuple(env),
        tuple(container),
    )


def _mesa_mode():
    mode = os.environ.get("CROSSTL_CONFORMANCE_MESA", "auto")
    if mode not in {"auto", "host", "container"}:
        raise ValueError(
            f"CROSSTL_CONFORMANCE_MESA must be auto, host or container, not {mode!r}"
        )
    return mode


def _vulkan(runtime):
    if runtime.id == "vulkan-lavapipe":
        icd = _host_lavapipe_icd() if _mesa_mode() != "container" else None
        if icd is None and _mesa_mode() == "host":
            return Provision(
                runtime.id,
                runtime.target,
                False,
                "host",
                "no host lavapipe ICD was found",
            )
        if icd is None:
            image = mesa_image()
            if not image.available:
                return Provision(
                    runtime.id,
                    runtime.target,
                    False,
                    None,
                    f"no host lavapipe ICD and no Mesa container: {image.reason}",
                )
            identity, reason = _probe_vulkan((), image.command)
            return _checked(runtime, "container", identity, reason, (), image.command)
    else:
        icd = _sdk_icd(
            {
                "vulkan-moltenvk": "MoltenVK_icd.json",
                "vulkan-kosmickrisp": "libkosmickrisp_icd.json",
            }[runtime.id]
        )
        if icd is None:
            return Provision(
                runtime.id,
                runtime.target,
                False,
                None,
                "the Vulkan SDK does not provide this ICD",
            )
    runner = vkrun()
    if not runner.available:
        return Provision(runtime.id, runtime.target, False, "host", runner.reason)
    env = (
        ("VK_DRIVER_FILES", str(icd)),
        ("VK_ICD_FILENAMES", str(icd)),
        ("CROSSTL_CONFORMANCE_VKRUN", runner.command[0]),
    )
    identity, reason = _probe_vulkan(env)
    return _checked(runtime, "host", identity, reason, env)


@functools.lru_cache(maxsize=None)
def provision(runtime):
    """Decide where a runtime executes on this machine, or why it cannot."""
    runtime = RUNTIMES[runtime] if isinstance(runtime, str) else runtime
    system = platform.system()
    if runtime.adapter == "vulkan":
        return _vulkan(runtime)
    if runtime.adapter == "metal":
        if system != "Darwin":
            return Provision(
                runtime.id, runtime.target, False, None, "Metal requires macOS"
            )
        module = python_module("Metal")
        if not module.available:
            return Provision(runtime.id, runtime.target, False, "host", module.reason)
        identity, reason = _probe_python("metal")
        return _checked(runtime, "host", identity, reason)
    if runtime.adapter == "d3d12":
        if system != "Windows":
            return Provision(
                runtime.id,
                runtime.target,
                False,
                None,
                "Direct3D 12 WARP requires Windows",
            )
        module = python_module("compushady")
        if not module.available:
            return Provision(runtime.id, runtime.target, False, "host", module.reason)
        identity, reason = _probe_python("d3d12")
        return _checked(runtime, "host", identity, reason)
    if (
        system == "Linux"
        and _mesa_mode() != "container"
        and python_module("moderngl").available
    ):
        identity, reason = _probe_python("opengl", MESA_ENV)
        if identity is not None or _mesa_mode() == "host":
            return _checked(runtime, "host", identity, reason, MESA_ENV)
    if _mesa_mode() == "host":
        return Provision(
            runtime.id, runtime.target, False, "host", "host Mesa OpenGL is unavailable"
        )
    image = mesa_image()
    if not image.available:
        return Provision(
            runtime.id,
            runtime.target,
            False,
            None,
            f"no host Mesa OpenGL and no Mesa container: {image.reason}",
        )
    identity, reason = _probe_python("opengl", container=image.command)
    return _checked(runtime, "container", identity, reason, (), image.command)


def _result(path, status):
    result_path = Path(path).with_suffix(".result.json")
    if status == "accepted" and result_path.is_file():
        return json.loads(result_path.read_text())
    return {
        "status": status if status in {"timeout", "unavailable"} else "worker-error"
    }


def execute(prepared, jobs, root, *, timeout=60):
    """Run execution jobs (paths of job JSON files under ``root``) and return their results in order."""
    if not prepared.available:
        return [{"status": "unavailable", "message": prepared.reason} for _ in jobs]
    worker = ["-m", "tests.conformance.worker"]
    if prepared.mode == "host":
        env = {**os.environ, **dict(prepared.env), "PYTHONPATH": str(REPOSITORY)}
        outcomes = []
        for job in jobs:
            process = run([sys.executable, *worker, job], timeout=timeout, env=env)
            outcome = _result(job, process.status)
            if outcome["status"] == "worker-error":
                outcome["process"] = process.to_json()
            outcomes.append(outcome)
        return outcomes
    lines = ["#!/bin/sh"]
    for job in jobs:
        quoted = shlex.quote(str(job))
        lines.append(
            f"timeout -s KILL {int(timeout)} python3 {' '.join(worker)} {quoted} "
            f"> {quoted}.log 2>&1; echo $? > {quoted}.rc"
        )
    script = Path(root) / f"execute-{prepared.runtime}.sh"
    script.write_text("\n".join(lines) + "\n")
    batch = run(
        _container_command(prepared.container, Path(root), "sh", str(script)),
        timeout=120 + timeout * len(jobs),
    )
    outcomes = []
    for job in jobs:
        code = Path(f"{job}.rc")
        status = int(code.read_text().strip()) if code.is_file() else None
        outcome = _result(
            job,
            (
                "accepted"
                if status == 0
                else "timeout" if status == 137 else "worker-error"
            ),
        )
        if outcome["status"] == "worker-error":
            log = Path(f"{job}.log")
            outcome["process"] = {
                "batch": batch.status,
                "returncode": status,
                "log": log.read_text()[-3000:] if log.is_file() else batch.log[-3000:],
            }
        outcomes.append(outcome)
    return outcomes
